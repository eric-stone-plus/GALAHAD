"""Paper session engine: bars → strategy targets → risk → book → journal."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from galahad_futures.book import FuturesPaperBook
from galahad_futures.data import load_bars, sample_kind_for_source
from galahad_futures.risk import RiskConfig, RiskGate
from galahad_futures.strategy import build_strategy, strategy_kwargs_from_config


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _risk_from_cfg(cfg: dict[str, Any]) -> RiskConfig:
    risk_cfg_raw = dict(cfg.get("risk") or {})
    mode = str(cfg.get("mode", "paper")).lower()
    return RiskConfig(
        max_order_notional=float(risk_cfg_raw.get("max_order_notional", 5000)),
        max_position_notional=float(risk_cfg_raw.get("max_position_notional", 15000)),
        max_daily_loss=float(risk_cfg_raw.get("max_daily_loss", 500)),
        max_leverage=float(cfg.get("max_leverage", 5.0)),
        max_drawdown_pct=float(risk_cfg_raw.get("max_drawdown_pct", 0.15)),
        kill_switch=bool(risk_cfg_raw.get("kill_switch", True)),
        enable_live=bool(risk_cfg_raw.get("enable_live", False)),
        mode=mode,
    )


def run_paper_on_bars(
    bars: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    symbol: str | None = None,
    strategy_name: str | None = None,
    strategy_kwargs: dict[str, Any] | None = None,
    evaluate_from: int = 0,
) -> dict[str, Any]:
    """Core loop: targets → risk → book on an in-memory OHLCV frame.

    evaluate_from: only count fills/equity scoring from this row index onward
    (warmup history still traded for continuity, but OOS metrics use the tail).
    """
    symbol = symbol or str(cfg.get("symbol", "BTCUSDT"))
    strat_cfg = dict(cfg.get("strategy") or {})
    name = strategy_name or str(strat_cfg.get("name", "dual_ma"))
    kw = strategy_kwargs if strategy_kwargs is not None else strategy_kwargs_from_config(strat_cfg)
    strategy = build_strategy(name, **kw)
    targets = strategy.targets(bars)

    risk_cfg = _risk_from_cfg(cfg)
    book = FuturesPaperBook(
        wallet=float(cfg.get("initial_equity", 10_000)),
        fee_bps=float(cfg.get("fee_bps", 4.0)),
        maintenance_margin_rate=float(cfg.get("maintenance_margin_rate", 0.005)),
        funding_rate_per_bar=float(cfg.get("funding_rate_per_bar", 0.0)),
        default_leverage=float(cfg.get("default_leverage", 3.0)),
        max_leverage=float(cfg.get("max_leverage", 5.0)),
    )
    book.set_leverage(symbol, float(cfg.get("default_leverage", 3.0)))
    gate = RiskGate(config=risk_cfg, day_start_equity=book.wallet)

    oos_fills = 0
    risk_decisions: list[dict[str, Any]] = []
    for i, row in bars.iterrows():
        ts = str(row["ts"])
        mark = float(row["close"])
        marks = {symbol: mark}
        # Mark equity before any trade this bar — invalidate first so we never
        # open/add risk on a bar that already breaches max_drawdown_pct.
        pre_eq = book.equity(marks)
        gate.update_equity(pre_eq, ts=ts)
        raw_target = float(targets.iloc[i]) if i < len(targets) else 0.0
        pos = book.position(symbol)
        qty_before = pos.qty
        decision = gate.filter_target(
            symbol=symbol,
            target_signed_leverage=raw_target,
            mark=mark,
            equity=max(pre_eq, 1e-9),
            current_qty=pos.qty,
            leverage=pos.leverage,
            ts=ts,
        )
        risk_decisions.append(
            {
                "ts": ts,
                "raw_target": raw_target,
                "allowed": decision.allowed,
                "final_target": decision.target_signed_leverage,
                "reason": decision.reason,
                "clipped": decision.clipped,
                "invalidated": gate.invalidated,
                "pre_trade_equity": pre_eq,
                "pre_trade_drawdown": gate.current_drawdown(pre_eq),
            }
        )
        n_fills_before = len(book.fills)
        if decision.allowed:
            book.apply_target(
                symbol,
                decision.target_signed_leverage,
                mark,
                ts=ts,
                note=f"{name}:{decision.reason}",
            )
        if int(i) >= int(evaluate_from) and len(book.fills) > n_fills_before:
            oos_fills += len(book.fills) - n_fills_before
        book.mark_to_market(marks, ts=ts)
        # Funding/wallet changes after MTM also update peak and running max DD
        gate.update_equity(book.equity(marks), ts=ts)
        if book.liquidated:
            break

    final_marks = {symbol: float(bars.iloc[-1]["close"])}
    if book.equity_curve:
        final_equity = book.equity_curve[-1]["equity"]
        oos_curve = [
            s for j, s in enumerate(book.equity_curve) if j >= int(evaluate_from)
        ]
    else:
        final_equity = book.equity(final_marks)
        oos_curve = []

    eq_series = [float(s["equity"]) for s in (oos_curve or book.equity_curve)]
    rets = []
    if len(eq_series) >= 2:
        import numpy as np

        a = np.asarray(eq_series, dtype=float)
        rets = (np.diff(a) / np.maximum(a[:-1], 1e-9)).tolist()

    peak = float(gate.peak_equity)
    # Session peak-to-trough max, not drawdown from peak at final equity alone
    max_dd = float(gate.max_drawdown_seen)

    return {
        "strategy": name,
        "strategy_kwargs": dict(kw),
        "symbol": symbol,
        "bars": len(bars),
        "evaluate_from": int(evaluate_from),
        "n_fills": len(book.fills),
        "n_fills_oos": oos_fills,
        "n_risk_rejects": len(gate.rejects),
        "liquidated": book.liquidated,
        "invalidated": gate.invalidated,
        "invalidation_reason": gate.invalidation_reason or None,
        "invalidation_events": list(gate.invalidation_events),
        "peak_equity": peak,
        "max_drawdown": max_dd,
        "initial_equity": float(cfg.get("initial_equity", 10_000)),
        "final_equity": float(final_equity),
        "equity_curve": book.equity_curve,
        "equity_curve_len": len(book.equity_curve),
        "oos_equity_curve_len": len(oos_curve),
        "returns_oos": rets,
        "total_funding": float(book.total_funding),
        "n_funding_events": len(book.funding_events),
        "funding_events": book.funding_events,
        "fills": [asdict(f) for f in book.fills],
        "risk_rejects": gate.rejects,
        "risk_decisions": risk_decisions,
        "risk_decisions_tail": risk_decisions[-20:],
        "liquidation_events": book.liquidation_events,
        "positions": book.to_dict(final_marks)["positions"],
        "book": book,
    }


def run_paper_session(
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
    force_source: str | None = None,
    output_dir: str | Path | None = None,
    force_strategy: str | None = None,
    force_symbol: str | None = None,
    force_lookback: int | None = None,
) -> dict[str, Any]:
    """Run one paper session. Returns summary dict; writes journal under output/."""
    root = project_root()
    cfg = config if config is not None else load_config(config_path)

    symbol = force_symbol or str(cfg.get("symbol", "BTCUSDT"))
    interval = str(cfg.get("interval", "1h"))
    bar_limit = int(cfg.get("bar_limit", 120))
    data_cfg = dict(cfg.get("data") or {})
    source = force_source or str(data_cfg.get("source", "auto"))
    fixture = data_cfg.get("fixture_path", "data/fixtures/btcusdt_1h.csv")
    rest_tmpl = data_cfg.get("rest_url_template") or None
    fetch_limit = int(data_cfg.get("fetch_limit", max(bar_limit, 500)))

    bars, source_used, data_note = load_bars(
        source=source,
        fixture_path=fixture,
        rest_url=None,
        rest_timeout=float(data_cfg.get("rest_timeout_sec", 12)),
        project_root=root,
        symbol=symbol,
        interval=interval,
        limit=fetch_limit,
        rest_url_template=rest_tmpl,
    )
    if len(bars) > bar_limit:
        bars = bars.iloc[-bar_limit:].reset_index(drop=True)
    sample_kind = sample_kind_for_source(source_used)

    strat_cfg = dict(cfg.get("strategy") or {})
    strat_name = force_strategy or str(strat_cfg.get("name", "dual_ma"))
    strat_kw = strategy_kwargs_from_config(strat_cfg)
    # tsmom_long is a fixed 7d (168×1h) preset — do not inherit short lookback from config
    sn = strat_name.lower().replace("-", "_")
    if sn in ("tsmom_long", "tsmom_7d", "tsmom_168"):
        if force_lookback is None:
            strat_kw["lookback"] = 168
        strat_kw.setdefault("max_target_leverage", 1.0)
        if strat_cfg.get("name", "").lower() not in ("tsmom_long", "tsmom_7d", "tsmom_168"):
            # config was short TSMOM; use long preset leverage default
            strat_kw["max_target_leverage"] = float(
                strat_kw.get("max_target_leverage") or 1.0
            )
            if force_lookback is None:
                strat_kw["max_target_leverage"] = 1.0
    if force_lookback is not None:
        strat_kw["lookback"] = int(force_lookback)
    result = run_paper_on_bars(
        bars,
        cfg,
        symbol=symbol,
        strategy_name=strat_name,
        strategy_kwargs=strat_kw,
    )

    out_dir = Path(output_dir) if output_dir else root / str(cfg.get("output_dir", "output"))
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    funding_rate = float(cfg.get("funding_rate_per_bar", 0.0))
    summary: dict[str, Any] = {
        "run_id": run_id,
        "mode": str(cfg.get("mode", "paper")).lower(),
        "strategy": strat_name,
        "strategy_kwargs": result.get("strategy_kwargs"),
        "symbol": symbol,
        "interval": interval,
        "bars": len(bars),
        "source_used": source_used,
        "sample_kind": sample_kind,
        "data_note": data_note,
        "n_fills": result["n_fills"],
        "n_risk_rejects": result["n_risk_rejects"],
        "liquidated": result["liquidated"],
        "invalidated": result.get("invalidated", False),
        "invalidation_reason": result.get("invalidation_reason"),
        "peak_equity": result.get("peak_equity"),
        "max_drawdown": result.get("max_drawdown"),
        "initial_equity": result["initial_equity"],
        "final_equity": result["final_equity"],
        "equity_curve_len": result["equity_curve_len"],
        "funding_rate_per_bar": funding_rate,
        "total_funding": result["total_funding"],
        "n_funding_events": result["n_funding_events"],
        "status": "ok",
    }
    if summary["n_fills"] == 0 and not result["liquidated"]:
        summary["status"] = "no-trade but risk-idle OK"
        summary["idle_reason"] = "strategy flat or risk blocked all targets"
    if result.get("invalidated"):
        summary["status"] = "ok_invalidated" if summary["n_fills"] else summary["status"]

    journal = {
        "summary": summary,
        "fills": result["fills"],
        "equity_curve": result["equity_curve"],
        "risk_rejects": result["risk_rejects"],
        "risk_decisions_tail": result["risk_decisions_tail"],
        "liquidation_events": result["liquidation_events"],
        "invalidation_events": result.get("invalidation_events") or [],
        "funding_events": result["funding_events"],
        "positions": result["positions"],
    }

    journal_path = out_dir / f"paper_journal_{run_id}.json"
    summary_path = out_dir / "paper_last_summary.json"
    with journal_path.open("w", encoding="utf-8") as f:
        json.dump(journal, f, indent=2, ensure_ascii=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if result["equity_curve"]:
        eq_df = pd.DataFrame(result["equity_curve"])
        eq_path = out_dir / f"equity_curve_{run_id}.csv"
        eq_df.to_csv(eq_path, index=False)
        summary["equity_curve_path"] = str(eq_path)

    summary["journal_path"] = str(journal_path)
    summary["summary_path"] = str(summary_path)
    return summary

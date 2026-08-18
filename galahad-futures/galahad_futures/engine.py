"""Paper session engine (reference backend): bars → targets → risk → book → journal.

This module hosts the *paper* execution backend. The per-bar decision
logic lives in ``decision.SessionRisk`` and is shared verbatim with the
NautilusTrader backend (``nautilus_backend``); only execution mechanics
differ. The reference book is the arbiter in parity runs.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from galahad_futures.book import FuturesPaperBook
from galahad_futures.data import load_bars, sample_kind_for_source
from galahad_futures.decision import SessionRisk
from galahad_futures.report import build_summary, write_journal
from galahad_futures.strategy import build_strategy, strategy_kwargs_from_config

ENGINE_NAME = "paper"
ENGINE_VERSION = "galahad-futures.book.v1"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    book = FuturesPaperBook(
        wallet=float(cfg.get("initial_equity", 10_000)),
        fee_bps=float(cfg.get("fee_bps", 4.0)),
        maintenance_margin_rate=float(cfg.get("maintenance_margin_rate", 0.005)),
        funding_rate_per_bar=float(cfg.get("funding_rate_per_bar", 0.0)),
        default_leverage=float(cfg.get("default_leverage", 3.0)),
        max_leverage=float(cfg.get("max_leverage", 5.0)),
    )
    book.set_leverage(symbol, float(cfg.get("default_leverage", 3.0)))
    session = SessionRisk.from_config(cfg, start_equity=book.wallet)
    gate = session.gate

    oos_fills = 0
    for i, row in bars.iterrows():
        ts = str(row["ts"])
        mark = float(row["close"])
        marks = {symbol: mark}
        # Mark equity before any trade this bar — invalidate first so we never
        # open/add risk on a bar that already breaches max_drawdown_pct.
        pre_eq = book.equity(marks)
        session.update_equity(pre_eq, ts=ts)
        raw_target = float(targets.iloc[i]) if i < len(targets) else 0.0
        pos = book.position(symbol)
        qty_before = pos.qty
        decision = session.evaluate_target(
            symbol=symbol,
            raw_target=raw_target,
            mark=mark,
            pre_trade_equity=pre_eq,
            current_qty=pos.qty,
            leverage=pos.leverage,
            ts=ts,
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
        session.update_equity(book.equity(marks), ts=ts)
        if book.liquidated:
            session.note_liquidation(ts=ts)
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
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
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
        "loss_halt_events": list(gate.loss_halt_events),
        "decision_phase_final": session.phase(),
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
        "risk_decisions": session.risk_decisions,
        "risk_decisions_tail": session.risk_decisions[-20:],
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
    engine: str | None = None,
) -> dict[str, Any]:
    """Run one paper session. Returns summary dict; writes journal under output/.

    engine: "paper" (default) | "nautilus". The nautilus backend requires
    the optional nautilus_trader dependency; a missing package raises a
    clear usage error (never a silent fallback).
    """
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

    engine_name = (engine or cfg.get("engine") or ENGINE_NAME).lower()
    if engine_name in ("paper", "reference", "book"):
        result = run_paper_on_bars(
            bars,
            cfg,
            symbol=symbol,
            strategy_name=strat_name,
            strategy_kwargs=strat_kw,
        )
        engine_tag, engine_ver = ENGINE_NAME, ENGINE_VERSION
    elif engine_name == "nautilus":
        from galahad_futures.nautilus_backend import (
            ENGINE_NAME as NAUTILUS_ENGINE_NAME,
            ENGINE_VERSION as NAUTILUS_ENGINE_VERSION,
            run_nautilus_on_bars,
        )

        result = run_nautilus_on_bars(
            bars,
            cfg,
            symbol=symbol,
            strategy_name=strat_name,
            strategy_kwargs=strat_kw,
        )
        engine_tag, engine_ver = NAUTILUS_ENGINE_NAME, NAUTILUS_ENGINE_VERSION
    else:
        raise ValueError(f"unknown engine: {engine_name!r} (expected paper | nautilus)")

    out_dir = Path(output_dir) if output_dir else root / str(cfg.get("output_dir", "output"))
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    summary = build_summary(
        result,
        cfg=cfg,
        symbol=symbol,
        interval=interval,
        source_used=source_used,
        sample_kind=sample_kind,
        data_note=data_note,
        out_dir=out_dir,
        engine=engine_tag,
        engine_version=engine_ver,
        strategy_name=strat_name,
    )
    summary = write_journal(summary, result, out_dir)
    return summary

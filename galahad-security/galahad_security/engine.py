"""Paper session engine (reference backend): bars → targets → risk → book → journal.

This module hosts the *paper* execution backend (offline reference cash
book). The per-bar decision logic lives in ``decision.SessionRisk`` and
is shared verbatim with the Alpaca paper venue backend
(``venue_alpaca``); only execution mechanics differ. The reference book
is the arbiter in venue reconciliation runs.

Multi-symbol: per-symbol daily bars are aligned on their common trading
calendar (inner join on ts); every bar marks all symbols, and each
symbol's decision in a bar shares the same pre-trade equity snapshot —
intra-bar cash contention resolves in config symbol order (documented
execution mechanics, deterministic).
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from galahad_security.book import CashEquityBook, tca_from_fills
from galahad_security.data import load_bars, sample_kind_for_source
from galahad_security.decision import SessionRisk
from galahad_security.report import build_summary, write_journal
from galahad_security.strategy import build_strategy, strategy_kwargs_from_config

ENGINE_NAME = "paper"
ENGINE_VERSION = "galahad-security.book.v1"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cost_bps(cfg: dict[str, Any]) -> tuple[float, float]:
    """costs.spread_bps / costs.impact_bps — hard error on invalid values.

    Both default 0.0 (opt-in): execution price == arrival price, which is
    bit-identical to pre-TCA behavior.
    """
    costs = dict(cfg.get("costs") or {})
    try:
        spread = float(costs.get("spread_bps", 0.0))
        impact = float(costs.get("impact_bps", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"costs.spread_bps/impact_bps must be numbers (got {costs!r})"
        ) from exc
    for name, value in (("spread_bps", spread), ("impact_bps", impact)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"costs.{name} must be a non-negative finite number (got {value!r})"
            )
    return spread, impact


def _align_bars(
    bars_by_symbol: dict[str, pd.DataFrame], symbols: list[str]
) -> tuple[list[str], dict[str, dict[str, pd.Series]]]:
    """Align per-symbol bars on the common trading calendar (inner join).

    Returns (sorted common ts list, {symbol: {ts: row}}). An empty
    intersection is a hard error — comparing symbols across disjoint
    calendars would be silently wrong.
    """
    calendars = []
    rows: dict[str, dict[str, pd.Series]] = {}
    for sym in symbols:
        df = bars_by_symbol[sym]
        rows[sym] = {str(r["ts"]): r for _, r in df.iterrows()}
        calendars.append(set(rows[sym]))
    common = sorted(set.intersection(*calendars)) if calendars else []
    if not common:
        raise ValueError(f"no common trading days across symbols {symbols}")
    return common, rows


def run_paper_on_bars(
    bars_by_symbol: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    *,
    symbols: list[str] | None = None,
    strategy_name: str | None = None,
    strategy_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Core loop: targets → risk → book on in-memory daily OHLCV frames."""
    symbols = list(symbols or cfg.get("symbols") or ["AAPL"])
    strat_cfg = dict(cfg.get("strategy") or {})
    name = strategy_name or str(strat_cfg.get("name", "dual_ma"))
    kw = strategy_kwargs if strategy_kwargs is not None else strategy_kwargs_from_config(strat_cfg)
    strategy = build_strategy(name, **kw)
    raw_weight: dict[str, dict[str, float]] = {}
    for sym in symbols:
        df = bars_by_symbol[sym]
        targets = strategy.targets(df)
        raw_weight[sym] = {
            str(row["ts"]): float(targets.iloc[j])
            for j, (_, row) in enumerate(df.iterrows())
        }

    common_ts, rows = _align_bars(bars_by_symbol, symbols)

    spread_bps, impact_bps = _cost_bps(cfg)
    book = CashEquityBook(
        cash=float(cfg.get("initial_equity", 100_000)),
        fee_bps=float(cfg.get("fee_bps", 0.0)),
        spread_bps=spread_bps,
        impact_bps=impact_bps,
    )
    session = SessionRisk.from_config(cfg, start_equity=book.cash)
    gate = session.gate

    for ts in common_ts:
        marks = {sym: float(rows[sym][ts]["close"]) for sym in symbols}
        # Mark equity before any trade this bar — invalidate first so we
        # never open/add risk on a bar that already breaches max_drawdown_pct.
        pre_eq = book.equity(marks)
        session.update_equity(pre_eq, ts=ts)
        for sym in symbols:
            mark = marks[sym]
            pos = book.position(sym)
            decision = session.evaluate_weight(
                symbol=sym,
                raw_weight=raw_weight[sym].get(ts, 0.0),
                mark=mark,
                pre_trade_equity=pre_eq,
                current_qty=float(pos.qty),
                ts=ts,
            )
            if decision.allowed:
                book.apply_target_weight(
                    sym,
                    decision.target_weight,
                    mark,
                    ts=ts,
                    note=f"{name}:{decision.reason}",
                    equity=pre_eq,  # the MTM equity the gate approved against
                )
        book.mark_to_market(marks, ts=ts)
        session.update_equity(book.equity(marks), ts=ts)

    final_marks = {sym: float(rows[sym][common_ts[-1]]["close"]) for sym in symbols}
    if book.equity_curve:
        final_equity = book.equity_curve[-1]["equity"]
    else:
        final_equity = book.equity(final_marks)

    eq_series = [float(s["equity"]) for s in book.equity_curve]
    rets: list[float] = []
    if len(eq_series) >= 2:
        import numpy as np

        a = np.asarray(eq_series, dtype=float)
        rets = (np.diff(a) / np.maximum(a[:-1], 1e-9)).tolist()

    fills = [asdict(f) for f in book.fills]
    return {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "strategy": name,
        "strategy_kwargs": dict(kw),
        "symbols": symbols,
        "bars": len(common_ts),
        "n_fills": len(book.fills),
        "n_risk_rejects": len(gate.rejects),
        "invalidated": gate.invalidated,
        "invalidation_reason": gate.invalidation_reason or None,
        "invalidation_events": list(gate.invalidation_events),
        "loss_halt_events": list(gate.loss_halt_events),
        "decision_phase_final": session.phase(),
        "peak_equity": float(gate.peak_equity),
        "max_drawdown": float(gate.max_drawdown_seen),
        "initial_equity": float(cfg.get("initial_equity", 100_000)),
        "final_equity": float(final_equity),
        "equity_curve": book.equity_curve,
        "equity_curve_len": len(book.equity_curve),
        "returns_oos": rets,
        "fills": fills,
        "tca": tca_from_fills(fills),
        "derisk": gate.derisk_summary(),
        "risk_rejects": gate.rejects,
        "risk_decisions": session.risk_decisions,
        "risk_decisions_tail": session.risk_decisions[-20:],
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
    force_symbols: list[str] | None = None,
    engine: str | None = None,
) -> dict[str, Any]:
    """Run one paper session. Returns summary dict; writes journal under output/.

    engine: "paper" (default) | "alpaca_paper". The venue backend requires
    ALPACA_PAPER_API_KEY/ALPACA_PAPER_API_SECRET env credentials AND
    risk.enable_alpaca_paper: true with the kill switch off; a missing
    precondition raises a clear usage error (never a silent fallback).
    """
    root = project_root()
    cfg = config if config is not None else load_config(config_path)

    symbols = list(force_symbols or cfg.get("symbols") or ["AAPL"])
    interval = str(cfg.get("interval", "1d"))
    bar_limit = int(cfg.get("bar_limit", 250))
    data_cfg = dict(cfg.get("data") or {})
    source = force_source or str(data_cfg.get("source", "auto"))
    fixture_seed = int(data_cfg.get("fixture_seed", 42))

    engine_name = (engine or cfg.get("engine") or ENGINE_NAME).lower()

    if engine_name == "alpaca_paper":
        # Venue path: credentials + gate checked before any HTTP, inside
        # venue_alpaca; latest bars come from the Alpaca data API there.
        from galahad_security.venue_alpaca import (
            ENGINE_NAME as VENUE_ENGINE_NAME,
            ENGINE_VERSION as VENUE_ENGINE_VERSION,
            run_venue_once,
        )

        cfg = {**cfg, "mode": "venue"}
        result = run_venue_once(
            cfg,
            symbols=symbols,
            strategy_name=force_strategy,
            bar_limit=bar_limit,
            project_root=root,  # venue bars write through to data/cache/
        )
        engine_tag, engine_ver = VENUE_ENGINE_NAME, VENUE_ENGINE_VERSION
        source_used, sample_kind, data_note = "venue", "venue", None
    elif engine_name in ("paper", "reference", "book"):
        bars_by_symbol, source_used, data_note = load_bars(
            source=source,
            project_root=root,
            symbols=symbols,
            limit=bar_limit,
            fixture_seed=fixture_seed,
        )
        sample_kind = sample_kind_for_source(source_used)
        result = run_paper_on_bars(
            bars_by_symbol,
            cfg,
            symbols=symbols,
            strategy_name=force_strategy,
        )
        engine_tag, engine_ver = ENGINE_NAME, ENGINE_VERSION
    else:
        raise ValueError(
            f"unknown engine: {engine_name!r} (expected paper | alpaca_paper)"
        )

    out_dir = Path(output_dir) if output_dir else root / str(cfg.get("output_dir", "output"))
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    summary = build_summary(
        result,
        cfg=cfg,
        symbols=symbols,
        interval=interval,
        source_used=source_used,
        sample_kind=sample_kind,
        data_note=data_note,
        out_dir=out_dir,
        engine=engine_tag,
        engine_version=engine_ver,
        strategy_name=result.get("strategy", force_strategy or "dual_ma"),
    )
    summary = write_journal(summary, result, out_dir)
    return summary

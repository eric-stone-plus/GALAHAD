#!/usr/bin/env python3
"""Paper↔testnet shadow-run harness.

Runs the same strategy decision stream through the paper reference
engine (offline, on the configured bars) and the Binance USDT-M futures
TESTNET engine (``nautilus_live``: a bounded live session, warmup-seeded
with the same bars), then reconciles fills and positions into a report
(schema ``galahad.shadow.v1``) — the shadow-run evidence artifact for
the delivery platform.

Real testnet execution is env-gated: without ``GALAHAD_TESTNET_IT=1``
(plus the testnet credentials, the nautilus extra, and an open testnet
gate in config) the script prints what is missing and exits non-zero.
Nothing here ever touches mainnet — the backend is testnet-only by
construction (see docs/architecture.md).

Usage:
    python scripts/run_shadow.py --source fixture            # refused: prints preconditions
    GALAHAD_TESTNET_IT=1 \
    BINANCE_TESTNET_API_KEY=... BINANCE_TESTNET_API_SECRET=... \
        python scripts/run_shadow.py --source cache --strategy tsmom --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galahad_futures.data import load_bars, sample_kind_for_source
from galahad_futures.engine import load_config, run_paper_on_bars
from galahad_futures.live_backend import (
    ENGINE_NAME as LIVE_ENGINE,
    ENGINE_VERSION as LIVE_VERSION,
    precheck_gate,
    run_live_testnet_session,
    testnet_credentials,
    testnet_max_minutes,
)
from galahad_futures.strategy import strategy_kwargs_from_config

KNOWN_DIVERGENCES: list[str] = [
    "windows differ: the paper leg evaluates the loaded historical bars; "
    "the testnet leg uses those bars as warmup history only and decides "
    "on live testnet bars arriving inside the bounded session — fills and "
    "positions are not expected to match 1:1 (unlike run_parity.py)",
    "the authoritative execution check is the testnet leg's own "
    "reconciliation block (orders_submitted/orders_filled/"
    "position_mismatch): decision-expected net qty vs venue-reported net "
    "position",
    "a bounded session may complete zero live bars (interval longer than "
    "testnet.max_minutes): a valid no-trade session, not an error",
    "live funding is settled by the venue inside the account balance; "
    "the paper leg applies the configured per-bar funding convention",
]


def _preflight(cfg: dict[str, Any]) -> list[str]:
    """Collect missing preconditions; empty list means the run is allowed."""
    problems: list[str] = []
    if os.environ.get("GALAHAD_TESTNET_IT") != "1":
        problems.append(
            "GALAHAD_TESTNET_IT=1 not set — real testnet execution is env-gated "
            "(this is the deliberate opt-in for order placement on the futures testnet)"
        )
    if importlib.util.find_spec("nautilus_trader") is None:
        problems.append(
            "optional dependency nautilus_trader==1.231.0 not installed "
            "(package extra 'nautilus')"
        )
    try:
        testnet_credentials()
    except RuntimeError as exc:
        problems.append(str(exc))
    try:
        precheck_gate({**cfg, "mode": "testnet"})
    except RuntimeError as exc:
        problems.append(str(exc))
    return problems


def _resolve_bars(
    args: argparse.Namespace, cfg: dict[str, Any], root: Path
) -> tuple[Any, dict[str, Any]]:
    data_cfg = dict(cfg.get("data") or {})
    symbol = str(cfg.get("symbol", "BTCUSDT"))
    interval = str(cfg.get("interval", "1h"))
    bar_limit = int(cfg.get("bar_limit", 120))
    source = args.source or str(data_cfg.get("source", "auto"))
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
    return bars, {
        "symbol": symbol,
        "interval": interval,
        "source_used": source_used,
        "sample_kind": sample_kind_for_source(source_used),
        "data_note": data_note,
    }


def _final_qty(positions: dict[str, Any]) -> float:
    if not positions:
        return 0.0
    return float(next(iter(positions.values())).get("qty", 0.0))


def build_shadow_report(
    cfg: dict[str, Any],
    bars: Any,
    inputs: dict[str, Any],
    *,
    force_strategy: str | None,
    force_lookback: int | None,
) -> dict[str, Any]:
    strat_cfg = dict(cfg.get("strategy") or {})
    strat_name = force_strategy or str(strat_cfg.get("name", "dual_ma"))
    strat_kw = strategy_kwargs_from_config(strat_cfg)
    if force_lookback is not None:
        strat_kw["lookback"] = int(force_lookback)

    paper = run_paper_on_bars(
        bars,
        cfg,
        symbol=inputs["symbol"],
        strategy_name=strat_name,
        strategy_kwargs=strat_kw,
    )
    cfg_live = {**cfg, "mode": "testnet"}
    live = run_live_testnet_session(
        bars,
        cfg_live,
        symbol=inputs["symbol"],
        strategy_name=strat_name,
        strategy_kwargs=strat_kw,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "schema": "galahad.shadow.v1",
        "run_id": run_id,
        "inputs": {
            "symbol": inputs["symbol"],
            "interval": inputs["interval"],
            "warmup_bars": len(bars),
            "source_used": inputs["source_used"],
            "sample_kind": inputs["sample_kind"],
            "strategy": strat_name,
            "strategy_kwargs": strat_kw,
            "testnet_max_minutes": testnet_max_minutes(cfg),
        },
        "engines": {
            "paper": {
                "engine": paper["engine"],
                "engine_version": paper["engine_version"],
                "n_fills": paper["n_fills"],
                "n_risk_rejects": paper["n_risk_rejects"],
                "final_equity": paper["final_equity"],
                "liquidated": paper["liquidated"],
                "invalidated": paper["invalidated"],
                "decision_phase_final": paper.get("decision_phase_final"),
                "final_qty": _final_qty(paper.get("positions") or {}),
            },
            LIVE_ENGINE: {
                "engine": live["engine"],
                "engine_version": live["engine_version"],
                "venue": live["venue"],
                "live_bars": live["bars"],
                "n_fills": live["n_fills"],
                "n_risk_rejects": live["n_risk_rejects"],
                "initial_equity": live["initial_equity"],
                "final_equity": live["final_equity"],
                "equity_source": live["equity_source"],
                "liquidated": live["liquidated"],
                "invalidated": live["invalidated"],
                "decision_phase_final": live.get("decision_phase_final"),
                "session_seconds": live["session_seconds"],
                "positions": live["positions"],
            },
        },
        "reconciliation": live["reconciliation"],
        "position_check": {
            "paper_final_qty": _final_qty(paper.get("positions") or {}),
            "testnet_expected_final_qty": live["expected_final_qty"],
            "testnet_venue_final_qty": live["venue_final_qty"],
            "note": "windows differ (see known_divergences) — the binding "
            "check is reconciliation.position_mismatch on the testnet leg",
        },
        "known_divergences": list(KNOWN_DIVERGENCES),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GALAHAD paper↔testnet shadow run")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--source", default=None, help="override data.source (warmup bars)")
    ap.add_argument("--strategy", default=None, help="override strategy.name")
    ap.add_argument("--lookback", type=int, default=None, help="override TSMOM lookback")
    ap.add_argument("--output-dir", default=None, help="override report output directory")
    ap.add_argument("--json", action="store_true", help="print report JSON only")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    cfg = load_config(args.config)

    problems = _preflight(cfg)
    if problems:
        print("shadow run refused — missing preconditions:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "hint: set risk.enable_testnet: true and risk.kill_switch: false in "
            "config.yaml, export GALAHAD_TESTNET_IT=1 plus "
            "BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET, and install "
            "the 'nautilus' extra. Mainnet keys are never read.",
            file=sys.stderr,
        )
        return 1

    bars, inputs = _resolve_bars(args, cfg, root)
    report = build_shadow_report(
        cfg, bars, inputs, force_strategy=args.strategy, force_lookback=args.lookback
    )

    out_dir = Path(args.output_dir) if args.output_dir else root / str(cfg.get("output_dir", "output"))
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shadow_{report['run_id']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    report["report_path"] = str(path)
    stable = out_dir / "shadow_last.json"
    with stable.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    report["stable_path"] = str(stable)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        e = report["engines"]
        r = report["reconciliation"]
        print("GALAHAD paper↔testnet shadow report")
        print(f"  run_id:               {report['run_id']}")
        print(f"  warmup bars:          {report['inputs']['warmup_bars']} ({report['inputs']['source_used']})")
        print(f"  paper    fills/qty:   {e['paper']['n_fills']} / {e['paper']['final_qty']}")
        print(f"  testnet  fills/bars:  {e[LIVE_ENGINE]['n_fills']} / {e[LIVE_ENGINE]['live_bars']}")
        print(
            f"  reconciliation:       submitted={r['orders_submitted']} "
            f"filled={r['orders_filled']} position_mismatch={r['position_mismatch']}"
        )
        if r["position_mismatch"]:
            print("  WARNING: testnet position mismatch — investigate before any further run")
        print(f"  report:               {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

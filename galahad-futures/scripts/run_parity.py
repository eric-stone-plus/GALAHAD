#!/usr/bin/env python3
"""Dual-engine parity reconciliation report.

Runs the paper reference book and the nautilus backend on the same
bars and configuration, then diffs the decision stream, fills, equity
curves, funding totals, and terminal flags. The report is an evidence
artifact for the delivery platform: divergences are expected only where
execution mechanics differ (see ``known_divergences``), never from
decision drift.

Usage:
    python scripts/run_parity.py [--source fixture] [--strategy tsmom] [--json]
    python scripts/run_parity.py --config config.yaml --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galahad_futures.engine import (
    ENGINE_NAME as PAPER_ENGINE,
    ENGINE_VERSION as PAPER_VERSION,
    load_config,
    run_paper_on_bars,
)
from galahad_futures.nautilus_backend import (
    ENGINE_NAME as NAUTILUS_ENGINE,
    ENGINE_VERSION as NAUTILUS_VERSION,
    _engine_precision,
    run_nautilus_on_bars,
)
from galahad_futures.data import load_bars, sample_kind_for_source
from galahad_futures.strategy import strategy_kwargs_from_config

KNOWN_DIVERGENCES: list[str] = [
    "price/size quantization: the nautilus instrument rounds to the "
    "configured engine_nautilus precisions; the reference book trades "
    "continuous floats",
    "funding: nautilus_trader 1.231.0 backtest has no funding-settlement "
    "path; the nautilus backend applies the reference per-bar convention "
    "in the harness and feeds funding-adjusted equity to the shared gate",
    "account reads inside the nautilus strategy callback are one bar "
    "stale (fills/account events apply after the callback returns); the "
    "backend projects a paper-convention equity curve from its own "
    "submitted deltas for the gate and parity comparison",
    "margin enforcement differs: the reference book caps same-side adds "
    "by available margin; nautilus enforces its margin_init entry check "
    "and its own liquidation machinery",
    "fee timing on flips: the reference book charges close + open legs "
    "in one fill record; nautilus reports one fill per order leg",
]


def _resolve_inputs(args: argparse.Namespace, root: Path) -> tuple[dict[str, Any], Any]:
    cfg = load_config(args.config)
    if args.parity_precision:
        opts = dict(cfg.get("engine_nautilus") or {})
        opts["price_precision"] = int(args.parity_precision)
        opts["size_precision"] = int(args.parity_precision)
        cfg["engine_nautilus"] = opts
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
    return cfg, {
        "bars": bars,
        "symbol": symbol,
        "interval": interval,
        "source_used": source_used,
        "sample_kind": sample_kind_for_source(source_used),
        "data_note": data_note,
    }


def _diff_decisions(paper: dict[str, Any], nautilus: dict[str, Any]) -> dict[str, Any]:
    p = paper["risk_decisions"]
    n = nautilus["risk_decisions"]
    fields = ("raw_target", "allowed", "final_target", "reason")
    diffs = 0
    first_divergence: str | None = None
    for a, b in zip(p, n):
        for k in fields:
            va, vb = a.get(k), b.get(k)
            same = (
                abs(float(va) - float(vb)) <= 1e-9
                if isinstance(va, (int, float)) and isinstance(vb, (int, float))
                else va == vb
            )
            if not same:
                diffs += 1
                if first_divergence is None:
                    first_divergence = str(a.get("ts"))
    return {
        "decision_fields_compared": len(p) * len(fields),
        "decision_field_diffs": diffs,
        "first_divergence_ts": first_divergence,
        "n_decisions": len(p),
    }


def _diff_equity(paper: dict[str, Any], nautilus: dict[str, Any]) -> dict[str, Any]:
    p = {s["ts"]: float(s["equity"]) for s in paper["equity_curve"]}
    n = {s["ts"]: float(s["equity"]) for s in nautilus["equity_curve"]}
    common = sorted(set(p) & set(n))
    if not common:
        return {"overlap": 0, "max_abs_diff": None, "final_diff": None, "max_diff_ts": None}
    diffs = [abs(p[t] - n[t]) for t in common]
    max_i = max(range(len(diffs)), key=diffs.__getitem__)
    return {
        "overlap": len(common),
        "max_abs_diff": float(diffs[max_i]),
        "max_diff_ts": common[max_i],
        "final_diff": float(p[common[-1]] - n[common[-1]]),
    }


def _diff_fills(paper: dict[str, Any], nautilus: dict[str, Any]) -> dict[str, Any]:
    pf = {f["ts"]: f for f in paper["fills"]}
    nf = {f["ts"]: f for f in nautilus["fills"]}
    only_paper = sorted(set(pf) - set(nf))
    only_nautilus = sorted(set(nf) - set(pf))
    max_price_diff = 0.0
    max_qty_diff = 0.0
    for t in sorted(set(pf) & set(nf)):
        max_price_diff = max(max_price_diff, abs(pf[t]["price"] - nf[t]["price"]))
        max_qty_diff = max(max_qty_diff, abs(pf[t]["qty"] - nf[t]["qty"]))
    return {
        "n_fills_paper": len(pf),
        "n_fills_nautilus": len(nf),
        "ts_only_paper": only_paper[:5],
        "ts_only_nautilus": only_nautilus[:5],
        "max_price_diff": float(max_price_diff),
        "max_qty_diff": float(max_qty_diff),
    }


def _detect_boundary_crossing(paper: dict[str, Any], nautilus: dict[str, Any]) -> dict[str, Any]:
    """Flag sessions where the engines land on opposite sides of a risk
    boundary (invalidation trip, daily-loss halt, liquidation).

    This is the boundary-sensitivity finding: execution-accounting
    differences of fractions of a percent can flip a terminal decision
    when the session runs near a trip line. Not an engine bug — a
    threshold-robustness signal.
    """
    crossing: dict[str, Any] = {"detected": False, "kinds": []}
    pd_ = {d["ts"]: d for d in paper["risk_decisions"]}
    nd_ = {d["ts"]: d for d in nautilus["risk_decisions"]}
    for ts in sorted(set(pd_) & set(nd_)):
        a, b = pd_[ts], nd_[ts]
        if a.get("invalidated") != b.get("invalidated"):
            crossing["detected"] = True
            if "invalidation" not in crossing["kinds"]:
                crossing["kinds"].append("invalidation")
                crossing["invalidation"] = {
                    "first_divergence_ts": ts,
                    "paper": {
                        "invalidated": a.get("invalidated"),
                        "dd_headroom": a.get("dd_headroom"),
                        "max_drawdown": paper["max_drawdown"],
                    },
                    "nautilus": {
                        "invalidated": b.get("invalidated"),
                        "dd_headroom": b.get("dd_headroom"),
                        "max_drawdown": nautilus["max_drawdown"],
                    },
                }
        if a.get("loss_halted") != b.get("loss_halted"):
            crossing["detected"] = True
            if "daily_loss" not in crossing["kinds"]:
                crossing["kinds"].append("daily_loss")
                crossing["daily_loss"] = {
                    "first_divergence_ts": ts,
                    "paper_loss_headroom": a.get("loss_headroom"),
                    "nautilus_loss_headroom": b.get("loss_headroom"),
                }
    if paper.get("liquidated") != nautilus.get("liquidated"):
        crossing["detected"] = True
        crossing["kinds"].append("liquidation")
        crossing["liquidation"] = {
            "paper_liquidated": paper.get("liquidated"),
            "nautilus_liquidated": nautilus.get("liquidated"),
        }
    if paper.get("decision_phase_final") != nautilus.get("decision_phase_final"):
        crossing["phase_final"] = {
            "paper": paper.get("decision_phase_final"),
            "nautilus": nautilus.get("decision_phase_final"),
        }
    return crossing


def _sensitivity_scan(cfg: dict[str, Any], inputs: dict[str, Any], strat_name: str, strat_kw: dict[str, Any]) -> dict[str, Any]:
    """Reference-engine outcomes across threshold bands around the
    configured trip lines. Shows whether the session's fate is robust to
    threshold selection — the flip points are the actionable output.
    """
    bars = inputs["bars"]
    dd_pcts = [0.13, 0.14, 0.15, 0.16, 0.17]
    loss_limits = [300.0, 400.0, 500.0, 600.0, 700.0]
    dd_rows = []
    for pct in dd_pcts:
        c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
        c.setdefault("risk", {})["max_drawdown_pct"] = pct
        r = run_paper_on_bars(bars, c, symbol=inputs["symbol"], strategy_name=strat_name, strategy_kwargs=strat_kw)
        dd_rows.append({
            "max_drawdown_pct": pct,
            "final_equity": r["final_equity"],
            "n_fills": r["n_fills"],
            "invalidated": r["invalidated"],
            "decision_phase_final": r.get("decision_phase_final"),
        })
    loss_rows = []
    for limit in loss_limits:
        c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
        c.setdefault("risk", {})["max_daily_loss"] = limit
        r = run_paper_on_bars(bars, c, symbol=inputs["symbol"], strategy_name=strat_name, strategy_kwargs=strat_kw)
        loss_rows.append({
            "max_daily_loss": limit,
            "final_equity": r["final_equity"],
            "n_fills": r["n_fills"],
            "loss_halted": bool(r.get("loss_halt_events") and any(e.get("event") == "halted" for e in r["loss_halt_events"])),
            "decision_phase_final": r.get("decision_phase_final"),
        })
    return {
        "max_drawdown_pct": dd_rows,
        "max_daily_loss": loss_rows,
    }


def build_parity_report(
    cfg: dict[str, Any],
    inputs: dict[str, Any],
    *,
    force_strategy: str | None,
    force_lookback: int | None,
) -> dict[str, Any]:
    bars = inputs["bars"]
    strat_cfg = dict(cfg.get("strategy") or {})
    strat_name = force_strategy or str(strat_cfg.get("name", "dual_ma"))
    strat_kw = strategy_kwargs_from_config(strat_cfg)
    if force_lookback is not None:
        strat_kw["lookback"] = int(force_lookback)

    paper = run_paper_on_bars(
        bars, cfg, symbol=inputs["symbol"], strategy_name=strat_name, strategy_kwargs=strat_kw
    )
    nautilus = run_nautilus_on_bars(
        bars, cfg, symbol=inputs["symbol"], strategy_name=strat_name, strategy_kwargs=strat_kw
    )

    price_precision, size_precision = _engine_precision(cfg)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "schema": "galahad.parity.v1",
        "run_id": run_id,
        "inputs": {
            "symbol": inputs["symbol"],
            "interval": inputs["interval"],
            "bars": len(bars),
            "source_used": inputs["source_used"],
            "sample_kind": inputs["sample_kind"],
            "strategy": strat_name,
            "strategy_kwargs": strat_kw,
            "funding_rate_per_bar": float(cfg.get("funding_rate_per_bar", 0.0)),
            "fee_bps": float(cfg.get("fee_bps", 4.0)),
            "default_leverage": float(cfg.get("default_leverage", 3.0)),
            "engine_nautilus": {
                "price_precision": price_precision,
                "size_precision": size_precision,
            },
        },
        "engines": {
            "paper": {
                "engine": PAPER_ENGINE,
                "engine_version": PAPER_VERSION,
                "n_fills": paper["n_fills"],
                "final_equity": paper["final_equity"],
                "total_funding": paper["total_funding"],
                "max_drawdown": paper["max_drawdown"],
                "peak_equity": paper["peak_equity"],
                "liquidated": paper["liquidated"],
                "invalidated": paper["invalidated"],
                "decision_phase_final": paper.get("decision_phase_final"),
            },
            "nautilus": {
                "engine": NAUTILUS_ENGINE,
                "engine_version": NAUTILUS_VERSION,
                "n_fills": nautilus["n_fills"],
                "final_equity": nautilus["final_equity"],
                "total_funding": nautilus["total_funding"],
                "max_drawdown": nautilus["max_drawdown"],
                "peak_equity": nautilus["peak_equity"],
                "liquidated": nautilus["liquidated"],
                "invalidated": nautilus["invalidated"],
                "decision_phase_final": nautilus.get("decision_phase_final"),
                "orders_submitted": nautilus.get("orders_submitted"),
                "orders_filled": nautilus.get("orders_filled"),
            },
        },
        "diffs": {
            "decisions": _diff_decisions(paper, nautilus),
            "equity": _diff_equity(paper, nautilus),
            "fills": _diff_fills(paper, nautilus),
            "total_funding_diff": float(paper["total_funding"] - nautilus["total_funding"]),
            "final_equity_diff": float(paper["final_equity"] - nautilus["final_equity"]),
        },
        "boundary_crossing": _detect_boundary_crossing(paper, nautilus),
        "sensitivity": _sensitivity_scan(cfg, inputs, strat_name, strat_kw),
        "known_divergences": list(KNOWN_DIVERGENCES),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GALAHAD dual-engine parity report")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--source", default=None, help="override data.source")
    ap.add_argument("--strategy", default=None, help="override strategy.name")
    ap.add_argument("--lookback", type=int, default=None, help="override TSMOM lookback")
    ap.add_argument(
        "--parity-precision",
        type=int,
        default=None,
        help="override price/size precision for the nautilus instrument "
        "(defaults to engine_nautilus config; raise to isolate "
        "execution-mechanics divergence from venue quantization)",
    )
    ap.add_argument("--output-dir", default=None, help="override report output directory")
    ap.add_argument("--json", action="store_true", help="print report JSON only")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    cfg, inputs = _resolve_inputs(args, root)
    report = build_parity_report(cfg, inputs, force_strategy=args.strategy, force_lookback=args.lookback)

    out_dir = Path(args.output_dir) if args.output_dir else root / str(cfg.get("output_dir", "output"))
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"parity_{report['run_id']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    report["report_path"] = str(path)
    # Stable pointer for operators who declare the report as review
    # evidence (STAMMTISCH brief stage evidence_roots): the review
    # product snapshots the file the operator points to.
    stable = out_dir / "parity_last.json"
    with stable.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    report["stable_path"] = str(stable)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        e = report["engines"]
        d = report["diffs"]
        print("GALAHAD dual-engine parity report")
        print(f"  run_id:                {report['run_id']}")
        print(f"  bars / strategy:       {report['inputs']['bars']} x {report['inputs']['strategy']}")
        print(f"  paper   final_equity:  {e['paper']['final_equity']:.4f}  fills={e['paper']['n_fills']}")
        print(f"  nautilus final_equity: {e['nautilus']['final_equity']:.4f}  fills={e['nautilus']['n_fills']}")
        print(f"  equity  max_abs_diff:  {d['equity']['max_abs_diff']}")
        print(f"  equity  final_diff:    {d['equity']['final_diff']}")
        print(f"  decisions field diffs: {d['decisions']['decision_field_diffs']} / {d['decisions']['decision_fields_compared']}")
        print(f"  report:                {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

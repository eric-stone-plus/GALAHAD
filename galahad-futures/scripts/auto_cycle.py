#!/usr/bin/env python3
"""Automated paper trading cycle: perception → signal → paper → report.

Self-contained, no external dependencies beyond galahad_futures.

Usage:
  quant-python scripts/auto_cycle.py                    # single cycle
  quant-python scripts/auto_cycle.py --strategy rsi     # override strategy
  quant-python scripts/auto_cycle.py --symbols ETHUSDT SOLUSDT  # multi-symbol
  quant-python scripts/auto_cycle.py --dry-run          # perception only, no paper

Exit codes:
  0  success (including a --dry-run whose perception succeeded)
  1  halted (HALT file present)
  2  fetch failed (no prices)
  3  internal error
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galahad_futures.data import load_bars, sample_kind_for_source
from galahad_futures.engine import run_paper_on_bars
from galahad_futures.perception import build_snapshot, write_snapshot
from galahad_futures.risk import RiskConfig, RiskGate
from galahad_futures.strategy import build_strategy, strategy_kwargs_from_config

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_halt(halt_path: Path) -> tuple[bool, str]:
    """Check the exact HALT file path. Returns (halted, reason)."""
    if halt_path.is_file():
        reason = halt_path.read_text(encoding="utf-8", errors="replace").strip()
        return True, reason or "HALT file present"
    return False, ""


def write_cycle_log(state_dir: Path, entry: dict) -> None:
    """Append one JSON line to cycle_log.jsonl."""
    log_path = state_dir / "cycle_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_snapshot_json(state_dir: Path, data: dict, name: str) -> Path:
    """Write JSON snapshot to state directory."""
    snap_dir = state_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"{name}_{utc_stamp()}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also write as "last" pointer
    last = state_dir / f"last_{name}.json"
    last.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_cycle(
    cfg: dict,
    *,
    state_dir: Path,
    force_strategy: str | None = None,
    force_symbols: list[str] | None = None,
    force_source: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run one complete cycle. Returns summary dict."""
    ts = utc_now()
    stamp = utc_stamp()
    symbol = cfg.get("symbol", "BTCUSDT")
    interval = cfg.get("interval", "1h")
    bar_limit = int(cfg.get("bar_limit", 400))
    data_cfg = dict(cfg.get("data") or {})
    source = force_source or str(data_cfg.get("source", "auto"))
    strat_cfg = dict(cfg.get("strategy") or {})
    strat_name = force_strategy or str(strat_cfg.get("name", "tsmom"))

    # Multi-symbol support
    symbols = force_symbols or [symbol]

    # --- Perception ---
    snap = build_snapshot(
        symbols=tuple(symbols),
        rest_timeout=float(data_cfg.get("rest_timeout_sec", 12)),
        force_offline=(source == "fixture"),
    )

    # Write perception snapshot
    perc_path = write_snapshot_json(state_dir, snap.to_dict(), "perception")

    result = {
        "ts": ts,
        "cycle_id": stamp,
        "halted": False,
        "perception": {
            "status": snap.status,
            "source": snap.source,
            "n_symbols": len(snap.symbols),
            "symbols": snap.symbols,
            "fetch_error": snap.fetch_error,
        },
        "paper_runs": [],
        "summary": "",
    }

    if snap.status == "fetch_failed" and not snap.symbols:
        result["summary"] = "FETCH FAILED — no prices available"
        write_cycle_log(state_dir, result)
        return result

    if dry_run:
        result["summary"] = f"DRY RUN — perception OK, {len(snap.symbols)} symbols"
        write_cycle_log(state_dir, result)
        return result

    # --- Paper runs per symbol ---
    strat_kw = strategy_kwargs_from_config(strat_cfg)
    risk_cfg = cfg.get("risk") or {}

    for sym in symbols:
        paper_result = {
            "symbol": sym,
            "strategy": strat_name,
            "status": "skipped",
        }

        try:
            bars, source_used, data_note = load_bars(
                source=source,
                fixture_path=data_cfg.get("fixture_path", "data/fixtures/btcusdt_1h.csv"),
                rest_timeout=float(data_cfg.get("rest_timeout_sec", 12)),
                project_root=ROOT,
                symbol=sym,
                interval=interval,
                limit=int(data_cfg.get("fetch_limit", max(bar_limit, 500))),
            )
            if len(bars) > bar_limit:
                bars = bars.iloc[-bar_limit:].reset_index(drop=True)

            if len(bars) < 50:
                paper_result["status"] = "insufficient_bars"
                paper_result["n_bars"] = len(bars)
                result["paper_runs"].append(paper_result)
                continue

            paper_cfg = {
                "mode": "paper",
                "symbol": sym,
                "interval": interval,
                "initial_equity": float(cfg.get("initial_equity", 10_000)),
                "fee_bps": float(cfg.get("fee_bps", 4.0)),
                "max_leverage": float(cfg.get("max_leverage", 5.0)),
                "default_leverage": float(cfg.get("default_leverage", 3.0)),
                "maintenance_margin_rate": float(cfg.get("maintenance_margin_rate", 0.005)),
                "funding_rate_per_bar": float(cfg.get("funding_rate_per_bar", 0.0001)),
                "risk": {
                    "max_drawdown_pct": float(risk_cfg.get("max_drawdown_pct", 0.15)),
                    "max_order_notional": float(risk_cfg.get("max_order_notional", 5000)),
                    "max_position_notional": float(risk_cfg.get("max_position_notional", 15000)),
                    "max_daily_loss": float(risk_cfg.get("max_daily_loss", 500)),
                    "kill_switch": bool(risk_cfg.get("kill_switch", True)),
                    "enable_live": bool(risk_cfg.get("enable_live", False)),
                },
                "strategy": {**strat_cfg, "name": strat_name, **strat_kw},
            }

            pr = run_paper_on_bars(bars, paper_cfg, symbol=sym, strategy_name=strat_name, strategy_kwargs=strat_kw)

            paper_result["status"] = "ok"
            paper_result["source"] = source_used
            paper_result["bars"] = len(bars)
            paper_result["n_fills"] = pr["n_fills"]
            paper_result["n_risk_rejects"] = pr["n_risk_rejects"]
            paper_result["liquidated"] = pr["liquidated"]
            paper_result["invalidated"] = pr.get("invalidated", False)
            paper_result["invalidation_reason"] = pr.get("invalidation_reason")
            paper_result["initial_equity"] = pr["initial_equity"]
            paper_result["final_equity"] = pr["final_equity"]
            paper_result["return_pct"] = (pr["final_equity"] / pr["initial_equity"] - 1) * 100
            paper_result["peak_equity"] = pr.get("peak_equity")
            paper_result["max_drawdown"] = pr.get("max_drawdown")
            paper_result["total_funding"] = pr.get("total_funding", 0)
            paper_result["n_funding_events"] = pr.get("n_funding_events", 0)

        except Exception as e:
            paper_result["status"] = "error"
            paper_result["error"] = f"{type(e).__name__}: {e}"

        result["paper_runs"].append(paper_result)

    # --- Write outputs ---
    write_snapshot_json(state_dir, result, "cycle")

    # Also write to project output/
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "last_cycle.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Build summary ---
    ok_runs = [r for r in result["paper_runs"] if r["status"] == "ok"]
    if ok_runs:
        parts = []
        for r in ok_runs:
            ret = r.get("return_pct", 0)
            dd = r.get("max_drawdown", 0) * 100 if r.get("max_drawdown") else 0
            inv = " ⚠" if r.get("invalidated") else ""
            parts.append(f"{r['symbol']}: {ret:+.1f}% (dd={dd:.1f}%){inv}")
        result["summary"] = " | ".join(parts)
    elif result["paper_runs"]:
        result["summary"] = f"All runs failed: {[r['status'] for r in result['paper_runs']]}"
    else:
        result["summary"] = "No paper runs executed"

    write_cycle_log(state_dir, result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GALAHAD automated paper trading cycle")
    ap.add_argument("--config", default=None, help="config.yaml path")
    ap.add_argument("--strategy", default=None, help="override strategy name")
    ap.add_argument("--symbols", nargs="+", default=None, help="symbols to trade")
    ap.add_argument(
        "--source", choices=("auto", "fixture", "rest", "cache", "parquet"), default=None
    )
    ap.add_argument("--dry-run", action="store_true", help="perception only, no paper")
    ap.add_argument("--state-dir", default=None, help="state directory (default: futures/state)")
    ap.add_argument("--halt-file", default=None, help="HALT file path")
    ap.add_argument("--json", action="store_true", help="output JSON only")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config) if args.config else None)
    state_dir = Path(args.state_dir) if args.state_dir else ROOT / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # HALT check: honour the exact --halt-file path (default: state/HALT)
    halt_path = Path(args.halt_file) if args.halt_file else state_dir / "HALT"
    halted, halt_reason = check_halt(halt_path)
    if halted:
        msg = {"ts": utc_now(), "halted": True, "halt_reason": halt_reason}
        if args.json:
            print(json.dumps(msg, indent=2))
        else:
            print(f"HALTED: {halt_reason}")
        write_cycle_log(state_dir, msg)
        return 1

    # Run cycle
    result = run_cycle(
        cfg,
        state_dir=state_dir,
        force_strategy=args.strategy,
        force_symbols=args.symbols,
        force_source=args.source,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"GALAHAD Auto Cycle {result['cycle_id']}")
        print(f"  perception: {result['perception']['status']} ({result['perception']['n_symbols']} symbols)")
        for pr in result["paper_runs"]:
            if pr["status"] == "ok":
                print(f"  {pr['symbol']}: {pr['return_pct']:+.1f}% fills={pr['n_fills']} dd={pr.get('max_drawdown',0)*100:.1f}%")
            else:
                print(f"  {pr['symbol']}: {pr['status']}")
        print(f"  summary: {result['summary']}")

    # Return code based on result
    if result.get("halted"):
        return 1
    if not result.get("paper_runs"):
        # --dry-run completes without paper runs by design; that is success
        # (exit 0), not a fetch failure — unless perception actually failed
        if args.dry_run and result["perception"]["status"] != "fetch_failed":
            return 0
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

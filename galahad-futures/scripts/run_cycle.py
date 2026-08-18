#!/usr/bin/env python3
"""One-shot cycle: perception → HALT check → paper (or skip).

Fail-closed discipline:
  - HALT file present → no paper advancement
  - perception still runs
  - news/attention never becomes an order

All cycle state lives under futures/state/ (runtime, gitignored).
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

from galahad_futures.engine import run_paper_session
from galahad_futures.perception import run_perception

DEFAULT_HALT = Path(__file__).resolve().parent.parent / "state" / "HALT"
DEFAULT_OPS = Path(__file__).resolve().parent.parent / "state"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GALAHAD perception+paper cycle (paper-only)")
    ap.add_argument("--offline", action="store_true", help="perception+paper offline/fixture")
    ap.add_argument("--source", choices=("auto", "fixture", "rest"), default=None)
    ap.add_argument("--halt-file", default=str(DEFAULT_HALT))
    ap.add_argument("--ops-state", default=str(DEFAULT_OPS))
    ap.add_argument("--skip-paper", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    halt_path = Path(args.halt_file) if args.halt_file else None
    ops = Path(args.ops_state) if args.ops_state else None
    halted = bool(halt_path and halt_path.is_file())
    halt_reason = ""
    if halted and halt_path:
        halt_reason = halt_path.read_text(encoding="utf-8", errors="replace").strip().split("\n")[0]

    perc = run_perception(
        project_root=ROOT,
        ops_state=ops,
        force_offline=args.offline,
        rest_timeout=8.0,
    )

    paper: dict | None = None
    trade_status = "skipped"
    if args.skip_paper:
        trade_status = "skipped_cli"
    elif halted:
        trade_status = "halted"
        paper = {
            "status": "halted",
            "halt_file": str(halt_path),
            "halt_reason": halt_reason,
            "n_fills": 0,
            "final_equity": None,
            "note": "HALT active — paper not advanced (fail-closed)",
        }
        if ops:
            ops.mkdir(parents=True, exist_ok=True)
            (ops / "last_galahad_paper.json").write_text(
                json.dumps({"ts": utc_now(), **paper}, indent=2), encoding="utf-8"
            )
    else:
        force_src = args.source or ("fixture" if args.offline else "fixture")
        # fixture default for reliable cycle; auto still available via --source auto
        paper = run_paper_session(force_source=force_src)
        trade_status = str(paper.get("status", "ok"))
        if ops:
            ops.mkdir(parents=True, exist_ok=True)
            slim = {
                "ts": utc_now(),
                "status": paper.get("status"),
                "final_equity": paper.get("final_equity"),
                "n_fills": paper.get("n_fills"),
                "equity_curve_len": paper.get("equity_curve_len"),
                "journal_path": paper.get("journal_path"),
                "halted": False,
            }
            (ops / "last_galahad_paper.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")

    cycle = {
        "ts": utc_now(),
        "halted": halted,
        "halt_reason": halt_reason or None,
        "trade_status": trade_status,
        "perception": {
            "status": perc.get("status"),
            "source": perc.get("source"),
            "n_symbols": perc.get("n_symbols"),
            "symbols": perc.get("symbols"),
            "fetch_error": perc.get("fetch_error"),
            "written": perc.get("written"),
        },
        "paper": paper,
    }
    # Always write cycle summary under futures/output
    out = ROOT / "output" / "last_cycle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cycle, indent=2, ensure_ascii=False), encoding="utf-8")
    if ops:
        (Path(ops) / "last_galahad_cycle.json").write_text(
            json.dumps(cycle, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.json:
        print(json.dumps(cycle, indent=2, ensure_ascii=False))
    else:
        print("GALAHAD cycle")
        print(f"  ts:           {cycle['ts']}")
        print(f"  halted:       {halted}")
        if halt_reason:
            print(f"  halt_reason:  {halt_reason[:120]}")
        print(f"  perception:   {perc.get('status')} source={perc.get('source')} n={perc.get('n_symbols')}")
        print(f"  trade_status: {trade_status}")
        if paper and paper.get("final_equity") is not None:
            print(f"  final_equity: {paper.get('final_equity')}")
            print(f"  n_fills:      {paper.get('n_fills')}")
        print(f"  cycle_json:   {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

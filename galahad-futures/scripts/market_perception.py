#!/usr/bin/env python3
"""One-shot market perception entry (prices + optional attention pointers)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galahad_futures.perception import run_perception

DEFAULT_OPS = Path(__file__).resolve().parent.parent / "state"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GALAHAD market perception (no orders)")
    ap.add_argument("--offline", action="store_true", help="skip REST; fixture only")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument(
        "--ops-state",
        default=str(DEFAULT_OPS),
        help="futures state dir (empty to skip)",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    ops = Path(args.ops_state) if args.ops_state else None
    summary = run_perception(
        project_root=ROOT,
        ops_state=ops,
        force_offline=args.offline,
        rest_timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("GALAHAD market perception")
        print(f"  status:     {summary['status']}")
        print(f"  source:     {summary['source']}")
        print(f"  ts:         {summary['ts']}")
        print(f"  symbols:    {summary['symbols']}")
        print(f"  fetch_error:{summary.get('fetch_error')}")
        print(f"  firecrawl:  {summary.get('firecrawl_ok')}")
        for w in summary.get("written", []):
            print(f"  wrote:      {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

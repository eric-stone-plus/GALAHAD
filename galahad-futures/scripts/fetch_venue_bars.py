#!/usr/bin/env python3
"""Fetch venue OHLCV and write durable cache under data/cache/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.data import fetch_and_cache_venue_bars, load_venue_cache


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    bars, src, note, path = fetch_and_cache_venue_bars(
        project_root=ROOT,
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        rest_timeout=args.timeout,
        prefer_vision=True,
    )
    out = {
        "source_used": src,
        "data_note": note,
        "cache_path": str(path) if path else None,
        "n_rows": len(bars),
        "last_close": float(bars["close"].iloc[-1]) if len(bars) else None,
        "first_ts": str(bars["ts"].iloc[0]) if len(bars) else None,
        "last_ts": str(bars["ts"].iloc[-1]) if len(bars) else None,
    }
    if src != "rest":
        cached = load_venue_cache(ROOT, symbol=args.symbol, interval=args.interval)
        if cached:
            out["cache_available"] = True
            out["cache_rows"] = len(cached[0])
        else:
            out["cache_available"] = False

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print("GALAHAD venue bar fetch")
        for k, v in out.items():
            print(f"  {k}: {v}")
    return 0 if src == "rest" or out.get("cache_available") else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Regenerate bundled OHLCV fixture (deterministic)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.data import write_synthetic_fixture


def main() -> int:
    path = ROOT / "data" / "fixtures" / "btcusdt_1h.csv"
    write_synthetic_fixture(path, n=120, start_price=40_000.0, seed=42)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

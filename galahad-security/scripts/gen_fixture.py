#!/usr/bin/env python3
"""Regenerate bundled daily OHLCV fixtures (deterministic)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from galahad_security.data import write_synthetic_fixture  # noqa: E402


def main() -> int:
    with (ROOT / "config.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    symbols = list(cfg.get("symbols") or ["AAPL"])
    seed = int((cfg.get("data") or {}).get("fixture_seed", 42))
    n = int(cfg.get("bar_limit", 250))
    for idx, symbol in enumerate(symbols):
        path = ROOT / "data" / "fixtures" / f"{symbol}_1d.csv"
        write_synthetic_fixture(
            path, n=max(n, 250), start_price=100.0 + 25.0 * idx, seed=seed + idx
        )
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

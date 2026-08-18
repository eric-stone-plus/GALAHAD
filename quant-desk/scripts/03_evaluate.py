#!/usr/bin/env python3
"""Return / risk evaluation (from the ledger equity curve or lifecycle NAV)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, ensure_layout  # noqa: E402

from quantkit.review import equity_performance


def _load_equity() -> pd.Series:
    # prefer lifecycle equity; else marks from paper book
    for name in ("lifecycle_equity.csv", "equity_marks.csv"):
        path = OUT / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "equity" not in df.columns:
            continue
        if "date" in df.columns:
            idx = pd.to_datetime(df["date"])
        elif "ts" in df.columns:
            idx = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
        else:
            idx = pd.RangeIndex(len(df))
        s = pd.Series(df["equity"].astype(float).values, index=idx, name="equity")
        return s.dropna()
    raise FileNotFoundError("no equity series found; run run_lifecycle or 02_trade first")


def main() -> int:
    ensure_layout()
    try:
        eq = _load_equity()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    perf = equity_performance(eq)
    out = perf.as_dict()
    # rolling stats
    rets = eq.pct_change().dropna()
    if len(rets) >= 21:
        out["ret_1m"] = float((1 + rets.tail(21)).prod() - 1)
    if len(rets) >= 63:
        out["ret_3m"] = float((1 + rets.tail(63)).prod() - 1)

    path = OUT / "performance.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    rets.to_csv(OUT / "strategy_returns.csv", header=["return"])
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

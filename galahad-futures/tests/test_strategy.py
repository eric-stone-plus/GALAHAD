"""Strategy unit tests — dual MA targets on known series."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.strategy import DualMAConfig, DualMAStrategy, build_strategy


def test_dual_ma_long_then_short():
    # Strictly increasing then decreasing so MAs cross
    n = 40
    closes = list(range(100, 100 + 20)) + list(range(119, 99, -1))
    assert len(closes) == n
    bars = pd.DataFrame(
        {
            "ts": [f"t{i}" for i in range(n)],
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1.0] * n,
        }
    )
    strat = DualMAStrategy(DualMAConfig(fast=3, slow=5, max_target_leverage=2.0))
    t = strat.targets(bars)
    assert t.iloc[:4].eq(0).all()  # warm-up incomplete for slow=5 → index 0..3 may be nan/0
    # After warm-up in uptrend should be long
    assert t.iloc[10] == pytest.approx(2.0)
    # Late downtrend should be short
    assert t.iloc[-1] == pytest.approx(-2.0)


def test_build_strategy():
    s = build_strategy("dual_ma", fast=5, slow=10, max_target_leverage=1.5)
    assert s.config.fast == 5
    assert s.config.max_target_leverage == 1.5
    t = build_strategy("tsmom", lookback=12)
    assert t.config.lookback == 12
    with pytest.raises(ValueError):
        build_strategy("unknown")

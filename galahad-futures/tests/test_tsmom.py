"""TSMOM strategy targets on fixed series."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.strategy import TSMOMConfig, TSMOMStrategy, build_strategy


def test_tsmom_long_then_short_on_trend():
    # 30 bars up then 30 down; lookback=10
    up = list(range(100, 130))
    down = list(range(129, 99, -1))
    closes = up + down
    n = len(closes)
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
    s = TSMOMStrategy(TSMOMConfig(lookback=10, max_target_leverage=1.5))
    t = s.targets(bars)
    assert t.iloc[:10].eq(0).all()  # warmup
    # mid uptrend after lookback
    assert t.iloc[20] == pytest.approx(1.5)
    # late downtrend: price below 10 bars ago
    assert t.iloc[-1] == pytest.approx(-1.5)


def test_build_tsmom():
    s = build_strategy("tsmom", lookback=24, max_target_leverage=1.0)
    assert s.config.lookback == 24
    assert s.config.max_target_leverage == 1.0


def test_walkforward_splits_smoke():
    from galahad_futures.walkforward import bar_walk_forward_splits

    folds = list(bar_walk_forward_splits(400, n_folds=4, min_train=120))
    assert len(folds) >= 2
    for tr, te in folds:
        assert len(tr) > 0 and len(te) > 0
        assert tr[-1] < te[0] or tr.max() < te.min()

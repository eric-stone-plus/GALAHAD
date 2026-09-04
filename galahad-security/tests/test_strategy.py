"""Strategy tests — the dual_ma null emits long-only target weights."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_security.strategy import build_strategy, strategy_kwargs_from_config


def _px_bars(closes):
    return pd.DataFrame(
        {
            "ts": [f"2026-01-{i + 1:02d}" for i in range(len(closes))],
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1.0,
        }
    )


def test_dual_ma_weights_long_only():
    strat = build_strategy("dual_ma", fast=2, slow=4, target_weight=0.2)
    rising = _px_bars([100.0 + i for i in range(10)])
    t = strat.targets(rising)
    assert t.name == "target_weight"
    assert (t >= 0.0).all() and (t.max() <= 0.2)
    assert t.iloc[:3].tolist() == [0.0, 0.0, 0.0]  # warmup flat
    assert t.iloc[-1] == 0.2  # fast > slow on a rising tail

    falling = _px_bars([110.0 - i for i in range(10)])
    assert (strat.targets(falling) == 0.0).all()  # long-only null goes flat, never short


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("martingale")


def test_strategy_kwargs_from_config():
    kw = strategy_kwargs_from_config(
        {"name": "dual_ma", "fast": 5, "slow": 30, "target_weight": 0.15, "junk": 1}
    )
    assert kw == {"fast": 5, "slow": 30, "target_weight": 0.15}

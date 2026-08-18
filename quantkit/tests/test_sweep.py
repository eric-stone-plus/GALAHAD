"""Tests for quantkit.sweep — vectorbt parameter sweeps (research use).

Contracts:
  - one row per lookback (tsmom) / per (fast, slow) pair (dual-MA)
  - stat columns present and finite on a trending synthetic series
  - fees charged per side: with nonzero fees and identical signals, the
    fee run's total_return is strictly lower than the zero-fee run
  - invalid inputs fail closed (bad price column, lookback < 1, no
    fast < slow pairs)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("vectorbt")

from quantkit.sweep import dual_ma_sweep, tsmom_sweep  # noqa: E402

STAT_COLS = [
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "trades",
    "win_rate",
]


def _synthetic_bars(n: int = 400, drift: float = 0.0008, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    px = 40_000 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": px,
            "high": px * 1.002,
            "low": px * 0.998,
            "close": px,
            "volume": 100.0,
        }
    )


def test_tsmom_sweep_shape_and_finite():
    bars = _synthetic_bars()
    df = tsmom_sweep(bars, [24, 48, 168], fee_rate=0.0004)
    assert list(df.index) == [24, 48, 168]
    assert set(STAT_COLS) <= set(df.columns)
    assert np.isfinite(df["sharpe"]).all()
    assert (df["trades"] > 0).all()


def test_tsmom_sweep_fees_reduce_returns():
    bars = _synthetic_bars()
    free = tsmom_sweep(bars, [48], fee_rate=0.0)
    paid = tsmom_sweep(bars, [48], fee_rate=0.0004)
    assert paid.loc[48, "total_return"] < free.loc[48, "total_return"]
    # Fees never create trades, only pay them.
    assert paid.loc[48, "trades"] == free.loc[48, "trades"]


def test_dual_ma_sweep_grid():
    bars = _synthetic_bars()
    df = dual_ma_sweep(bars, [8, 20], [21, 55], fee_rate=0.0004)
    assert list(df.index) == [(8, 21), (8, 55), (20, 21), (20, 55)]
    assert set(STAT_COLS) <= set(df.columns)


def test_sweep_fail_closed():
    bars = _synthetic_bars()
    with pytest.raises(KeyError):
        tsmom_sweep(bars, [48], price_col="nope")
    with pytest.raises(ValueError):
        tsmom_sweep(bars, [0])
    with pytest.raises(ValueError):
        dual_ma_sweep(bars, [55], [21])  # no fast < slow pairs

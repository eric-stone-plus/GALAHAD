"""Smoke tests for existing quantkit engines (small synthetic frames).

These guard the shared foundation that finance projects build on:
quantkit.backtest.run_long_only, quantkit.portfolio.run_portfolio,
quantkit.cn_market.tradeable_mask.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantkit.backtest import dual_ma_signal, run_long_only
from quantkit.cn_market import CNTradeRules, tradeable_mask
from quantkit.portfolio import equal_weight_targets, run_portfolio


def _trending_close(n: int = 300) -> pd.Series:
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(0.001 + 0.01 * rng.normal(size=n)))
    return pd.Series(px, index=idx, name="close")


def test_run_long_only_always_long_matches_buyhold():
    close = _trending_close()
    signal = pd.Series(1.0, index=close.index)
    bt = run_long_only(close, signal, fee_bps=0.0)
    # always long, no fees → total return ≈ buy & hold (first bar is flat)
    bnh = close.iloc[-1] / close.iloc[0] - 1.0
    assert bt.total_return == pytest.approx(bnh, rel=0.02)
    assert bt.trades >= 1
    assert 0.0 <= bt.win_rate <= 1.0
    assert np.isfinite(bt.sharpe)
    assert -1.0 <= bt.max_drawdown <= 0.0


def test_run_long_only_flat_signal_no_trades():
    close = _trending_close()
    bt = run_long_only(close, pd.Series(0.0, index=close.index))
    assert bt.total_return == pytest.approx(0.0, abs=1e-12)
    assert bt.trades == 0
    assert bt.equity.iloc[-1] == pytest.approx(1.0)


def test_dual_ma_signal_binary_and_aligned():
    close = _trending_close()
    sig = dual_ma_signal(close, fast=5, slow=20)
    assert set(sig.unique()) <= {0.0, 1.0}
    assert sig.index.equals(close.index)


def test_run_portfolio_equal_weight_monthly():
    idx = pd.date_range("2022-01-03", periods=260, freq="B")
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {
            "AAA": 50 * np.exp(np.cumsum(0.0005 + 0.01 * rng.normal(size=260))),
            "BBB": 80 * np.exp(np.cumsum(0.0003 + 0.012 * rng.normal(size=260))),
        },
        index=idx,
    )
    tw = equal_weight_targets(idx, ["AAA", "BBB"])
    res = run_portfolio(prices, tw, rebalance="M", fee_bps=5.0)
    assert np.isfinite(res.sharpe)
    assert res.equity.iloc[-1] > 0
    # weights stay within [0, 1] and sum to ~1 once invested
    assert (res.weights >= -1e-12).all().all()
    invested = res.weights.sum(axis=1)
    assert invested.max() <= 1.0 + 1e-9
    assert res.stats["n_assets"] == 2


def _cn_ohlcv_with_events() -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=6, freq="B")
    close = [10.00, 11.00, 11.00, 10.45, 10.45, 10.66]
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1e6, 1e6, 1e6, 0.0, 1e6, 1e6],
        },
        index=idx,
    )
    return df


def test_tradeable_mask_blocks_limit_up_and_suspension():
    ohlcv = _cn_ohlcv_with_events()
    rules = CNTradeRules(board="main")
    mask_buy = tradeable_mask(ohlcv, rules, side="buy")
    # day 1: close 11.00 = prev 10.00 * 1.10 → main-board limit-up, no opening buys
    assert not bool(mask_buy.iloc[1])
    # day 3: zero volume → suspended → not tradeable
    assert not bool(mask_buy.iloc[3])
    # ordinary days tradeable
    assert bool(mask_buy.iloc[4])
    assert bool(mask_buy.iloc[5])


def test_tradeable_mask_sell_side_blocks_limit_down():
    idx = pd.date_range("2023-02-06", periods=3, freq="B")
    close = [10.00, 9.00, 9.10]
    ohlcv = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1e6},
        index=idx,
    )
    mask_sell = tradeable_mask(ohlcv, CNTradeRules(board="main"), side="sell")
    # day 1: 9.00 = 10.00 * 0.90 → limit-down, cannot exit
    assert not bool(mask_sell.iloc[1])
    assert bool(mask_sell.iloc[2])

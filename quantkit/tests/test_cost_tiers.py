"""Tests for the 2026 A-share cost basis in quantkit.backtest.run_long_only.

Contracts:
  - COST_TIERS is the two-sided all-in rate table (low 0.2% / mid 0.4% /
    high 0.5% of traded notional); higher tier ⇒ strictly lower equity, and
    the per-bar return gap between tiers is exactly turnover × Δrate
  - leaving the new parameters at their defaults reproduces the legacy
    fee_bps/slippage_bps behaviour bit-for-bit
  - cancel_fee_per_order converts to bps as fee / avg_order_notional × 1e4
    (5 CNY at 100k CNY average order size = 0.5 bps) and lowers equity
  - ambiguous / invalid cost configurations are fail-closed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantkit.backtest import COST_TIERS, run_long_only


def _close_and_signal(n: int = 300) -> tuple[pd.Series, pd.Series]:
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(0.001 + 0.01 * rng.normal(size=n)))
    close = pd.Series(px, index=idx, name="close")
    # position flips every 30 bars → a handful of round trips
    signal = pd.Series((np.arange(n) % 60 < 30).astype(float), index=idx)
    return close, signal


def _turnover(close: pd.Series, signal: pd.Series) -> pd.Series:
    pos = signal.reindex(close.index).fillna(0.0).clip(-1.0, 1.0).shift(1).fillna(0.0)
    return pos.diff().abs().fillna(pos.abs())


def test_cost_tiers_ordering_and_exact_magnitude():
    close, signal = _close_and_signal()
    bt = {t: run_long_only(close, signal, cost_tier=t) for t in COST_TIERS}
    # direction: higher two-sided rate ⇒ lower equity
    assert bt["low"].equity.iloc[-1] > bt["mid"].equity.iloc[-1] > bt["high"].equity.iloc[-1]
    # magnitude: per-bar return gap equals turnover × Δrate exactly
    turnover = _turnover(close, signal)
    gap = bt["low"].returns - bt["high"].returns
    pd.testing.assert_series_equal(
        gap, (turnover * (0.005 - 0.002)).rename("returns"), atol=1e-15
    )
    # stats report the effective basis
    assert bt["high"].stats["cost_tier"] == "high"
    assert bt["high"].stats["two_sided_bps"] == pytest.approx(50.0)
    assert bt["high"].stats["cost_bps_effective"] == pytest.approx(50.0)


def test_default_parameters_match_legacy_behaviour():
    close, signal = _close_and_signal()
    bt_default = run_long_only(close, signal)
    bt_legacy = run_long_only(close, signal, fee_bps=5.0, slippage_bps=0.0)
    pd.testing.assert_series_equal(bt_default.equity, bt_legacy.equity)
    # and both equal the hand-computed legacy formula (5 bps on turnover)
    pos = signal.reindex(close.index).fillna(0.0).clip(-1.0, 1.0).shift(1).fillna(0.0)
    ret = close.astype(float).pct_change().fillna(0.0)
    expected = (1.0 + pos * ret - _turnover(close, signal) * 5e-4).cumprod()
    pd.testing.assert_series_equal(bt_default.equity, expected.rename("equity"))
    assert bt_default.stats["cost_tier"] is None
    assert bt_default.stats["cost_bps_effective"] == pytest.approx(5.0)


def test_two_sided_bps_equivalent_to_named_tier():
    close, signal = _close_and_signal()
    bt_bps = run_long_only(close, signal, two_sided_bps=20.0)
    bt_tier = run_long_only(close, signal, cost_tier="low")
    pd.testing.assert_series_equal(bt_bps.equity, bt_tier.equity)
    assert bt_bps.stats["cost_bps_effective"] == pytest.approx(20.0)


def test_cancel_fee_lowers_equity_and_converts_to_bps():
    close, signal = _close_and_signal()
    bt_off = run_long_only(close, signal)
    bt_on = run_long_only(close, signal, cancel_fee_per_order=5.0)
    assert bt_on.equity.iloc[-1] < bt_off.equity.iloc[-1]
    # 5 CNY / 100k CNY average order = 0.5 bps on traded notional
    turnover = _turnover(close, signal)
    gap = bt_off.returns - bt_on.returns
    pd.testing.assert_series_equal(gap, (turnover * 0.5e-4).rename("returns"), atol=1e-15)
    assert bt_on.stats["cancel_fee_bps"] == pytest.approx(0.5)
    # smaller average order size ⇒ larger converted fee ⇒ even lower equity
    bt_small_orders = run_long_only(
        close, signal, cancel_fee_per_order=5.0, avg_order_notional=50_000.0
    )
    assert bt_small_orders.stats["cancel_fee_bps"] == pytest.approx(1.0)
    assert bt_small_orders.equity.iloc[-1] < bt_on.equity.iloc[-1]


def test_invalid_cost_configuration_fail_closed():
    close, signal = _close_and_signal()
    with pytest.raises(ValueError):
        run_long_only(close, signal, cost_tier="extreme")
    with pytest.raises(ValueError):
        run_long_only(close, signal, cost_tier="low", two_sided_bps=20.0)
    with pytest.raises(ValueError):
        run_long_only(close, signal, two_sided_bps=-5.0)
    with pytest.raises(ValueError):
        run_long_only(close, signal, cancel_fee_per_order=-1.0)
    with pytest.raises(ValueError):
        run_long_only(close, signal, cancel_fee_per_order=5.0, avg_order_notional=0.0)

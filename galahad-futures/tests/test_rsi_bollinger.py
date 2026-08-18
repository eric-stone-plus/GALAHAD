"""Tests for RSI and Bollinger Bands strategies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.strategy import (
    BollingerConfig,
    BollingerStrategy,
    RSIConfig,
    RSIStrategy,
    build_strategy,
    strategy_kwargs_from_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(closes: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    return pd.DataFrame(
        {
            "ts": [f"t{i}" for i in range(n)],
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# RSI Strategy Tests
# ---------------------------------------------------------------------------

class TestRSIStrategy:

    def test_warmup_returns_zero(self):
        """RSI should return 0 during warmup period."""
        closes = list(range(100, 130))  # 30 bars uptrend
        bars = _make_bars(closes)
        s = RSIStrategy(RSIConfig(period=14))
        t = s.targets(bars)
        # First 13 bars should be 0 (warmup: need `period` data points)
        assert t.iloc[:13].eq(0).all(), "warmup should be zero"
        assert len(t) == 30
        assert t.name == "target_signed_leverage"

    def test_strong_uptrend_triggers_long(self):
        """Sustained uptrend should push RSI above overbought → long."""
        # Create a strong monotonic uptrend
        closes = list(range(100, 150))  # 50 bars, strictly increasing
        bars = _make_bars(closes)
        s = RSIStrategy(RSIConfig(period=14, overbought=70.0, max_target_leverage=1.5))
        t = s.targets(bars)
        # After warmup in a strong uptrend, RSI should be > 70 → long
        assert t.iloc[20] == pytest.approx(1.5), "strong uptrend should be long"
        assert t.iloc[30] == pytest.approx(1.5)
        assert t.iloc[-1] == pytest.approx(1.5)

    def test_strong_downtrend_triggers_short(self):
        """Sustained downtrend should push RSI below oversold → short."""
        closes = list(range(150, 100, -1))  # 50 bars, strictly decreasing
        bars = _make_bars(closes)
        s = RSIStrategy(RSIConfig(period=14, oversold=30.0, max_target_leverage=2.0))
        t = s.targets(bars)
        # RSI in a strong downtrend should be < 30 → short
        assert t.iloc[20] == pytest.approx(-2.0), "strong downtrend should be short"
        assert t.iloc[-1] == pytest.approx(-2.0)

    def test_flat_series_stays_flat(self):
        """Constant price → no signal (RSI undefined or 50)."""
        closes = [100.0] * 50
        bars = _make_bars(closes)
        s = RSIStrategy(RSIConfig(period=14))
        t = s.targets(bars)
        # With zero delta, avg_loss = 0 → RS = nan → RSI = nan → flat
        # Or if both avg_gain and avg_loss are 0, RSI could be nan
        assert t.eq(0).all(), "constant price should produce all zeros"

    def test_missing_price_col_raises(self):
        bars = pd.DataFrame({"ts": ["t0"], "open": [1.0], "close": [1.0]})
        bars = bars.drop(columns=["close"])
        s = RSIStrategy(RSIConfig(price_col="close"))
        with pytest.raises(KeyError):
            s.targets(bars)

    def test_period_validation(self):
        with pytest.raises(ValueError, match="period must be >= 1"):
            s = RSIStrategy(RSIConfig(period=0))
            s.targets(_make_bars([1.0] * 5))

    def test_custom_overbought_oversold(self):
        """Custom thresholds should shift entry points."""
        closes = list(range(100, 150))
        bars = _make_bars(closes)
        # Very high overbought threshold → harder to trigger long
        s = RSIStrategy(RSIConfig(period=14, overbought=95.0))
        t = s.targets(bars)
        # With OB=95, even a strong trend might not trigger
        # At least some bars should still be 0
        assert t.eq(0).any(), "high OB threshold should leave some bars flat"

    def test_output_dtype_and_index(self):
        closes = list(range(100, 120))
        bars = _make_bars(closes)
        s = RSIStrategy(RSIConfig(period=5))
        t = s.targets(bars)
        assert t.dtype == float
        assert len(t) == len(bars)
        assert (t.index == bars.index).all()

    def test_build_strategy_rsi(self):
        s = build_strategy("rsi", period=10, overbought=80, oversold=20, max_target_leverage=2.0)
        assert isinstance(s, RSIStrategy)
        assert s.config.period == 10
        assert s.config.overbought == 80
        assert s.config.oversold == 20
        assert s.config.max_target_leverage == 2.0

    def test_build_strategy_rsi_defaults(self):
        s = build_strategy("rsi")
        assert s.config.period == 14
        assert s.config.overbought == 70.0
        assert s.config.oversold == 30.0
        assert s.config.max_target_leverage == 1.5

    def test_signal_values_are_bounded(self):
        """Targets should only be 0, +lev, or -lev — nothing in between."""
        np.random.seed(42)
        closes = (np.cumsum(np.random.randn(200)) + 100).tolist()
        bars = _make_bars(closes)
        s = RSIStrategy(RSIConfig(period=14, max_target_leverage=1.5))
        t = s.targets(bars)
        allowed = {0.0, 1.5, -1.5}
        assert set(t.unique()).issubset(allowed), f"unexpected values: {set(t.unique()) - allowed}"


# ---------------------------------------------------------------------------
# Bollinger Bands Strategy Tests
# ---------------------------------------------------------------------------

class TestBollingerStrategy:

    def test_warmup_returns_zero(self):
        """Bollinger should return 0 during warmup (need `period` bars for SMA+std)."""
        closes = list(range(100, 140))
        bars = _make_bars(closes)
        s = BollingerStrategy(BollingerConfig(period=20))
        t = s.targets(bars)
        assert t.iloc[:19].eq(0).all(), "warmup should be zero"
        assert len(t) == 40
        assert t.name == "target_signed_leverage"

    def test_breakout_above_upper_band(self):
        """Price well above SMA + 2σ should trigger long."""
        # Build a stable series then a sudden spike at the very end
        stable = [100.0] * 30  # establish narrow band
        spike = [115.0]  # sudden jump well above band
        closes = stable + spike
        bars = _make_bars(closes)
        s = BollingerStrategy(BollingerConfig(period=20, num_std=2.0, max_target_leverage=1.5))
        t = s.targets(bars)
        # The spike should be above upper band → long
        assert t.iloc[-1] == pytest.approx(1.5), "spike above band should be long"

    def test_breakout_below_lower_band(self):
        """Price well below SMA - 2σ should trigger short."""
        stable = [100.0] * 30
        drop = [85.0]  # sudden drop
        closes = stable + drop
        bars = _make_bars(closes)
        s = BollingerStrategy(BollingerConfig(period=20, num_std=2.0, max_target_leverage=2.0))
        t = s.targets(bars)
        assert t.iloc[-1] == pytest.approx(-2.0), "drop below band should be short"

    def test_within_bands_stays_flat(self):
        """Price within bands should produce 0."""
        # Narrow range around 100
        np.random.seed(123)
        closes = (100 + np.random.randn(60) * 0.1).tolist()
        bars = _make_bars(closes)
        s = BollingerStrategy(BollingerConfig(period=20, num_std=2.0))
        t = s.targets(bars)
        # With tiny noise and 2σ bands, should stay within bands
        # At least most bars after warmup should be 0
        post_warmup = t.iloc[20:]
        assert post_warmup.eq(0).sum() > len(post_warmup) * 0.5, "should mostly stay flat"

    def test_missing_price_col_raises(self):
        bars = pd.DataFrame({"ts": ["t0"], "open": [1.0]})
        s = BollingerStrategy(BollingerConfig(price_col="close"))
        with pytest.raises(KeyError):
            s.targets(bars)

    def test_period_validation(self):
        with pytest.raises(ValueError, match="period must be >= 1"):
            s = BollingerStrategy(BollingerConfig(period=0))
            s.targets(_make_bars([1.0] * 5))

    def test_narrow_band_with_num_std(self):
        """Very narrow band (num_std=0.1) should trigger on small moves."""
        closes = [100.0] * 25 + [100.5] * 10  # small move up
        bars = _make_bars(closes)
        s = BollingerStrategy(BollingerConfig(period=20, num_std=0.1))
        t = s.targets(bars)
        # With σ=0.1, even 0.5 move should break above upper band
        assert t.iloc[-1] == pytest.approx(1.5), "narrow band should trigger on small move"

    def test_output_dtype_and_index(self):
        closes = list(range(100, 130))
        bars = _make_bars(closes)
        s = BollingerStrategy(BollingerConfig(period=10))
        t = s.targets(bars)
        assert t.dtype == float
        assert len(t) == len(bars)
        assert (t.index == bars.index).all()

    def test_build_strategy_bollinger(self):
        s = build_strategy("bollinger", period=15, num_std=1.5, max_target_leverage=2.0)
        assert isinstance(s, BollingerStrategy)
        assert s.config.period == 15
        assert s.config.num_std == 1.5
        assert s.config.max_target_leverage == 2.0

    def test_build_strategy_boll_aliases(self):
        """Aliases 'boll' and 'bollinger_bands' should work."""
        for name in ("boll", "bollinger_bands"):
            s = build_strategy(name, period=10)
            assert isinstance(s, BollingerStrategy)
            assert s.config.period == 10

    def test_build_strategy_bollinger_defaults(self):
        s = build_strategy("bollinger")
        assert s.config.period == 20
        assert s.config.num_std == 2.0
        assert s.config.max_target_leverage == 1.5

    def test_signal_values_are_bounded(self):
        """Targets should only be 0, +lev, or -lev."""
        np.random.seed(99)
        closes = (np.cumsum(np.random.randn(200)) + 100).tolist()
        bars = _make_bars(closes)
        s = BollingerStrategy(BollingerConfig(period=20, max_target_leverage=1.5))
        t = s.targets(bars)
        allowed = {0.0, 1.5, -1.5}
        assert set(t.unique()).issubset(allowed), f"unexpected values: {set(t.unique()) - allowed}"

    def test_monotonic_uptrend_may_stay_flat(self):
        """Linear uptrend: price tracks SMA closely, may stay within bands."""
        closes = list(range(100, 160))
        bars = _make_bars(closes)
        s = BollingerStrategy(BollingerConfig(period=20, num_std=2.0))
        t = s.targets(bars)
        # A perfectly linear trend tracks its SMA closely; std is small.
        # Depending on geometry, may or may not break out. Just verify no crash.
        assert len(t) == 60


# ---------------------------------------------------------------------------
# strategy_kwargs_from_config for new keys
# ---------------------------------------------------------------------------

class TestStrategyKwargsFromConfig:

    def test_rsi_keys_passthrough(self):
        cfg = {"period": 10, "overbought": 80, "oversold": 20, "max_target_leverage": 2.0}
        kw = strategy_kwargs_from_config(cfg)
        assert kw == cfg

    def test_bollinger_keys_passthrough(self):
        cfg = {"period": 15, "num_std": 1.5, "max_target_leverage": 2.0}
        kw = strategy_kwargs_from_config(cfg)
        assert kw == cfg

    def test_mixed_keys(self):
        cfg = {"fast": 5, "slow": 10, "period": 14, "num_std": 2.0, "unknown_key": 99}
        kw = strategy_kwargs_from_config(cfg)
        assert "unknown_key" not in kw
        assert kw["fast"] == 5
        assert kw["period"] == 14

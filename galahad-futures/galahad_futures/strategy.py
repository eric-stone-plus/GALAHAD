"""Strategy layer — emits target signed leverage only. Never places orders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union

import numpy as np
import pandas as pd


class Strategy(Protocol):
    def targets(self, bars: pd.DataFrame) -> pd.Series:
        """Return series of target_signed_leverage aligned to bars index."""
        ...


@dataclass
class DualMAConfig:
    fast: int = 8
    slow: int = 21
    max_target_leverage: float = 2.0
    price_col: str = "close"


@dataclass
class DualMAStrategy:
    """Deliberately naive: long when fast MA > slow MA, short otherwise, flat until warm."""

    config: DualMAConfig

    def targets(self, bars: pd.DataFrame) -> pd.Series:
        if self.config.price_col not in bars.columns:
            raise KeyError(f"missing price column {self.config.price_col}")
        px = bars[self.config.price_col].astype(float)
        fast = px.rolling(self.config.fast, min_periods=self.config.fast).mean()
        slow = px.rolling(self.config.slow, min_periods=self.config.slow).mean()
        lev = float(self.config.max_target_leverage)
        out = pd.Series(0.0, index=bars.index, dtype=float)
        ready = fast.notna() & slow.notna()
        out.loc[ready & (fast > slow)] = lev
        out.loc[ready & (fast < slow)] = -lev
        out.name = "target_signed_leverage"
        return out


@dataclass
class TSMOMConfig:
    """Time-series momentum with a single pre-specified lookback (no grid search).

    Classic rule: sign of return over ``lookback`` bars → full target leverage.
    Defaults map to ~7d on 1h bars (168) — longer-horizon than dual-MA toy.
    """

    lookback: int = 168
    max_target_leverage: float = 1.5
    price_col: str = "close"
    # optional: stay flat if |return| below threshold (reduces churn)
    min_abs_return: float = 0.0


@dataclass
class TSMOMStrategy:
    """Pre-specified TSMOM — long if past lookback return > 0, else short."""

    config: TSMOMConfig

    def targets(self, bars: pd.DataFrame) -> pd.Series:
        if self.config.price_col not in bars.columns:
            raise KeyError(f"missing price column {self.config.price_col}")
        px = bars[self.config.price_col].astype(float)
        lb = int(self.config.lookback)
        if lb < 1:
            raise ValueError("lookback must be >= 1")
        past = px.shift(lb)
        ret = px / past - 1.0
        lev = float(self.config.max_target_leverage)
        thr = float(self.config.min_abs_return)
        out = pd.Series(0.0, index=bars.index, dtype=float)
        ready = ret.notna()
        if thr > 0:
            strong = ready & (ret.abs() >= thr)
            out.loc[strong & (ret > 0)] = lev
            out.loc[strong & (ret < 0)] = -lev
        else:
            out.loc[ready & (ret > 0)] = lev
            out.loc[ready & (ret < 0)] = -lev
            # ret == 0 stays flat
        out.name = "target_signed_leverage"
        return out


@dataclass
class RSIConfig:
    """RSI strategy with Wilder's smoothing and threshold-based entry/exit.

    Long when RSI crosses above ``overbought`` from below (momentum confirmation),
    short when RSI crosses below ``oversold`` from above.  Flat in between.
    ``period`` controls the Wilder smoothing window; warm-up returns 0.
    """

    period: int = 14
    overbought: float = 70.0
    oversold: float = 30.0
    max_target_leverage: float = 1.5
    price_col: str = "close"


@dataclass
class RSIStrategy:
    """RSI-based strategy: long above overbought, short below oversold, flat otherwise."""

    config: RSIConfig

    def targets(self, bars: pd.DataFrame) -> pd.Series:
        if self.config.price_col not in bars.columns:
            raise KeyError(f"missing price column {self.config.price_col}")
        px = bars[self.config.price_col].astype(float)
        period = int(self.config.period)
        if period < 1:
            raise ValueError("period must be >= 1")

        delta = px.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        # Wilder's smoothing (exponential moving average with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        # Handle edge cases: when avg_loss==0 and avg_gain>0 → RSI=100
        # (all gains, no losses); when avg_gain==0 and avg_loss>0 → RSI=0
        rsi = pd.Series(np.nan, index=bars.index, dtype=float)
        both_pos = (avg_gain > 0) & (avg_loss > 0)
        gain_zero = (avg_gain == 0) & avg_loss.gt(0) & avg_loss.notna()
        loss_zero = (avg_loss == 0) & avg_gain.gt(0) & avg_gain.notna()
        rsi.loc[both_pos] = 100.0 - 100.0 / (1.0 + avg_gain[both_pos] / avg_loss[both_pos])
        rsi.loc[gain_zero] = 0.0
        rsi.loc[loss_zero] = 100.0

        lev = float(self.config.max_target_leverage)
        out = pd.Series(0.0, index=bars.index, dtype=float)
        ready = rsi.notna()
        out.loc[ready & (rsi >= self.config.overbought)] = lev
        out.loc[ready & (rsi <= self.config.oversold)] = -lev
        out.name = "target_signed_leverage"
        return out


@dataclass
class BollingerConfig:
    """Bollinger Bands breakout strategy.

    Long when price breaks above upper band (SMA + k·σ),
    short when price breaks below lower band (SMA − k·σ), flat in between.
    ``period`` controls the rolling window; warm-up returns 0.
    """

    period: int = 20
    num_std: float = 2.0
    max_target_leverage: float = 1.5
    price_col: str = "close"


@dataclass
class BollingerStrategy:
    """Bollinger Bands breakout: long above upper band, short below lower band."""

    config: BollingerConfig

    def targets(self, bars: pd.DataFrame) -> pd.Series:
        if self.config.price_col not in bars.columns:
            raise KeyError(f"missing price column {self.config.price_col}")
        px = bars[self.config.price_col].astype(float)
        period = int(self.config.period)
        if period < 1:
            raise ValueError("period must be >= 1")

        sma = px.rolling(period, min_periods=period).mean()
        std = px.rolling(period, min_periods=period).std()

        upper = sma + float(self.config.num_std) * std
        lower = sma - float(self.config.num_std) * std

        lev = float(self.config.max_target_leverage)
        out = pd.Series(0.0, index=bars.index, dtype=float)
        ready = upper.notna() & lower.notna()
        out.loc[ready & (px > upper)] = lev
        out.loc[ready & (px < lower)] = -lev
        out.name = "target_signed_leverage"
        return out


AnyStrategy = Union[DualMAStrategy, TSMOMStrategy, RSIStrategy, BollingerStrategy]


def build_strategy(name: str, **kwargs) -> AnyStrategy:
    n = (name or "dual_ma").lower().replace("-", "_")
    if n in ("dual_ma", "dma"):
        return DualMAStrategy(
            DualMAConfig(
                fast=int(kwargs.get("fast", 8)),
                slow=int(kwargs.get("slow", 21)),
                max_target_leverage=float(kwargs.get("max_target_leverage", 2.0)),
            )
        )
    if n in ("tsmom", "ts_mom", "timeseries_momentum", "time_series_momentum"):
        return TSMOMStrategy(
            TSMOMConfig(
                lookback=int(kwargs.get("lookback", 48)),
                max_target_leverage=float(kwargs.get("max_target_leverage", 1.5)),
                min_abs_return=float(kwargs.get("min_abs_return", 0.0)),
            )
        )
    # Pre-specified longer horizon (7d on 1h bars) — fixed lookback, not a search
    if n in ("tsmom_long", "tsmom_7d", "tsmom_168"):
        return TSMOMStrategy(
            TSMOMConfig(
                lookback=int(kwargs.get("lookback", 168)),
                max_target_leverage=float(kwargs.get("max_target_leverage", 1.0)),
                min_abs_return=float(kwargs.get("min_abs_return", 0.0)),
            )
        )
    if n in ("rsi",):
        return RSIStrategy(
            RSIConfig(
                period=int(kwargs.get("period", 14)),
                overbought=float(kwargs.get("overbought", 70.0)),
                oversold=float(kwargs.get("oversold", 30.0)),
                max_target_leverage=float(kwargs.get("max_target_leverage", 1.5)),
            )
        )
    if n in ("bollinger", "boll", "bollinger_bands"):
        return BollingerStrategy(
            BollingerConfig(
                period=int(kwargs.get("period", 20)),
                num_std=float(kwargs.get("num_std", 2.0)),
                max_target_leverage=float(kwargs.get("max_target_leverage", 1.5)),
            )
        )
    raise ValueError(f"unknown strategy: {name}")


def strategy_kwargs_from_config(strat_cfg: dict) -> dict:
    """Pass-through known keys for build_strategy."""
    keys = (
        "fast",
        "slow",
        "lookback",
        "max_target_leverage",
        "min_abs_return",
        "price_col",
        "period",
        "overbought",
        "oversold",
        "num_std",
    )
    return {k: strat_cfg[k] for k in keys if k in strat_cfg}

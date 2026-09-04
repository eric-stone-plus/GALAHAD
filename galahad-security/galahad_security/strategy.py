"""Strategy layer — emits target weights only. Never places orders.

Targets are long-only weights: fraction of equity per symbol in [0, 1].
The shipped null is dual_ma on daily closes — the same deliberately-naive
doctrine as the futures nulls: a plumbing benchmark, not an edge claim.
Multi-symbol from day one; the null evaluates each symbol independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union

import pandas as pd


class Strategy(Protocol):
    def targets(self, bars: pd.DataFrame) -> pd.Series:
        """Return series of target weights (0..1) aligned to bars index."""
        ...


@dataclass
class DualMAConfig:
    fast: int = 8
    slow: int = 21
    target_weight: float = 0.2  # fraction of equity when the signal is long
    price_col: str = "close"


@dataclass
class DualMAStrategy:
    """Deliberately naive long-only null: target weight when fast MA > slow
    MA, flat otherwise (and flat until both MAs are warm)."""

    config: DualMAConfig

    def targets(self, bars: pd.DataFrame) -> pd.Series:
        if self.config.price_col not in bars.columns:
            raise KeyError(f"missing price column {self.config.price_col}")
        px = bars[self.config.price_col].astype(float)
        fast = px.rolling(self.config.fast, min_periods=self.config.fast).mean()
        slow = px.rolling(self.config.slow, min_periods=self.config.slow).mean()
        w = float(self.config.target_weight)
        out = pd.Series(0.0, index=bars.index, dtype=float)
        ready = fast.notna() & slow.notna()
        out.loc[ready & (fast > slow)] = w
        out.name = "target_weight"
        return out


AnyStrategy = Union[DualMAStrategy]


def build_strategy(name: str, **kwargs) -> AnyStrategy:
    n = (name or "dual_ma").lower().replace("-", "_")
    if n in ("dual_ma", "dma"):
        return DualMAStrategy(
            DualMAConfig(
                fast=int(kwargs.get("fast", 8)),
                slow=int(kwargs.get("slow", 21)),
                target_weight=float(kwargs.get("target_weight", 0.2)),
            )
        )
    raise ValueError(f"unknown strategy: {name}")


def strategy_kwargs_from_config(strat_cfg: dict) -> dict:
    """Pass-through known keys for build_strategy."""
    keys = ("fast", "slow", "target_weight", "price_col")
    return {k: strat_cfg[k] for k in keys if k in strat_cfg}

"""Lightweight vectorized backtest (no broker simulation).

For research / mid-low frequency strategies. Not for HFT.

Cost model: the 2026 A-share research conclusion is that the only legal
backtest cost basis is a two-sided all-in rate — see ``COST_TIERS``
(0.2% / 0.4% / 0.5% by capital & order-splitting profile) — plus an
optional per-order cancel fee. Legacy ``fee_bps``/``slippage_bps`` behaviour
is unchanged when the new parameters are left at their defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

# Two-sided all-in cost tiers (fraction of traded notional, both legs):
#   low  0.2% — small/mid capital, no order splitting
#   mid  0.4% — micro-caps / high-frequency split orders
#   high 0.5% — >50M CNY, heavy splitting
COST_TIERS: dict[str, float] = {"low": 0.002, "mid": 0.004, "high": 0.005}


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: int
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    stats: dict[str, Any]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "equity": self.equity,
                "returns": self.returns,
                "position": self.positions,
            }
        )


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _cagr(equity: pd.Series, periods_per_year: float = 252.0) -> float:
    if equity.empty or equity.iloc[0] <= 0:
        return 0.0
    n = len(equity)
    if n < 2:
        return 0.0
    years = n / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def _sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    r = returns.dropna()
    if r.empty or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def run_long_only(
    close: pd.Series,
    signal: pd.Series,
    *,
    fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    initial_capital: float = 1.0,
    periods_per_year: float = 252.0,
    cost_tier: Literal["low", "mid", "high"] | None = None,
    two_sided_bps: float | None = None,
    cancel_fee_per_order: float = 0.0,
    avg_order_notional: float = 100_000.0,
) -> BacktestResult:
    """Vectorized long-only (or long/flat) backtest.

    Parameters
    ----------
    close :
        Price series (DatetimeIndex).
    signal :
        Desired position in [0, 1] (or {-1,0,1}); evaluated at bar close,
        position taken next bar (shift 1) to avoid look-ahead.
    fee_bps :
        Round-trip is not assumed; fee charged on position change *notional*.
    cost_tier :
        Named two-sided all-in cost tier from ``COST_TIERS`` (2026 A-share
        cost-basis conclusion): "low" 0.2% / "mid" 0.4% / "high" 0.5% of
        traded notional, both legs included. Overrides ``fee_bps`` +
        ``slippage_bps`` when set; mutually exclusive with
        ``two_sided_bps``. Default None keeps the legacy behaviour
        bit-identical.
    two_sided_bps :
        Explicit two-sided all-in rate in bps (20.0 = 0.2%). Same override
        semantics as ``cost_tier``; takes precedence over it.
    cancel_fee_per_order :
        Cancel fee in CNY per order (default 0 → off). Conversion
        assumption: every position-change bar generates one order and one
        cancel (conservative), so the fee is charged as
        ``cancel_fee_per_order / avg_order_notional`` of traded notional,
        i.e. ``fee / avg_order_notional × 1e4`` bps, scaled by turnover.
        Example: 5 CNY/order at 100k CNY average order size ≈ 0.5 bps
        (1 bps at 50k, 0.1 bps at 500k for large split orders).
    avg_order_notional :
        Assumed average executed order size in CNY used for the cancel-fee
        conversion above. Must be > 0 when the cancel fee is on.
    """
    if cost_tier is not None and two_sided_bps is not None:
        raise ValueError("cost_tier and two_sided_bps are mutually exclusive")
    if cost_tier is not None:
        if cost_tier not in COST_TIERS:
            raise ValueError(f"unknown cost_tier: {cost_tier!r} (use one of {sorted(COST_TIERS)})")
        notional_bps = COST_TIERS[cost_tier] * 10_000.0
    elif two_sided_bps is not None:
        if two_sided_bps < 0:
            raise ValueError("two_sided_bps must be >= 0")
        notional_bps = float(two_sided_bps)
    else:  # legacy path, unchanged
        notional_bps = fee_bps + slippage_bps
    if cancel_fee_per_order < 0:
        raise ValueError("cancel_fee_per_order must be >= 0")
    cancel_bps = 0.0
    if cancel_fee_per_order > 0:
        if avg_order_notional <= 0:
            raise ValueError("avg_order_notional must be > 0 when cancel_fee_per_order is set")
        cancel_bps = cancel_fee_per_order / avg_order_notional * 10_000.0

    px = close.astype(float).dropna()
    pos_target = signal.reindex(px.index).fillna(0.0).astype(float).clip(-1.0, 1.0)
    # trade on next bar
    pos = pos_target.shift(1).fillna(0.0)
    ret = px.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * ((notional_bps + cancel_bps) / 10_000.0)
    strat_ret = pos * ret - cost
    equity = (1.0 + strat_ret).cumprod() * initial_capital

    # simple trade count: whenever position changes
    trades = int((pos.diff().fillna(pos).abs() > 1e-12).sum())
    # win rate on days with non-zero position
    active = strat_ret[pos.abs() > 1e-12]
    win_rate = float((active > 0).mean()) if len(active) else 0.0

    stats = {
        "total_return": float(equity.iloc[-1] / initial_capital - 1) if len(equity) else 0.0,
        "cagr": _cagr(equity, periods_per_year),
        "sharpe": _sharpe(strat_ret, periods_per_year),
        "max_drawdown": _max_drawdown(equity),
        "trades": trades,
        "win_rate": win_rate,
        "final_equity": float(equity.iloc[-1]) if len(equity) else initial_capital,
        "fee_bps": fee_bps,
        # effective cost actually charged per traded notional (bps)
        "cost_bps_effective": notional_bps + cancel_bps,
        "cost_tier": cost_tier,
        "two_sided_bps": notional_bps,
        "cancel_fee_bps": cancel_bps,
    }
    return BacktestResult(
        equity=equity.rename("equity"),
        returns=strat_ret.rename("returns"),
        positions=pos.rename("position"),
        trades=trades,
        total_return=stats["total_return"],
        cagr=stats["cagr"],
        sharpe=stats["sharpe"],
        max_drawdown=stats["max_drawdown"],
        win_rate=win_rate,
        stats=stats,
    )


def dual_ma_signal(
    close: pd.Series, fast: int = 20, slow: int = 50
) -> pd.Series:
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    sig = (f > s).astype(float)
    return sig.rename("dual_ma")


def rsi_mean_reversion_signal(
    close: pd.Series, window: int = 14, low: float = 30.0, high: float = 70.0
) -> pd.Series:
    from quantkit.indicators import rsi

    r = rsi(close, window)
    # long when oversold, exit when overbought
    pos = pd.Series(np.nan, index=close.index)
    pos = pos.mask(r < low, 1.0)
    pos = pos.mask(r > high, 0.0)
    return pos.ffill().fillna(0.0).rename("rsi_mr")


def summary_dict(result: BacktestResult) -> dict[str, Any]:
    base = {
        "trades": result.trades,
        "total_return": result.total_return,
        "cagr": result.cagr,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
    }
    extra = {
        k: v
        for k, v in result.stats.items()
        if k not in base
    }
    return {**base, **extra}

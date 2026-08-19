"""Vectorized parameter sweeps (vectorbt) for research signal families.

Internal research tooling only. vectorbt is distributed under
"fair-code": Apache-2.0 with the Commons Clause, which forbids selling
products/services consisting primarily of vectorbt itself. Research use
— running sweeps to compare signal families, and building our own
engines around the results — is fine; do not wrap vectorbt in a
standalone product for resale. See docs/roadmap.md for the evaluation
context.

Optional dependency: ``vectorbt`` (pinned to 1.0.0 via requirements.txt
and the ``sweep`` extra in pyproject.toml). Import this module lazily
from research code; ``import quantkit.sweep`` raises a clear error when
vectorbt is absent.

Exposure convention: sweep statistics are computed at 1x exposure. The
sweeps run cash-constrained vbt portfolios — no margin, no leverage — so
all reported stats (total_return, cagr, sharpe, max_drawdown, ...) are
per 1x unit of notional. To compare against levered strategies, scale
returns linearly by the strategy's leverage (e.g. double a 2x position's
per-bar returns before computing stats). Funding and margin effects are
NOT in the sweep; they are settled in the futures engines
(``galahad_futures`` book/nautilus backends), which trade at configured
leverage (e.g. 1.5x/3x) and net those costs per bar.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except ImportError as exc:  # pragma: no cover - depends on venv contents
    raise ImportError(
        "quantkit.sweep requires the optional dependency vectorbt, pinned "
        "to 1.0.0 (pip install vectorbt==1.0.0, or "
        "pip install -e './quantkit[sweep]')"
    ) from exc




def _cagr(total_return: float, close: pd.Series, freq: str) -> float:
    """Annualized growth from a total return over the series span."""
    seconds_per_bar = pd.Timedelta(freq).total_seconds()
    years = len(close) * seconds_per_bar / (365.0 * 24.0 * 3600.0)
    if years <= 0 or total_return <= -1.0:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def tsmom_sweep(
    bars: pd.DataFrame,
    lookbacks: Sequence[int],
    *,
    fee_rate: float = 0.0,
    price_col: str = "close",
    freq: str = "1h",
) -> pd.DataFrame:
    """Long/short TSMOM sweep over lookback horizons via vectorbt.

    Signal: sign of the ``lookback``-bar return; 100% exposure either
    side (long or short), flat while the return is exactly zero. vbt
    portfolios are cash-constrained — no margin — so leverage is out of
    scope here; scale returns linearly for leverage comparisons. Fees
    are charged per side on turnover. No broker simulation: vectorized
    fills at bar close, matching the quantkit backtest cost conventions
    (pass a two-sided all-in ``fee_rate`` from
    ``quantkit.backtest.COST_TIERS`` for A-share work).

    Returns one row per lookback: total_return, cagr, sharpe, sortino,
    max_drawdown, calmar, trades, win_rate.
    """
    if price_col not in bars.columns:
        raise KeyError(f"missing price column {price_col}")
    close = bars[price_col].astype(float)
    lookbacks = [int(lb) for lb in lookbacks]
    if any(lb < 1 for lb in lookbacks):
        raise ValueError("lookbacks must be >= 1")

    rows: list[dict[str, float]] = []
    for lb in lookbacks:
        ret = close.pct_change(lb)
        direction = np.sign(ret).shift(1).fillna(0.0)
        entries = (direction > 0).fillna(False)
        exits = (direction <= 0).fillna(False)
        short_entries = (direction < 0).fillna(False)
        short_exits = (direction >= 0).fillna(False)
        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            fees=float(fee_rate),
            freq=freq,
        )
        rows.append(
            {
                "lookback": lb,
                "total_return": float(pf.total_return()),
                "cagr": _cagr(float(pf.total_return()), close, freq),
                "sharpe": float(pf.sharpe_ratio()),
                "sortino": float(pf.sortino_ratio()),
                "max_drawdown": float(pf.max_drawdown()),
                "calmar": float(pf.calmar_ratio()),
                "trades": int(pf.trades.count()),
                "win_rate": float(pf.trades.win_rate() or 0.0),
            }
        )
    return pd.DataFrame(rows).set_index("lookback")


def dual_ma_sweep(
    bars: pd.DataFrame,
    fast_periods: Sequence[int],
    slow_periods: Sequence[int],
    *,
    fee_rate: float = 0.0,
    price_col: str = "close",
    freq: str = "1h",
) -> pd.DataFrame:
    """Dual-MA cross sweep (fast × slow grid) via vectorbt.

    Long when fast MA > slow MA, short otherwise, after both warm up.
    Returns one row per (fast, slow) pair with the same stat columns as
    :func:`tsmom_sweep`.
    """
    if price_col not in bars.columns:
        raise KeyError(f"missing price column {price_col}")
    close = bars[price_col].astype(float)

    rows: list[dict[str, float]] = []
    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue
            fast_ma = close.rolling(int(fast), min_periods=int(fast)).mean()
            slow_ma = close.rolling(int(slow), min_periods=int(slow)).mean()
            direction = np.where(fast_ma > slow_ma, 1.0, np.where(fast_ma < slow_ma, -1.0, 0.0))
            direction = pd.Series(direction, index=close.index).shift(1).fillna(0.0)
            entries = (direction > 0).fillna(False)
            exits = (direction <= 0).fillna(False)
            short_entries = (direction < 0).fillna(False)
            short_exits = (direction >= 0).fillna(False)
            pf = vbt.Portfolio.from_signals(
                close,
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
                fees=float(fee_rate),
                freq=freq,
            )
            rows.append(
                {
                    "fast": int(fast),
                    "slow": int(slow),
                    "total_return": float(pf.total_return()),
                    "cagr": _cagr(float(pf.total_return()), close, freq),
                    "sharpe": float(pf.sharpe_ratio()),
                    "sortino": float(pf.sortino_ratio()),
                    "max_drawdown": float(pf.max_drawdown()),
                    "calmar": float(pf.calmar_ratio()),
                    "trades": int(pf.trades.count()),
                    "win_rate": float(pf.trades.win_rate() or 0.0),
                }
            )
    if not rows:
        raise ValueError("no (fast, slow) pairs with fast < slow")
    return pd.DataFrame(rows).set_index(["fast", "slow"])

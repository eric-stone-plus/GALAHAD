"""Multi-asset portfolio backtest with target weights and rebalancing.

Designed for mid/low-frequency research (daily bars). Not a broker simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantkit.backtest import BacktestResult, _cagr, _max_drawdown, _sharpe

Rebalance = Literal["D", "W", "M", "Q", "none"]


@dataclass
class PortfolioResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    trades: int
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    stats: dict[str, Any]

    def to_backtest_result(self) -> BacktestResult:
        """Flatten to single-asset BacktestResult shape (positions = net exposure)."""
        net = self.weights.abs().sum(axis=1)
        return BacktestResult(
            equity=self.equity,
            returns=self.returns,
            positions=net.rename("position"),
            trades=self.trades,
            total_return=self.total_return,
            cagr=self.cagr,
            sharpe=self.sharpe,
            max_drawdown=self.max_drawdown,
            win_rate=self.win_rate,
            stats=self.stats,
        )


def align_prices(prices: pd.DataFrame | dict[str, pd.Series]) -> pd.DataFrame:
    """Build a wide close-price panel (columns = symbols), forward-fill gaps."""
    if isinstance(prices, dict):
        prices = pd.DataFrame(prices)
    px = prices.astype(float).sort_index()
    # drop all-NaN rows; ffill limited gaps after first valid
    px = px.dropna(how="all")
    px = px.ffill()
    return px


def equal_weight_targets(
    price_index: pd.DatetimeIndex,
    symbols: list[str],
    mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Constant equal weight across symbols (optionally masked tradeable)."""
    n = len(symbols)
    if n == 0:
        return pd.DataFrame(index=price_index)
    base = pd.DataFrame(1.0 / n, index=price_index, columns=symbols)
    if mask is not None:
        m = mask.reindex(index=price_index, columns=symbols).fillna(False)
        w = base.where(m, 0.0)
        row_sum = w.sum(axis=1).replace(0, np.nan)
        w = w.div(row_sum, axis=0).fillna(0.0)
        return w
    return base


def signal_to_weights(
    signals: pd.DataFrame,
    *,
    long_only: bool = True,
    max_weight: float | None = None,
    neutralize: bool = False,
) -> pd.DataFrame:
    """Map per-asset signal scores → portfolio weights each day.

    - long_only: clip negative to 0, renormalize positive score sum to 1
    - neutralize: demean cross-section then L1-normalize (gross ≈ 1)
    - max_weight: cap absolute weight per name, then renorm
    """
    s = signals.astype(float)
    if long_only:
        s = s.clip(lower=0.0)
        row = s.sum(axis=1).replace(0, np.nan)
        w = s.div(row, axis=0).fillna(0.0)
    elif neutralize:
        s = s.sub(s.mean(axis=1), axis=0)
        row = s.abs().sum(axis=1).replace(0, np.nan)
        w = s.div(row, axis=0).fillna(0.0)
    else:
        row = s.abs().sum(axis=1).replace(0, np.nan)
        w = s.div(row, axis=0).fillna(0.0)

    if max_weight is not None and max_weight > 0:
        w = w.clip(lower=-max_weight, upper=max_weight)
        if long_only:
            row = w.sum(axis=1).replace(0, np.nan)
            w = w.div(row, axis=0).fillna(0.0)
        else:
            row = w.abs().sum(axis=1).replace(0, np.nan)
            w = w.div(row, axis=0).fillna(0.0)
    return w


def _rebalance_mask(index: pd.DatetimeIndex, freq: Rebalance) -> pd.Series:
    if freq == "none" or freq == "D":
        return pd.Series(True, index=index)
    s = pd.Series(index, index=index)
    if freq == "W":
        # last trading day of each ISO week
        key = s.dt.isocalendar().week.astype(str) + "-" + s.dt.isocalendar().year.astype(str)
    elif freq == "M":
        key = s.dt.to_period("M").astype(str)
    elif freq == "Q":
        key = s.dt.to_period("Q").astype(str)
    else:
        return pd.Series(True, index=index)
    # True on last bar of each group
    return key != key.shift(-1)


def run_portfolio(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    rebalance: Rebalance = "M",
    fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    initial_capital: float = 1.0,
    periods_per_year: float = 252.0,
    tradeable: pd.DataFrame | None = None,
) -> PortfolioResult:
    """Vectorized multi-asset backtest.

    Parameters
    ----------
    prices :
        Wide close panel (DatetimeIndex × symbols).
    target_weights :
        Desired weights per day (same columns). Signals are shifted by 1 bar
        (trade next open/close) to reduce look-ahead.
    rebalance :
        ``D`` daily, ``W`` weekly, ``M`` monthly, ``Q`` quarterly, ``none`` = hold
        first non-zero weights only (no scheduled rebalance; still follows targets
        when rebalance mask is always true for D).
    tradeable :
        Optional bool panel; False freezes weight to 0 for that name (e.g. suspended).
    """
    px = align_prices(prices)
    tw = target_weights.reindex(index=px.index, columns=px.columns).fillna(0.0)
    if tradeable is not None:
        m = tradeable.reindex(index=px.index, columns=px.columns).fillna(False)
        tw = tw.where(m, 0.0)
        # renorm long-only rows
        row = tw.clip(lower=0).sum(axis=1)
        long_rows = row > 0
        tw.loc[long_rows] = tw.loc[long_rows].clip(lower=0).div(row[long_rows], axis=0)

    # desired weights known at close t → hold from t+1
    desired = tw.shift(1).fillna(0.0)

    rebal = _rebalance_mask(px.index, rebalance)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    current = pd.Series(0.0, index=px.columns)
    for i, dt in enumerate(px.index):
        if rebal.iloc[i] or current.abs().sum() < 1e-15:
            current = desired.iloc[i].copy()
        weights.iloc[i] = current.values

    asset_ret = px.pct_change().fillna(0.0)
    # portfolio return before costs
    gross = (weights * asset_ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost = turnover * ((fee_bps + slippage_bps) / 10_000.0)
    port_ret = gross - cost
    equity = (1.0 + port_ret).cumprod() * initial_capital

    trades = int((turnover > 1e-12).sum())
    active = port_ret[weights.abs().sum(axis=1) > 1e-12]
    win_rate = float((active > 0).mean()) if len(active) else 0.0

    stats = {
        "total_return": float(equity.iloc[-1] / initial_capital - 1) if len(equity) else 0.0,
        "cagr": _cagr(equity, periods_per_year),
        "sharpe": _sharpe(port_ret, periods_per_year),
        "max_drawdown": _max_drawdown(equity),
        "trades": trades,
        "win_rate": win_rate,
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "final_equity": float(equity.iloc[-1]) if len(equity) else initial_capital,
        "fee_bps": fee_bps,
        "rebalance": rebalance,
        "n_assets": int(px.shape[1]),
        "avg_gross_exposure": float(weights.abs().sum(axis=1).mean()),
    }
    return PortfolioResult(
        equity=equity.rename("equity"),
        returns=port_ret.rename("returns"),
        weights=weights,
        turnover=turnover.rename("turnover"),
        trades=trades,
        total_return=stats["total_return"],
        cagr=stats["cagr"],
        sharpe=stats["sharpe"],
        max_drawdown=stats["max_drawdown"],
        win_rate=win_rate,
        stats=stats,
    )


def dual_ma_panel(
    prices: pd.DataFrame, fast: int = 20, slow: int = 50
) -> pd.DataFrame:
    """Per-asset dual-MA binary signal panel (1 = fast > slow)."""
    sig = {}
    for col in prices.columns:
        f = prices[col].rolling(fast).mean()
        s = prices[col].rolling(slow).mean()
        sig[col] = (f > s).astype(float)
    return pd.DataFrame(sig, index=prices.index)


def momentum_panel(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Simple total-return momentum score panel."""
    return prices.pct_change(lookback)


# ---------------------------------------------------------------------------
# Conformal weight policy — ACI/DtACI interval width → position scaling
# ---------------------------------------------------------------------------

def conformal_weight_policy(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    alpha: float = 0.10,
    halflife: float = 20.0,
    gamma: float = 0.01,
    min_scale: float = 0.1,
    max_scale: float = 1.0,
) -> pd.DataFrame:
    """Scale portfolio weights by conformal prediction confidence.

    Uses DtACI (dynamically-tuned ACI) to track per-asset prediction
    intervals: each asset carries its own DtACI state (advanced once per
    bar), and the coverage indicator compares each observation against the
    interval formed with that asset's current adaptive miscoverage level
    ``alpha_t`` — so persistent misses widen the interval (alpha_t falls)
    and hits narrow it again.  When the interval is wide (high
    uncertainty), the weight is scaled down; when narrow (high
    confidence), weight stays at target.

    Parameters
    ----------
    target_weights : T×N target weight panel (pre-conformal).
    returns : T×N realized returns (same index/columns as target_weights).
    alpha : nominal miscoverage level (0.10 = 90% interval).
    halflife : EWM halflife for volatility estimation.
    gamma : ACI step size for online alpha update.
    min_scale : floor on confidence scaling factor.
    max_scale : ceiling on confidence scaling factor.

    Returns
    -------
    Scaled weight panel, same shape as target_weights.
    """
    from scipy.stats import norm

    from quantkit.conformal import DtACIState, dtaci_update

    aligned = target_weights.index.intersection(returns.index)
    tw = target_weights.loc[aligned].astype(float)
    ret = returns.loc[aligned].astype(float)

    scales = pd.DataFrame(1.0, index=aligned, columns=tw.columns)
    # One independent DtACI state per asset: a single shared state would be
    # advanced N times per bar (alpha evolving N× too fast) with coverage
    # errors from one asset polluting every other asset's interval.
    states = {
        col: DtACIState(alpha_target=alpha, eta=gamma * 10) for col in tw.columns
    }
    # current aggregated miscoverage per asset; each new observation is
    # judged against the interval formed with this alpha_t (not the fixed
    # nominal alpha_target), closing the online widen/narrow feedback loop
    alpha_t: dict[str, float] = {col: alpha for col in tw.columns}

    # Rolling volatility for prediction interval estimation
    vol = ret.ewm(halflife=halflife, min_periods=5).std().shift(1).bfill().clip(lower=1e-8)

    for t in range(1, len(aligned)):
        prev_date = aligned[t - 1]
        curr_date = aligned[t]
        for col in tw.columns:
            actual = ret.loc[prev_date, col] if prev_date in ret.index else 0.0
            sigma = vol.loc[prev_date, col] if prev_date in vol.index else 0.01
            # z-score: how many sigma away from zero
            z = abs(actual) / max(sigma, 1e-8)
            # Coverage indicator: 0 if within interval, 1 if outside.
            # Interval width scales with the adaptive alpha_t: a miss pushes
            # alpha_t down (wider interval), a hit pushes it back up.
            threshold = norm.ppf(1 - alpha_t[col] / 2)
            err = 0 if z <= threshold else 1
            alpha_t[col] = dtaci_update(states[col], err)
            # Lower alpha_t = better coverage = higher confidence
            # Map to [min_scale, max_scale]
            conf = 1.0 - alpha_t[col]
            scale = min_scale + (max_scale - min_scale) * max(0.0, min(1.0, conf))
            scales.loc[curr_date, col] = scale

    return tw * scales


def fetch_price_panel(
    symbols: list[str],
    *,
    market: str = "us",
    provider: str = "auto",
    start: str | None = "2020-01-01",
    end: str | None = None,
    data_dir: Any = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Convenience: download multiple symbols into a close panel."""
    from quantkit.data import fetch_ohlcv

    series: dict[str, pd.Series] = {}
    for sym in symbols:
        df = fetch_ohlcv(
            sym,
            market=market,  # type: ignore[arg-type]
            provider=provider,  # type: ignore[arg-type]
            start=start,
            end=end,
            data_dir=data_dir,
            force_refresh=force_refresh,
        )
        if not df.empty:
            series[sym] = df["close"].rename(sym)
    return align_prices(series)

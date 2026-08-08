"""Cross-sectional stock selection (universe → factors → rank → weights).

Research-grade mid/low frequency screener. Not a live broker feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from quantkit.data.panel import fetch_close_panel
from quantkit.indicators import add_core_indicators, rsi

ScoreMode = Literal["momentum", "quality_trend", "mean_reversion", "composite"]


@dataclass
class SelectionResult:
    """One cross-section selection snapshot."""

    asof: pd.Timestamp
    scores: pd.DataFrame  # symbol-level factor table + total score
    selected: list[str]
    target_weights: pd.Series
    rejected: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return self.scores.copy()


def _cross_z(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd is None or sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def score_universe(
    prices: pd.DataFrame,
    *,
    mode: ScoreMode = "composite",
    asof: pd.Timestamp | str | None = None,
    mom_lookback: int = 60,
    vol_lookback: int = 20,
    ma_fast: int = 20,
    ma_slow: int = 50,
) -> pd.DataFrame:
    """Score each symbol on the last available bar (or ``asof``).

    Returns a DataFrame indexed by symbol with factor columns and ``score``.
    """
    px = prices.astype(float).sort_index().ffill()
    if asof is not None:
        ts = pd.Timestamp(asof)
        px = px.loc[:ts]
    if px.empty or len(px) < max(mom_lookback, ma_slow) + 5:
        return pd.DataFrame()

    last = px.index[-1]
    rows: list[dict[str, Any]] = []
    for sym in px.columns:
        s = px[sym].dropna()
        if len(s) < max(mom_lookback, ma_slow) + 5:
            continue
        close = s
        ret_20 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else np.nan
        ret_60 = (
            float(close.iloc[-1] / close.iloc[-1 - mom_lookback] - 1)
            if len(close) > mom_lookback
            else np.nan
        )
        rets = close.pct_change()
        vol = float(rets.tail(vol_lookback).std() * np.sqrt(252)) if len(rets) >= vol_lookback else np.nan
        sma_f = float(close.tail(ma_fast).mean())
        sma_s = float(close.tail(ma_slow).mean())
        dist_ma = close.iloc[-1] / sma_s - 1.0 if sma_s else np.nan
        trend = 1.0 if sma_f > sma_s else 0.0
        r = float(rsi(close, 14).iloc[-1])
        # risk-adj momentum
        mom_qa = ret_60 / vol if vol and vol > 1e-8 else np.nan
        rows.append(
            {
                "symbol": sym,
                "asof": last,
                "close": float(close.iloc[-1]),
                "ret_20": ret_20,
                "ret_60": ret_60,
                "vol_ann": vol,
                "mom_quality": mom_qa,
                "dist_sma50": dist_ma,
                "trend_bull": trend,
                "rsi_14": r,
            }
        )

    df = pd.DataFrame(rows).set_index("symbol")
    if df.empty:
        return df

    # cross-sectional z-scores (higher better unless noted)
    z_mom = _cross_z(df["ret_60"])
    z_mom_q = _cross_z(df["mom_quality"])
    z_trend = _cross_z(df["trend_bull"])
    z_dist = _cross_z(df["dist_sma50"])
    # mean-reversion: prefer lower RSI / lower short ret
    z_mr = _cross_z(-df["rsi_14"]) + 0.5 * _cross_z(-df["ret_20"])
    # low vol quality tilt
    z_lowvol = _cross_z(-df["vol_ann"])

    if mode == "momentum":
        score = 0.7 * z_mom + 0.3 * z_mom_q
    elif mode == "quality_trend":
        score = 0.4 * z_mom_q + 0.3 * z_trend + 0.3 * z_lowvol
    elif mode == "mean_reversion":
        score = z_mr
    else:  # composite
        score = 0.35 * z_mom + 0.25 * z_mom_q + 0.2 * z_trend + 0.1 * z_lowvol + 0.1 * z_dist

    df["score"] = score
    df["rank"] = df["score"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("rank")


def apply_filters(
    scores: pd.DataFrame,
    *,
    min_price: float | None = 5.0,
    max_vol_ann: float | None = 1.2,
    require_uptrend: bool = False,
    min_ret_60: float | None = None,
    exclude: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (passed, rejected_with_reason)."""
    if scores.empty:
        return scores, scores
    ok = pd.Series(True, index=scores.index)
    reasons = pd.Series("", index=scores.index, dtype=object)

    def fail(mask: pd.Series, reason: str) -> None:
        nonlocal ok, reasons
        hit = mask.fillna(False)
        reasons = reasons.mask(hit & ok, reason)
        ok = ok & ~hit

    if min_price is not None and "close" in scores:
        fail(scores["close"] < min_price, f"price<{min_price}")
    if max_vol_ann is not None and "vol_ann" in scores:
        fail(scores["vol_ann"] > max_vol_ann, f"vol>{max_vol_ann}")
    if require_uptrend and "trend_bull" in scores:
        fail(scores["trend_bull"] < 0.5, "not_uptrend")
    if min_ret_60 is not None and "ret_60" in scores:
        fail(scores["ret_60"] < min_ret_60, f"ret60<{min_ret_60}")
    if exclude:
        fail(scores.index.isin(list(exclude)), "excluded")

    passed = scores.loc[ok].copy()
    rejected = scores.loc[~ok].copy()
    rejected["reject_reason"] = reasons.loc[~ok]
    return passed, rejected


def select_top_n(
    scores: pd.DataFrame,
    n: int = 5,
    *,
    weight_scheme: Literal["equal", "score"] = "equal",
    max_weight: float = 0.35,
) -> tuple[list[str], pd.Series]:
    """Pick top-N by score; return symbols + target weights (sum≈1)."""
    if scores.empty or n <= 0:
        return [], pd.Series(dtype=float)
    top = scores.sort_values("score", ascending=False).head(n)
    syms = list(top.index)
    if weight_scheme == "score":
        raw = top["score"].clip(lower=0)
        if raw.sum() <= 0:
            w = pd.Series(1.0 / len(syms), index=syms)
        else:
            w = raw / raw.sum()
    else:
        w = pd.Series(1.0 / len(syms), index=syms)
    w = w.clip(upper=max_weight)
    if w.sum() > 0:
        w = w / w.sum()
    return syms, w.rename("target_weight")


def run_selection(
    universe: Sequence[str],
    *,
    start: str = "2020-01-01",
    mode: ScoreMode = "composite",
    top_n: int = 5,
    market: str = "us",
    provider: str = "yahoo",
    data_dir: Any = None,
    asof: str | None = None,
    weight_scheme: Literal["equal", "score"] = "equal",
    max_weight: float = 0.35,
    min_price: float | None = 5.0,
    max_vol_ann: float | None = 1.2,
    require_uptrend: bool = False,
    exclude: Sequence[str] | None = None,
    force_refresh: bool = False,
) -> SelectionResult:
    """End-to-end: download panel → score → filter → top-N weights."""
    prices = fetch_close_panel(
        list(universe),
        market=market,
        provider=provider,
        start=start,
        data_dir=data_dir,
        force_refresh=force_refresh,
    )
    scores = score_universe(prices, mode=mode, asof=asof)
    passed, rejected = apply_filters(
        scores,
        min_price=min_price,
        max_vol_ann=max_vol_ann,
        require_uptrend=require_uptrend,
        exclude=exclude,
    )
    selected, weights = select_top_n(
        passed, n=top_n, weight_scheme=weight_scheme, max_weight=max_weight
    )
    asof_ts = scores["asof"].iloc[0] if not scores.empty else pd.Timestamp.utcnow()
    # attach weight column on full score table
    out = scores.copy()
    out["selected"] = out.index.isin(selected)
    out["target_weight"] = out.index.map(lambda s: float(weights.get(s, 0.0)))
    return SelectionResult(
        asof=pd.Timestamp(asof_ts),
        scores=out,
        selected=selected,
        target_weights=weights,
        rejected=rejected,
        meta={
            "mode": mode,
            "top_n": top_n,
            "universe_size": len(universe),
            "scored": len(scores),
            "passed_filters": len(passed),
            "weight_scheme": weight_scheme,
        },
    )

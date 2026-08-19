"""Factor engineering and simple ML ranking helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from quantkit.indicators import add_core_indicators

logger = logging.getLogger(__name__)


FEATURE_COLS_DEFAULT = [
    "rsi_14",
    "macd",
    "macd_hist",
    "ret_5",
    "ret_20",
    "vol_20",
    "sma_20",
    "sma_50",
]


def build_feature_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → indicator features + forward return label (5-day)."""
    feat = add_core_indicators(ohlcv)
    feat["fwd_ret_5"] = feat["close"].pct_change(5).shift(-5)
    # price-normalized distances
    feat["dist_sma20"] = feat["close"] / feat["sma_20"] - 1.0
    feat["dist_sma50"] = feat["close"] / feat["sma_50"] - 1.0
    return feat


def zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mu = series.rolling(window, min_periods=max(10, window // 3)).mean()
    sd = series.rolling(window, min_periods=max(10, window // 3)).std()
    return (series - mu) / sd.replace(0, np.nan)


def combine_factors(
    feat: pd.DataFrame,
    cols: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
) -> pd.Series:
    """Equal-weight (or weighted) z-scored factor composite."""
    cols = list(cols or ["rsi_14", "dist_sma20", "ret_20", "macd_hist"])
    # invert RSI so high RSI is not always "long"
    pieces = []
    for c in cols:
        if c not in feat.columns:
            continue
        s = feat[c].astype(float)
        if c == "rsi_14":
            s = 50 - s  # prefer mid/mean-reversion tilt as demo default
        pieces.append(zscore(s))
    if not pieces:
        return pd.Series(index=feat.index, dtype=float)
    mat = pd.concat(pieces, axis=1)
    if weights is None:
        score = mat.mean(axis=1)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        score = (mat * w[: mat.shape[1]]).sum(axis=1)
    return score.rename("factor_score")


@dataclass
class MLRankResult:
    model: object
    feature_cols: list[str]
    predictions: pd.Series
    importance: pd.DataFrame | None


def train_lightgbm_ranker(
    feat: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    label_col: str = "fwd_ret_5",
    min_train: int = 120,
    oos_only: bool = True,
) -> MLRankResult | None:
    """Hold-out LightGBM regression on forward returns.

    Trains on the first 70% of non-null rows. By default ``predictions`` are
    **out-of-sample only** (last 30%) to avoid trivial in-sample leakage in demos.
    Returns None if data is too short or lightgbm is unavailable.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    cols = list(feature_cols or FEATURE_COLS_DEFAULT)
    cols = [c for c in cols if c in feat.columns]
    if label_col not in feat.columns or not cols:
        return None

    data = feat[cols + [label_col]].dropna()
    if len(data) < min_train:
        return None

    split = int(len(data) * 0.7)
    train, test = data.iloc[:split], data.iloc[split:]
    if test.empty:
        return None
    model = lgb.LGBMRegressor(
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train[cols], train[label_col])
    if oos_only:
        pred = pd.Series(model.predict(test[cols]), index=test.index, name="ml_score")
    else:
        pred = pd.Series(model.predict(data[cols]), index=data.index, name="ml_score")
    imp = pd.DataFrame(
        {"feature": cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return MLRankResult(model=model, feature_cols=cols, predictions=pred, importance=imp)


def signal_from_score(
    score: pd.Series,
    long_quantile: float = 0.7,
    short_quantile: float = 0.3,
    long_only: bool = True,
) -> pd.Series:
    """Map continuous score → position {-1, 0, +1} using expanding quantiles."""
    s = score.astype(float)
    hi = s.expanding(min_periods=40).quantile(long_quantile)
    lo = s.expanding(min_periods=40).quantile(short_quantile)
    pos = pd.Series(0.0, index=s.index)
    pos = pos.mask(s >= hi, 1.0)
    if not long_only:
        pos = pos.mask(s <= lo, -1.0)
    return pos.rename("position")


# ---------------------------------------------------------------------------
# Huatai unified cross-sectional preprocessing pipeline
# (docs/library_digest/finance_ml_models.md transferable list 1 — all 7 Huatai AI
# reports shared; Huatai21 'Genetic Programming-based Stock Selection Factor Mining' fitness pipeline)
# ---------------------------------------------------------------------------


def _ols_residual_per_date(
    s: pd.Series, industry: pd.Series, mktcap: pd.Series
) -> pd.Series:
    """Residual of ``factor ~ 1 + log(mktcap) + industry dummies`` per date.

    Industry/size neutralization residual (step 3 of the Huatai (华泰) unified
    preprocessing pipeline). Dates with too few valid names to regress are
    left as NaN (excluded downstream) rather than silently passed through
    un-neutralized.
    """
    logcap = np.log(mktcap.where(mktcap > 0))
    out = pd.Series(np.nan, index=s.index, dtype=float)
    for _, cs in s.groupby(level=0):
        ind = industry.loc[cs.index]
        cap = logcap.loc[cs.index]
        valid = cs.notna() & ind.notna() & cap.notna()
        n = int(valid.sum())
        y = cs[valid].to_numpy(dtype=float)
        dummies = pd.get_dummies(ind[valid].astype(str), drop_first=True, dtype=float)
        X = np.column_stack(
            [np.ones(n), cap[valid].to_numpy(dtype=float), dummies.to_numpy(dtype=float)]
        )
        if n <= X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[cs.index[valid.to_numpy()]] = y - X @ beta
    return out


def preprocess_cross_section(
    frame: pd.DataFrame,
    factor_cols: Sequence[str],
    industry_col: str | None = None,
    mktcap_col: str | None = None,
    mad_k: float = 5.0,
) -> pd.DataFrame:
    """Huatai (华泰) unified factor preprocessing pipeline, applied per date cross-section.

    Steps for each column of ``factor_cols`` (the shared flow across all 7
    reports of the Huatai AI series; transferable list 1 of
    finance_ml_models.md):

      1. **MAD winsorization** — clip at ``median ± mad_k · MAD`` with
         ``MAD = median(|x − median|)`` (report convention DM ± 5·DM1).
      2. **Missing-value fill** — industry-date mean when ``industry_col`` is
         given, falling back to the date median (global fallback).
      3. **Industry/size neutralization** — OLS residual of
         ``factor ~ 1 + log(mktcap) + industry dummies`` per date. SKIPPED
         with an explicit log note unless BOTH ``industry_col`` and
         ``mktcap_col`` are provided.
      4. **Cross-sectional zscore standardization** (constant cross-section → 0, no information).

    ``frame`` must be a long-form panel with a 2-level ``(date, asset)``
    MultiIndex. Returns a copy; non-factor columns pass through unchanged.
    """
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ValueError("frame must have a 2-level (date, asset) MultiIndex")
    out = frame.copy()
    for col in factor_cols:
        if col not in out.columns:
            raise ValueError(f"factor column missing: {col}")
        s = out[col].astype(float)
        # 1. MAD winsorize per date
        med = s.groupby(level=0).transform("median")
        mad = (s - med).abs().groupby(level=0).transform("median")
        s = s.clip(med - mad_k * mad, med + mad_k * mad)
        # 2. fill missing: industry mean → date median
        if industry_col is not None:
            ind_mean = s.groupby(
                [out.index.get_level_values(0), out[industry_col]]
            ).transform("mean")
            s = s.fillna(ind_mean)
        s = s.fillna(s.groupby(level=0).transform("median"))
        # 3. industry / size neutralization (only when both provided)
        if industry_col is not None and mktcap_col is not None:
            s = _ols_residual_per_date(
                s, out[industry_col], out[mktcap_col].astype(float)
            )
        else:
            logger.info(
                "preprocess_cross_section: neutralization skipped for %r "
                "(needs BOTH industry_col and mktcap_col)",
                col,
            )
        # 4. cross-sectional zscore
        mu = s.groupby(level=0).transform("mean")
        sd = s.groupby(level=0).transform("std")
        z = (s - mu) / sd.where(sd > 0)
        out[col] = z.where(sd > 0, 0.0)
    return out


# ---------------------------------------------------------------------------
# Walk-forward LightGBM ranker with equal-weight seed ensemble
# (docs/library_digest/finance_factor_mining_models.md transferable list 1 —
# AlphaNet/TFT unified training protocol: rolling window + semi-annual rolling retraining + 10 seeds equal-weight ensemble)
# ---------------------------------------------------------------------------


@dataclass
class WFRankResult:
    """Result of :func:`train_lightgbm_ranker_walkforward`.

    ``predictions`` covers OOS test blocks only (NaN elsewhere). ``summary``
    aggregates per-fold IC stats and contrasts the seed-ensemble per-fold
    RankIC std against the mean single-seed per-fold std (Huatai discipline:
    the ensemble is not only best on the metric but also has the smallest std
    — transferable list 3 of finance_ml_models.md).
    """

    predictions: pd.Series
    fold_metrics: pd.DataFrame
    seed_metrics: pd.DataFrame
    summary: dict
    feature_cols: list[str] = field(default_factory=list)
    n_seeds: int = 0


def _mean_daily_ic(pred: pd.Series, label: pd.Series) -> tuple[float, float]:
    """Mean per-date Pearson IC and Spearman RankIC of pred vs label."""
    df = pd.DataFrame({"p": pred, "y": label}).dropna()
    ics, rics = [], []
    for _, g in df.groupby(level=0):
        if len(g) < 5 or g["p"].nunique() < 2 or g["y"].nunique() < 2:
            continue
        ics.append(g["p"].corr(g["y"]))
        rics.append(g["p"].corr(g["y"], method="spearman"))
    if not ics:
        return float("nan"), float("nan")
    return float(np.mean(ics)), float(np.mean(rics))


def train_lightgbm_ranker_walkforward(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    n_splits: int = 4,
    val_months: int = 6,
    purge_bars: int = 5,
    n_seeds: int = 10,
    lgbm_params: dict | None = None,
    seed0: int = 42,
) -> WFRankResult:
    """Walk-forward LightGBM regression ranker with equal-weight seed ensemble.

    Training protocol (AlphaNet trilogy / TFT unified protocol; transferable
    list 1 of finance_factor_mining_models.md): the source reports' "1500-day
    window / semi-annual rolling / 10-seed equal weight" is adapted to the
    expanding windows of ``quantkit.validation.walk_forward_splits`` — splits
    fall only on month boundaries, the test block rolls forward ``val_months``
    months (≈half a year) each time, and ``purge_bars`` leaves an embargo gap
    at the tail of each training set to prevent overlapping-label leakage
    (the dual protocol of update_ml_validation.md: purged walk-forward as the
    final deployable OOS). Each fold trains ``n_seeds`` models with different
    ``random_state``; OOS predictions are the equal-weight average.

    Labels should be **regression** labels (standardized next-period/excess
    returns), not classification (Huatai series 17: regression labels beat
    classification overall).

    ``frame`` must be a long-form panel with a 2-level ``(date, asset)``
    MultiIndex sorted by date. Rows with NaN in any feature/label are dropped.

    Returns per-fold OOS ensemble predictions + per-fold IC/RankIC table +
    aggregated stats. Raises ImportError if lightgbm is unavailable (the
    legacy :func:`train_lightgbm_ranker` silently returns None instead).
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "train_lightgbm_ranker_walkforward requires lightgbm; "
            "install it into the shared venv or use the legacy "
            "train_lightgbm_ranker (returns None when missing)"
        ) from exc

    from quantkit.validation import walk_forward_splits

    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ValueError("frame must have a 2-level (date, asset) MultiIndex")
    cols = [c for c in feature_cols if c in frame.columns]
    if label_col not in frame.columns or not cols:
        raise ValueError("label_col missing or no feature_cols present in frame")

    data = frame[cols + [label_col]].dropna()
    if data.empty:
        raise ValueError("no rows left after dropna on features + label")
    dates = data.index.get_level_values(0)
    uniq = dates.unique()

    params = dict(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        verbosity=-1,
    )
    params.update(lgbm_params or {})

    preds = pd.Series(np.nan, index=data.index, dtype=float, name="ml_score_wf")
    fold_rows: list[dict] = []
    seed_rows: list[dict] = []
    for k, (tr_d, te_d) in enumerate(
        walk_forward_splits(
            uniq, n_splits=n_splits, test_months=val_months, purge_bars=purge_bars
        )
    ):
        tr_dates, te_dates = uniq[tr_d], uniq[te_d]
        train = data[dates.isin(tr_dates)]
        test = data[dates.isin(te_dates)]
        if train.empty or test.empty:
            continue
        seed_pred: dict[int, pd.Series] = {}
        for i in range(n_seeds):
            model = lgb.LGBMRegressor(random_state=seed0 + i, **params)
            model.fit(train[cols], train[label_col])
            seed_pred[seed0 + i] = pd.Series(
                model.predict(test[cols]), index=test.index
            )
        ens = sum(seed_pred.values()) / float(n_seeds)  # seed equal-weight ensemble
        preds.loc[ens.index] = ens
        ic, ric = _mean_daily_ic(ens, test[label_col])
        fold_rows.append(
            dict(
                fold=k,
                test_start=te_dates[0],
                test_end=te_dates[-1],
                n_train=len(train),
                n_test=len(test),
                ic=ic,
                rank_ic=ric,
            )
        )
        for sd_seed, sp in seed_pred.items():
            ic_s, ric_s = _mean_daily_ic(sp, test[label_col])
            seed_rows.append(dict(fold=k, seed=sd_seed, ic=ic_s, rank_ic=ric_s))

    if not fold_rows:
        raise ValueError(
            "no walk-forward folds produced — check the date span against "
            "n_splits x val_months"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    seed_metrics = pd.DataFrame(seed_rows)
    ric_mean = float(fold_metrics["rank_ic"].mean())
    ric_std = float(fold_metrics["rank_ic"].std(ddof=1))
    seed_std = seed_metrics.groupby("seed")["rank_ic"].std(ddof=1)
    summary = dict(
        n_folds=int(len(fold_metrics)),
        n_seeds=int(n_seeds),
        ic_mean=float(fold_metrics["ic"].mean()),
        ic_std=float(fold_metrics["ic"].std(ddof=1)),
        rank_ic_mean=ric_mean,
        rank_ic_std=ric_std,
        rank_ic_ir=float(ric_mean / ric_std) if ric_std > 0 else float("nan"),
        single_seed_rank_ic_std_mean=float(seed_std.mean()),
    )
    return WFRankResult(
        predictions=preds,
        fold_metrics=fold_metrics,
        seed_metrics=seed_metrics,
        summary=summary,
        feature_cols=cols,
        n_seeds=n_seeds,
    )


# ---------------------------------------------------------------------------
# Style factor families (A-share compatible)
# ---------------------------------------------------------------------------

def style_factors(
    ohlcv: pd.DataFrame,
    *,
    market_cap: pd.Series | None = None,
    industry: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute style factors from a single asset's OHLCV history.

    Returns a DataFrame with one column per factor, indexed same as ohlcv.
    Each factor is z-scored against its own trailing history (rolling
    60-bar time-series z-score, NOT a per-date cross-sectional z-score);
    values are therefore comparable across time for this asset, not
    across assets — cross-sectional standardization for a panel happens
    downstream (e.g. ``preprocess_cross_section``).

    Factors produced:
    - value: 1/PE proxy via inverse of trailing 60d return (cheaper = higher)
    - momentum: 12-1 month total return
    - quality: return stability = -rolling vol of daily returns
    - size: log market cap (or proxy via rolling average volume × close)
    - volatility: 20d realized vol (lower = low-vol factor)
    - liquidity: 20d average dollar volume (log)
    """
    close = ohlcv["close"].astype(float)
    volume = ohlcv.get("volume", pd.Series(0.0, index=ohlcv.index)).astype(float)
    ret = close.pct_change()

    factors = pd.DataFrame(index=ohlcv.index)

    # Value: inverse of trailing 60d return (mean-reversion proxy)
    factors["value"] = -close.pct_change(60)

    # Momentum: 12-1 month return (skip last month)
    mom_12 = close.pct_change(252)
    mom_1 = close.pct_change(21)
    factors["momentum"] = mom_12 - mom_1

    # Quality: negative rolling volatility (stability)
    factors["quality"] = -ret.rolling(60, min_periods=20).std()

    # Size: log market cap or proxy
    if market_cap is not None:
        factors["size"] = np.log(market_cap.reindex(ohlcv.index).clip(lower=1))
    else:
        # Proxy: rolling 60d avg dollar volume
        dollar_vol = (close * volume).rolling(60, min_periods=20).mean()
        factors["size"] = np.log(dollar_vol.clip(lower=1))

    # Low volatility: negative 20d realized vol
    factors["low_vol"] = -ret.rolling(20, min_periods=10).std()

    # Liquidity: log 20d average dollar volume
    factors["liquidity"] = np.log(
        (close * volume).rolling(20, min_periods=10).mean().clip(lower=1)
    )

    # Rolling time-series z-score per factor (trailing 60 bars)
    for col in factors.columns:
        factors[col] = zscore(factors[col], window=60)

    return factors


# ---------------------------------------------------------------------------
# GP factor mining (optional, requires gplearn)
# ---------------------------------------------------------------------------

def gp_mine_factors(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    population_size: int = 500,
    generations: int = 10,
    tournament_size: int = 20,
    parsimony_coefficient: float = 0.01,
    n_factors: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Genetic programming factor mining via gplearn.

    Searches for nonlinear feature combinations that maximize IC with
    forward returns.  Returns top N GP factors as a DataFrame.

    Parameters
    ----------
    features : T×K feature matrix (cross-sectional factors).
    labels : T-vector of forward returns.
    population_size : GP population per generation.
    generations : number of GP generations.
    tournament_size : tournament selection size.
    parsimony_coefficient : complexity penalty (higher = simpler trees).
    n_factors : number of top GP factors to return.
    random_state : seed.

    Returns
    -------
    DataFrame with n_factors columns, each a GP-derived factor.
    """
    try:
        from gplearn.genetic import SymbolicTransformer
    except ImportError:
        raise ImportError(
            "gplearn is required for GP factor mining. "
            "Install with: pip install gplearn"
        )

    aligned = features.dropna().index.intersection(labels.dropna().index)
    X = features.loc[aligned].values
    y = labels.loc[aligned].values

    st = SymbolicTransformer(
        population_size=population_size,
        generations=generations,
        tournament_size=tournament_size,
        parsimony_coefficient=parsimony_coefficient,
        hall_of_fame=n_factors * 2,
        n_components=n_factors,
        feature_names=list(features.columns),
        function_set=["add", "sub", "mul", "div", "sqrt", "log", "abs", "neg"],
        random_state=random_state,
        verbose=1,
    )
    st.fit(X, y)

    gp_arr = st.transform(X)
    cols = [f"gp_factor_{i}" for i in range(gp_arr.shape[1])]
    return pd.DataFrame(gp_arr, index=aligned, columns=cols)

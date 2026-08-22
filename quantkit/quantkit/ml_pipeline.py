"""ML Pipeline — multi-model factor-based prediction for GALAHAD/STAMMTISCH.

Inspired by quant-flow's QLib integration (Alpha158-style factors + gradient
boosting + rolling retraining), adapted for A-share/HK/US exchange calendars.

Anti-overfitting discipline is implemented locally in this module:

- unique-date three-way train/valid/test splits — a trading day is never
  torn across blocks, and ``purged_gap`` trading days (>= the label horizon)
  separate train from valid and valid from test so overlapping forward-return
  labels cannot leak across blocks;
- early stopping and model selection use the validation block only;
- reported test metrics are mean daily (cross-sectional) IC / RankIC with
  ICIR, computed on the held-out test block;
- features are standardized with train-block statistics only (per-date
  cross-sectional z-score for panels, train-fitted z-score for a single
  asset), clipped to [-3, 3]; labels are z-scored with train-block
  statistics (QLib CSZScoreNorm convention), which leaves IC/RankIC
  unchanged.

For full walk-forward / PurgedKFold / DSR / PBO evaluation of the resulting
signal, see ``quantkit.validation``.

Dependencies: lightgbm, xgboost, scikit-learn (all optional; the module
degrades gracefully when a candidate's library is missing).

Usage::

    from quantkit.ml_pipeline import Alpha158, MLPipeline

    alpha = Alpha158()
    features = alpha.calculate(ohlcv_df)

    pipe = MLPipeline(model_dir="/path/to/models")
    pipe.train_cross_sectional({"AAPL": ohlcv_df, "MSFT": ohlcv_df2})
    signal = pipe.predict_panel({"AAPL": ohlcv_df, "MSFT": ohlcv_df2})
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quantkit.indicators import atr, macd, rsi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alpha158 factor calculator (pandas-native, no QLib dependency)
# ---------------------------------------------------------------------------

# K-line pattern factors
_KBAR_EXPRS = {
    "KMID": lambda o, h, l, c, v: (c - o) / o.replace(0, np.nan),
    "KLEN": lambda o, h, l, c, v: (h - l) / o.replace(0, np.nan),
    "KSFT": lambda o, h, l, c, v: (c - o) / (h - l + 1e-12),
    "KUP": lambda o, h, l, c, v: (h - np.maximum(o, c)) / o.replace(0, np.nan),
    "KLOW": lambda o, h, l, c, v: (np.minimum(o, c) - l) / o.replace(0, np.nan),
    "KSHUP": lambda o, h, l, c, v: (h - np.maximum(o, c)) / (h - l + 1e-12),
    "KSHDN": lambda o, h, l, c, v: (np.minimum(o, c) - l) / (h - l + 1e-12),
}

# ROC windows
_ROC_WINDOWS = [1, 2, 3, 5, 10, 20, 30, 60]

# MA windows for deviation factors
_MA_WINDOWS = [5, 10, 20, 30, 60]

# Volatility windows
_VOL_WINDOWS = [5, 10, 20, 30, 60]

# Rolling statistics windows
_ROLL_WINDOWS = [5, 10, 20, 30, 60]

# Z-score clip bound applied to every standardized feature
_ZSCORE_CLIP = 3.0


class Alpha158:
    """Alpha158-style factor calculator for A-share/HK/US OHLCV data.

    Produces 62 factors from raw OHLCV: K-line patterns, price momentum,
    MA deviations, volatility, rolling statistics, volume-price relationships,
    and technical indicators (RSI/MACD/ATR reused from
    ``quantkit.indicators``).  All pandas-native, no external factor library.
    """

    def calculate(self, df: pd.DataFrame, label_periods: int = 5) -> pd.DataFrame:
        """Compute all factors from OHLCV DataFrame.

        Args:
            df: Must have columns: open, high, low, close, volume
            label_periods: Forward return period for the label column

        Returns:
            DataFrame with all factor columns + 'fwd_ret_N' label
        """
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        v = df["volume"].astype(float).replace(0, np.nan)

        result = pd.DataFrame(index=df.index)

        # --- K-line pattern factors ---
        for name, fn in _KBAR_EXPRS.items():
            result[name] = fn(o, h, l, c, v)

        # --- Price momentum (ROC) ---
        for w in _ROC_WINDOWS:
            result[f"ROC_{w}"] = c.shift(w) / c - 1

        # --- MA deviation factors ---
        for w in _MA_WINDOWS:
            ma = c.rolling(w, min_periods=max(3, w // 3)).mean()
            result[f"MA_{w}"] = ma
            result[f"MADEV_{w}"] = (c - ma) / ma.replace(0, np.nan)

        # --- Volatility factors ---
        log_ret = np.log(c / c.shift(1))
        for w in _VOL_WINDOWS:
            result[f"VOL_{w}"] = log_ret.rolling(w, min_periods=max(3, w // 3)).std()

        # --- Rolling statistics ---
        for w in _ROLL_WINDOWS:
            r = c.pct_change()
            result[f"SKEW_{w}"] = r.rolling(w, min_periods=max(5, w // 3)).skew()
            result[f"KURT_{w}"] = r.rolling(w, min_periods=max(5, w // 3)).kurt()

        # --- Volume-price relationship ---
        for w in [5, 10, 20]:
            result[f"CORR_VOL_{w}"] = c.rolling(w, min_periods=max(3, w // 3)).corr(v)
            result[f"VOL_RATIO_{w}"] = v / v.rolling(w, min_periods=max(3, w // 3)).mean()

        # --- Technical indicators (reused from quantkit.indicators) ---
        # RSI (Wilder smoothing, same convention as quantkit.indicators.rsi);
        # to_numeric converts the pd.NA placeholders to NaN
        for w in [6, 14, 24]:
            result[f"RSI_{w}"] = pd.to_numeric(rsi(c, window=w), errors="coerce")

        # MACD
        m = macd(c)
        result["MACD"] = m["macd"]
        result["MACD_SIGNAL"] = m["macd_signal"]
        result["MACD_HIST"] = m["macd_hist"]

        # Bollinger Bands
        for w in [20]:
            ma = c.rolling(w).mean()
            std = c.rolling(w).std()
            result[f"BB_UPPER_{w}"] = (ma + 2 * std - c) / c
            result[f"BB_LOWER_{w}"] = (c - ma + 2 * std) / c
            result[f"BB_WIDTH_{w}"] = 4 * std / ma.replace(0, np.nan)

        # ATR, relative to close
        for w in [14, 24]:
            result[f"ATR_{w}"] = atr(df, window=w) / c

        # --- Volume factors ---
        result["VOL_CHANGE"] = v.pct_change()
        result["VOL_MA5_RATIO"] = v / v.rolling(5, min_periods=3).mean()
        result["VOL_MA20_RATIO"] = v / v.rolling(20, min_periods=5).mean()

        # --- Price range factors ---
        result["RANGE"] = (h - l) / c
        result["RANGE_MA5"] = result["RANGE"].rolling(5, min_periods=3).mean()

        # --- Label: forward return ---
        result[f"fwd_ret_{label_periods}"] = c.pct_change(label_periods).shift(-label_periods)

        return result

    @property
    def feature_names(self) -> list[str]:
        """Return list of all factor column names (excluding label)."""
        # Dynamically compute from a dummy DataFrame
        dummy = pd.DataFrame({
            "open": [1.0] * 100,
            "high": [1.1] * 100,
            "low": [0.9] * 100,
            "close": [1.0] * 100,
            "volume": [1000.0] * 100,
        })
        result = self.calculate(dummy, label_periods=5)
        return [c for c in result.columns if not c.startswith("fwd_ret_")]


# ---------------------------------------------------------------------------
# Split / normalization / evaluation helpers
# ---------------------------------------------------------------------------

def _label_periods_from_col(label_col: str) -> int:
    """Extract the forward-return horizon from a label column name."""
    return int(label_col.rsplit("_", 1)[-1])


def _three_way_date_split(
    index: pd.Index,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    purged_gap: int = 5,
    label_periods: int = 5,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Split a date index into train/valid/test date blocks with purged gaps.

    Splits on unique dates — a trading day (all of its cross-sectional rows)
    is never torn across blocks.  ``purged_gap`` trading days are dropped
    between train/valid and between valid/test so that overlapping
    forward-return labels cannot leak across blocks.

    Raises:
        ValueError: If ``purged_gap < label_periods`` (purge would be
            ineffective) or there are too few unique dates for the split.
    """
    if purged_gap < label_periods:
        raise ValueError(
            f"purged_gap ({purged_gap}) must be >= label_periods "
            f"({label_periods}), otherwise forward-return labels overlap "
            "across the split boundary"
        )
    uniq = pd.Index(index.unique()).sort_values()
    n = len(uniq)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    i_valid = n_train + purged_gap
    i_test = i_valid + n_valid + purged_gap
    train_dates = uniq[:n_train]
    valid_dates = uniq[i_valid:i_valid + n_valid]
    test_dates = uniq[i_test:]
    if len(train_dates) == 0 or len(valid_dates) == 0 or len(test_dates) == 0:
        raise ValueError(
            f"not enough unique dates ({n}) for a purged three-way split with "
            f"train_ratio={train_ratio}, valid_ratio={valid_ratio}, "
            f"purged_gap={purged_gap}"
        )
    return train_dates, valid_dates, test_dates


def _zscore_cross_section(data: pd.DataFrame, feature_cols: list[str],
                          clip: float = _ZSCORE_CLIP) -> None:
    """Per-date cross-sectional z-score in place, clipped to [-clip, clip].

    Must be applied per split segment (after the date split), so no date
    outside a segment ever contributes statistics to it.  NaNs (including
    zero-variance dates) become 0, the cross-sectional mean.
    """
    grouped = data.groupby(level=0)
    for col in feature_cols:
        mu = grouped[col].transform("mean")
        sd = grouped[col].transform("std")
        data.loc[:, col] = ((data[col] - mu) / sd.where(sd > 0)).clip(-clip, clip).fillna(0.0)


def _zscore_train_fit(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    feature_cols: list[str],
    clip: float = _ZSCORE_CLIP,
) -> dict:
    """Time-series z-score: fit mean/std on train, transform all segments.

    Returns the fitted statistics as a JSON-serializable normalization
    descriptor so inference can apply the exact same transform.
    """
    mu = train[feature_cols].mean()
    sd = train[feature_cols].std(ddof=1).replace(0, np.nan)
    for seg in (train, *others):
        seg.loc[:, feature_cols] = (
            ((seg[feature_cols] - mu) / sd).clip(-clip, clip).fillna(0.0)
        )
    return {
        "kind": "train_fit",
        "clip": clip,
        "mean": {k: float(v) for k, v in mu.items()},
        "std": {k: float(v) for k, v in sd.items()},
    }


def _zscore_label(train: pd.DataFrame, *others: pd.DataFrame,
                  label_col: str) -> None:
    """Z-score the label with train-block statistics, in place.

    QLib CSZScoreNorm convention: keeps gradient boosting numerically
    well-conditioned (L1/L2 penalties assume O(1) targets).  IC/RankIC are
    invariant to affine label transforms, so metrics are unaffected.
    """
    mu = float(train[label_col].mean())
    sd = float(train[label_col].std(ddof=1))
    if not sd > 0:
        return
    for seg in (train, *others):
        seg.loc[:, label_col] = (seg[label_col] - mu) / sd


def _daily_ic_series(pred: pd.Series, label: pd.Series, method: str = "pearson") -> pd.Series:
    """Per-date cross-sectional IC between predictions and labels.

    Rows are grouped by index level 0 (the date); dates with fewer than 5
    observations or with constant pred/label are skipped.  Returns an empty
    Series for single-asset input (one row per date).
    """
    df = pd.DataFrame({"p": pred, "y": label}).dropna()
    ics = {}
    for d, g in df.groupby(level=0):
        if len(g) < 5 or g["p"].nunique() < 2 or g["y"].nunique() < 2:
            continue
        ics[d] = g["p"].corr(g["y"], method=method)
    return pd.Series(ics, dtype=float)


def _ic_stats(pred: pd.Series, label: pd.Series) -> dict:
    """Mean daily IC / RankIC and ICIR of pred vs label.

    ICIR is mean(RankIC) / std(RankIC) over dates (the convention of
    ``quantkit.factors.train_lightgbm_ranker_walkforward``).  Falls back to
    pooled correlation for single-asset series, where per-date IC is
    undefined and ICIR is reported as NaN.
    """
    ic = _daily_ic_series(pred, label)
    ric = _daily_ic_series(pred, label, method="spearman")
    if ic.empty:
        df = pd.DataFrame({"p": pred, "y": label}).dropna()
        if len(df) < 5 or df["p"].nunique() < 2 or df["y"].nunique() < 2:
            return {"ic": float("nan"), "rank_ic": float("nan"),
                    "icir": float("nan"), "n_dates": 0}
        return {
            "ic": float(df["p"].corr(df["y"])),
            "rank_ic": float(df["p"].corr(df["y"], method="spearman")),
            "icir": float("nan"),
            "n_dates": 0,
        }
    ric_mean = float(ric.mean())
    ric_std = float(ric.std(ddof=1)) if len(ric) > 1 else float("nan")
    return {
        "ic": float(ic.mean()),
        "rank_ic": ric_mean,
        "icir": float(ric_mean / ric_std) if ric_std > 0 else float("nan"),
        "n_dates": int(len(ic)),
    }


# ---------------------------------------------------------------------------
# Multi-model trainer
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Result of training a single model.

    IC metrics are mean daily (cross-sectional) IC for panel training and
    pooled IC for single-asset training; ICIR is mean(RankIC)/std(RankIC)
    over dates (NaN for single-asset training).
    """
    model_type: str
    model: Any
    feature_cols: list[str]
    train_ic: float = 0.0
    valid_ic: float = 0.0
    test_ic: float = 0.0
    valid_rank_ic: float = 0.0
    test_rank_ic: float = 0.0
    valid_icir: float = float("nan")
    test_icir: float = float("nan")
    importance: pd.DataFrame | None = None
    train_time_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


class MLPipeline:
    """Multi-model factor-based prediction pipeline.

    Trains LightGBM, XGBoost, and Linear models in competition on a
    purged three-way date split: early stopping and model selection use the
    validation block only, and held-out test IC/ICIR is reported for the
    selected model.  Persists models for online inference.
    """

    MODEL_CONFIGS = {
        "lightgbm": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 128,
            "max_depth": 6,
            "colsample_bytree": 0.85,
            "subsample": 0.85,
            "lambda_l1": 10,
            "lambda_l2": 10,
            "num_threads": 4,
            "verbose": -1,
        },
        "xgboost": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "colsample_bytree": 0.85,
            "subsample": 0.85,
            "reg_alpha": 10,
            "reg_lambda": 10,
            "early_stopping_rounds": 50,
            "verbosity": 0,
        },
    }

    def __init__(
        self,
        model_dir: str | Path = "models/ml",
        candidates: Sequence[str] | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.candidates = list(candidates or ["lightgbm", "xgboost"])
        self.alpha = Alpha158()
        self._best_model: ModelResult | None = None
        self._model_versions: list[dict] = []
        versions_path = self.model_dir / "versions.json"
        if versions_path.exists():
            try:
                self._model_versions = json.loads(versions_path.read_text())
            except Exception as e:
                logger.warning(f"Could not load model version history: {e}")

    @property
    def best_model(self) -> ModelResult | None:
        return self._best_model

    @property
    def model_versions(self) -> list[dict]:
        return list(self._model_versions)

    def train(
        self,
        df: pd.DataFrame,
        label_col: str = "fwd_ret_5",
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2,
        purged_gap: int = 5,
        custom_params: dict | None = None,
    ) -> list[ModelResult]:
        """Train all candidate models on a single asset and select the best.

        Args:
            df: OHLCV DataFrame
            label_col: Label column name
            train_ratio: Fraction of unique dates for training
            valid_ratio: Fraction of unique dates for validation (the rest,
                minus two ``purged_gap`` embargoes, is the held-out test block)
            purged_gap: Purge gap in bars between blocks; must be >= the
                label horizon
            custom_params: Override model params

        Returns:
            List of ModelResult for each candidate
        """
        label_periods = _label_periods_from_col(label_col)
        features = self.alpha.calculate(df, label_periods=label_periods)
        feature_cols = [c for c in features.columns if not c.startswith("fwd_ret_")]
        data = features[feature_cols + [label_col]].dropna()

        if len(data) < 100:
            logger.warning(f"Too few samples ({len(data)}), need >= 100")
            return []

        train_dates, valid_dates, test_dates = _three_way_date_split(
            data.index, train_ratio, valid_ratio, purged_gap, label_periods
        )
        train_data, valid_data, test_data = self._slice(data, train_dates, valid_dates, test_dates)

        # Fit standardization on the train block only, then transform
        # valid/test; clipped to [-3, 3] so outliers cannot dominate.
        norm_meta = _zscore_train_fit(train_data, valid_data, test_data,
                                      feature_cols=feature_cols)
        _zscore_label(train_data, valid_data, test_data, label_col=label_col)

        logger.info(
            f"Training {len(self.candidates)} models on {len(train_data)} train / "
            f"{len(valid_data)} valid / {len(test_data)} test samples, "
            f"{len(feature_cols)} factors"
        )

        results = self._train_candidates(
            train_data, valid_data, test_data, feature_cols, label_col, custom_params
        )
        for r in results:
            r.metadata["normalization"] = norm_meta
        self._select_best(results)
        return results

    def train_cross_sectional(
        self,
        data_dict: dict[str, pd.DataFrame],
        label_col: str = "fwd_ret_5",
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2,
        purged_gap: int = 5,
        custom_params: dict | None = None,
    ) -> list[ModelResult]:
        """Train on a cross-sectional panel (multiple stocks stacked).

        This is the correct way to use Alpha158 — factors are designed
        for cross-sectional ranking, not single-stock time-series.

        The panel is split by unique trading dates into train/valid/test
        blocks with ``purged_gap`` trading-day embargoes, and each block is
        z-scored per date within itself (clipped to [-3, 3]) so no block
        contributes normalization statistics to another.

        Args:
            data_dict: {symbol: ohlcv_df} dict
            label_col: Label column name
            train_ratio: Fraction of unique dates for training
            valid_ratio: Fraction of unique dates for validation
            purged_gap: Purge gap in trading days between blocks; must be
                >= the label horizon
            custom_params: Override model params

        Returns:
            List of ModelResult for each candidate
        """
        label_periods = _label_periods_from_col(label_col)
        panels = []
        for symbol, df in data_dict.items():
            if len(df) < 100:
                continue
            features = self.alpha.calculate(df, label_periods=label_periods)
            features["symbol"] = symbol
            panels.append(features)

        if not panels:
            logger.warning("No valid data for cross-sectional training")
            return []

        panel = pd.concat(panels, axis=0).sort_index()
        feature_cols = [c for c in panel.columns
                        if not c.startswith("fwd_ret_") and c != "symbol"]

        data = panel[feature_cols + [label_col]].dropna()

        if len(data) < 200:
            logger.warning(f"Too few samples ({len(data)}), need >= 200")
            return []

        train_dates, valid_dates, test_dates = _three_way_date_split(
            data.index, train_ratio, valid_ratio, purged_gap, label_periods
        )
        train_data, valid_data, test_data = self._slice(data, train_dates, valid_dates, test_dates)

        # Per-date cross-sectional z-score, applied within each block
        # separately so blocks never share normalization statistics.
        for seg in (train_data, valid_data, test_data):
            _zscore_cross_section(seg, feature_cols)
        norm_meta = {"kind": "cross_sectional", "clip": _ZSCORE_CLIP}
        _zscore_label(train_data, valid_data, test_data, label_col=label_col)

        logger.info(
            f"Cross-sectional training: {len(panels)} symbols, "
            f"{len(train_data)} train / {len(valid_data)} valid / "
            f"{len(test_data)} test samples, {len(feature_cols)} factors"
        )

        results = self._train_candidates(
            train_data, valid_data, test_data, feature_cols, label_col, custom_params
        )
        for r in results:
            r.metadata["normalization"] = norm_meta
        self._select_best(results)
        return results

    @staticmethod
    def _slice(data, train_dates, valid_dates, test_dates):
        idx = data.index
        return (
            data.loc[idx.isin(train_dates)].copy(),
            data.loc[idx.isin(valid_dates)].copy(),
            data.loc[idx.isin(test_dates)].copy(),
        )

    def _train_candidates(self, train_data, valid_data, test_data,
                          feature_cols, label_col, custom_params) -> list[ModelResult]:
        results = []
        for model_type in self.candidates:
            result = self._train_one(
                model_type, train_data, valid_data, test_data,
                feature_cols, label_col, custom_params,
            )
            if result:
                results.append(result)
        return results

    def _select_best(self, results: list[ModelResult]) -> None:
        """Select the best candidate by validation IC and persist it."""
        if not results:
            return
        self._best_model = max(results, key=lambda r: r.valid_ic)
        self._save_best()
        logger.info(
            f"Best model: {self._best_model.model_type} "
            f"(valid_ic={self._best_model.valid_ic:.4f}, "
            f"test_ic={self._best_model.test_ic:.4f})"
        )

    def _train_one(
        self,
        model_type: str,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        custom_params: dict | None,
    ) -> ModelResult | None:
        """Train a single model type; early stopping uses the valid block."""
        t0 = time.time()
        params = {**self.MODEL_CONFIGS.get(model_type, {})}
        if custom_params and model_type in custom_params:
            params.update(custom_params[model_type])

        X_train, y_train = train_data[feature_cols], train_data[label_col]
        X_valid, y_valid = valid_data[feature_cols], valid_data[label_col]
        X_test, y_test = test_data[feature_cols], test_data[label_col]

        try:
            if model_type == "lightgbm":
                model = self._train_lightgbm(X_train, y_train, X_valid, y_valid, params)
            elif model_type == "xgboost":
                model = self._train_xgboost(X_train, y_train, X_valid, y_valid, params)
            elif model_type == "linear":
                model = self._train_linear(X_train, y_train)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            return None

        elapsed = time.time() - t0

        # Evaluate: train/valid/test blocks stay strictly separate
        train_stats = _ic_stats(pd.Series(model.predict(X_train), index=X_train.index), y_train)
        valid_stats = _ic_stats(pd.Series(model.predict(X_valid), index=X_valid.index), y_valid)
        test_stats = _ic_stats(pd.Series(model.predict(X_test), index=X_test.index), y_test)

        # Feature importance
        importance = None
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)

        result = ModelResult(
            model_type=model_type,
            model=model,
            feature_cols=feature_cols,
            train_ic=train_stats["ic"],
            valid_ic=valid_stats["ic"],
            test_ic=test_stats["ic"],
            valid_rank_ic=valid_stats["rank_ic"],
            test_rank_ic=test_stats["rank_ic"],
            valid_icir=valid_stats["icir"],
            test_icir=test_stats["icir"],
            importance=importance,
            train_time_seconds=elapsed,
            metadata={
                "n_dates_valid": valid_stats["n_dates"],
                "n_dates_test": test_stats["n_dates"],
            },
        )

        logger.info(
            f"  {model_type}: train_ic={result.train_ic:.4f} "
            f"valid_ic={result.valid_ic:.4f} test_ic={result.test_ic:.4f} "
            f"time={elapsed:.1f}s"
        )
        return result

    def _train_lightgbm(self, X_train, y_train, X_valid, y_valid, params):
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return model

    def _train_xgboost(self, X_train, y_train, X_valid, y_valid, params):
        import xgboost as xgb
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        return model

    def _train_linear(self, X_train, y_train):
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return model

    # --- Inference ---

    def _check_features(self, features: pd.DataFrame) -> list[str]:
        feature_cols = self._best_model.feature_cols
        available = [c for c in feature_cols if c in features.columns]
        if len(available) < len(feature_cols) * 0.5:
            raise RuntimeError(
                f"Too few matching features: {len(available)}/{len(feature_cols)}"
            )
        return available

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate predictions for a single asset using the best model.

        Applies the same train-fitted standardization used during training.
        If the best model was trained cross-sectionally, use
        :meth:`predict_panel` instead — per-date cross-sectional
        normalization is undefined for a single asset.

        Args:
            df: OHLCV DataFrame (same schema as training)

        Returns:
            Series of predicted returns, indexed like input
        """
        if self._best_model is None:
            raise RuntimeError("No model trained yet. Call train() first.")

        norm = self._best_model.metadata.get("normalization", {})
        if norm.get("kind") == "cross_sectional":
            raise RuntimeError(
                "Best model was trained cross-sectionally; use predict_panel() "
                "so features can be z-scored per date"
            )

        features = self.alpha.calculate(df)
        available = self._check_features(features)
        X = features[available]
        if norm.get("kind") == "train_fit":
            mu = pd.Series(norm["mean"])
            sd = pd.Series(norm["std"])
            X = ((X - mu[available]) / sd[available]).clip(-norm["clip"], norm["clip"])
        pred = pd.Series(
            self._best_model.model.predict(X.fillna(0)),
            index=X.index,
            name="ml_signal",
        )
        return pred

    def predict_panel(self, data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Cross-sectional predictions for a {symbol: ohlcv_df} dict.

        Applies the same per-date z-score (clipped to +/-3) used in
        :meth:`train_cross_sectional` and returns a date x symbol
        prediction matrix.
        """
        if self._best_model is None:
            raise RuntimeError("No model trained yet. Call train() first.")

        panels = []
        for symbol, df in data_dict.items():
            feats = self.alpha.calculate(df)
            feats["symbol"] = symbol
            panels.append(feats)
        panel = pd.concat(panels, axis=0).sort_index()

        available = self._check_features(panel)
        data = panel[available + ["symbol"]].copy()
        _zscore_cross_section(data, available)
        data["pred"] = self._best_model.model.predict(data[available])
        return (data.assign(date=data.index)
                    .pivot(index="date", columns="symbol", values="pred"))

    def generate_signal(
        self,
        df: pd.DataFrame,
        long_threshold: float = 0.7,
        short_threshold: float = 0.3,
    ) -> pd.Series:
        """Generate trading signal from predictions.

        Uses expanding quantile thresholds — only extreme predictions
        trigger positions, most bars stay flat.

        Args:
            df: OHLCV DataFrame
            long_threshold: Quantile threshold for long signal
            short_threshold: Quantile threshold for short signal

        Returns:
            Series of {-1, 0, +1} positions
        """
        pred = self.predict(df)
        # Need enough history for meaningful quantiles
        min_periods = max(60, len(pred) // 5)
        hi = pred.expanding(min_periods=min_periods).quantile(long_threshold)
        lo = pred.expanding(min_periods=min_periods).quantile(short_threshold)
        signal = pd.Series(0, index=pred.index)
        signal[pred >= hi] = 1
        signal[pred <= lo] = -1
        return signal.rename("ml_position")

    # --- Persistence ---

    def _save_best(self):
        """Save the best model and metadata."""
        if self._best_model is None:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"best_{self._best_model.model_type}_{ts}.pkl"
        meta_path = self.model_dir / "best_model.json"

        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self._best_model.model,
                "feature_cols": self._best_model.feature_cols,
                "model_type": self._best_model.model_type,
                "train_ic": self._best_model.train_ic,
                "valid_ic": self._best_model.valid_ic,
                "test_ic": self._best_model.test_ic,
                "valid_icir": self._best_model.valid_icir,
                "timestamp": ts,
                "metadata": self._best_model.metadata,
            }, f)

        meta = {
            "model_type": self._best_model.model_type,
            "model_path": str(model_path),
            "feature_cols": self._best_model.feature_cols,
            "train_ic": self._best_model.train_ic,
            "valid_ic": self._best_model.valid_ic,
            "test_ic": self._best_model.test_ic,
            "valid_icir": self._best_model.valid_icir,
            "train_time_seconds": self._best_model.train_time_seconds,
            "timestamp": ts,
            "metadata": self._best_model.metadata,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Version tracking
        self._model_versions.append(meta)
        versions_path = self.model_dir / "versions.json"
        with open(versions_path, "w") as f:
            json.dump(self._model_versions[-20:], f, indent=2)

        logger.info(f"Saved best model to {model_path}")

    def load_best(self) -> bool:
        """Load the best model from disk.

        Returns:
            True if loaded successfully
        """
        meta_path = self.model_dir / "best_model.json"
        if not meta_path.exists():
            return False

        with open(meta_path) as f:
            meta = json.load(f)

        model_path = Path(meta["model_path"])
        if not model_path.exists():
            return False

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self._best_model = ModelResult(
            model_type=data["model_type"],
            model=data["model"],
            feature_cols=data["feature_cols"],
            train_ic=data.get("train_ic", 0),
            valid_ic=data.get("valid_ic", 0),
            test_ic=data.get("test_ic", 0),
            valid_icir=data.get("valid_icir", float("nan")),
            metadata=data.get("metadata", {}),
        )
        logger.info(
            f"Loaded best model: {self._best_model.model_type} "
            f"(valid_ic={self._best_model.valid_ic:.4f})"
        )
        return True


# ---------------------------------------------------------------------------
# Online model manager (rolling retraining)
# ---------------------------------------------------------------------------

class OnlineModelManager:
    """Manages rolling retraining of ML models with a quality gate.

    Checks if enough time has passed since last training and retrains with
    the latest data — preferring the cross-sectional path (pass a
    ``{symbol: ohlcv_df}`` dict), which is the correct way to use Alpha158.

    A newly trained model only replaces the incumbent when it passes the
    quality gate: valid IC > 0, valid ICIR > 0 (when defined), and valid
    ICIR >= ``keep_ratio`` x the incumbent's.  Otherwise the incumbent model
    and its persisted metadata are kept and the rejection reason is logged
    (``last_reject_reason``).

    State (last train time, retrain count, last rejection reason) persists
    to ``online_manager.json`` inside the pipeline's model_dir and is
    restored on construction.
    """

    STATE_FILE = "online_manager.json"

    def __init__(
        self,
        pipeline: MLPipeline,
        retrain_interval_hours: float = 168,  # weekly
        min_samples: int = 200,
        keep_ratio: float = 0.8,
    ):
        self.pipeline = pipeline
        self.retrain_interval_hours = retrain_interval_hours
        self.min_samples = min_samples
        self.keep_ratio = keep_ratio
        self._last_train_time: float = 0
        self._retrain_count: int = 0
        self.last_reject_reason: str | None = None
        self._load_state()
        # Restore the incumbent so the gate can compare against it
        if self.pipeline.best_model is None:
            self.pipeline.load_best()

    @property
    def _state_path(self) -> Path:
        return self.pipeline.model_dir / self.STATE_FILE

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text())
            self._last_train_time = float(state.get("last_train_time", 0))
            self._retrain_count = int(state.get("retrain_count", 0))
            self.last_reject_reason = state.get("last_reject_reason")
        except Exception as e:
            logger.warning(f"Could not load online-manager state: {e}")

    def _save_state(self) -> None:
        state = {
            "last_train_time": self._last_train_time,
            "retrain_count": self._retrain_count,
            "last_reject_reason": self.last_reject_reason,
        }
        self._state_path.write_text(json.dumps(state, indent=2))

    def should_retrain(self) -> bool:
        """Check if retraining is due."""
        if self._last_train_time == 0:
            return True
        elapsed_hours = (time.time() - self._last_train_time) / 3600
        return elapsed_hours >= self.retrain_interval_hours

    def _passes_gate(self, new: ModelResult | None,
                     old: ModelResult | None) -> tuple[bool, str]:
        """Quality gate: accept a new model only if it is good and not degraded.

        Requires valid IC > 0, valid ICIR > 0 (when defined — ICIR is NaN
        for single-asset training), and, when an incumbent exists, the new
        ICIR (or IC when ICIR is undefined) must be >= ``keep_ratio`` x the
        incumbent's.
        """
        if new is None:
            return False, "training produced no model"
        if not np.isfinite(new.valid_ic) or new.valid_ic <= 0:
            return False, f"valid IC {new.valid_ic:.4f} <= 0"
        if np.isfinite(new.valid_icir) and new.valid_icir <= 0:
            return False, f"valid ICIR {new.valid_icir:.4f} <= 0"
        if old is not None:
            new_metric = new.valid_icir if np.isfinite(new.valid_icir) else new.valid_ic
            old_metric = old.valid_icir if np.isfinite(old.valid_icir) else old.valid_ic
            if np.isfinite(old_metric) and new_metric < old_metric * self.keep_ratio:
                return False, (
                    f"degraded: new {new_metric:.4f} < {self.keep_ratio} x "
                    f"incumbent {old_metric:.4f}"
                )
        return True, ""

    def maybe_retrain(
        self, data: pd.DataFrame | dict[str, pd.DataFrame], **kwargs
    ) -> list[ModelResult] | None:
        """Retrain if due, switching models only if the quality gate passes.

        Args:
            data: ``{symbol: ohlcv_df}`` dict (preferred — cross-sectional
                training) or a single OHLCV DataFrame (legacy single-asset
                path)

        Returns:
            List of ModelResult if retrained, None if not due
        """
        if not self.should_retrain():
            return None
        n = (sum(len(d) for d in data.values())
             if isinstance(data, dict) else len(data))
        if n < self.min_samples:
            logger.warning(f"Not enough data for retraining ({n} < {self.min_samples})")
            return None
        return self._retrain(data, **kwargs)

    def force_retrain(
        self, data: pd.DataFrame | dict[str, pd.DataFrame], **kwargs
    ) -> list[ModelResult]:
        """Force retrain regardless of schedule (quality gate still applies)."""
        logger.info("Force retraining models")
        return self._retrain(data, **kwargs)

    def _retrain(
        self, data: pd.DataFrame | dict[str, pd.DataFrame], **kwargs
    ) -> list[ModelResult]:
        logger.info(f"Retraining models (retrain #{self._retrain_count + 1})")
        old_best = self.pipeline.best_model
        meta_path = self.pipeline.model_dir / "best_model.json"
        meta_backup = meta_path.read_bytes() if meta_path.exists() else None

        if isinstance(data, dict):
            results = self.pipeline.train_cross_sectional(data, **kwargs)
        else:
            results = self.pipeline.train(data, **kwargs)
        if not results:
            return []

        accepted, reason = self._passes_gate(self.pipeline.best_model, old_best)
        if accepted:
            self.last_reject_reason = None
        else:
            # Roll back to the incumbent model and its persisted metadata
            self.pipeline._best_model = old_best
            if meta_backup is not None:
                meta_path.write_bytes(meta_backup)
            else:
                meta_path.unlink(missing_ok=True)
            self.last_reject_reason = reason
            logger.warning(
                f"New model rejected by quality gate ({reason}); "
                "keeping the incumbent"
            )

        self._last_train_time = time.time()
        self._retrain_count += 1
        self._save_state()
        return results

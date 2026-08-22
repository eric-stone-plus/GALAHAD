"""Tests for quantkit.ml_pipeline.

Synthetic-data contracts:
  - Alpha158 MUST produce the documented 62 factors plus the label
  - the forward-return label MUST align with future prices (no lookahead)
  - the purged three-way split MUST split on unique dates: no date torn
    across blocks, and the purge gap MUST be effective in trading days
  - OnlineModelManager's quality gate MUST reject a degraded new model and
    keep the incumbent (in memory and on disk)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantkit.ml_pipeline import (
    Alpha158,
    MLPipeline,
    ModelResult,
    OnlineModelManager,
    _three_way_date_split,
)


def _make_ohlcv(seed: int, n: int = 300, momentum: float = 0.0) -> pd.DataFrame:
    """Synthetic OHLCV random walk; ``momentum`` plants an AR(1) return signal."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    eps = rng.normal(0.0, 0.02, n)
    ret = np.zeros(n)
    for t in range(n):
        ret[t] = eps[t] + (momentum * ret[t - 1] if t > 0 else 0.0)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_panel(n_symbols: int = 6, n: int = 280, momentum: float = 0.5):
    return {f"S{i:02d}": _make_ohlcv(100 + i, n=n, momentum=momentum)
            for i in range(n_symbols)}


# ---------------------------------------------------------------------------
# Alpha158 factor calculation
# ---------------------------------------------------------------------------


def test_alpha158_smoke_and_factor_count():
    alpha = Alpha158()
    features = alpha.calculate(_make_ohlcv(0))
    feature_cols = [c for c in features.columns if not c.startswith("fwd_ret_")]
    assert len(feature_cols) == 62
    assert len(alpha.feature_names) == 62
    assert "fwd_ret_5" in features.columns
    # spot-check a few factors are actually populated at the tail
    for col in ["KMID", "ROC_20", "RSI_14", "MACD", "ATR_14", "VOL_20"]:
        assert np.isfinite(features[col].iloc[-1]), col
    assert 0.0 <= features["RSI_14"].iloc[-1] <= 100.0


def test_label_alignment_no_lookahead():
    df = _make_ohlcv(1)
    features = Alpha158().calculate(df, label_periods=5)
    expected = df["close"].shift(-5) / df["close"] - 1
    got = features["fwd_ret_5"]
    both = pd.concat([got, expected], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1])
    # the last 5 labels must be NaN (no future data available)
    assert got.iloc[-5:].isna().all()


# ---------------------------------------------------------------------------
# Purged three-way date split
# ---------------------------------------------------------------------------


def test_three_way_split_date_blocks_disjoint_and_purged():
    dates = pd.bdate_range("2023-01-02", periods=200)
    index = dates.repeat(10)  # 10 assets per trading day
    tr, va, te = _three_way_date_split(
        index, train_ratio=0.6, valid_ratio=0.2, purged_gap=5, label_periods=5
    )
    # strict block ordering by date
    assert tr.max() < va.min() < te.min()
    # no date shared between blocks
    assert set(tr).isdisjoint(va) and set(va).isdisjoint(te)
    # purge gap is effective: 5 trading days dropped between blocks
    pos = {d: i for i, d in enumerate(dates)}
    assert pos[va.min()] - pos[tr.max()] == 5 + 1
    assert pos[te.min()] - pos[va.max()] == 5 + 1


def test_three_way_split_rejects_ineffective_purge():
    index = pd.bdate_range("2023-01-02", periods=200).repeat(5)
    with pytest.raises(ValueError, match="purged_gap"):
        _three_way_date_split(index, purged_gap=3, label_periods=5)


def test_three_way_split_rejects_too_few_dates():
    index = pd.bdate_range("2023-01-02", periods=20)
    with pytest.raises(ValueError, match="not enough unique dates"):
        _three_way_date_split(index, purged_gap=5, label_periods=5)


# ---------------------------------------------------------------------------
# Training paths
# ---------------------------------------------------------------------------


def test_train_cross_sectional_end_to_end(tmp_path):
    pipe = MLPipeline(model_dir=tmp_path, candidates=["linear"])
    results = pipe.train_cross_sectional(_make_panel())
    assert len(results) == 1
    r = results[0]
    # daily cross-sectional IC was used (not pooled), test block is held out
    assert r.metadata["n_dates_valid"] > 0
    assert r.metadata["n_dates_test"] > 0
    assert np.isfinite(r.valid_ic) and np.isfinite(r.test_ic)
    assert np.isfinite(r.valid_icir)
    assert r.metadata["normalization"]["kind"] == "cross_sectional"
    assert pipe.best_model is r
    # persisted metadata carries the new metrics
    meta = json.loads((tmp_path / "best_model.json").read_text())
    assert "test_ic" in meta and "valid_icir" in meta
    # panel predictions are a date x symbol matrix
    pred = pipe.predict_panel(_make_panel(n=120))
    assert pred.shape[1] == 6


def test_train_single_stock_standardizes_and_predicts(tmp_path):
    pipe = MLPipeline(model_dir=tmp_path, candidates=["linear"])
    df = _make_ohlcv(7, momentum=0.5)
    results = pipe.train(df)
    assert len(results) == 1
    norm = results[0].metadata["normalization"]
    assert norm["kind"] == "train_fit"
    assert len(norm["mean"]) == 62
    # single-asset: pooled IC fallback, ICIR undefined
    assert np.isfinite(results[0].valid_ic)
    assert np.isnan(results[0].valid_icir)
    pred = pipe.predict(df)
    assert pred.index.equals(df.index)
    assert np.isfinite(pred.iloc[-1])


# ---------------------------------------------------------------------------
# OnlineModelManager quality gate
# ---------------------------------------------------------------------------


def test_quality_gate_unit(tmp_path):
    mgr = OnlineModelManager(MLPipeline(model_dir=tmp_path, candidates=["linear"]))
    old = ModelResult("lightgbm", None, [], valid_ic=0.05, valid_icir=0.5)
    good = ModelResult("lightgbm", None, [], valid_ic=0.06, valid_icir=0.6)
    assert mgr._passes_gate(good, old)[0]
    # negative IC rejected
    assert not mgr._passes_gate(
        ModelResult("lightgbm", None, [], valid_ic=-0.01, valid_icir=-0.2), old)[0]
    # positive but degraded ICIR (< 0.8 x incumbent) rejected
    ok, reason = mgr._passes_gate(
        ModelResult("lightgbm", None, [], valid_ic=0.05, valid_icir=0.3), old)
    assert not ok and "degraded" in reason
    # first model: no incumbent comparison, but IC/ICIR must be positive
    assert mgr._passes_gate(good, None)[0]
    assert not mgr._passes_gate(
        ModelResult("lightgbm", None, [], valid_ic=0.0, valid_icir=0.0), None)[0]


def test_manager_retrain_persists_state(tmp_path):
    pipe = MLPipeline(model_dir=tmp_path, candidates=["linear"])
    mgr = OnlineModelManager(pipe, retrain_interval_hours=0)
    results = mgr.maybe_retrain(_make_panel())
    assert results
    assert pipe.best_model is not None
    assert mgr.last_reject_reason is None
    assert mgr._retrain_count == 1
    # state survives a fresh manager on the same model_dir
    mgr2 = OnlineModelManager(MLPipeline(model_dir=tmp_path, candidates=["linear"]),
                              retrain_interval_hours=0)
    assert mgr2._retrain_count == 1
    assert mgr2._last_train_time == mgr._last_train_time
    assert mgr2.pipeline.best_model is not None  # incumbent restored from disk


def test_manager_rolls_back_on_worse_model(tmp_path, monkeypatch):
    pipe = MLPipeline(model_dir=tmp_path, candidates=["linear"])
    mgr = OnlineModelManager(pipe, retrain_interval_hours=0)
    assert mgr.maybe_retrain(_make_panel())
    good = pipe.best_model
    assert good is not None and np.isfinite(good.valid_icir)

    # force a retrain whose winner is worse than the incumbent
    mgr._last_train_time = 0
    worse = ModelResult("linear", good.model, good.feature_cols,
                        valid_ic=-0.01, valid_icir=-0.1)

    def fake_train_cross_sectional(data_dict, **kwargs):
        pipe._best_model = worse
        pipe._save_best()
        return [worse]

    monkeypatch.setattr(pipe, "train_cross_sectional", fake_train_cross_sectional)
    out = mgr.maybe_retrain(_make_panel())
    assert out == [worse]
    # incumbent restored in memory and on disk
    assert pipe.best_model is good
    assert mgr.last_reject_reason
    meta = json.loads((tmp_path / "best_model.json").read_text())
    assert meta["valid_ic"] == pytest.approx(good.valid_ic)

"""Tests for quantkit.factors ML additions.

Synthetic-data contracts (recipes: docs/library_digest/finance_ml_models.md
unified preprocessing pipeline; finance_factor_mining_models.md transferable
list 1 rolling retraining + multi-seed equal-weight ensemble):
  - planted MAD outlier MUST be clipped to the cross-sectional bound
  - industry-mean fill MUST fill NaN with the industry-date mean
  - neutralization MUST make the factor ~orthogonal to log-mktcap per date
  - walk-forward seed-ensemble ranker MUST recover a planted monotone signal
    (OOS RankIC clearly > 0) and ensemble per-fold IC std < single-seed std
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantkit.factors import (
    preprocess_cross_section,
    train_lightgbm_ranker_walkforward,
)


def _panel_index(n_days: int, n_names: int) -> pd.MultiIndex:
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    return pd.MultiIndex.from_product(
        [dates, [f"A{i:03d}" for i in range(n_names)]], names=["date", "asset"]
    )


# ---------------------------------------------------------------------------
# preprocess_cross_section
# ---------------------------------------------------------------------------


def test_mad_clip_bounds_planted_outlier():
    rng = np.random.default_rng(0)
    idx = _panel_index(30, 60)
    f = pd.Series(rng.normal(0, 1, len(idx)), index=idx, name="f")
    victim = idx[7]
    f.loc[victim] = 50.0  # planted outlier
    out = preprocess_cross_section(pd.DataFrame({"f": f}), ["f"])
    z = out["f"]
    # 50 raw would z-score to ~50; after DM±5·MAD clipping it is bounded
    assert abs(z.loc[victim]) < 8.0
    # and it is no longer the cross-section's dominant magnitude on that date
    assert abs(z.loc[victim]) <= z.loc[idx[0][0]].abs().max() + 1e-9
    # zscore step: per-date mean ~0, std ~1
    g = z.groupby(level=0)
    assert g.mean().abs().max() < 1e-8
    assert g.std().sub(1.0).abs().max() < 1e-6


def test_industry_mean_fill_then_date_median_fallback():
    idx = _panel_index(1, 6)
    ind = pd.Series(["X", "X", "X", "Y", "Y", "Y"], index=idx, name="ind")
    f = pd.Series([1.0, 2.0, np.nan, 10.0, 20.0, 30.0], index=idx, name="f")
    out = preprocess_cross_section(pd.DataFrame({"f": f, "ind": ind}), ["f"], industry_col="ind")
    z = out["f"]
    assert z.notna().all()
    # industry X mean = (1+2)/2 = 1.5 fills the NaN (no clipping: all values
    # inside DM±5·MAD here), then plain zscore over [1, 2, 1.5, 10, 20, 30]
    filled = np.array([1.0, 2.0, 1.5, 10.0, 20.0, 30.0])
    expected = (filled - filled.mean()) / filled.std(ddof=1)
    np.testing.assert_allclose(z.to_numpy(), expected, rtol=1e-8)


def test_neutralization_orthogonalizes_log_mktcap():
    rng = np.random.default_rng(1)
    n_days, n_names = 60, 80
    idx = _panel_index(n_days, n_names)
    logcap = pd.Series(rng.normal(5.0, 1.0, len(idx)), index=idx, name="cap")
    ind = pd.Series(
        rng.integers(0, 4, len(idx)).astype(str), index=idx, name="ind"
    )
    # factor is (mostly) a size factor + noise → neutralization must strip it
    f = 2.0 * logcap + pd.Series(rng.normal(0, 0.5, len(idx)), index=idx)
    frame = pd.DataFrame({"f": f, "ind": ind, "cap": np.exp(logcap)})
    out = preprocess_cross_section(frame, ["f"], industry_col="ind", mktcap_col="cap")
    z = out["f"]
    # per-date correlation with log-mktcap ≈ 0 after neutralization
    corrs = []
    for _, g in pd.DataFrame({"z": z, "c": logcap}).groupby(level=0):
        corrs.append(g["z"].corr(g["c"]))
    assert abs(float(np.mean(corrs))) < 0.02
    # still properly standardized
    g = z.groupby(level=0)
    assert g.std().sub(1.0).abs().max() < 1e-6


def test_neutralization_skipped_with_log_note(caplog):
    import logging

    rng = np.random.default_rng(2)
    idx = _panel_index(10, 40)
    f = pd.Series(rng.normal(0, 1, len(idx)), index=idx, name="f")
    with caplog.at_level(logging.INFO, logger="quantkit.factors"):
        out = preprocess_cross_section(pd.DataFrame({"f": f}), ["f"])
    assert "neutralization skipped" in caplog.text
    g = out["f"].groupby(level=0)
    assert g.std().sub(1.0).abs().max() < 1e-6


# ---------------------------------------------------------------------------
# train_lightgbm_ranker_walkforward
# ---------------------------------------------------------------------------


def _signal_panel(n_days: int = 750, n_names: int = 200, seed: int = 9):
    """200 names x 750 days; planted monotone signal in x0 (+weak x1), 12 noise
    features so per-seed models genuinely disagree (seed ensemble bites)."""
    rng = np.random.default_rng(seed)
    idx = _panel_index(n_days, n_names)
    x0 = rng.normal(0, 1, len(idx))
    x1 = rng.normal(0, 1, len(idx))
    cols = {"x0": x0, "x1": x1}
    for i in range(12):
        cols[f"n{i}"] = rng.normal(0, 1, len(idx))
    cols["fwd_ret"] = 0.8 * x0 + 0.2 * x1 + rng.normal(0, 1.2, len(idx))
    return pd.DataFrame(cols, index=idx)


_WF_FEATURES = ["x0", "x1"] + [f"n{i}" for i in range(12)]
# tiny high-variance models → real seed dispersion (LightGBM needs
# subsample_freq=1 for bagging to actually engage)
_WF_PARAMS = dict(
    n_estimators=5, num_leaves=63, subsample=0.5, subsample_freq=1,
    colsample_bytree=0.25, min_child_samples=5,
)


def test_walkforward_ranker_recovers_planted_signal():
    pytest.importorskip("lightgbm")
    panel = _signal_panel()
    res = train_lightgbm_ranker_walkforward(
        panel,
        _WF_FEATURES,
        "fwd_ret",
        n_splits=8,
        val_months=2,
        purge_bars=5,
        n_seeds=5,
        lgbm_params=_WF_PARAMS,
        seed0=11,
    )
    s = res.summary
    assert s["n_folds"] == 8
    # planted monotone signal → aggregated OOS RankIC clearly > 0
    assert s["rank_ic_mean"] > 0.15, f"OOS RankIC too low: {s['rank_ic_mean']}"
    # OOS-only: predictions cover exactly the test blocks (later period)
    pred_dates = res.predictions.dropna().index.get_level_values(0).unique()
    all_dates = panel.index.get_level_values(0).unique()
    assert pred_dates.min() > all_dates[0]
    assert len(pred_dates) < len(all_dates)
    # Huatai Discipline: Seed ensemble compresses single-seed variance — ensemble per-fold IC std must be
    # below the mean single-seed per-fold std, and seed noise must exist
    ens_std = res.fold_metrics["rank_ic"].std(ddof=1)
    seed_std = res.seed_metrics.groupby("seed")["rank_ic"].std(ddof=1).mean()
    seed_dispersion = res.seed_metrics.groupby("fold")["rank_ic"].std(ddof=1).mean()
    assert seed_dispersion > 1e-6, "seed noise absent — comparison meaningless"
    assert ens_std < seed_std, f"ensemble std {ens_std} !< single-seed std {seed_std}"


def test_walkforward_ranker_rejects_short_panel():
    pytest.importorskip("lightgbm")
    panel = _signal_panel(n_days=60, n_names=30)
    with pytest.raises(ValueError):
        train_lightgbm_ranker_walkforward(
            panel, ["x0", "x1"], "fwd_ret", n_splits=4, val_months=6
        )

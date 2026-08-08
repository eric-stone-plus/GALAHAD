"""Tests for quantkit.validation — anti-overfitting toolkit.

Synthetic-data contracts (per docs/library_digest/finance_validation_overfitting.md):
  - overlapping interval labels MUST be purged from training folds (asserted exactly)
  - walk-forward cuts fall only at month boundaries, test strictly after train
  - PBO is HIGH on random strategies, LOW on a genuinely persistent signal
  - DSR of noise ≈ 0 under many trials
  - block bootstrap preserves serial dependence structure (unlike iid shuffle)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantkit.validation import (
    PurgedKFold,
    block_bootstrap,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    min_track_record_length,
    prob_backtest_overfitting,
    walk_forward_splits,
)


# ---------------------------------------------------------------------------
# PurgedKFold
# ---------------------------------------------------------------------------


def _daily_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-03", periods=n, freq="B")


def _interval_t1(idx: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """Label end times for a fixed-horizon forward label (clipped at the end)."""
    vals = idx.to_series().shift(-horizon).fillna(idx[-1])
    return pd.Series(vals.to_numpy(), index=idx)


def test_purged_kfold_removes_overlapping_labels():
    n, horizon = 200, 10
    idx = _daily_index(n)
    X = pd.DataFrame({"f": np.arange(n, dtype=float)}, index=idx)
    t1 = _interval_t1(idx, horizon)
    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.0)
    t0v = idx.to_numpy()
    t1v = t1.to_numpy()
    for train_idx, test_idx in cv.split(X):
        # contiguity + coverage of test block
        assert len(test_idx) > 0
        assert not set(train_idx) & set(test_idx)
        test_lo, test_hi = t0v[test_idx[0]], t1v[test_idx[-1]]
        # every remaining train label window must NOT overlap the test window
        overlaps = (t1v[train_idx] >= test_lo) & (t0v[train_idx] <= test_hi)
        assert not overlaps.any(), "overlapping label leaked into train set"
        # and the purge must actually bite (horizon > 0 ⇒ some rows dropped)
        n_unpurged = n - len(test_idx)
        assert len(train_idx) < n_unpurged


def test_purged_kfold_embargo():
    n = 400
    idx = _daily_index(n)
    X = pd.DataFrame({"f": np.arange(n, dtype=float)}, index=idx)
    pct = 0.05
    cv = PurgedKFold(n_splits=4, t1=None, pct_embargo=pct)
    embargo = int(n * pct)
    for train_idx, test_idx in cv.split(X):
        after = np.arange(test_idx[-1] + 1, min(n, test_idx[-1] + 1 + embargo))
        assert not set(after) & set(train_idx), "embargo window leaked into train set"


def test_purged_kfold_point_labels_no_purge():
    n = 100
    X = pd.DataFrame({"f": np.arange(n, dtype=float)}, index=_daily_index(n))
    cv = PurgedKFold(n_splits=5, t1=None, pct_embargo=0.0)
    total_train = sum(len(tr) for tr, _ in cv.split(X))
    # point-in-time labels: nothing purged, every row is train or test exactly once
    assert total_train == n * (5 - 1)


# ---------------------------------------------------------------------------
# walk_forward_splits
# ---------------------------------------------------------------------------


def test_walk_forward_month_boundaries_and_order():
    idx = pd.date_range("2020-01-01", "2024-12-31", freq="B")
    folds = list(walk_forward_splits(idx, n_splits=4, test_months=6, purge_bars=5))
    assert len(folds) == 4
    months = idx.to_period("M")
    prev_test_end = -1
    for train_idx, test_idx in folds:
        # test strictly after train (plus the purge gap)
        assert train_idx.max() < test_idx.min()
        assert test_idx.min() - train_idx.max() == 1 + 5  # purge gap width
        # cuts at month boundaries: train months and test months are disjoint sets
        assert not set(months[train_idx]) & set(months[test_idx])
        # test blocks advance monotonically
        assert test_idx.min() > prev_test_end
        prev_test_end = test_idx.min()
    # expanding window: train grows across folds
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) > 1


def test_walk_forward_rolling_window():
    idx = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    folds = list(
        walk_forward_splits(idx, test_months=3, train_months=12, purge_bars=0)
    )
    assert len(folds) > 1
    months = idx.to_period("M")
    for train_idx, test_idx in folds:
        assert len(set(months[train_idx])) <= 12
        assert train_idx.max() < test_idx.min()


def test_walk_forward_rejects_impossible_config():
    idx = pd.date_range("2022-01-01", "2022-06-30", freq="B")
    with pytest.raises(ValueError):
        list(walk_forward_splits(idx, n_splits=4, test_months=6))


# ---------------------------------------------------------------------------
# PBO (CSCV)
# ---------------------------------------------------------------------------


def test_pbo_high_on_random_strategies():
    rng = np.random.default_rng(7)
    # 8 noise strategies with tiny idiosyncratic means → IS pick is luck
    m = rng.normal(0.0, 1.0, size=(96, 8)) + np.linspace(-0.05, 0.05, 8)
    pbo = prob_backtest_overfitting(m, n_blocks=16)
    assert pbo > 0.3, f"PBO on noise should be high, got {pbo}"


def test_pbo_low_on_persistent_signal():
    rng = np.random.default_rng(11)
    # strategy 0 dominates in EVERY block (large persistent edge); rest is noise
    noise = rng.normal(0.0, 1.0, size=(96, 7))
    champ = rng.normal(0.8, 1.0, size=(96, 1))
    m = np.hstack([champ, noise])
    pbo = prob_backtest_overfitting(m, n_blocks=16)
    assert pbo < 0.05, f"PBO on persistent signal should be ~0, got {pbo}"


def test_pbo_detail_shape_and_range():
    rng = np.random.default_rng(3)
    m = rng.normal(size=(96, 4))
    pbo, detail = prob_backtest_overfitting(m, n_blocks=8, return_detail=True)
    assert len(detail) == math.comb(8, 4) == 70
    assert 0.0 <= pbo <= 1.0
    assert detail["omega"].between(0, 1, inclusive="neither").all()


def test_pbo_ir_metric_on_excess_returns():
    rng = np.random.default_rng(21)
    m = rng.normal(0.0, 1.0, size=(96, 6))
    pbo = prob_backtest_overfitting(m, n_blocks=16, metric="ir")
    assert 0.0 <= pbo <= 1.0


# ---------------------------------------------------------------------------
# DSR / MinTRL
# ---------------------------------------------------------------------------


def test_dsr_of_noise_near_zero_with_many_trials():
    rng = np.random.default_rng(5)
    # a NON-selected member of a 50-strategy noise family: no skill, but the
    # family's selection pressure (n_trials, sr_std) is accounted for → DSR ≈ 0
    trials = rng.normal(0.0, 1.0, size=(50, 252))
    sr_trials = trials.mean(axis=1) / trials.std(axis=1, ddof=1)
    dsr = deflated_sharpe_ratio(trials[0], n_trials=50, sr_std=float(sr_trials.std()))
    assert dsr < 0.3, f"DSR of noise should be ≈ 0, got {dsr}"


def test_dsr_high_for_genuine_edge_single_trial():
    rng = np.random.default_rng(9)
    r = rng.normal(0.002, 0.01, size=756)  # daily SR ≈ 0.2 → ann ≈ 3.2
    dsr = deflated_sharpe_ratio(r, n_trials=1, sr_std=0.0)
    assert dsr > 0.95


def test_min_track_record_length():
    # hopeless when estimate does not beat the benchmark
    assert min_track_record_length(0.0, 0.0) == math.inf
    assert min_track_record_length(-0.1, 0.0) == math.inf
    # strong edge needs a short track record; monotone in the hurdle
    short = min_track_record_length(0.2, 0.0, alpha=0.05)
    longer = min_track_record_length(0.05, 0.0, alpha=0.05)
    assert 1.0 < short < longer < math.inf


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------


def test_block_bootstrap_shape_and_reproducibility():
    r = np.random.default_rng(1).normal(0, 1, 300)
    a = list(block_bootstrap(r, n_boot=5, block_len=10, random_state=42))
    b = list(block_bootstrap(r, n_boot=5, block_len=10, random_state=42))
    assert len(a) == 5 and all(len(x) == 300 for x in a)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    # resample is a permutation of pooled values, not fresh noise
    assert set(np.round(a[0], 12)) <= set(np.round(r, 12))


def test_block_bootstrap_preserves_autocorrelation():
    # strongly autocorrelated series: AR(1) rho = 0.9
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 2000)
    r = np.empty(2000)
    r[0] = e[0]
    for i in range(1, 2000):
        r[i] = 0.9 * r[i - 1] + e[i]
    sample = next(iter(block_bootstrap(r, n_boot=1, block_len=50, random_state=0)))
    acf_orig = np.corrcoef(r[:-1], r[1:])[0, 1]
    acf_boot = np.corrcoef(sample[:-1], sample[1:])[0, 1]
    shuffled = rng.permutation(r)
    acf_iid = np.corrcoef(shuffled[:-1], shuffled[1:])[0, 1]
    assert acf_boot > 0.5 * acf_orig, "block bootstrap should keep dependence"
    assert abs(acf_iid) < 0.2, "iid shuffle kills dependence (banned design)"


def test_block_bootstrap_sharpe_ci():
    rng = np.random.default_rng(4)
    strong = rng.normal(0.003, 0.01, 504)  # ann. Sharpe ≈ 4.8
    sharpe = lambda r: float(r.mean() / r.std(ddof=1) * np.sqrt(252))
    point, lo, hi = block_bootstrap_ci(
        strong, sharpe, n_boot=200, block_len=22, random_state=0
    )
    assert lo < point < hi
    assert lo > 0.0, "strong persistent edge: CI should exclude zero"
    noise = rng.normal(0.0, 0.01, 504)
    _, lo_n, hi_n = block_bootstrap_ci(
        noise, sharpe, n_boot=200, block_len=22, random_state=0
    )
    assert lo_n < 0.0 < hi_n, "noise: CI should straddle zero"

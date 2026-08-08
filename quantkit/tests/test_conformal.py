"""Tests for quantkit.conformal — vol-standardized conformal + ACI.

Synthetic-data contracts (recipe: docs/library_digest/updates/update_ml_validation.md §D;
raw absolute-residual conformal is INVALIDATED — Chernozhukov et al. PNAS 2021
show coverage collapsing to ~50% vs 90% nominal in high-vol regimes):
  - vol-standardized scores keep coverage near nominal ACROSS vol regimes
  - raw absolute-residual conformal (implemented inline, comparison only)
    collapses in the high-vol segment; standardized beats it decisively
  - after a coverage-breaking regime shift, ACI (Gibbs & Candès) coverage
    recovers toward nominal within T steps while fixed-alpha does not
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantkit.conformal import (
    ACIState,
    aci_update,
    conformal_interval,
    conformal_quantile,
    vol_standardized_scores,
)


def _garch(n: int, seed: int, vol_lo: float = 0.008, vol_hi: float = 0.032,
           heavy_tails_after: bool = False) -> pd.Series:
    """GARCH(1,1) with a mid-sample vol-regime shift (vol clustering)."""
    rng = np.random.default_rng(seed)
    a, b = 0.12, 0.85
    eps = rng.normal(0, 1, n)
    if heavy_tails_after:  # heavier innovation tails in the high-vol regime
        eps[n // 2:] = rng.standard_t(3, n - n // 2) / np.sqrt(3)
    target = np.where(np.arange(n) < n // 2, vol_lo, vol_hi)
    omega = target**2 * (1 - a - b)
    s2 = np.empty(n)
    s2[0] = target[0] ** 2
    r = np.empty(n)
    r[0] = target[0] * eps[0]
    for t in range(1, n):
        s2[t] = omega[t] + a * (r[t - 1] ** 2) + b * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * eps[t]
    return pd.Series(r, name="ret")


# ---------------------------------------------------------------------------
# vol_standardized_scores / conformal_quantile / conformal_interval
# ---------------------------------------------------------------------------


def test_scores_use_given_sigma_and_floor():
    r = pd.Series([0.01, -0.02, 0.03])
    s = vol_standardized_scores(r, sigma=pd.Series([0.01, 0.01, 0.0]))
    # third sigma = 0 → floored at min_sigma (no inf)
    assert np.isfinite(s).all()
    assert s.iloc[0] == pytest.approx(1.0)
    assert s.iloc[1] == pytest.approx(2.0)


def test_conformal_quantile_finite_sample_level():
    scores = np.arange(1.0, 11.0)  # 1..10, n=10
    # level = ceil((n+1)(1-alpha))/n: alpha=0.1 → 1.0 → max
    assert conformal_quantile(scores, 0.1) == pytest.approx(10.0)
    # monotone in alpha
    assert conformal_quantile(scores, 0.05) >= conformal_quantile(scores, 0.2)
    with pytest.raises(ValueError):
        conformal_quantile([], 0.1)
    with pytest.raises(ValueError):
        conformal_quantile(scores, 1.5)


def test_vol_standardized_coverage_stable_across_vol_regimes():
    r = _garch(1600, seed=0)
    n_cal = 600
    scores = vol_standardized_scores(r, halflife=25)
    sig = r.ewm(halflife=25, min_periods=5).std().shift(1).bfill().clip(lower=1e-8)
    q = conformal_quantile(scores.iloc[:n_cal], alpha=0.1)
    lo, hi = conformal_interval(0.0, q, sig.iloc[n_cal:])
    te = r.iloc[n_cal:]
    covered = (te >= lo) & (te <= hi)
    low_seg, high_seg = covered.iloc[:200], covered.iloc[200:]  # shift at t=800
    # nominal 0.90: coverage holds within ±0.06 overall AND per regime
    for name, seg in [("overall", covered), ("low-vol", low_seg), ("high-vol", high_seg)]:
        cov = float(seg.mean())
        assert 0.84 <= cov <= 0.96, f"{name} coverage {cov} outside tolerance"

    # comparison only: raw absolute-residual conformal (INVALIDATED default)
    q_raw = conformal_quantile(r.iloc[:n_cal].abs().to_numpy(), alpha=0.1)
    raw_high = float((high_seg.index.to_series().map(r.abs()) <= q_raw).mean())
    assert raw_high < 0.60, f"raw conformal should collapse in high-vol, got {raw_high}"
    assert float(high_seg.mean()) > raw_high + 0.2


# ---------------------------------------------------------------------------
# ACI (Gibbs & Candès): regime-shift recovery
# ---------------------------------------------------------------------------


def test_aci_update_direction_and_clip():
    st = ACIState(alpha=0.1, gamma=0.05, alpha_target=0.1)
    a1 = aci_update(st, err=1)  # a miss → alpha DOWN (widen)
    assert a1 < 0.1
    st2 = ACIState(alpha=0.1, gamma=0.05, alpha_target=0.1)
    a2 = aci_update(st2, err=0)  # a hit → alpha slightly UP (narrow)
    assert a2 > 0.1
    for _ in range(1000):  # persistent misses → clipped near 0, never negative
        aci_update(st, err=1)
    assert st.alpha >= 1e-4
    for _ in range(10000):
        aci_update(st, err=0)
    assert st.alpha <= 1.0 - 1e-4


def test_aci_recovers_coverage_after_regime_shift():
    n, t0 = 2000, 500
    r = _garch(n, seed=3, heavy_tails_after=True)
    # frozen pre-shift vol model — the new regime is genuinely unseen
    sig_hat = r.ewm(halflife=40, min_periods=5).std().shift(1)
    sigma_frozen = float(sig_hat.iloc[t0 - 1])
    scores = vol_standardized_scores(r, sigma=pd.Series(sigma_frozen, index=r.index))

    fixed = ACIState(alpha=0.1, gamma=0.0, alpha_target=0.1)
    aci = ACIState(alpha=0.1, gamma=0.05, alpha_target=0.1)
    err_f, err_a = [], []
    for t in range(t0, n):
        pool = scores.iloc[:t].to_numpy()  # calibration + expanding online history
        q_f = conformal_quantile(pool, fixed.alpha)
        err_f.append(int(abs(r.iloc[t]) > q_f * sigma_frozen))
        q_a = conformal_quantile(pool, aci.alpha)
        e = int(abs(r.iloc[t]) > q_a * sigma_frozen)
        err_a.append(e)
        aci_update(aci, e)

    shift = n // 2 - t0  # regime shift position within the error arrays
    seg = slice(shift + 100, shift + 500)  # T = 100..500 steps after the shift
    cov_fixed = 1.0 - float(np.mean(err_f[seg]))
    cov_aci = 1.0 - float(np.mean(err_a[seg]))
    # ACI recovers toward nominal 0.90 within T steps; fixed-alpha does not
    assert 0.84 <= cov_aci <= 0.96, f"ACI coverage {cov_aci} did not recover"
    assert cov_fixed < 0.75, f"fixed-alpha coverage {cov_fixed} should stay broken"
    assert cov_aci > cov_fixed + 0.15

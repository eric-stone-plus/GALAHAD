"""Tests for quantkit.optimizer — LW shrinkage covariance + GF Securities (广发) index-enhanced optimizer.

Synthetic-data contracts (recipe sources:
docs/library_digest/finance_factor_mining_models.md — GF Securities deep-learning research report 6,
docs/library_digest/finance_factors_microstructure_timing.md — Xinghuo-8 (星火8) / Caitong LW):

  - LW shrinkage pulls extreme sample correlations toward the constant-
    correlation target and stays well-conditioned when N > T (Xinghuo-8: LW_ConstCoeff
    best GMV out-of-sample)
  - the optimizer respects EVERY constraint on a 50-name panel: sum(w)=1,
    w≥0, TE ≤ cap, per-industry |dev| ≤ ±10%-style cap, size |dev| ≤ cap,
    per-rebalance turnover ½‖w−w0‖₁ ≤ 24%-style cap (GF Securities recipe)
  - infeasible configs (TE vs turnover conflict) raise OptimizationError
    naming the binding constraints
  - λ sweep reproduces GF Securities' qualitative result on planted alpha+cost:
    net IR peaks at λ=1, lower at λ=0 (cost ignored → churn drag) and at
    λ=2 (over-penalized → alpha cut)
  - attribution recovers the planted factor vs residual-Alpha split
    (GF Securities: the CSI1000 excess is dominated by residual Alpha)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantkit.optimizer import (
    OptimizationError,
    alpha_attribution,
    index_enhanced_weights,
    lambda_sweep,
    lw_shrinkage_cov,
)


# ---------------------------------------------------------------------------
# lw_shrinkage_cov
# ---------------------------------------------------------------------------


def _corr(cov: np.ndarray, i: int, j: int) -> float:
    return float(cov[i, j] / np.sqrt(cov[i, i] * cov[j, j]))


def test_lw_shrinkage_pulls_extreme_correlation_toward_target():
    rng = np.random.default_rng(0)
    T, N = 80, 6
    X = rng.normal(0.0, 0.02, (T, N))
    X[:, 1] = X[:, 0] + rng.normal(0.0, 1e-4, T)  # planted |corr| ≈ 1 pair
    sample = np.cov(X.T, ddof=0)  # lw_shrinkage_cov uses the 1/T convention
    shrunk, info = lw_shrinkage_cov(X, return_info=True)

    c_sample = _corr(sample, 0, 1)
    c_shrunk = _corr(np.asarray(shrunk), 0, 1)
    r_bar = info["avg_corr"]
    assert c_sample > 0.99, f"setup: planted corr too low ({c_sample})"
    assert 0.0 < info["alpha"] <= 1.0
    # shrunk correlation must lie BETWEEN the target r̄ and the sample corr
    assert r_bar - 1e-9 <= c_shrunk < c_sample
    # variances are untouched by the constant-correlation target
    np.testing.assert_allclose(np.diag(np.asarray(shrunk)), np.diag(sample), rtol=1e-10)


def test_lw_shrinkage_partial_on_mild_data_and_diagonal_variant():
    rng = np.random.default_rng(3)
    T, N = 400, 8
    # mild population correlation 0.5 between one pair
    z = rng.normal(0.0, 1.0, (T, N))
    z[:, 1] = 0.5 * z[:, 0] + np.sqrt(0.75) * z[:, 1]
    X = z * 0.02
    sample = np.cov(X.T)
    shrunk, info = lw_shrinkage_cov(X, return_info=True)
    c_sample, c_shrunk = _corr(sample, 0, 1), _corr(np.asarray(shrunk), 0, 1)
    assert 0.0 <= info["alpha"] <= 1.0
    assert info["avg_corr"] - 1e-9 <= c_shrunk <= c_sample + 1e-9
    # diagonal target: off-diagonals shrink toward 0
    shrunk_d, info_d = lw_shrinkage_cov(X, method="diagonal", return_info=True)
    assert 0.0 <= info_d["alpha"] <= 1.0
    assert abs(_corr(np.asarray(shrunk_d), 0, 1)) <= abs(c_sample) + 1e-12
    with pytest.raises(ValueError):
        lw_shrinkage_cov(X, method="nonsense")


def test_lw_shrinkage_well_conditioned_when_n_gt_t():
    rng = np.random.default_rng(5)
    T, N = 10, 30  # sample covariance is singular
    X = rng.normal(0.0, 0.02, (T, N))
    sample = np.cov(X.T)
    assert np.linalg.matrix_rank(sample) < N
    shrunk = np.asarray(lw_shrinkage_cov(X))
    assert np.linalg.eigvalsh(shrunk).min() > 0.0
    # DataFrame in → DataFrame out with labels preserved
    df = pd.DataFrame(X, columns=[f"s{i}" for i in range(N)])
    out = lw_shrinkage_cov(df)
    assert isinstance(out, pd.DataFrame) and list(out.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# index_enhanced_weights — 50-name panel, every constraint
# ---------------------------------------------------------------------------


def _panel() -> dict:
    rng = np.random.default_rng(11)
    N, T = 50, 150
    # factor-structured returns so the covariance is non-trivial
    f = rng.normal(0.0, 0.012, (T, 3))
    b = rng.normal(0.0, 0.8, (N, 3))
    e = rng.normal(0.0, 0.018, (T, N))
    rets = f @ b.T + e
    cov = lw_shrinkage_cov(rets)
    wb = np.full(N, 1.0 / N)
    scores = rng.normal(0.0, 1.0, N)
    industry = np.array(["g0"] * 15 + ["g1"] * 15 + ["g2"] * 10 + ["g3"] * 10)
    log_mktcap = rng.normal(10.0, 1.0, N)
    # prev holdings: 10% single-side turnover away from benchmark, all ≥ 0
    prev = wb.copy()
    prev[:5] += 0.02
    prev[5:15] -= 0.01
    assert abs(prev.sum() - 1.0) < 1e-12 and prev.min() > 0
    assert abs(0.5 * np.abs(prev - wb).sum() - 0.10) < 1e-12
    return dict(N=N, cov=cov, wb=wb, scores=scores, industry=industry,
                log_mktcap=log_mktcap, prev=prev)


def test_optimizer_respects_every_constraint_50_names():
    p = _panel()
    N, cov, wb, s = p["N"], np.asarray(p["cov"]), p["wb"], p["scores"]
    ind, lm, prev = p["industry"], p["log_mktcap"], p["prev"]

    te_max, ind_cap, sty_cap, to_cap = 0.004, 0.05, 0.05, 0.15
    w = index_enhanced_weights(
        s, wb, cov, industry=ind, log_mktcap=lm, te_max=te_max,
        industry_max_dev=ind_cap, style_max_dev=sty_cap,
        turnover_cap=to_cap, prev_weights=prev, cost_lambda=1.0,
    )
    d = w - wb
    assert abs(w.sum() - 1.0) < 1e-8
    assert w.min() >= -1e-9
    te = float(np.sqrt(d @ cov @ d))
    assert te <= te_max + 1e-6, f"TE {te} > cap {te_max}"
    for g in ("g0", "g1", "g2", "g3"):
        dev = abs(float(d[ind == g].sum()))
        assert dev <= ind_cap + 1e-7, f"industry {g} dev {dev} > {ind_cap}"
    sty = abs(float(lm @ d))
    assert sty <= sty_cap + 1e-7, f"size dev {sty} > {sty_cap}"
    to = 0.5 * float(np.abs(w - prev).sum())
    assert to <= to_cap + 1e-7, f"turnover {to} > {to_cap}"

    # non-vacuous: the unconstrained max-score corner would breach the TE cap,
    # and the optimizer actually traded toward the scores
    corner = np.zeros(N)
    corner[int(np.argmax(s))] = 1.0
    dc = corner - wb
    assert np.sqrt(dc @ cov @ dc) > te_max
    assert to > 1e-3 and float(s @ w) > float(s @ wb)


def test_optimizer_infeasible_raises_with_constraint_info():
    p = _panel()
    cov, wb, s, prev = np.asarray(p["cov"]), p["wb"], p["scores"], p["prev"]
    # TE cap = 0 forces w == wb (10% away); turnover cap 2% forbids the move
    with pytest.raises(OptimizationError) as exc:
        index_enhanced_weights(
            s, wb, cov, te_max=0.0, turnover_cap=0.02, prev_weights=prev
        )
    msg = str(exc.value)
    assert "turnover" in msg
    assert "tracking" in msg.lower() or "TE" in msg


def test_optimizer_input_validation():
    p = _panel()
    cov, wb, s = np.asarray(p["cov"]), p["wb"], p["scores"]
    with pytest.raises(ValueError):
        index_enhanced_weights(s, wb * 2.0, cov)  # benchmark not summing to 1
    with pytest.raises(ValueError):
        index_enhanced_weights(s[:-1], wb, cov)  # length mismatch
    bad_prev = p["prev"].copy()
    bad_prev[0] -= 0.5  # negative holding under long_only
    with pytest.raises(ValueError):
        index_enhanced_weights(s, wb, cov, prev_weights=bad_prev)


# ---------------------------------------------------------------------------
# lambda_sweep — Guangfa: IR peaks at λ=1
# ---------------------------------------------------------------------------


def _alpha_scenario(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Planted persistent alpha + small observation noise + 0.3% cost.

    Construction so the λ penalty differentiates (GF Securities mechanism):
    - score CHANGES are alpha-dominated (sig_obs ≪ Δa), so chasing them earns
      real return — λ=2's over-penalty cuts profitable trades (right side);
    - the marginal small-Δ trades have gain < tc, so λ=0's churn loses money
      net of cost (left side; the modest side, as in GF Securities' own numbers).
    - turnover cap non-binding: GF Securities ran the λ sweep with interior
      turnover (their 24%/rebalance hard cap was a separate experiment).
    """
    N, T = 40, 36
    sig_a, sig_obs, sig_eps, rho = 0.006, 0.0015, 0.012, 0.75
    rng = np.random.default_rng(seed)
    a = np.zeros((T, N))
    a[0] = rng.normal(0.0, sig_a, N)
    for t in range(1, T):
        a[t] = rho * a[t - 1] + np.sqrt(1 - rho**2) * rng.normal(0.0, sig_a, N)
    s = a + rng.normal(0.0, sig_obs, (T, N))
    r = a + rng.normal(0.0, sig_eps, (T, N))
    wb = np.full(N, 1.0 / N)
    rho_c = 0.3
    cov = (sig_eps**2) * ((1 - rho_c) * np.eye(N) + rho_c * np.ones((N, N)))
    return pd.DataFrame(s), pd.DataFrame(r), wb, cov


def test_lambda_sweep_ir_peaks_at_one():
    irs: dict[int, pd.Series] = {}
    for seed in (7, 23):
        s, r, wb, cov = _alpha_scenario(seed)
        tbl = lambda_sweep(
            s, r, wb, cov, lambdas=(0.0, 1.0, 2.0), periods_per_year=24,
            te_max=0.010, turnover_cap=1.0, tc_rate=0.003,
        )
        ir = tbl["ir_net"]
        # right side is the robust one (Guangfa's dramatic effect: CSI300 IR 1.74→0.74)
        assert ir.loc[1.0] > ir.loc[2.0], f"seed {seed}: λ=2 should cut alpha: {ir.to_dict()}"
        # turnover strictly decreasing in λ (Guangfa: 18.64→15.07→13.94 annual)
        to = tbl["avg_turnover"]
        assert to.loc[0.0] > to.loc[1.0] > to.loc[2.0], f"seed {seed}: {to.to_dict()}"
        irs[seed] = ir
    mean_ir = pd.DataFrame(irs).mean(axis=1)
    # left side (λ=0 churns on observation noise) is the modest one — assert
    # on the seed-averaged IR, which is the statistically meaningful object
    assert mean_ir.loc[1.0] > mean_ir.loc[0.0], f"λ=0 should churn: {mean_ir.to_dict()}"
    assert mean_ir.loc[1.0] > mean_ir.loc[2.0], f"peak should be at λ=1: {mean_ir.to_dict()}"


# ---------------------------------------------------------------------------
# alpha_attribution — planted factor vs residual split
# ---------------------------------------------------------------------------


def test_alpha_attribution_recovers_planted_split():
    rng = np.random.default_rng(13)
    N, T = 12, 20
    ind = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    X = np.zeros((N, 5))
    X[:, 0] = rng.normal(0.0, 1.0, N)  # style 1 (e.g. size)
    X[:, 1] = rng.normal(0.0, 1.0, N)  # style 2 (e.g. beta)
    X[np.arange(N), 2 + ind] = 1.0     # industry one-hots
    F = rng.normal(0.0, 0.01, (T, 5))
    eps = rng.normal(0.0, 0.02, (T, N))
    R = F @ X.T + eps
    wb = np.full(N, 1.0 / N)
    w = wb.copy()
    w[:4] += 0.01
    w[4:8] -= 0.01

    res = alpha_attribution(w, X, F, asset_returns=R, benchmark_weights=wb)
    a = w - wb
    np.testing.assert_allclose(
        res.per_period["explained_return"], (F * (a @ X)).sum(axis=1), atol=1e-12
    )
    np.testing.assert_allclose(res.per_period["residual_alpha"], eps @ a, atol=1e-12)
    np.testing.assert_allclose(
        res.per_period["total_return"],
        res.per_period["explained_return"] + res.per_period["residual_alpha"],
        atol=1e-12,
    )
    assert res.summary["residual_mean"] == pytest.approx(float((eps @ a).mean()))
    assert 0.0 <= res.summary["residual_share"] <= 1.0

    # single-period (1-D) inputs broadcast to T=1
    one = alpha_attribution(w, X, F[0], asset_returns=R[0], benchmark_weights=wb)
    assert one.per_period.shape == (1, 3)
    # residual needs realized returns
    with pytest.raises(ValueError):
        alpha_attribution(w, X, F)

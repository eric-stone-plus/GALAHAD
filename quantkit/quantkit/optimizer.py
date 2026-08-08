"""Index-enhancement portfolio optimizer (pure numpy/scipy, no cvxpy).

Recipe sources (docs/library_digest/):

- ``finance_factor_mining_models.md`` — 广发深度学习研究报告6 (2019):
  max ``score·w − λ·TC(w, w0)`` with ``TC = tc × ½‖w − w0‖₁`` (tc = 0.3%,
  single-side), s.t. tracking-error cap (annualized 7.75% for 指增 products),
  industry deviation ±10%, weighted size deviation ≤1%, ``w ≥ 0``, fully
  invested (``Σw = 1``), per-rebalance turnover hard cap ``½‖w − w0‖₁ ≤ 24%``.
  Empirical: cost-penalty λ = 1 (deduct real cost only) maximizes IR;
  λ = 2 is counterproductive (CSI300 IR 1.74 → 0.74). 7-style + industry
  attribution with residual Alpha as the main excess-return source.
- ``finance_factors_microstructure_timing.md`` — 星火8 (财通, covariance
  estimator comparison): ``Σ_shrink = α·F + (1−α)·S`` with structured target
  F; LW constant-correlation (``LW_ConstCoeff``) had the best GMV
  out-of-sample behaviour and is simple to implement.

Tracking-error annualization convention
----------------------------------------
广发's 7.75% TE cap is ANNUALIZED (semi-monthly rebalance, 24 periods/yr).
``index_enhanced_weights`` takes ``te_max`` in the SAME frequency as ``cov``:
if ``cov`` is estimated from daily returns, the equivalent daily cap is
``0.0775 / sqrt(252) ≈ 0.49%``; at semi-monthly frequency
``0.0775 / sqrt(24) ≈ 1.58%``. Convert before calling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "OptimizationError",
    "AttributionResult",
    "lw_shrinkage_cov",
    "index_enhanced_weights",
    "lambda_sweep",
    "alpha_attribution",
]

_TC_RATE = 0.003  # 广发 recipe: tc = 千分之三 (single-side)


class OptimizationError(RuntimeError):
    """Infeasible problem or solver failure; message lists binding constraints."""


# ---------------------------------------------------------------------------
# Ledoit-Wolf-style shrinkage covariance (星火8 recipe: Σ = αF + (1−α)S)
# ---------------------------------------------------------------------------


def lw_shrinkage_cov(
    returns: pd.DataFrame | np.ndarray,
    method: str = "constant_correlation",
    *,
    return_info: bool = False,
) -> pd.DataFrame | np.ndarray | tuple[Any, dict[str, float]]:
    """Ledoit-Wolf-style shrinkage of the sample covariance matrix.

    ``Σ_shrink = α·F + (1−α)·S`` with analytic α under Frobenius loss
    (Ledoit & Wolf 2004, "Honey, I Shrunk the Sample Covariance Matrix";
    implemented from scratch — no sklearn dependency).

    Parameters
    ----------
    returns :
        ``T × N`` return observations (rows = periods). DataFrame input keeps
        its column labels on the output.
    method :
        ``constant_correlation`` (default; 星火8's ``LW_ConstCoeff`` — best
        GMV out-of-sample): F has sample variances and a single averaged
        pairwise correlation. ``diagonal``: F = diag(S) (shrink covariances
        to zero).

    Notes
    -----
    Sample moments use the ``1/T`` convention throughout (consistent for the
    π/ρ/γ estimators). Zero-variance columns are floored only inside the
    ratio computations so a constant series yields a zero risk row instead
    of NaNs.
    """
    is_frame = isinstance(returns, pd.DataFrame)
    cols = list(returns.columns) if is_frame else None
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("returns must be a T×N array with T ≥ 2")
    if not np.isfinite(X).all():
        raise ValueError("returns contain NaN/inf — clean before shrinking")

    X = X - X.mean(axis=0, keepdims=True)
    T, N = X.shape
    S = (X.T @ X) / T
    d = np.diag(S).copy()
    d_safe = np.maximum(d, 1e-30)
    sd = np.sqrt(d_safe)

    # estimation error of S:  π̂_ij = mean_t (x_it x_jt − s_ij)²
    X2 = X * X
    Pi = (X2.T @ X2) / T - S * S
    sum_pi = float(Pi.sum())

    if method == "constant_correlation":
        # average pairwise sample correlation (off-diagonal)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = S / np.outer(sd, sd)
        iu = np.triu_indices(N, k=1)
        r_bar = float(np.mean(corr[iu])) if N > 1 else 0.0
        r_bar = float(np.clip(r_bar, -1.0, 1.0))

        # target F: sample variances, constant correlation r̄
        F = np.outer(sd, sd) * r_bar
        np.fill_diagonal(F, d)

        # ρ̂_ij (i≠j) = (r̄/2)·[√(s_jj/s_ii)·θ̂^{ii,ij} + √(s_ii/s_jj)·θ̂^{jj,ij}]
        # θ̂^{ii,ij} = mean_t[(x_it² − s_ii)(x_it x_jt − s_ij)]
        X3 = X2 * X
        theta1 = (X3.T @ X) / T - d[:, None] * S  # [i, j] = θ̂^{ii,ij}
        theta2 = theta1.T  # [i, j] = θ̂^{jj,ij}
        ratio = sd[None, :] / sd[:, None]  # [i, j] = √(s_jj/s_ii)
        Rho = (r_bar / 2.0) * (ratio * theta1 + ratio.T * theta2)
        np.fill_diagonal(Rho, np.diag(Pi))  # ρ̂_ii = π̂_ii
        sum_rho = float(Rho.sum())
        sum_gamma = float(((F - S) ** 2).sum())
    elif method == "diagonal":
        F = np.diag(d)
        r_bar = 0.0
        sum_rho = float(np.diag(Pi).sum())  # f_ij = 0 const ⇒ ρ̂_ij = 0 off-diag
        off = S - np.diag(d)
        sum_gamma = float((off**2).sum())
    else:
        raise ValueError(f"unknown method: {method!r}")

    if sum_gamma <= 0.0:
        alpha = 1.0  # F == S: intensity irrelevant, take full shrink
    else:
        alpha = float(np.clip((sum_pi - sum_rho) / sum_gamma, 0.0, 1.0))
    shrunk = alpha * F + (1.0 - alpha) * S
    shrunk = 0.5 * (shrunk + shrunk.T)
    # Ensure positive-definiteness: fix tiny negative eigenvalues from numerics
    eigvals = np.linalg.eigvalsh(shrunk)
    min_eig = eigvals.min()
    if min_eig < 1e-15:
        shrunk += (1e-15 - min_eig) * np.eye(N)

    info = {"alpha": alpha, "avg_corr": r_bar, "method": method}
    if is_frame:
        out: Any = pd.DataFrame(shrunk, index=cols, columns=cols)
    else:
        out = shrunk
    return (out, info) if return_info else out


# ---------------------------------------------------------------------------
# Index-enhancement optimizer (广发深度学习6 recipe)
# ---------------------------------------------------------------------------


def _vec(x: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.shape[0] != n:
        raise ValueError(f"{name}: expected length {n}, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/inf")
    return arr


def _align(
    expected_scores: Any,
    benchmark_weights: Any,
    cov: Any,
    industry: Any,
    log_mktcap: Any,
    prev_weights: Any,
) -> tuple[np.ndarray, ...]:
    """Align pandas inputs on the benchmark index; trust raw-array order."""
    if isinstance(benchmark_weights, pd.Series):
        idx = benchmark_weights.index

        def align1(x: Any, name: str) -> Any:
            if isinstance(x, pd.Series):
                x = x.reindex(idx)
                if x.isna().any():
                    raise ValueError(f"{name}: missing symbols vs benchmark index")
            return x

        expected_scores = align1(expected_scores, "expected_scores")
        prev_weights = align1(prev_weights, "prev_weights") if prev_weights is not None else None
        log_mktcap = align1(log_mktcap, "log_mktcap") if log_mktcap is not None else None
        if isinstance(industry, pd.Series):
            industry = industry.reindex(idx)
            if industry.isna().any():
                raise ValueError("industry: missing symbols vs benchmark index")
        if isinstance(cov, pd.DataFrame):
            cov = cov.reindex(index=idx, columns=idx)
            if cov.isna().any().any():
                raise ValueError("cov: missing symbols vs benchmark index")
    wb = np.asarray(benchmark_weights, dtype=float).ravel()
    n = wb.shape[0]
    s = _vec(expected_scores, n, "expected_scores")
    C = np.asarray(cov, dtype=float)
    if C.shape != (n, n):
        raise ValueError(f"cov: expected shape ({n},{n}), got {C.shape}")
    if not np.isfinite(C).all():
        raise ValueError("cov contains NaN/inf")
    ind = None if industry is None else np.asarray(industry).ravel()
    if ind is not None and len(ind) != n:
        raise ValueError("industry: wrong length")
    lm = None if log_mktcap is None else _vec(log_mktcap, n, "log_mktcap")
    pw = None if prev_weights is None else _vec(prev_weights, n, "prev_weights")
    return s, wb, C, ind, lm, pw


def index_enhanced_weights(
    expected_scores: Any,
    benchmark_weights: Any,
    cov: Any,
    industry: Any = None,
    log_mktcap: Any = None,
    te_max: float = 0.02,
    industry_max_dev: float = 0.10,
    style_max_dev: float = 0.01,
    turnover_cap: float = 0.24,
    prev_weights: Any = None,
    cost_lambda: float = 1.0,
    tc_rate: float = _TC_RATE,
    long_only: bool = True,
) -> np.ndarray:
    """Maximize ``score·w − λ·tc·½‖w−w0‖₁`` s.t. 广发指增 constraints.

    Parameters
    ----------
    expected_scores :
        Cross-sectional scores (higher = better), aligned to the benchmark
        order (pandas Series are reindexed on the benchmark index).
    benchmark_weights :
        Benchmark weights; must sum to 1 (±1e-6) and be ≥ 0 if ``long_only``.
    cov :
        ``N×N`` asset-return covariance (any single frequency — see the
        module docstring for the TE annualization convention).
    industry :
        Industry label per asset; per-group ``|Σ_{i∈g}(w_i − wb_i)|``
        ≤ ``industry_max_dev`` (广发: ±10%).
    log_mktcap :
        Log market cap per asset; constraint on the weighted-average
        deviation ``|Σ(w_i − wb_i)·logcap_i|`` ≤ ``style_max_dev``
        (广发: ≤1%; shift-invariant since weights sum to 1).
    te_max :
        Tracking-error cap, same frequency as ``cov`` (广发 product cap
        7.75% is annualized — convert, e.g. daily: ``0.0775/√252``).
    turnover_cap :
        Per-rebalance single-side turnover hard cap ``½‖w−w0‖₁`` (广发: 24%).
        Ignored when ``prev_weights`` is None.
    prev_weights :
        Current holdings ``w0``. None ⇒ no turnover cap and no cost term.
    cost_lambda :
        Cost-penalty multiplier λ (广发: λ=1 deducts real cost only and
        maximizes IR; λ=2 is counterproductive).
    tc_rate :
        Single-side transaction-cost rate (广发: 千三 = 0.003).

    Returns
    -------
    np.ndarray
        Optimal weights (same order as the benchmark).

    Raises
    ------
    OptimizationError
        When infeasible or the solver fails; the message reports each
        constraint's required bound vs realized value so the binding one is
        identifiable.
    """
    s, wb, C, ind, lm, pw = _align(
        expected_scores, benchmark_weights, cov, industry, log_mktcap, prev_weights
    )
    n = len(wb)
    if te_max < 0 or industry_max_dev < 0 or style_max_dev < 0 or turnover_cap < 0:
        raise ValueError("caps/deviations must be ≥ 0")
    if abs(wb.sum() - 1.0) > 1e-6:
        raise ValueError(f"benchmark_weights must sum to 1, got {wb.sum():.8f}")
    if long_only and (wb < -1e-9).any():
        raise ValueError("benchmark_weights must be ≥ 0 under long_only")
    if pw is not None:
        if abs(pw.sum() - 1.0) > 1e-6:
            raise ValueError(f"prev_weights must sum to 1, got {pw.sum():.8f}")
        if long_only and (pw < -1e-9).any():
            raise ValueError("prev_weights must be ≥ 0 under long_only")
    C = 0.5 * (C + C.T)

    has_prev = pw is not None
    w0 = pw if has_prev else wb.copy()

    # industry one-hot (G × n)
    G = None
    if ind is not None:
        groups = list(dict.fromkeys(ind.tolist()))
        G = np.zeros((len(groups), n))
        for gi, g in enumerate(groups):
            G[gi, ind == g] = 1.0

    # decision vector x = [w (n), u (n)]; u_i ≥ |w_i − w0_i| linearizes the L1
    m = 2 * n

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x[:n], x[n:]

    cost_coef = cost_lambda * tc_rate * 0.5

    def objective(x: np.ndarray) -> float:
        w, u = unpack(x)
        return float(-s @ w + (cost_coef * u.sum() if has_prev else 0.0))

    def obj_jac(x: np.ndarray) -> np.ndarray:
        g = np.zeros(m)
        g[:n] = -s
        if has_prev:
            g[n:] = cost_coef
        return g

    cons: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda x: float(x[:n].sum() - 1.0),
         "jac": lambda x: np.r_[np.ones(n), np.zeros(n)]},
    ]

    # TE (normalized by te_max² for conditioning): 1 − ΔwᵀΣΔw/te_max² ≥ 0
    te_scale = max(te_max * te_max, 1e-30)

    def te_fun(x: np.ndarray) -> float:
        d = x[:n] - wb
        return 1.0 - float(d @ C @ d) / te_scale

    def te_jac(x: np.ndarray) -> np.ndarray:
        d = x[:n] - wb
        return np.r_[-2.0 * (C @ d) / te_scale, np.zeros(n)]

    cons.append({"type": "ineq", "fun": te_fun, "jac": te_jac})

    if G is not None:
        g_mat = G

        def ind_fun(x: np.ndarray) -> np.ndarray:
            dev = g_mat @ (x[:n] - wb)
            return np.r_[industry_max_dev - dev, industry_max_dev + dev]

        def ind_jac(x: np.ndarray) -> np.ndarray:
            z = np.zeros_like(g_mat)
            return np.block([[-g_mat, z], [g_mat, z]])

        cons.append({"type": "ineq", "fun": ind_fun, "jac": ind_jac})

    if lm is not None:
        z = lm

        def sty_fun(x: np.ndarray) -> np.ndarray:
            dev = float(z @ (x[:n] - wb))
            return np.array([style_max_dev - dev, style_max_dev + dev])

        def sty_jac(x: np.ndarray) -> np.ndarray:
            zz = np.zeros(n)
            return np.block([[-z, zz], [z, zz]])

        cons.append({"type": "ineq", "fun": sty_fun, "jac": sty_jac})

    if has_prev:
        def to_fun(x: np.ndarray) -> float:
            return turnover_cap - 0.5 * float(x[n:].sum())

        def to_jac(x: np.ndarray) -> np.ndarray:
            return np.r_[np.zeros(n), -0.5 * np.ones(n)]

        cons.append({"type": "ineq", "fun": to_fun, "jac": to_jac})

        eye = np.eye(n)

        def link_fun(x: np.ndarray) -> np.ndarray:
            w, u = unpack(x)
            d = w - w0
            return np.r_[u - d, u + d]

        def link_jac(x: np.ndarray) -> np.ndarray:
            return np.block([[-eye, eye], [eye, eye]])

        cons.append({"type": "ineq", "fun": link_fun, "jac": link_jac})

    w_lb = 0.0 if long_only else -1.0
    bounds = [(w_lb, 1.0)] * n + [(0.0, 1.0)] * n

    # Start ladder: benchmark when reachable under the turnover cap; else
    # points on the prev→benchmark segment strictly INSIDE the budget, then
    # prev itself (starting exactly on the turnover boundary makes SLSQP's
    # QP subproblem fail with status 8; a ladder of interior starts is the
    # robust remedy). A result is accepted only if SLSQP converged AND the
    # point passes the constraint checks below.
    starts: list[np.ndarray] = []
    if has_prev:
        dist_wb = 0.5 * float(np.abs(wb - w0).sum())
        if dist_wb > turnover_cap:
            base_theta = turnover_cap / dist_wb
            starts.extend(
                w0 + (frac * base_theta) * (wb - w0)
                for frac in (0.9, 0.6, 0.3, 0.1)
            )
            starts.append(w0.copy())
        else:
            starts.append(wb.copy())
    else:
        starts.append(wb.copy())

    def _solve(x0_w: np.ndarray, ftol: float) -> Any:
        x0 = np.r_[x0_w, np.abs(x0_w - w0) + 1e-9]
        return minimize(
            objective, x0, jac=obj_jac, bounds=bounds, constraints=cons,
            method="SLSQP", options={"maxiter": 80, "ftol": ftol},
        )

    res = None
    w_raw: np.ndarray | None = None
    violations: dict[str, str] = {}
    solved = False
    for w_start in starts:
        # ftol ladder: 1e-14 is too tight for SLSQP's QP subproblem here
        # (status 8 despite a feasible, already-optimal iterate)
        best_viol_mag = float("inf")
        for ftol in (1e-12, 1e-10, 1e-8):
            cand = _solve(w_start, ftol)
            cand_w = cand.x[:n] if cand.x is not None else w_start
            cand_viol = _constraint_violations(
                cand_w, wb, C, G, lm, w0 if has_prev else None,
                te_max, industry_max_dev, style_max_dev, turnover_cap, long_only,
            )
            res, w_raw, violations = cand, cand_w, cand_viol
            if cand.success and not cand_viol:
                solved = True
                break
            # Early exit: if this ftol didn't reduce violations, tighter won't help
            n_viol = len(cand_viol) if cand_viol else 0
            if n_viol >= best_viol_mag and not cand.success:
                break
            best_viol_mag = min(best_viol_mag, n_viol)
        if solved:
            break
    assert res is not None and w_raw is not None
    if not res.success or violations:
        raise OptimizationError(
            _infeasibility_report(res, violations, wb, C, w0 if has_prev else None,
                                  te_max, turnover_cap)
        )
    # cosmetic cleanup only (≤1e-9 relative, far below the check tolerances)
    w = np.clip(w_raw, w_lb, 1.0)
    if w.sum() > 0:
        w = w / w.sum()
    return w


def _constraint_violations(
    w: np.ndarray,
    wb: np.ndarray,
    C: np.ndarray,
    G: np.ndarray | None,
    lm: np.ndarray | None,
    w0: np.ndarray | None,
    te_max: float,
    industry_max_dev: float,
    style_max_dev: float,
    turnover_cap: float,
    long_only: bool,
) -> dict[str, str]:
    """Return {constraint: detail} for every constraint breached beyond tol."""
    out: dict[str, str] = {}
    d = w - wb
    sum_err = abs(float(w.sum()) - 1.0)
    if sum_err > 1e-6:
        out["fully_invested"] = f"Σw−1 = {sum_err:.3e} > 1e-6"
    if long_only and w.min() < -1e-7:
        out["long_only"] = f"min(w) = {w.min():.3e} < 0"
    te = float(np.sqrt(max(d @ C @ d, 0.0)))
    if te > te_max + max(1e-9, 1e-6 * max(te_max, 1e-12)):
        out["tracking_error"] = f"TE = {te:.6g} > cap {te_max:.6g}"
    if G is not None:
        dev = np.abs(G @ d)
        k = int(np.argmax(dev))
        if dev[k] > industry_max_dev + 1e-7:
            out[f"industry[{k}]"] = (
                f"|dev| = {dev[k]:.6g} > cap {industry_max_dev:.6g}"
            )
    if lm is not None:
        dev = abs(float(lm @ d))
        if dev > style_max_dev + 1e-7:
            out["size_style"] = f"|dev| = {dev:.6g} > cap {style_max_dev:.6g}"
    if w0 is not None:
        to = 0.5 * float(np.abs(w - w0).sum())
        if to > turnover_cap + 1e-7:
            out["turnover"] = f"½‖w−w0‖₁ = {to:.6g} > cap {turnover_cap:.6g}"
    return out


def _infeasibility_report(
    res: Any,
    violations: dict[str, str],
    wb: np.ndarray,
    C: np.ndarray,
    w0: np.ndarray | None,
    te_max: float,
    turnover_cap: float,
) -> str:
    lines = [
        f"index_enhanced_weights failed (solver status "
        f"{getattr(res, 'status', '?')}: {getattr(res, 'message', '?')}).",
    ]
    if violations:
        lines.append("Violated constraints at the final iterate:")
        lines += [f"  - {k}: {v}" for k, v in sorted(violations.items())]
    else:
        lines.append("Solver reported failure before constraint checks.")
    if w0 is not None:
        dist = 0.5 * float(np.abs(wb - w0).sum())
        seg = wb - w0
        te_full = float(np.sqrt(max(seg @ C @ seg, 0.0)))
        theta = min(1.0, turnover_cap / dist) if dist > 1e-12 else 1.0
        lines.append(
            f"turnover budget ½‖wb−w0‖₁ = {dist:.4f}; reachable fraction of the "
            f"prev→benchmark gap θ = {theta:.3f}; min TE along that segment "
            f"≈ {theta * te_full:.6g} (cap {te_max:.6g})."
        )
        if theta * te_full > te_max:
            lines.append(
                "→ tracking_error and turnover caps CONFLICT: relax te_max, "
                "raise turnover_cap, or move prev_weights closer to benchmark."
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# λ sweep (广发: IR peaks at λ=1, degrades at λ=2)
# ---------------------------------------------------------------------------


def lambda_sweep(
    scores: pd.DataFrame,
    period_returns: pd.DataFrame,
    benchmark_weights: Any,
    cov: Any,
    *,
    lambdas: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    periods_per_year: int = 12,
    **opt_kwargs: Any,
) -> pd.DataFrame:
    """Backtest the optimizer over a λ grid; one row per λ.

    Parameters
    ----------
    scores :
        ``T×N`` per-rebalance scores (row t is used to build the portfolio
        held over period t).
    period_returns :
        ``T×N`` realized asset returns over the same periods (row-aligned
        with ``scores``). Keeping the period frequency explicit makes the
        sweep frequency-agnostic (广发 rebalanced semi-monthly).
    periods_per_year :
        Annualization for the reported IR / TE (24 for semi-monthly).

    Returns
    -------
    DataFrame indexed by λ with columns: ``gross_active_ann``, ``cost_ann``,
    ``net_active_ann``, ``te_ann`` (realized, on gross active), ``ir_net``
    (mean/std of net active × √ppy), ``avg_turnover``, ``avg_objective``,
    ``n_periods``.

    Note
    ----
    广发 ran the λ sweep WITHOUT the hard per-rebalance turnover cap (their
    annual turnover fell 18.6 → 15.1 → 13.9 for CSI1000 as λ rose 0 → 1 → 2 —
    turnover was interior, limited by the cost penalty itself). The 24% hard
    cap was a SEPARATE experiment. To reproduce the sweep's shape, leave
    ``turnover_cap`` non-binding (e.g. 1.0); with a binding cap all λ trade
    to the same cap and the IR peak flattens or inverts.
    """
    sc = np.asarray(scores, dtype=float)
    rt = np.asarray(period_returns, dtype=float)
    if sc.shape != rt.shape:
        raise ValueError(f"scores {sc.shape} and period_returns {rt.shape} differ")
    wb = np.asarray(benchmark_weights, dtype=float).ravel()
    n = len(wb)
    if sc.shape[1] != n:
        raise ValueError("scores/returns columns must match benchmark length")
    tc = float(opt_kwargs.get("tc_rate", _TC_RATE))
    ppy = float(periods_per_year)

    rows: list[dict[str, float]] = []
    for lam in lambdas:
        w_prev = wb.copy()
        gross = np.zeros(sc.shape[0])
        net = np.zeros(sc.shape[0])
        tos = np.zeros(sc.shape[0])
        objs = np.zeros(sc.shape[0])
        for t in range(sc.shape[0]):
            w = index_enhanced_weights(
                sc[t], wb, cov, prev_weights=w_prev, cost_lambda=lam, **opt_kwargs
            )
            to = 0.5 * float(np.abs(w - w_prev).sum())
            cost = tc * to
            gross[t] = float((w - wb) @ rt[t])
            net[t] = gross[t] - cost
            tos[t] = to
            objs[t] = float(sc[t] @ w - lam * cost)
            w_prev = w
        sd = net.std(ddof=1) if len(net) > 1 else np.nan
        ir = float(net.mean() / sd * np.sqrt(ppy)) if sd and sd > 0 else np.nan
        rows.append(
            {
                "lambda": float(lam),
                "gross_active_ann": float(gross.mean() * ppy),
                "cost_ann": float((gross - net).mean() * ppy),
                "net_active_ann": float(net.mean() * ppy),
                "te_ann": float(gross.std(ddof=1) * np.sqrt(ppy)) if len(gross) > 1 else np.nan,
                "ir_net": ir,
                "avg_turnover": float(tos.mean()),
                "avg_objective": float(objs.mean()),
                "n_periods": float(sc.shape[0]),
            }
        )
    return pd.DataFrame(rows).set_index("lambda")


# ---------------------------------------------------------------------------
# Residual-Alpha attribution (广发: 7 style + industry factors + residual)
# ---------------------------------------------------------------------------


@dataclass
class AttributionResult:
    """Per-period split of (active) return into factor-explained + residual."""

    per_period: pd.DataFrame  # total_return, explained_return, residual_alpha
    factor_contrib: pd.DataFrame  # T×K contribution per factor
    summary: dict[str, Any] = field(default_factory=dict)


def alpha_attribution(
    weights: Any,
    factor_exposures: Any,
    factor_returns: Any,
    asset_returns: Any = None,
    benchmark_weights: Any = None,
) -> AttributionResult:
    """Decompose portfolio (active) return: explained part + residual Alpha.

    广发深度学习6 attribution: the per-period (excess) return is split into
    the part explained by 7 style factors + industry factors and a residual
    Alpha (their CSI1000 enhancement's excess is dominated by residual Alpha:
    单期 Alpha 均值 0.73% / 标准差 1.65%).

    Parameters
    ----------
    weights :
        ``(N,)`` or ``(T,N)`` portfolio weights.
    factor_exposures :
        ``(N,K)`` exposure matrix X (style columns + industry one-hots).
    factor_returns :
        ``(K,)`` or ``(T,K)`` realized factor returns f.
    asset_returns :
        ``(N,)`` or ``(T,N)`` realized asset returns. REQUIRED — the residual
        is total minus explained, which is unidentified without realized
        returns (the brief's 3-arg sketch omits it; it is positional-4 here).
    benchmark_weights :
        Optional benchmark weights; when given, the ACTIVE return
        ``(w − wb)′r`` is attributed (广发 attributes 超额).

    Returns
    -------
    AttributionResult with per-period totals and the summary keys
    ``residual_mean`` / ``residual_std`` (广发's 单期 Alpha 均值/标准差),
    ``explained_share`` / ``residual_share`` of the summed absolute split.
    """
    if asset_returns is None:
        raise ValueError(
            "asset_returns is required: residual alpha = total − explained "
            "needs realized asset returns"
        )
    W = np.atleast_2d(np.asarray(weights, dtype=float))
    X = np.asarray(factor_exposures, dtype=float)
    F = np.atleast_2d(np.asarray(factor_returns, dtype=float))
    R = np.atleast_2d(np.asarray(asset_returns, dtype=float))
    if X.ndim != 2:
        raise ValueError("factor_exposures must be (N,K)")
    n, k = X.shape
    if W.shape[1] != n or R.shape[1] != n or F.shape[1] != k:
        raise ValueError(
            f"shape mismatch: weights {W.shape}, exposures {X.shape}, "
            f"factor_returns {F.shape}, asset_returns {R.shape}"
        )
    T = max(W.shape[0], F.shape[0], R.shape[0])
    if W.shape[0] not in (1, T) or F.shape[0] not in (1, T) or R.shape[0] not in (1, T):
        raise ValueError("T dimension must be 1 or consistent across inputs")
    W = np.broadcast_to(W, (T, n))
    F = np.broadcast_to(F, (T, k))
    R = np.broadcast_to(R, (T, n))

    A = W if benchmark_weights is None else W - np.broadcast_to(
        np.atleast_2d(np.asarray(benchmark_weights, dtype=float)), (T, n)
    )
    expo = A @ X  # (T,K) portfolio factor exposures
    contrib = expo * F  # (T,K) per-factor contribution
    explained = contrib.sum(axis=1)
    total = np.einsum("tn,tn->t", A, R)
    residual = total - explained

    per_period = pd.DataFrame(
        {
            "total_return": total,
            "explained_return": explained,
            "residual_alpha": residual,
        }
    )
    factor_contrib = pd.DataFrame(contrib, columns=[f"factor_{j}" for j in range(k)])
    tot_abs = float(np.abs(total).sum())
    summary = {
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std(ddof=1)) if T > 1 else 0.0,
        "explained_share": float(np.abs(explained).sum() / tot_abs) if tot_abs > 0 else np.nan,
        "residual_share": float(np.abs(residual).sum() / tot_abs) if tot_abs > 0 else np.nan,
        "n_periods": int(T),
    }
    return AttributionResult(
        per_period=per_period, factor_contrib=factor_contrib, summary=summary
    )

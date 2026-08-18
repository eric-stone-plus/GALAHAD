"""DSR / PBO statistical evaluation for futures paper strategies.

Mirror of ``quantkit.validation`` (DSR = Bailey & López de Prado 2014
deflated Sharpe; PBO = Bailey et al. 2017 CSCV), implemented with numpy +
stdlib only: quantkit is not importable from this package's venv (no scipy
there), so the definitions below are re-implemented 1:1 — same formulas,
same conventions (sample std ddof=1, Pearson kurtosis, Euler–Mascheroni
approximation for E[max SR], omega >= 0.5 for overfit) — with scipy's
``norm.cdf`` / ``norm.ppf`` replaced by local erf-based / Acklam
implementations (max abs error ~1.15e-9, deterministic).

Input convention for :func:`evaluate_oos_statistics`: a list of out-of-sample
per-bar return series, one per walk-forward window (each produced by
``engine.run_paper_on_bars(..., evaluate_from=oos_start)["returns_oos"]``).

DSR mapping: each walk-forward window is one "trial"; ``n_trials`` = number
of windows, ``sr_std`` = sample std of the per-window per-bar Sharpes
(identical windows -> sr_std = 0 -> DSR reduces to PSR vs 0, matching
quantkit's degenerate branch).

PBO mapping: the windows are aligned into a T x N returns matrix (rows =
positions on the common window timeline, shorter windows NaN-padded then
zero-filled exactly as quantkit does) and CSCV is run over that matrix —
the per-window ranking must not flip between the first and second half of
the windows' timelines.

Advisory gates only (roadmap P1): ``dsr_pass = DSR > 0`` and
``pbo_flag = PBO > 0.5`` are flags in the report, never promotion gates.

No funding/margin terms enter these statistics; the paper engine's per-bar
returns already net fees, funding, and margin effects (book.py), so the
statistics inherit them.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable, Literal, Sequence

import numpy as np

__all__ = [
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "prob_backtest_overfitting",
    "evaluate_oos_statistics",
]

_EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# scipy-free normal CDF / PPF (mirrors scipy.stats.norm used by quantkit)
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (max error ~1e-15)."""


# The normal quantile function follows the rational approximation of
# Acklam, P.J. (2004), "An algorithm for computing the inverse normal
# cumulative distribution function" — a published, freely usable
# algorithm; attributed here per the repository's third-party
# attribution policy.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam rational approximation, abs err ~1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    else:
        q = p - 0.5
        r = q * q
        x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    return x


# ---------------------------------------------------------------------------
# Sharpe inference under multiple testing (quantkit.validation mirror)
# ---------------------------------------------------------------------------


def _moments(returns: np.ndarray) -> tuple[float, float, float, float]:
    """(mean, std, skew, kurtosis) with kurtosis = Pearson (normal = 3)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or len(r) < 3:
        return mu, sd, 0.0, 3.0
    z = (r - mu) / sd
    return mu, sd, float((z**3).mean()), float((z**4).mean())


def probabilistic_sharpe_ratio(
    sr: float,
    sr_benchmark: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """P(SR > sr_benchmark) given non-normal returns (Bailey & LdP 2012)."""
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2))
    z = (sr - sr_benchmark) * math.sqrt(max(1, n_obs - 1)) / denom
    return float(_norm_cdf(z))


def deflated_sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    *,
    n_trials: int,
    sr_std: float | None = None,
    periods_per_year: float = 1.0,
) -> float:
    """Deflated Sharpe Ratio: P(true SR > E[max SR | no skill]).

    Same definition and defaults as ``quantkit.validation.deflated_sharpe_ratio``
    (returns-based path): Sharpe moments are estimated from ``returns`` in
    per-period units (pass ``periods_per_year`` only to convert an annualized
    ``sr`` back to per-period internally — this mirror uses the returns path,
    so ``periods_per_year`` is accepted for API parity). ``sr_std`` is the
    sample std of Sharpe estimates across the ``n_trials`` candidates (e.g.
    per-window Sharpes); when ``sr_std <= 0`` (single trial, or all trials
    identical) DSR reduces to PSR vs 0.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    r = np.asarray([float(v) for v in returns])
    r = r[~np.isnan(r)]
    if len(r) < 3:
        raise ValueError("need >= 3 observations")
    mu, sd, skew, kurt = _moments(r)
    if sd == 0:
        return 0.0  # zero variance -> DSR is undefined as a test of skill (quantkit mirror)
    sr_hat = mu / sd
    n_obs = len(r)

    if sr_std is None:
        raise ValueError("sr_std (std of Sharpe estimates across trials) required")
    if sr_std <= 0:
        # all trials identical -> no selection; DSR reduces to PSR vs 0
        return probabilistic_sharpe_ratio(sr_hat, 0.0, n_obs, skew, kurt)

    # E[max SR] under the null of no skill (Euler-Mascheroni approximation)
    sr_star = (
        sr_std
        * (
            (1.0 - _EULER_MASCHERONI) * _norm_ppf(1.0 - 1.0 / n_trials)
            + _EULER_MASCHERONI * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        )
        if n_trials > 1
        else 0.0
    )
    return probabilistic_sharpe_ratio(sr_hat, sr_star, n_obs, skew, kurt)


# ---------------------------------------------------------------------------
# CSCV / Probability of Backtest Overfitting (quantkit.validation mirror)
# ---------------------------------------------------------------------------


def _sharpe_from_sums(s1: np.ndarray, s2: np.ndarray, n: int) -> np.ndarray:
    """Sharpe (mean/std, ddof=1) of each column given block-sum aggregates."""
    mean = s1 / n
    var = (s2 - n * mean**2) / max(n - 1, 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mean / sd, 0.0)
    return out


def prob_backtest_overfitting(
    returns_matrix: np.ndarray,
    *,
    n_blocks: int = 16,
    metric: Literal["sharpe", "ir"] = "sharpe",
    return_detail: bool = False,
) -> float | tuple[float, list[dict[str, Any]]]:
    """PBO via Combinatorially Symmetric Cross-Validation (Bailey et al. 2017).

    Mirror of ``quantkit.validation.prob_backtest_overfitting``: T x N matrix
    of per-period returns for N strategy variants; rows split into
    ``n_blocks`` contiguous blocks; PBO = fraction of combinations where the
    train-optimal variant's relative rank on the test half falls in the worse
    half (omega >= 0.5). Rule of thumb (ht22): < 0.2 credible, > 0.5
    basically overfit.

    ``metric="ir"`` ranks by mean/std of excess-return series (the caller
    passes excess returns); ``"sharpe"`` ranks plain mean/std. Fails closed:
    needs >= 2 variants and T/n_blocks >= 2 rows per block.
    """
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T x N)")
    t_total, n_strat = m.shape
    if n_strat < 2:
        raise ValueError("need >= 2 strategy columns")
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even and >= 2")
    block_len = t_total // n_blocks
    if block_len < 2:
        raise ValueError("T/n_blocks too small (need >= 2 rows per block)")
    m = np.nan_to_num(m[: block_len * n_blocks], nan=0.0)

    blocks = m.reshape(n_blocks, block_len, n_strat)
    b_sum = blocks.sum(axis=1)  # S x N
    b_sumsq = (blocks**2).sum(axis=1)  # S x N
    half = n_blocks // 2
    n_train = half * block_len
    n_test = (n_blocks - half) * block_len

    n_overfit = 0
    rows: list[dict[str, Any]] = []
    for combo in itertools.combinations(range(n_blocks), half):
        mask = np.zeros(n_blocks, dtype=bool)
        mask[list(combo)] = True
        tr_metric = _sharpe_from_sums(
            b_sum[mask].sum(axis=0), b_sumsq[mask].sum(axis=0), n_train
        )
        te_metric = _sharpe_from_sums(
            b_sum[~mask].sum(axis=0), b_sumsq[~mask].sum(axis=0), n_test
        )
        n_star = int(np.argmax(tr_metric))
        rank = int((te_metric > te_metric[n_star]).sum()) + 1
        omega = rank / (n_strat + 1.0)
        n_overfit += int(omega >= 0.5)
        if return_detail:
            rows.append(
                {
                    "combo": combo,
                    "train_pick": n_star,
                    "train_metric": float(tr_metric[n_star]),
                    "test_metric": float(te_metric[n_star]),
                    "test_rank": rank,
                    "omega": omega,
                }
            )
    total = math.comb(n_blocks, half)
    pbo = n_overfit / total
    if return_detail:
        return pbo, rows
    return pbo


# ---------------------------------------------------------------------------
# Walk-forward wrapper
# ---------------------------------------------------------------------------


def _pick_n_blocks(t_total: int, default: int) -> int:
    """Largest even n <= default with t_total // n >= 2 (fails closed)."""
    for n in range(default, 1, -2):
        if t_total // n >= 2:
            return n
    raise ValueError(
        f"T={t_total} too small for CSCV: need >= 4 rows for n_blocks=2"
    )


def evaluate_oos_statistics(
    oos_return_series: Iterable[Sequence[float] | np.ndarray],
    *,
    n_blocks: int = 16,
    periods_per_year: float = 1.0,
    min_obs_per_window: int = 3,
) -> dict[str, Any]:
    """DSR / PBO / Sharpe from a list of per-window OOS return series.

    Fail-closed: empty input, no window with >= ``min_obs_per_window`` finite
    returns, fewer than 2 usable windows (PBO is undefined for a single
    variant), or a timeline too short for CSCV raises ValueError with a
    clear message (a statistics report is never emitted from nothing).

    Parameters
    ----------
    oos_return_series :
        One per walk-forward window: per-bar out-of-sample returns from the
        paper engine (``result["returns_oos"]``). Windows of different
        lengths are aligned by row position; shorter windows are NaN-padded
        and zero-filled for CSCV exactly as quantkit does.
    n_blocks :
        CSCV blocks (default 16, quantkit's default); auto-reduced to the
        largest even n <= ``n_blocks`` that leaves >= 2 rows per block.
    periods_per_year :
        Annualization factor for the reported ``oos_sharpe`` (e.g. 8760 for
        1h bars, matching ``scripts/walkforward_runner.py``). DSR is always
        computed in per-period units (mirrors quantkit defaults).

    Returns
    -------
    dict with ``dsr``, ``pbo``, ``oos_sharpe`` (annualized), ``n_windows``,
    ``dsr_pass`` (DSR > 0), ``pbo_flag`` (PBO > 0.5), plus diagnostics
    (``oos_obs``, ``n_blocks_used``, ``windows``, ``warnings``).
    """
    series = [
        np.asarray([float(v) for v in s], dtype=float)
        for s in oos_return_series
    ]
    series = [s[~np.isnan(s)] for s in series]
    series = [s for s in series if len(s) >= min_obs_per_window]
    if not series:
        raise ValueError(
            "evaluate_oos_statistics: no usable OOS return series "
            f"(need >= 1 window with >= {min_obs_per_window} finite returns)"
        )

    oos = np.concatenate(series)
    mu, sd, _skew, _kurt = _moments(oos)
    oos_sharpe_per_bar = mu / sd if sd > 0 else 0.0
    oos_sharpe = oos_sharpe_per_bar * math.sqrt(max(1.0, periods_per_year))

    # Per-window per-bar Sharpes -> selection deflation across windows
    window_stats = [_moments(s) for s in series]
    window_sharpes = np.array(
        [mu / sd if sd > 0 else 0.0 for mu, sd, _sk, _ku in window_stats]
    )
    sr_std = float(window_sharpes.std(ddof=1)) if len(window_sharpes) > 1 else 0.0
    dsr = deflated_sharpe_ratio(
        oos, n_trials=len(series), sr_std=sr_std, periods_per_year=1.0
    )

    # PBO: windows-as-variants CSCV over the row-aligned matrix
    if len(series) < 2:
        raise ValueError(
            f"evaluate_oos_statistics: need >= 2 windows for PBO (CSCV), got {len(series)}"
        )
    t_max = max(len(s) for s in series)
    matrix = np.full((t_max, len(series)), np.nan, dtype=float)
    for j, s in enumerate(series):
        matrix[: len(s), j] = s
    if matrix.shape[0] < 4:
        raise ValueError(
            f"evaluate_oos_statistics: timeline too short for CSCV (T={matrix.shape[0]})"
        )
    n_blocks_used = _pick_n_blocks(matrix.shape[0], n_blocks)
    warnings: list[str] = []
    if n_blocks_used < n_blocks:
        warnings.append(
            f"n_blocks reduced {n_blocks} -> {n_blocks_used} "
            f"(T={matrix.shape[0]} < {2 * n_blocks} rows for default block size)"
        )
    pbo = prob_backtest_overfitting(matrix, n_blocks=n_blocks_used)

    return {
        "dsr": float(dsr),
        "pbo": float(pbo),
        "oos_sharpe": float(oos_sharpe),
        "oos_sharpe_per_bar": float(oos_sharpe_per_bar),
        "n_windows": len(series),
        "oos_obs": int(len(oos)),
        "n_blocks_used": n_blocks_used,
        "dsr_pass": float(dsr) > 0.0,
        "pbo_flag": float(pbo) > 0.5,
        "windows": [
            {
                "window": j,
                "n_obs": int(len(s)),
                "sharpe_per_bar": float(mu / sd if sd > 0 else 0.0),
            }
            for j, (s, (mu, sd, _sk, _ku)) in enumerate(zip(series, window_stats))
        ],
        "warnings": warnings,
    }

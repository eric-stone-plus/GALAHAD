"""Validation / anti-overfitting toolkit (AFML ch.7 + Bailey et al. + ht14/16/19/22).

Implements the workspace-standard dual protocol
(see docs/library_digest/finance_validation_overfitting.md and
updates/update_ml_validation.md):

  - ``PurgedKFold``         purged k-fold CV with embargo for interval labels
  - ``walk_forward_splits`` month-aligned purged walk-forward OOS splits (final gate)
  - ``deflated_sharpe_ratio`` / ``min_track_record_length``  Sharpe inference under
    multiple testing (Bailey & López de Prado)
  - ``prob_backtest_overfitting``  CSCV / PBO (Bailey–Borwein–LdP–Zhu 2017; ht22 params)
  - ``block_bootstrap`` / ``block_bootstrap_ci``  stationary / circular block
    bootstrap — IID bootstrap is banned for return series (IAJ 2025-10 bias evidence)

Pure numpy/pandas/scipy. No new heavy deps.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Callable, Iterator, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "PurgedKFold",
    "walk_forward_splits",
    "deflated_sharpe_ratio",
    "min_track_record_length",
    "prob_backtest_overfitting",
    "block_bootstrap",
    "block_bootstrap_ci",
]

_EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# 1. Purged k-fold CV with embargo (AFML ch.7)
# ---------------------------------------------------------------------------


class PurgedKFold:
    """K-fold CV for time series with interval labels.

    Samples are taken in time order; each fold is a contiguous test block.
    Training samples whose label window ``[t0_i, t1_i]`` overlaps the test
    window are dropped (purge), and ``pct_embargo`` of T observations after
    each test block are dropped as well (embargo).

    Parameters
    ----------
    n_splits :
        Number of folds (contiguous test blocks).
    t1 :
        Label end times, a ``pd.Series`` aligned positionally with ``X``
        (e.g. ``index + horizon`` for a fixed-horizon forward-return label).
        ``None`` = point-in-time labels (t1 = sample time).
    pct_embargo :
        Embargo size as a fraction of T (default 1%, AFML recommendation).
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: pd.Series | None = None,
        pct_embargo: float = 0.01,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.0 <= pct_embargo < 1.0:
            raise ValueError("pct_embargo must be in [0, 1)")
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def get_n_splits(self) -> int:
        return self.n_splits

    def split(
        self, X, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        if n < self.n_splits:
            raise ValueError("n_samples < n_splits")
        if isinstance(X, (pd.DataFrame, pd.Series)):
            t0 = pd.DatetimeIndex(X.index)
        else:
            if self.t1 is None:
                raise ValueError("array-like X requires t1 with a DatetimeIndex")
            t0 = pd.DatetimeIndex(self.t1.index)
        if self.t1 is None:
            t1 = pd.Series(t0, index=t0)
        else:
            t1 = pd.Series(self.t1).reset_index(drop=True)
            if len(t1) != n:
                raise ValueError("t1 length must match X")
            if t1.isna().any():
                raise ValueError("t1 must not contain NaT/NaN (invalid label window)")
        t0v = t0.to_numpy()
        t1v = pd.DatetimeIndex(t1).to_numpy()

        sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        sizes[: n % self.n_splits] += 1
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        embargo = int(n * self.pct_embargo)
        all_idx = np.arange(n)

        for k in range(self.n_splits):
            test_idx = np.arange(bounds[k], bounds[k + 1])
            test_t0 = t0v[test_idx[0]]
            test_t1 = t1v[test_idx[-1]]
            train_idx = np.setdiff1d(all_idx, test_idx)
            # purge: drop train samples whose [t0, t1] overlaps [test_t0, test_t1]
            overlap = (t1v[train_idx] >= test_t0) & (t0v[train_idx] <= test_t1)
            train_idx = train_idx[~overlap]
            # embargo: drop train rows in the window right after the test block
            if embargo > 0:
                after = train_idx[train_idx > test_idx[-1]]
                train_idx = train_idx[
                    ~np.isin(train_idx, after[:embargo])
                ]
            yield train_idx, test_idx


# ---------------------------------------------------------------------------
# 2. Walk-forward OOS splits, cuts at month boundaries (ht16 GroupTimeSeriesSplit)
# ---------------------------------------------------------------------------


def walk_forward_splits(
    index: Sequence,
    *,
    n_splits: int = 4,
    test_months: int = 6,
    train_months: int | None = None,
    purge_bars: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` position arrays for walk-forward OOS.

    Cuts fall only at month boundaries (groups = calendar months, per ht16 —
    sklearn's TimeSeriesSplit can split inside a month). Test blocks advance
    by ``test_months`` each fold. Training is expanding (``train_months=None``,
    initial train = all months before the first test block) or a rolling
    window of the last ``train_months`` months. ``purge_bars`` drops the rows
    of the training set immediately preceding each test block (gap for
    overlapping labels, e.g. a 5-bar forward-return label → ``purge_bars=5``).

    Folds with empty train or test are skipped silently.
    """
    idx = pd.DatetimeIndex(index)
    if not idx.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")
    months = idx.to_period("M")
    uniq = months.unique()
    n_months = len(uniq)
    if train_months is None:
        init_train = n_months - n_splits * test_months
        if init_train < 1:
            raise ValueError(
                f"not enough months ({n_months}) for n_splits={n_splits} "
                f"x test_months={test_months}"
            )
    else:
        if train_months < 1:
            raise ValueError("train_months must be >= 1")
        init_train = train_months
        n_splits = max(0, (n_months - init_train) // test_months)

    pos = np.arange(len(idx))
    for k in range(n_splits):
        test_start_m = init_train + k * test_months
        test_month_range = uniq[test_start_m : test_start_m + test_months]
        if len(test_month_range) == 0:
            break
        test_mask = months.isin(test_month_range)
        test_idx = pos[test_mask]
        if len(test_idx) == 0:
            continue
        if train_months is None:
            train_mask = months < test_month_range[0]
        else:
            train_month_range = uniq[test_start_m - train_months : test_start_m]
            train_mask = months.isin(train_month_range)
        train_idx = pos[np.asarray(train_mask)]
        if purge_bars > 0 and len(train_idx) > purge_bars:
            train_idx = train_idx[:-purge_bars]
        elif purge_bars > 0:
            train_idx = train_idx[:0]
        if len(train_idx) == 0:
            continue
        yield train_idx, test_idx


# ---------------------------------------------------------------------------
# 3. Sharpe inference under multiple testing (Bailey & López de Prado)
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
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray | None = None,
    *,
    sr: float | None = None,
    n_trials: int,
    sr_std: float | None = None,
    n_obs: int | None = None,
    skew: float | None = None,
    kurt: float | None = None,
    periods_per_year: float = 1.0,
) -> float:
    """Deflated Sharpe Ratio: P(true SR > E[max SR | no skill]).

    ``sr*`` (expected maximum Sharpe across ``n_trials`` under the null) uses
    the Euler–Mascheroni approximation of Bailey & López de Prado (2014).
    Sharpe values are in per-period units; pass ``periods_per_year`` only to
    convert an annualized ``sr`` back to per-period internally.

    Provide either ``returns`` (sr/skew/kurt/n_obs estimated from the series,
    ``sr_std`` then **required** — e.g. std of per-trial Sharpes) or explicit
    ``sr`` + ``n_obs`` (+ optional ``skew``/``kurt``).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if returns is not None:
        r = pd.Series(returns).dropna().to_numpy(dtype=float)
        if len(r) < 3:
            raise ValueError("need >= 3 observations")
        mu, sd, sk, ku = _moments(r)
        if sd == 0:
            return 0.0
        sr_hat = mu / sd
        n_obs = len(r)
        skew = sk if skew is None else skew
        kurt = ku if kurt is None else kurt
    else:
        if sr is None or n_obs is None:
            raise ValueError("provide returns or (sr, n_obs)")
        sr_hat = sr / math.sqrt(periods_per_year)
        skew = 0.0 if skew is None else skew
        kurt = 3.0 if kurt is None else kurt

    if sr_std is None:
        raise ValueError("sr_std (std of Sharpe estimates across trials) required")
    if sr_std <= 0:
        # all trials identical → no selection; DSR reduces to PSR vs 0
        return probabilistic_sharpe_ratio(sr_hat, 0.0, n_obs, skew, kurt)

    # E[max SR] under the null of no skill
    sr_star = sr_std * (
        (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
        + _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    ) if n_trials > 1 else 0.0
    return probabilistic_sharpe_ratio(sr_hat, sr_star, n_obs, skew, kurt)


def min_track_record_length(
    sr: float,
    sr_benchmark: float = 0.0,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    alpha: float = 0.05,
) -> float:
    """Minimum number of observations for PSR(sr vs benchmark) >= 1 - alpha.

    Returns ``inf`` when the point estimate does not beat the benchmark.
    ``sr`` is per-period (same frequency as the observations being counted).
    """
    if sr <= sr_benchmark:
        return math.inf
    z = norm.ppf(1.0 - alpha)
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    return float(1.0 + var_term * (z / (sr - sr_benchmark)) ** 2)


# ---------------------------------------------------------------------------
# 4. CSCV / Probability of Backtest Overfitting (Bailey et al. 2017; ht22)
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
    returns_matrix: pd.DataFrame | np.ndarray,
    *,
    n_blocks: int = 16,
    metric: Literal["sharpe", "ir"] = "sharpe",
    return_detail: bool = False,
) -> float | tuple[float, pd.DataFrame]:
    """PBO via Combinatorially Symmetric Cross-Validation.

    Parameters
    ----------
    returns_matrix :
        T × N matrix of per-period returns (or excess returns) for N strategy
        variants. Rows are split into ``n_blocks`` contiguous blocks (order
        inside blocks preserved). For excess-return series (e.g. index
        enhancement vs benchmark) use ``metric="ir"`` — ranking by
        information ratio is the ht22 practical definition; ``"sharpe"``
        ranks by plain mean/std.
    n_blocks :
        S, even. Default 16 → C(16,8)=12870 combinations (ht22 numerics for
        T=96 months, T/S=6). Avoid T/S too large (S=2 leaves 2 combinations).
    return_detail :
        Also return per-combination rank dataframe (omega, train/test picks).

    PBO = fraction of combinations where the train-optimal strategy's
    relative rank on the test half falls in the worse half (omega >= 0.5).
    Rule of thumb (ht22): < 0.2 credible, > 0.5 basically overfit.
    """
    m = (
        returns_matrix.to_numpy(dtype=float)
        if isinstance(returns_matrix, pd.DataFrame)
        else np.asarray(returns_matrix, dtype=float)
    )
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

    # sharpe and ir share the mean/std form here; for "ir" the caller passes
    # excess returns, so mean/std of the series IS the information ratio.
    n_overfit = 0
    rows: list[dict] = []
    for combo in combinations(range(n_blocks), half):
        mask = np.zeros(n_blocks, dtype=bool)
        mask[list(combo)] = True
        tr_metric = _sharpe_from_sums(b_sum[mask].sum(axis=0), b_sumsq[mask].sum(axis=0), n_train)
        te_metric = _sharpe_from_sums(b_sum[~mask].sum(axis=0), b_sumsq[~mask].sum(axis=0), n_test)
        n_star = int(np.argmax(tr_metric))
        # relative rank of n_star on the test half, descending (0 = best)
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
        return pbo, pd.DataFrame(rows)
    return pbo


# ---------------------------------------------------------------------------
# 5. Block bootstrap (stationary / circular) — IID bootstrap is banned
# ---------------------------------------------------------------------------


def _default_block_len(n: int) -> int:
    """sqrt(T) heuristic; refine by autocorrelation decay when it matters."""
    return max(1, int(round(math.sqrt(n))))


def block_bootstrap(
    series: pd.Series | np.ndarray,
    *,
    n_boot: int = 1000,
    block_len: int | None = None,
    method: Literal["stationary", "circular"] = "stationary",
    random_state: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield bootstrap resamples (same length as input) preserving dependence.

    ``stationary`` : Politis–Romano stationary bootstrap (geometric block
    lengths with mean ``block_len``). ``circular`` : Künsch circular block
    bootstrap (fixed length, wrap-around). Choose ``block_len`` from the
    decay of the return autocorrelation; default sqrt(T) is a floor, not a
    recommendation.
    """
    r = (
        series.to_numpy(dtype=float)
        if isinstance(series, pd.Series)
        else np.asarray(series, dtype=float)
    )
    r = r[~np.isnan(r)]
    n = len(r)
    if n == 0:
        raise ValueError("empty series")
    b = block_len or _default_block_len(n)
    if b < 1:
        raise ValueError("block_len must be >= 1")
    rng = np.random.default_rng(random_state)

    for _ in range(n_boot):
        out = np.empty(n, dtype=float)
        filled = 0
        while filled < n:
            start = int(rng.integers(0, n))
            if method == "stationary":
                length = int(rng.geometric(1.0 / b))
            elif method == "circular":
                length = b
            else:
                raise ValueError(f"unknown method: {method}")
            take = min(length, n - filled)
            idx = (start + np.arange(take)) % n  # circular indexing
            out[filled : filled + take] = r[idx]
            filled += take
        yield out


def block_bootstrap_ci(
    series: pd.Series | np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    n_boot: int = 1000,
    block_len: int | None = None,
    method: Literal["stationary", "circular"] = "stationary",
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    """Percentile CI of ``stat_fn`` under block bootstrap.

    Returns ``(point_estimate, lo, hi)``. For a Sharpe CI pass e.g.
    ``lambda r: r.mean() / r.std() * np.sqrt(252)``.
    """
    r = (
        series.to_numpy(dtype=float)
        if isinstance(series, pd.Series)
        else np.asarray(series, dtype=float)
    )
    r = r[~np.isnan(r)]
    point = float(stat_fn(r))
    stats = np.array(
        [
            float(stat_fn(sample))
            for sample in block_bootstrap(
                r,
                n_boot=n_boot,
                block_len=block_len,
                method=method,
                random_state=random_state,
            )
        ]
    )
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return point, lo, hi

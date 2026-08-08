"""Split-conformal prediction intervals with volatility-standardized scores + ACI.

Recipe: docs/library_digest/updates/update_ml_validation.md §D (conformal).

  - **Vol-standardized nonconformity scores** (``|resid| / sigma_t``): raw
    absolute-residual split-conformal is INVALIDATED by the freshness audit —
    coverage collapses in high-vol regimes (Chernozhukov et al., PNAS 2021:
    ~50% realized vs 90% nominal on daily returns). Standardizing by a
    volatility estimate keeps coverage roughly stable across vol regimes, so
    it is the ONLY default offered here.
  - **ACI** (Adaptive Conformal Inference, Gibbs & Candès, NeurIPS 2021 /
    JMLR 2024): online update ``alpha_{t+1} = alpha_t + gamma·(alpha_target −
    err_t)`` so intervals re-widen after a coverage-breaking regime shift.
  - **DtACI** (Dynamically-tuned ACI, same paper §A.1): a grid of ACI
    experts with different step sizes ``gamma``, aggregated by exponential
    weighting on their cumulative pinball losses — removes the manual
    ``gamma`` choice, which is the one sensitive knob of plain ACI.

Pure numpy/pandas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "vol_standardized_scores",
    "conformal_quantile",
    "conformal_interval",
    "ACIState",
    "aci_update",
    "DtACIState",
    "dtaci_update",
]


def vol_standardized_scores(
    residuals: pd.Series | np.ndarray,
    sigma: pd.Series | np.ndarray | None = None,
    *,
    halflife: float = 20.0,
    min_sigma: float = 1e-8,
) -> pd.Series:
    """Nonconformity scores ``|resid_t| / sigma_t``.

    ``sigma`` : per-t volatility estimate. When None, estimated in-sample as
    the *lagged* ewm std of ``residuals`` (``halflife``), back-filled over the
    warm-up head so the calibration block stays usable. Floored at
    ``min_sigma`` to avoid division by zero in dead markets.
    """
    r = (
        residuals.astype(float)
        if isinstance(residuals, pd.Series)
        else pd.Series(np.asarray(residuals, dtype=float))
    )
    if sigma is None:
        sig = r.ewm(halflife=halflife, min_periods=5).std().shift(1).bfill()
    else:
        sig = (
            sigma.astype(float)
            if isinstance(sigma, pd.Series)
            else pd.Series(np.asarray(sigma, dtype=float), index=r.index)
        )
    sig = sig.clip(lower=min_sigma)
    return (r.abs() / sig).rename("score")


def conformal_quantile(
    scores: pd.Series | np.ndarray, alpha: float = 0.1
) -> float:
    """Finite-sample split-conformal quantile at miscoverage ``alpha``.

    Uses the standard level ``ceil((n+1)(1−alpha))/n`` (Vovk) with the
    "higher" order statistic, so realized coverage ≥ 1−alpha under
    exchangeability of the calibration scores.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    s = pd.Series(np.asarray(scores, dtype=float)).dropna().to_numpy()
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def conformal_interval(
    center: pd.Series | np.ndarray | float,
    quantile: float,
    sigma: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vol-standardized interval ``center ± quantile · sigma_t``.

    ``quantile`` comes from :func:`conformal_quantile` on vol-standardized
    scores; ``sigma`` is the same volatility estimate used there (evaluated
    at the prediction times).
    """
    if quantile < 0:
        raise ValueError("quantile must be >= 0")
    c = np.asarray(center, dtype=float)
    half = quantile * np.asarray(sigma, dtype=float)
    return c - half, c + half


@dataclass
class ACIState:
    """Online state for Adaptive Conformal Inference.

    ``alpha``        current miscoverage level (start at ``alpha_target``);
    ``gamma``        step size (Gibbs & Candès use ~0.005–0.05 in practice);
    ``alpha_target`` target miscoverage (e.g. 0.1 for 90% intervals).
    """

    alpha: float
    gamma: float
    alpha_target: float


def aci_update(state: ACIState, err: int | float) -> float:
    """One ACI step: ``alpha_{t+1} = alpha_t + gamma·(alpha_target − err_t)``.

    ``err_t`` = 1 when the realized value fell OUTSIDE the interval at t
    (a miss pushes ``alpha`` DOWN → wider intervals), 0 otherwise (pushes
    ``alpha`` slightly up → narrower). ``state`` is updated in place and the
    new alpha is returned; it is clipped to (0, 1) for numerical sanity.
    """
    new_alpha = state.alpha + state.gamma * (state.alpha_target - float(err))
    state.alpha = float(min(max(new_alpha, 1e-4), 1.0 - 1e-4))
    return state.alpha


# Default expert grid: spans ~2 orders of magnitude around the γ=0.005–0.05
# range Gibbs & Candès found workable for single-ACI.
_DTACI_DEFAULT_GAMMAS = (0.001, 0.005, 0.01, 0.05, 0.1)


@dataclass
class DtACIState:
    """Online state for DtACI (multi-expert ACI, Gibbs & Candès 2021 §A.1).

    ``alpha_target`` target miscoverage (e.g. 0.1 for 90% intervals);
    ``eta``          learning rate of the exponential expert weighting
                     (per-step losses are ≤ 1, so 0.1 discriminates between
                     experts over a few hundred steps without instant
                     collapse onto a single one);
    ``gammas``       per-expert ACI step sizes (set on first update call);
    ``alphas``       current per-expert miscoverage levels;
    ``weights``      aggregation weights, softmax(−eta·cum_loss);
    ``cum_loss``     per-expert cumulative pinball loss.

    Expert arrays are built lazily on the first :func:`dtaci_update` call
    from the ``gammas`` argument passed there.
    """

    alpha_target: float
    eta: float = 0.1
    gammas: tuple[float, ...] = ()
    alphas: np.ndarray | None = field(default=None, repr=False)
    weights: np.ndarray | None = field(default=None, repr=False)
    cum_loss: np.ndarray | None = field(default=None, repr=False)


def dtaci_update(
    state: DtACIState,
    err: int | float,
    *,
    gammas: tuple[float, ...] = _DTACI_DEFAULT_GAMMAS,
) -> float:
    """One DtACI step: update all ACI experts, re-weight, return aggregated α̂.

    ``err_t`` = 1 when the realized value fell OUTSIDE the interval built at
    the previous aggregated α̂ (same sign convention as :func:`aci_update`).
    Each expert k takes its own ACI step ``alpha_k += gamma_k·(alpha_target −
    err_t)``; the returned α̂ is the weighted mean of the updated expert
    alphas and feeds the next 1−α̂ conformal quantile.

    Expert losses are the pinball losses ``α_target·(β−α_k) − min(0, β−α_k)``
    that underlie the ACI gradient identity. With only the coverage
    indicator available, the realized quantile level β is replaced by its
    err-consistent extreme ``β̂ = 1 − err_t`` (a miss means the score fell
    beyond the quantile, a hit means inside it): a miss therefore costs
    ``(1−α_target)·α_k`` and a hit costs ``α_target·(1−α_k)``. Under ideal
    calibration the expected loss is minimized exactly at α_k = α_target,
    so weight concentrates on the expert whose γ tracks the shift best.

    Fail-closed: ``err`` must be exactly 0 or 1 (NaN raises).
    """
    e = float(err)
    if e not in (0.0, 1.0):
        raise ValueError(f"err must be 0 or 1 (coverage indicator), got {err!r}")
    if not 0.0 < state.alpha_target < 1.0:
        raise ValueError("alpha_target must be in (0, 1)")
    if state.eta <= 0:
        raise ValueError("eta must be > 0")
    if state.alphas is None:
        g = tuple(float(x) for x in gammas)
        if not g or any(x <= 0 for x in g):
            raise ValueError("gammas must be a non-empty tuple of positive floats")
        k = len(g)
        state.gammas = g
        state.alphas = np.full(k, state.alpha_target)
        state.weights = np.full(k, 1.0 / k)
        state.cum_loss = np.zeros(k)
    alphas = state.alphas
    assert state.weights is not None and state.cum_loss is not None

    # score the experts on the alphas they held when the interval was formed
    u = (1.0 - e) - alphas
    state.cum_loss += state.alpha_target * u - np.minimum(0.0, u)
    z = -state.eta * state.cum_loss
    z -= z.max()  # stable softmax(−eta·cum_loss)
    w = np.exp(z)
    state.weights = w / w.sum()

    # per-expert ACI step, same clip as aci_update
    g = np.asarray(state.gammas)
    state.alphas = np.clip(
        alphas + g * (state.alpha_target - e), 1e-4, 1.0 - 1e-4
    )
    return float(np.dot(state.weights, state.alphas))

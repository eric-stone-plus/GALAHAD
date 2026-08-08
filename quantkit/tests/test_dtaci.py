"""Tests for quantkit.conformal DtACI (multi-expert ACI, Gibbs & Candès 2021 §A.1).

Contracts under synthetic miscoverage streams (``err`` is exogenous here, so
trajectories are deterministic given the stream):
  - persistent misses push the aggregated alpha DOWN (intervals widen),
    persistent hits push it UP — same sign convention as single-expert ACI
  - in a block regime (long stretches of misses then hits), the fast experts
    (large gamma) track each transition first and accumulate strictly less
    pinball loss, so the exponential weights concentrate on them
  - with a single expert, DtACI reduces step-for-step to plain ACI
  - invalid err (NaN, non-binary) is fail-closed
"""

from __future__ import annotations

import numpy as np
import pytest

from quantkit.conformal import ACIState, DtACIState, aci_update, dtaci_update

GAMMAS = (0.001, 0.005, 0.01, 0.05, 0.1)


def _run(errs, state=None, gammas=GAMMAS):
    st = state if state is not None else DtACIState(alpha_target=0.1)
    out = [dtaci_update(st, e, gammas=gammas) for e in errs]
    return st, np.array(out)


def test_dtaci_direction_persistent_misses_widen_intervals():
    # every step is a miss → alpha must fall (wider intervals), down to the clip floor
    st, ahat = _run([1] * 1000)
    assert ahat[-1] < 0.01
    assert ahat[-1] < ahat[0]
    assert np.all(ahat >= 1e-4 - 1e-12) and np.all(ahat <= 1.0 - 1e-4 + 1e-12)


def test_dtaci_direction_persistent_hits_narrow_intervals():
    # every step is a hit → alpha must rise (narrower intervals)
    st, ahat = _run([0] * 1000)
    assert ahat[-1] > 0.5
    assert ahat[-1] > ahat[0]
    assert np.all(ahat >= 1e-4 - 1e-12) and np.all(ahat <= 1.0 - 1e-4 + 1e-12)


def test_dtaci_weights_concentrate_on_best_gamma():
    # block regime: 100 misses / 100 hits, repeated. Fast experts reach the
    # favourable alpha range first in BOTH phases → strictly less cumulative
    # loss per cycle → weights must end up monotone increasing in gamma.
    errs = ([1] * 100 + [0] * 100) * 10
    st, _ = _run(errs)
    w = st.weights
    assert w is not None
    assert float(w.sum()) == pytest.approx(1.0)
    assert int(np.argmax(w)) == len(GAMMAS) - 1  # largest gamma wins
    assert np.all(np.diff(w) > 0), f"weights not monotone in gamma: {w}"
    assert w[-1] > 1.0 / len(GAMMAS)  # beats the uniform baseline decisively


def test_dtaci_nan_and_nonbinary_err_fail_closed():
    st = DtACIState(alpha_target=0.1)
    for bad in (np.nan, 0.5, -1, 2):
        with pytest.raises(ValueError):
            dtaci_update(st, bad)
    # a rejected call must not mutate the state
    assert st.alphas is None


def test_dtaci_rejects_bad_configuration():
    with pytest.raises(ValueError):
        dtaci_update(DtACIState(alpha_target=1.5), 0)
    with pytest.raises(ValueError):
        dtaci_update(DtACIState(alpha_target=0.1, eta=0.0), 0)
    with pytest.raises(ValueError):
        dtaci_update(DtACIState(alpha_target=0.1), 0, gammas=())
    with pytest.raises(ValueError):
        dtaci_update(DtACIState(alpha_target=0.1), 0, gammas=(0.05, -0.1))


def test_dtaci_single_expert_matches_plain_aci():
    # interface-interchangeability smoke: with one gamma the aggregation is
    # trivial and the alpha trajectory must equal aci_update step for step
    rng = np.random.default_rng(7)
    errs = rng.binomial(1, 0.2, 300)
    dt, ahat = _run(errs.tolist(), gammas=(0.05,))
    aci = ACIState(alpha=0.1, gamma=0.05, alpha_target=0.1)
    for e, a_dt in zip(errs, ahat):
        a_aci = aci_update(aci, int(e))
        assert a_dt == pytest.approx(a_aci, abs=1e-15)
    assert float(dt.weights[0]) == pytest.approx(1.0)

"""Unit tests for ExecutionEnv and the training pipeline (fully offline,
synthetic data)."""

import time

import numpy as np
import pandas as pd
import pytest

from execution_env import (
    ACTIONS,
    ExecutionConfig,
    ExecutionEnv,
    StateDiscretizer,
    evaluate_entries,
    make_twap_policy,
    policy_all_now,
    run_episode,
    synthetic_ohlcv,
)
from train_eval import run_experiment, train_q, make_greedy_policy


def _trend_df(prices, vol=1000.0):
    n = len(prices)
    idx = pd.date_range("2022-01-01", periods=n, freq="4h")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices,
         "volume": np.full(n, vol)},
        index=idx,
    )


# ---------------------------------------------------------------------------
# 1. Environment determinism: same-seed synthetic data + same policy ->
#    identical results
# ---------------------------------------------------------------------------


def test_env_determinism_fixed_seed():
    df = synthetic_ohlcv(seed=3, n_bars=400)
    cfg = ExecutionConfig()
    entries = np.arange(30, 380, 25)

    def random_policy_factory(seed):
        rng = np.random.default_rng(seed)

        def policy(obs, env):
            return float(rng.choice(ACTIONS))

        return policy

    is1 = evaluate_entries(ExecutionEnv(df, cfg), entries, random_policy_factory(11))
    is2 = evaluate_entries(ExecutionEnv(df, cfg), entries, random_policy_factory(11))
    np.testing.assert_array_equal(is1, is2)


def test_train_q_determinism():
    df = synthetic_ohlcv(seed=5, n_bars=500)
    cfg = ExecutionConfig()
    disc = StateDiscretizer(cfg).fit(df.iloc[:350])
    entries = np.arange(30, 340, 20)
    envs = [ExecutionEnv(df, cfg, disc)]
    q1 = train_q(envs, [entries], gamma=0.5, lr=0.2, episodes=60, seed=99)
    q2 = train_q(envs, [entries], gamma=0.5, lr=0.2, episodes=60, seed=99)
    np.testing.assert_array_equal(q1, q2)


# ---------------------------------------------------------------------------
# 2. Accounting identities: executed fractions sum to 1; bar N forces
#    completion; shortfall sign convention
# ---------------------------------------------------------------------------


def test_forced_completion_and_weight_sum():
    df = synthetic_ohlcv(seed=3, n_bars=100)
    env = ExecutionEnv(df, ExecutionConfig())
    # A policy that always "waits" (action=0) -> the final bar must force
    # execution of the entire remaining order
    env.reset(50)
    done = False
    while not done:
        _, _, done, _ = env.step(0.0)
    weights = [h.executed_frac for h in env.history]
    assert len(weights) == env.cfg.n_bars
    assert sum(weights) == pytest.approx(1.0)
    assert weights[-1] == pytest.approx(1.0)  # first 5 bars zero, last bar 100%
    assert env.remaining <= 1e-12


def test_weight_sum_random_policy():
    df = synthetic_ohlcv(seed=4, n_bars=100)
    rng = np.random.default_rng(0)
    env = ExecutionEnv(df, ExecutionConfig())
    env.reset(50)
    done = False
    while not done:
        _, _, done, _ = env.step(float(rng.random()))
    assert sum(h.executed_frac for h in env.history) == pytest.approx(1.0)


def test_shortfall_sign_convention_buy():
    # Monotonically rising price: buying later costs more -> waiting has a
    # higher IS than immediate full execution (both positive costs)
    up = _trend_df(np.linspace(100, 110, 60))
    cfg = ExecutionConfig(fee_bps=0.0, impact_k_bps=0.0)
    entries = np.array([10, 20, 30])
    is_aon = evaluate_entries(ExecutionEnv(up, cfg), entries, policy_all_now)
    is_wait = evaluate_entries(ExecutionEnv(up, cfg), entries, lambda o, e: 0.0)
    assert np.all(is_aon > 0)
    assert np.all(is_wait > is_aon)
    # Monotonically falling price: waiting = buying cheaper -> negative IS
    # (a gain), below AON
    down = _trend_df(np.linspace(110, 100, 60))
    is_aon_d = evaluate_entries(ExecutionEnv(down, cfg), entries, policy_all_now)
    is_wait_d = evaluate_entries(ExecutionEnv(down, cfg), entries, lambda o, e: 0.0)
    assert np.all(is_wait_d < 0)
    assert np.all(is_wait_d < is_aon_d)


# ---------------------------------------------------------------------------
# 3. Fee/impact monotonicity
# ---------------------------------------------------------------------------


def test_fee_monotonicity():
    df = synthetic_ohlcv(seed=3, n_bars=100)
    entries = np.array([20, 40, 60])
    cfg0 = ExecutionConfig(fee_bps=0.0, impact_k_bps=0.0)
    cfg10 = ExecutionConfig(fee_bps=10.0, impact_k_bps=0.0)
    twap = make_twap_policy(6)
    is0 = evaluate_entries(ExecutionEnv(df, cfg0), entries, twap)
    is10 = evaluate_entries(ExecutionEnv(df, cfg10), entries, twap)
    # Full execution -> total fee is exactly fee_bps x 1
    np.testing.assert_allclose(is10 - is0, 10.0, atol=1e-9)


def test_impact_monotonic_in_q():
    df = synthetic_ohlcv(seed=3, n_bars=100)
    env = ExecutionEnv(df, ExecutionConfig(impact_k_bps=50.0))
    env.reset(50)
    impacts = [env.impact_bps(f, 51) for f in (0.0, 0.1, 0.25, 0.5, 1.0)]
    assert impacts[0] == 0.0
    assert all(b > a for a, b in zip(impacts, impacts[1:]))
    # Square-root concavity: two half-sized slices cost less total impact
    # than one full-size slice
    half_twice = 2 * 0.5 * env.impact_bps(0.5, 51)
    full_once = env.impact_bps(1.0, 51)
    assert half_twice < full_once


# ---------------------------------------------------------------------------
# 4. Q-learning update sanity: a monotonically falling toy market teaches
#    the agent to wait
# ---------------------------------------------------------------------------


def test_q_learning_waits_in_falling_market():
    down = _trend_df(np.linspace(200, 100, 300))
    cfg = ExecutionConfig(fee_bps=0.0, impact_k_bps=0.0)
    entries = np.arange(30, 190, 10)
    # Toy environment without a discretizer: vol/trend buckets stay 0 and
    # the state reduces to (bars_remaining, rem_bucket), so Q updates
    # converge quickly over 30 states (the real experiment enables the
    # vol/trend buckets).
    envs = [ExecutionEnv(down, cfg)]
    # gamma=0.9: the terminal value of "waiting" across the 6-step chain is
    # not crushed by discounting (at gamma=0.5, 0.5^5 ~ 3%, discounting
    # would prefer immediate execution — exactly why gamma must be swept).
    q = train_q(envs, [entries], gamma=0.9, lr=0.3, episodes=600, seed=1)
    pol = make_greedy_policy(q, np.random.default_rng(2))
    eval_entries = np.arange(210, 280, 10)
    is_rl = evaluate_entries(envs[0], eval_entries, pol)
    is_aon = evaluate_entries(envs[0], eval_entries, policy_all_now)
    # In a falling market "wait until the last bar" is optimal: the learned
    # policy must beat immediate full execution by a wide margin
    assert is_rl.mean() < is_aon.mean() - 50  # price move ~10%/24bar: huge slack


# ---------------------------------------------------------------------------
# 5. Offline end-to-end: synthetic data + minimal configuration, < 60s
# ---------------------------------------------------------------------------


def test_offline_experiment_fast():
    t0 = time.time()
    s = run_experiment(
        symbols=["SYN-A/USDT", "SYN-B/USDT"],
        start="2022-01-01",
        n_seeds=2,
        episodes=200,
        variant_seeds=1,
        variant_episodes=100,
        force_synthetic=True,
        n_boot=100,
        write_output=False,
    )
    elapsed = time.time() - t0
    assert elapsed < 60, f"offline experiment took {elapsed:.1f}s (over budget)"
    assert s["verdict"] in ("PASS", "CAUTION", "FAIL")
    assert s["n_oos_entries"] > 0
    assert np.isfinite(s["twap_minus_rl_bootstrap"]["point"])

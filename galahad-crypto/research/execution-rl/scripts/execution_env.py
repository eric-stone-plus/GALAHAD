"""Execution-layer RL sandbox environment (ExecutionEnv) — no gymnasium
dependency, reset/step two-phase API.

Design recipe:
- Ahlawat, *Reinforcement Learning for Finance*, §5.5.2 MDP template:
  costs inside the reward, forced liquidation at the terminal bar,
  episodic structure, time-based train/test split.
- RL is used for the execution layer only (not timing/alpha): reward =
  negative implementation shortfall, mandatory baselines = execute-all-now
  and TWAP, gamma treated as a swept hyperparameter.
- Square-root market-impact model impact_bps = k * sqrt(q / ADV_bar)
  (Almgren / Obizhaeva-Wang family); k is a calibrated constant with a
  sensitivity sweep in train_eval.py.

MDP definition:
- The parent-order decision is given: at bar t's close the desk decides to
  buy quantity Q (base units); arrival price = close[t].
- Execution window: bars t+1 ... t+N (N = 6 four-hour bars); each bar
  fills at its close price.
- Action a in {0, 0.25, 0.5, 0.75, 1.0} = fraction of the remaining
  quantity executed this bar; the N-th bar forces full execution
  (action ignored, everything remaining is filled).
- State = (bars_remaining, remaining_frac bucket, short-vol bucket, trend
  bucket); vol/trend features only use data closed before the execution
  bar (MOC decisions, no look-ahead). Bucket edges come from training-set
  quantiles only (no OOS data, no leakage).
- Per-step reward r_i = -w_i * (slip_bps_i + fee_bps + impact_bps_i),
  where w_i is this bar's executed fraction of the parent order; the
  episode reward sums to -IS_bps (implementation shortfall, buy side:
  positive = cost).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACTIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
"""Action grid: fraction of the remaining quantity executed this bar."""

REMAINING_EDGES: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
"""Bucket edges for remaining_frac (fixed, 5 buckets)."""


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution environment parameters. k calibration and sensitivity:
    see the k-sweep in train_eval.py."""

    n_bars: int = 6                 # execution window length (6 x 4h bars = must finish within 24h)
    fee_bps: float = 10.0           # one-way taker fee (conservative venue baseline)
    impact_k_bps: float = 50.0      # square-root impact coefficient: impact_bps = k*sqrt(q/ADV_bar)
    parent_participation: float = 0.01  # parent size = 1% x training-set median bar volume
    vol_lookback: int = 12          # short-term volatility feature window (bars)
    trend_lookback: int = 6         # short-term trend feature window (bars)
    n_vol_buckets: int = 3          # 3 buckets each for vol/trend (train-quantile edges)
    n_trend_buckets: int = 3


# ---------------------------------------------------------------------------
# Synthetic data (deterministic fallback: GBM with volatility clustering,
# runnable offline)
# ---------------------------------------------------------------------------


def synthetic_ohlcv(
    *,
    seed: int = 7,
    n_bars: int = 6000,
    s0: float = 100.0,
    bar_hours: int = 4,
) -> pd.DataFrame:
    """Deterministic synthetic 4h OHLCV: two-state Markov volatility
    regime (clustering) + volume scaling with volatility.

    Fully reproducible for a given seed; used as the offline fallback and
    by all unit tests.
    """
    rng = np.random.default_rng(seed)
    # Two-state vol regime (low/high); the transition matrix creates clustering
    p_stay = 0.97
    state = 0
    sigmas = np.empty(n_bars)
    sig_low, sig_high = 0.006, 0.022  # per 4h bar
    for t in range(n_bars):
        if rng.random() > p_stay:
            state = 1 - state
        sigmas[t] = sig_low if state == 0 else sig_high
    drift = 2e-5
    rets = drift + sigmas * rng.standard_normal(n_bars)
    close = s0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[s0], close[:-1]])
    spread = np.abs(rets) * 0.5 + 0.001
    high = np.maximum(open_, close) * (1 + spread * rng.random(n_bars))
    low = np.minimum(open_, close) * (1 - spread * rng.random(n_bars))
    # Volume: baseline x (1 + volatility amplification) x noise
    base_vol = 5000.0
    volume = base_vol * (1.0 + 60.0 * sigmas) * np.exp(0.3 * rng.standard_normal(n_bars))
    idx = pd.date_range("2022-01-01", periods=n_bars, freq=f"{bar_hours}h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Entry signal: RSI(14) + Bollinger(20, 2.5 sigma), parameterized exactly
# like the round-2 crypto backtest indicators (Wilder RSI, BB on rolling
# std). The signal confirms at bar t's close -> parent BUY order born at t.
# ---------------------------------------------------------------------------


def rsi_bb_entries(
    df: pd.DataFrame,
    *,
    rsi_n: int = 14,
    rsi_buy: float = 25.0,
    bb_n: int = 20,
    bb_k: float = 2.5,
    horizon: int = 6,
) -> np.ndarray:
    """Return bar positions where RSI < rsi_buy and close < lower BB, with
    the full execution window inside the sample."""
    close = df["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / rsi_n, min_periods=rsi_n).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / rsi_n, min_periods=rsi_n).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    sma = close.rolling(bb_n).mean()
    std = close.rolling(bb_n).std()
    bb_lower = sma - bb_k * std
    sig = (rsi < rsi_buy) & (close < bb_lower)
    pos = np.flatnonzero(sig.to_numpy())
    # Needs horizon execution bars + at least 1 feature-history bar
    pos = pos[(pos + horizon < len(df)) & (pos >= 1)]
    return pos


# ---------------------------------------------------------------------------
# State discretization: vol/trend bucket edges come from training-set
# quantiles only (no leakage)
# ---------------------------------------------------------------------------


class StateDiscretizer:
    """fit(train_df) estimates vol/trend bucket edges; transform is usable
    on any period."""

    def __init__(self, cfg: ExecutionConfig):
        self.cfg = cfg
        self.vol_edges: np.ndarray | None = None
        self.trend_edges: np.ndarray | None = None

    def _features(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """vol_t = std of returns over the past vol_lookback bars;
        trend_t = mean over the past trend_lookback bars.

        Both use data <= t only (when deciding at bar j, features from
        j-1 are passed in).
        """
        r = df["close"].pct_change()
        vol = r.rolling(self.cfg.vol_lookback).std()
        trend = r.rolling(self.cfg.trend_lookback).mean()
        return vol, trend

    def fit(self, train_df: pd.DataFrame) -> "StateDiscretizer":
        vol, trend = self._features(train_df)
        qs = np.linspace(0, 1, self.cfg.n_vol_buckets + 1)[1:-1]
        qt = np.linspace(0, 1, self.cfg.n_trend_buckets + 1)[1:-1]
        self.vol_edges = np.nanquantile(vol.to_numpy(), qs)
        self.trend_edges = np.nanquantile(trend.to_numpy(), qt)
        return self

    def bucket(self, df: pd.DataFrame, j: int, remaining_frac: float) -> tuple[int, int, int]:
        """Feature buckets for the execution decision at bar j (using data
        <= j-1 only) + the fixed remaining_frac bucket."""
        assert self.vol_edges is not None and self.trend_edges is not None, "fit() first"
        r = df["close"].pct_change()
        hi = j  # uses r[j-L .. j-1]
        lo_v = max(0, hi - self.cfg.vol_lookback)
        lo_t = max(0, hi - self.cfg.trend_lookback)
        window_v = r.iloc[lo_v:hi].to_numpy()
        window_t = r.iloc[lo_t:hi].to_numpy()
        vol_val = float(np.nanstd(window_v)) if len(window_v) else 0.0
        trend_val = float(np.nanmean(window_t)) if len(window_t) else 0.0
        vol_b = int(np.searchsorted(self.vol_edges, vol_val))
        trend_b = int(np.searchsorted(self.trend_edges, trend_val))
        rem_b = int(np.searchsorted(np.asarray(REMAINING_EDGES), remaining_frac))
        return rem_b, vol_b, trend_b


# ---------------------------------------------------------------------------
# Execution environment
# ---------------------------------------------------------------------------


@dataclass
class StepInfo:
    bar: int
    price: float
    executed_frac: float      # fraction w_i of the parent order
    slip_bps: float
    impact_bps: float
    fee_bps: float


class ExecutionEnv:
    """Single-instrument execution environment. reset(entry_idx) opens an
    episode; step(action_frac) advances one bar.

    action_frac in [0, 1] is the fraction of the remaining quantity
    executed this bar (the RL side maps a discrete action grid onto it;
    baselines pass exact fractions directly, e.g. TWAP's 1/(N-i+1)).
    The n_bars-th bar forces full execution.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: ExecutionConfig | None = None,
        discretizer: StateDiscretizer | None = None,
        q_base: float | None = None,
    ):
        self.cfg = cfg or ExecutionConfig()
        self.df = df.reset_index(drop=True)
        self.close = self.df["close"].to_numpy(dtype=float)
        self.volume = self.df["volume"].to_numpy(dtype=float)
        self.disc = discretizer
        # Parent base quantity: default = parent_participation x median bar volume
        med_vol = float(np.median(self.volume[self.volume > 0])) if (self.volume > 0).any() else 1.0
        self.q_base = q_base if q_base is not None else self.cfg.parent_participation * med_vol
        self._reset_state()

    # -- internal state ----------------------------------------------------
    def _reset_state(self) -> None:
        self.entry_idx = -1
        self.arrival = np.nan
        self.i = 0                    # execution bars advanced (0..n_bars)
        self.remaining = 1.0          # remaining fraction of the parent order
        self.done = True
        self.total_is_bps = 0.0
        self.history: list[StepInfo] = []

    # -- gym-style API ------------------------------------------------------
    def reset(self, entry_idx: int) -> tuple[int, int, int, int]:
        if entry_idx < 1 or entry_idx + self.cfg.n_bars >= len(self.df):
            raise ValueError(f"entry_idx {entry_idx} has no complete execution window")
        self._reset_state()
        self.entry_idx = int(entry_idx)
        self.arrival = float(self.close[entry_idx])
        self.done = False
        return self._obs()

    def _obs(self) -> tuple[int, int, int, int]:
        bars_remaining = self.cfg.n_bars - self.i
        if self.disc is not None:
            rem_b, vol_b, trend_b = self.disc.bucket(
                self.df, self.entry_idx + self.i + 1, self.remaining
            )
        else:  # fixed buckets without a discretizer (tests/baselines)
            rem_b = int(np.searchsorted(np.asarray(REMAINING_EDGES), self.remaining))
            vol_b, trend_b = 0, 0
        return (bars_remaining, rem_b, vol_b, trend_b)

    def impact_bps(self, executed_frac: float, bar: int) -> float:
        """Square-root impact: impact_bps = k * sqrt(q_bar / ADV_bar), with
        ADV_bar taken as this bar's volume."""
        q = executed_frac * self.q_base
        adv = max(float(self.volume[bar]), 1e-12)
        return self.cfg.impact_k_bps * float(np.sqrt(q / adv))

    def step(self, action_frac: float) -> tuple[tuple[int, int, int, int], float, bool, StepInfo]:
        if self.done:
            raise RuntimeError("episode already finished; call reset() first")
        self.i += 1
        bar = self.entry_idx + self.i
        last = self.i == self.cfg.n_bars
        frac = float(np.clip(action_frac, 0.0, 1.0))
        executed = self.remaining if last else frac * self.remaining  # forced liquidation
        executed = min(executed, self.remaining)
        price = float(self.close[bar])
        slip_bps = (price / self.arrival - 1.0) * 1e4  # buy side: positive = paid up = cost
        impact = self.impact_bps(executed, bar) if executed > 0 else 0.0
        fee = self.cfg.fee_bps if executed > 0 else 0.0
        cost_bps = executed * (slip_bps + fee + impact)
        reward = -cost_bps
        self.total_is_bps += cost_bps
        self.remaining -= executed
        self.done = last or self.remaining <= 1e-12
        self.history.append(
            StepInfo(bar=bar, price=price, executed_frac=executed,
                     slip_bps=slip_bps, impact_bps=impact, fee_bps=fee)
        )
        return self._obs(), reward, self.done, self.history[-1]


# ---------------------------------------------------------------------------
# Episode evaluation: policy(state, env) -> action_frac; baselines and RL
# share the same accounting
# ---------------------------------------------------------------------------

Policy = "callable"  # typing placeholder: policy(obs, env) -> float


def run_episode(env: ExecutionEnv, entry_idx: int, policy) -> float:
    """Run one episode, return IS_bps (positive = cost).
    policy(obs, env) -> action_frac."""
    obs = env.reset(entry_idx)
    while True:
        action = policy(obs, env)
        obs, _, done, _ = env.step(action)
        if done:
            break
    return env.total_is_bps


def policy_all_now(obs, env) -> float:
    """execute-all-now: full quantity on the first execution bar."""
    return 1.0


def make_twap_policy(n_bars: int):
    """TWAP-N: equal slices per bar (1/N of the parent) = 1/(N-i+1) of the
    remaining quantity."""

    def policy(obs, env) -> float:
        bars_remaining = obs[0]
        return 1.0 / bars_remaining

    return policy


def evaluate_entries(
    env: ExecutionEnv,
    entries: np.ndarray,
    policy,
) -> np.ndarray:
    """Evaluate a set of entry points episode by episode; returns per-episode
    IS_bps."""
    return np.array([run_episode(env, int(t), policy) for t in entries])

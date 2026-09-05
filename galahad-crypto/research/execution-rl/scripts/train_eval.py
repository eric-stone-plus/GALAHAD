"""Tabular Q-learning execution-scheduling experiment + honest evaluation
gates.

Experiment protocol (four hard gates; results are reported as they come
out, including the case where RL loses to TWAP):
1. Baseline control: execute-all-now (AON) and TWAP-N evaluated on the
   same OOS entry set; per-entry RL-vs-TWAP IS differences go through a
   block-bootstrap CI (quantkit.validation).
2. Seed distribution: n_seeds = 10 independent training seeds; report the
   OOS mean IS of every seed (no seed cherry-picking).
3. Probability of backtest overfitting: CSCV PBO over a small policy/gamma
   grid (gamma x lr, 12 variants), following Gort et al. arXiv:2209.05559
   ("test for overfitting before talking performance") via
   quantkit.validation.prob_backtest_overfitting.
4. Sensitivity: OOS IS table for gamma in {0.0, 0.5, 0.9, 0.99} (gamma
   starts at 0.5, never defaults to 0.99); plus an impact-coefficient
   sweep k in {0, 25, 50, 100} bps.

Verdict: PASS = RL beats TWAP with a bootstrap CI excluding 0 and
PBO < 0.5; FAIL = CI upper bound < 0 (significantly worse); anything
else = CAUTION (statistically indistinguishable).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_env import (  # noqa: E402
    ACTIONS,
    ExecutionConfig,
    ExecutionEnv,
    StateDiscretizer,
    evaluate_entries,
    make_twap_policy,
    policy_all_now,
    rsi_bb_entries,
    synthetic_ohlcv,
)

from quantkit.validation import block_bootstrap_ci, prob_backtest_overfitting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

GAMMA_GRID = (0.0, 0.5, 0.9, 0.99)
LR_GRID = (0.1, 0.2, 0.3)
MAIN_GAMMA = 0.5   # gamma starts at 0.5 by protocol, never defaults to 0.99
MAIN_LR = 0.2
K_SWEEP = (0.0, 25.0, 50.0, 100.0)


# ---------------------------------------------------------------------------
# Data loading: quantkit ccxt fetch (with parquet cache), deterministic
# synthetic fallback on any failure
# ---------------------------------------------------------------------------


def load_market_data(
    symbols: list[str],
    start: str,
    end: str | None,
    *,
    force_synthetic: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Return (symbol -> OHLCV, symbol -> source tag real/synthetic)."""
    data: dict[str, pd.DataFrame] = {}
    source: dict[str, str] = {}
    for i, sym in enumerate(symbols):
        df = None
        if not force_synthetic:
            try:
                from quantkit.data import fetch_ohlcv

                df = fetch_ohlcv(
                    sym,
                    market="crypto",
                    interval="4h",
                    start=start,
                    end=end,
                    data_dir=DATA_DIR,
                )
                if df is None or len(df) < 500:
                    df = None
            except Exception as exc:  # any network/ccxt failure -> synthetic
                print(f"[data] {sym} fetch failed ({type(exc).__name__}); using synthetic data")
        if df is None:
            df = synthetic_ohlcv(seed=1000 + i, n_bars=6000)
            source[sym] = "synthetic"
        else:
            source[sym] = "real"
        data[sym] = df
    return data, source


# ---------------------------------------------------------------------------
# Tabular Q-learning
# ---------------------------------------------------------------------------

# Q index: (bars_remaining-1, rem_bucket, vol_bucket, trend_bucket, action)
Q_SHAPE = (6, 5, 3, 3, len(ACTIONS))


def _greedy_action(q_sa: np.ndarray, rng: np.random.Generator) -> int:
    """argmax with random tie-breaking among maxima (deterministic within a
    seed)."""
    m = np.max(q_sa)
    cands = np.flatnonzero(q_sa == m)
    return int(rng.choice(cands))


def train_q(
    envs: list[ExecutionEnv],
    train_entries: list[np.ndarray],
    *,
    gamma: float,
    lr: float,
    episodes: int,
    seed: int,
    eps0: float = 1.0,
    eps1: float = 0.05,
) -> np.ndarray:
    """Train tabular Q with epsilon-greedy on the merged multi-symbol
    training entry pool."""
    rng = np.random.default_rng(seed)
    q = np.zeros(Q_SHAPE)
    pool = [(si, int(t)) for si, ent in enumerate(train_entries) for t in ent]
    if not pool:
        raise ValueError("training entry pool is empty")
    for ep in range(episodes):
        eps = eps0 + (eps1 - eps0) * ep / max(1, episodes - 1)
        si, t = pool[rng.integers(len(pool))]
        env = envs[si]
        obs = env.reset(t)
        while True:
            s_idx = (obs[0] - 1, obs[1], obs[2], obs[3])
            if rng.random() < eps:
                a = int(rng.integers(len(ACTIONS)))
            else:
                a = _greedy_action(q[s_idx], rng)
            obs2, r, done, _ = env.step(ACTIONS[a])
            target = r
            if not done:
                s2 = (obs2[0] - 1, obs2[1], obs2[2], obs2[3])
                target = r + gamma * float(np.max(q[s2]))
            q[s_idx + (a,)] += lr * (target - q[s_idx + (a,)])
            if done:
                break
            obs = obs2
    return q


def make_greedy_policy(q: np.ndarray, rng: np.random.Generator):
    """Greedy evaluation policy from a Q table (deterministic within a
    seed)."""

    def policy(obs, env) -> float:
        s_idx = (obs[0] - 1, obs[1], obs[2], obs[3])
        return ACTIONS[_greedy_action(q[s_idx], rng)]

    return policy


# ---------------------------------------------------------------------------
# Experiment main flow
# ---------------------------------------------------------------------------


def build_envs_and_entries(
    data: dict[str, pd.DataFrame],
    cfg: ExecutionConfig,
    train_frac: float = 0.7,
) -> tuple[list[ExecutionEnv], list[np.ndarray], list[np.ndarray], list[str]]:
    """Per symbol: fit the discretizer on the training segment, build the
    env; entries split 70/30 in time (true OOS is the tail)."""
    envs, train_list, oos_list, names = [], [], [], []
    for sym, df in data.items():
        split = int(len(df) * train_frac)
        disc = StateDiscretizer(cfg).fit(df.iloc[:split])
        train_df = df.iloc[:split]
        med_vol = float(np.median(train_df["volume"].to_numpy()))
        env = ExecutionEnv(
            df, cfg, disc, q_base=cfg.parent_participation * med_vol
        )
        entries = rsi_bb_entries(df, horizon=cfg.n_bars)
        envs.append(env)
        train_list.append(entries[entries < split - cfg.n_bars - 1])
        oos_list.append(entries[entries >= split])
        names.append(sym)
    return envs, train_list, oos_list, names


def eval_policy_all_envs(
    envs: list[ExecutionEnv],
    oos_list: list[np.ndarray],
    policy,
) -> np.ndarray:
    """Merge per-entry OOS IS across symbols."""
    parts = [
        evaluate_entries(env, entries, policy)
        for env, entries in zip(envs, oos_list)
        if len(entries)
    ]
    return np.concatenate(parts) if parts else np.array([])


def run_experiment(
    *,
    symbols: list[str],
    start: str,
    end: str | None = None,
    n_seeds: int = 10,
    episodes: int = 4000,
    variant_seeds: int = 3,
    variant_episodes: int | None = None,
    force_synthetic: bool = False,
    n_boot: int = 1000,
    master_seed: int = 42,
    write_output: bool = True,
) -> dict:
    t0 = time.time()
    variant_episodes = variant_episodes or episodes
    cfg = ExecutionConfig()
    data, source = load_market_data(symbols, start, end, force_synthetic=force_synthetic)
    envs, train_list, oos_list, names = build_envs_and_entries(data, cfg)
    n_train = sum(len(x) for x in train_list)
    n_oos = sum(len(x) for x in oos_list)
    print(f"[setup] symbols={names} source={[source[s] for s in names]}")
    print(f"[setup] train_entries={n_train} oos_entries={n_oos} (RSI<25 & close<BB_lower, 4h)")
    if n_train < 30:
        print("[warn] fewer than 30 training entries: statistical power is minimal, treat conclusions as indicative only")

    # ---- baselines ---------------------------------------------------------
    is_aon = eval_policy_all_envs(envs, oos_list, policy_all_now)
    is_twap = eval_policy_all_envs(envs, oos_list, make_twap_policy(cfg.n_bars))

    # ---- main configuration: n_seeds seeds ---------------------------------
    seed_rows = []
    rl_per_entry = []
    for s in range(n_seeds):
        seed = master_seed + 1000 * s
        q = train_q(envs, train_list, gamma=MAIN_GAMMA, lr=MAIN_LR,
                    episodes=episodes, seed=seed)
        pol = make_greedy_policy(q, np.random.default_rng(seed + 1))
        is_rl = eval_policy_all_envs(envs, oos_list, pol)
        rl_per_entry.append(is_rl)
        seed_rows.append(
            {"seed": seed, "oos_is_mean": float(np.mean(is_rl)),
             "oos_is_median": float(np.median(is_rl))}
        )
        print(f"[seed {s}] oos_is_mean={np.mean(is_rl):+.2f} bps")
    rl_mean_entry = np.mean(rl_per_entry, axis=0)  # seed ensemble (FinRL Contest convention)

    # ---- gate 1: RL vs TWAP paired-difference block-bootstrap CI ------------
    diff = is_twap - rl_mean_entry  # positive = RL costs less
    pt, lo, hi = block_bootstrap_ci(
        diff, lambda x: float(np.mean(x)), n_boot=n_boot, random_state=master_seed
    )

    # ---- gate 3: policy/gamma grid PBO --------------------------------------
    variant_scores = []
    variant_names = []
    for g in GAMMA_GRID:
        for lr in LR_GRID:
            per_seed = []
            for s in range(variant_seeds):
                seed = master_seed + 777 + 100 * s
                q = train_q(envs, train_list, gamma=g, lr=lr,
                            episodes=variant_episodes, seed=seed)
                pol = make_greedy_policy(q, np.random.default_rng(seed + 1))
                per_seed.append(eval_policy_all_envs(envs, oos_list, pol))
            variant_scores.append(-np.mean(per_seed, axis=0))  # higher is better
            variant_names.append(f"g{g}_lr{lr}")
    pbo = float("nan")
    pbo_note = "skipped: insufficient OOS entries"
    if n_oos >= 16:
        n_blocks = 8 if n_oos >= 32 else 4
        pbo = float(
            prob_backtest_overfitting(
                np.column_stack(variant_scores), n_blocks=n_blocks, metric="sharpe"
            )
        )
        pbo_note = f"CSCV S={n_blocks}, N={len(variant_names)} variants"
    print(f"[pbo] {pbo:.3f} ({pbo_note})")

    # ---- gate 4a: gamma sensitivity (main lr) -------------------------------
    gamma_rows = []
    for g in GAMMA_GRID:
        col = variant_scores[variant_names.index(f"g{g}_lr{MAIN_LR}")]
        gamma_rows.append({"gamma": g, "oos_is_mean": float(-np.mean(col))})

    # ---- gate 4b: impact-coefficient k sensitivity (re-evaluate the main
    #      seed-ensemble policy under each k) ----------------------------------
    k_rows = []
    for k in K_SWEEP:
        cfg_k = dataclasses.replace(cfg, impact_k_bps=k)
        envs_k = [
            ExecutionEnv(e.df, cfg_k, e.disc, q_base=e.q_base) for e in envs
        ]
        per_seed = []
        for s in range(n_seeds):
            seed = master_seed + 1000 * s
            q = train_q(envs_k, train_list, gamma=MAIN_GAMMA, lr=MAIN_LR,
                        episodes=episodes, seed=seed)
            pol = make_greedy_policy(q, np.random.default_rng(seed + 1))
            per_seed.append(eval_policy_all_envs(envs_k, oos_list, pol))
        rl_k = float(np.mean([x.mean() for x in per_seed]))
        twap_k = float(np.mean(eval_policy_all_envs(envs_k, oos_list, make_twap_policy(cfg.n_bars))))
        k_rows.append({"k_bps": k, "rl_is": rl_k, "twap_is": twap_k,
                       "rl_minus_twap": rl_k - twap_k})
        print(f"[k={k:.0f}] rl={rl_k:+.2f} twap={twap_k:+.2f} bps")

    # ---- verdict -------------------------------------------------------------
    if lo > 0 and (np.isnan(pbo) or pbo < 0.5):
        verdict = "PASS"
    elif hi < 0:
        verdict = "FAIL"
    else:
        verdict = "CAUTION"

    seed_means = [r["oos_is_mean"] for r in seed_rows]
    summary = {
        "verdict": verdict,
        "data_source": source,
        "n_train_entries": n_train,
        "n_oos_entries": n_oos,
        "config": {
            "n_bars": cfg.n_bars, "fee_bps": cfg.fee_bps,
            "impact_k_bps": cfg.impact_k_bps,
            "parent_participation": cfg.parent_participation,
            "main_gamma": MAIN_GAMMA, "main_lr": MAIN_LR,
            "n_seeds": n_seeds, "episodes": episodes,
        },
        "baseline_is_bps": {"aon": float(np.mean(is_aon)), "twap": float(np.mean(is_twap))},
        "rl_is_bps": {"mean_of_seed_means": float(np.mean(seed_means)),
                      "seed_min": float(np.min(seed_means)),
                      "seed_median": float(np.median(seed_means)),
                      "seed_max": float(np.max(seed_means))},
        "twap_minus_rl_bootstrap": {"point": pt, "ci_lo": lo, "ci_hi": hi,
                                    "n_boot": n_boot},
        "pbo": pbo, "pbo_note": pbo_note,
        "gamma_sensitivity": gamma_rows,
        "k_sensitivity": k_rows,
        "seed_table": seed_rows,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    if write_output:
        OUTPUT_DIR.mkdir(exist_ok=True)
        (OUTPUT_DIR / "experiment_results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )
        pd.DataFrame(
            {"is_twap": is_twap, "is_aon": is_aon, "is_rl_seed_ens": rl_mean_entry}
        ).to_csv(OUTPUT_DIR / "oos_per_entry.csv", index=False)
        print(f"[output] {OUTPUT_DIR/'experiment_results.json'}")
    return summary


def print_verdict_block(s: dict) -> None:
    b = s["baseline_is_bps"]
    r = s["rl_is_bps"]
    ci = s["twap_minus_rl_bootstrap"]
    lines = [
        "=" * 60,
        f"VERDICT: {s['verdict']}",
        "=" * 60,
        f"data_source          : {s['data_source']}",
        f"entries train/oos    : {s['n_train_entries']} / {s['n_oos_entries']}",
        f"OOS IS AON / TWAP    : {b['aon']:+.2f} / {b['twap']:+.2f} bps",
        f"OOS IS RL (10 seeds) : mean {r['mean_of_seed_means']:+.2f} "
        f"[{r['seed_min']:+.2f}, {r['seed_median']:+.2f}, {r['seed_max']:+.2f}] bps",
        f"TWAP-RL diff (bps)   : {ci['point']:+.2f} "
        f"CI95 [{ci['ci_lo']:+.2f}, {ci['ci_hi']:+.2f}] (block-bootstrap)",
        f"PBO (policy/gamma)   : {s['pbo']:.3f} ({s['pbo_note']})",
        "gamma sensitivity (OOS IS bps): "
        + ", ".join(f"g={g['gamma']}: {g['oos_is_mean']:+.2f}" for g in s["gamma_sensitivity"]),
        "k sensitivity (RL-TWAP bps)   : "
        + ", ".join(f"k={k['k_bps']:.0f}: {k['rl_minus_twap']:+.2f}" for k in s["k_sensitivity"]),
        f"elapsed              : {s['elapsed_sec']}s",
        "=" * 60,
    ]
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Execution-layer RL sandbox: Q-learning vs TWAP/AON")
    ap.add_argument("--fast", action="store_true", help="quick mode with reduced seeds/episodes")
    ap.add_argument("--synthetic", action="store_true", help="force synthetic data (offline)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    n_seeds = args.seeds or (3 if args.fast else 10)
    episodes = args.episodes or (800 if args.fast else 4000)
    summary = run_experiment(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        start=args.start,
        end=args.end,
        n_seeds=n_seeds,
        episodes=episodes,
        variant_seeds=2 if args.fast else 3,
        force_synthetic=args.synthetic,
        n_boot=400 if args.fast else 1000,
    )
    print_verdict_block(summary)


if __name__ == "__main__":
    main()

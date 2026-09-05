# crypto_execution_rl — execution-layer RL sandbox

The first reinforcement-learning experiment in this workspace.
**Positioning: an execution-layer sandbox, NOT a timing/alpha agent.
Pure sandbox — it never places a real order.**

## Scope statement

- This project answers exactly one question: **given a parent buy order
  already decided** (produced by an external signal — the experiment
  replays the RSI+BB rules from the round-2 crypto backtest), can RL
  schedule the execution over the following six 4h bars at a lower
  implementation shortfall than mechanical baselines (execute-all-now /
  TWAP)?
- What it does NOT do: predict direction, choose entries, size
  positions, or touch any live-trading interface. Timing-layer DRL has
  no stable post-cost evidence in the 2024–2026 literature consensus;
  the execution layer is where RL has a credible landing zone (the LOXM
  lineage; the Almgren–Chriss-corrected route of Hendricks & Wilcox).
- The environment follows the §5.5.2 MDP template of Ahlawat,
  *Reinforcement Learning for Finance*: costs inside the reward, forced
  terminal liquidation, episodic structure, time-based train/test split.

## Method

### MDP

- **Parent order**: a signal at bar t's close triggers a buy of fixed
  quantity Q (= 1% x median training-set bar volume, base units);
  arrival price = close[t].
- **Execution window**: bars t+1 … t+6 (N = 6, 24 hours); each bar fills
  at its close (MOC assumption — decisions use data up to the previous
  bar's close only, no look-ahead).
- **Actions**: a ∈ {0, 0.25, 0.5, 0.75, 1.0} = fraction of the remaining
  quantity executed this bar; bar 6 forces full execution.
- **State**: (bars_remaining, remaining_frac bucket, short-vol bucket,
  trend bucket). Vol/trend bucket edges come from training-set quantiles
  only (no OOS data, no leakage).
- **Costs** (buy side, bps, positive = cost):
  `IS = Σ_i w_i·[(p_i/p_arr − 1)·1e4 + fee_bps + impact_bps_i]`,
  fee = 10 bps one-way taker (conservative venue baseline); impact =
  square-root model `impact_bps = k·sqrt(q_i/ADV_bar)`, k = 50 bps
  (Almgren / Obizhaeva-Wang family calibration; sensitivity swept below).
  Per-step reward = −step cost; episode reward = −IS.
- **Learner**: tabular Q-learning (discrete state, 1350 Q entries),
  ε-greedy (1.0 → 0.05 linear decay), one shared Q table across symbols
  (buckets are per-symbol training-normalized).

### Experiment protocol (four honesty gates)

1. **Baseline control**: execute-all-now (AON) and TWAP-6 evaluated on
   the same OOS entries; per-entry RL-vs-TWAP IS differences go through
   a block-bootstrap 95% CI (`quantkit.validation.block_bootstrap_ci`).
2. **Seed distribution**: 10 independent training seeds, each one's OOS
   mean IS reported — no seed cherry-picking.
3. **Probability of backtest overfitting**: CSCV PBO over a small
   policy/γ grid (γ ∈ {0, 0.5, 0.9, 0.99} × lr ∈ {0.1, 0.2, 0.3}, 12
   variants) via `quantkit.validation.prob_backtest_overfitting` (the
   Gort et al. arXiv:2209.05559 "test for overfitting before talking
   performance" principle).
4. **γ / k sensitivity**: full-grid OOS IS table over γ (starting at
   0.5, never defaulting to 0.99); main policy re-evaluated under
   k ∈ {0, 25, 50, 100} bps.

**Verdict rule**: PASS = CI lower bound > 0 and PBO < 0.5; FAIL = CI
upper bound < 0 (RL significantly worse); anything else = CAUTION
(statistically indistinguishable). **Results are reported as they come
out, including RL losing to TWAP.**

### Data

- Real path: `quantkit.data.fetch_ohlcv` (ccxt, 4h, BTC/USDT + ETH/USDT
  from 2022-01-01), parquet-cached under `data/`.
- Fallback: any network/fetch failure switches to deterministic
  synthetic data (two-state Markov volatility-clustering GBM, fixed
  seed, fully reproducible) so the experiment runs anywhere. Entry
  signal: RSI(14, Wilder) < 25 and close < BB(20, 2.5σ) lower band —
  parameterized exactly like the round-2 crypto backtest indicators.
  Time split 70/30, OOS is the later segment.

## Results summary

<!-- RESULTS:START -->
Full run (2026-07-26, real BTC/USDT + ETH/USDT 4h data 2022-01-01 →
2026-07-26, 10 seeds x 4000 episodes, train/oos entries 75/30) —
**verbatim verdict block**:

```
============================================================
VERDICT: FAIL
============================================================
data_source          : {'BTC/USDT': 'real', 'ETH/USDT': 'real'}
entries train/oos    : 75 / 30
OOS IS AON / TWAP    : -15.20 / +31.39 bps
OOS IS RL (10 seeds) : mean +80.39 [+26.91, +81.03, +102.87] bps
TWAP-RL diff (bps)   : -49.00 CI95 [-95.89, -7.89] (block-bootstrap)
PBO (policy/gamma)   : 0.167 (CSCV S=4, N=12 variants)
gamma sensitivity (OOS IS bps): g=0.0: +94.07, g=0.5: +86.71, g=0.9: +49.48, g=0.99: +48.15
k sensitivity (RL-TWAP bps)   : k=0: +52.15, k=25: +37.38, k=50: +49.00, k=100: +34.72
elapsed              : 1038.2s
============================================================
```

**Verdict: FAIL — RL execution scheduling is significantly worse than
TWAP (by 49 bps; bootstrap CI upper bound −7.89 < 0), and also worse
than execute-all-now.** Reported as-is per protocol. Mechanism
diagnosis (inline probing, not part of the committed pipeline —
comparing train/OOS baselines of AON/TWAP/wait-all plus the learned
per-bar execution weights): in the 24h window after an RSI<25 + BB-lower
entry, price drift is positive (train: AON −1.5 / TWAP +23.4 / wait-all
+75.2 bps; OOS same direction), so the optimal policy is actually "buy
everything immediately"; but costs-in-reward + γ<1 credit assignment
makes the agent prefer deferral (each step saves fee immediately while
the drift cost gets discounted). The main configuration learned to push
88% of the quantity into the final bar — the worst possible schedule in
this regime. Higher γ reduces the damage (g=0.99: +48 vs g=0.0: +94)
but no setting beats the baselines. Seed variance is huge (+27 to +103
bps): the policy is unstable. This matches the 2024–2026 literature
consensus exactly (simple baselines remain unbeaten, costs are the
largest erosion term of any RL advantage, γ must be swept).
**Production implication: for this signal family, plain AON inside the
24h window is sufficient — RL is not needed. TWAP only overtakes AON
when k is large (impact-dominated; see twap_is in the k-sensitivity
table).**
<!-- RESULTS:END -->

## Reproduce

```bash
cd research/execution-rl
python -m pytest tests/ -q               # 9 unit tests (fully offline, synthetic data)
python scripts/train_eval.py --fast      # quick mode (3 seeds x 800 episodes)
python scripts/train_eval.py             # full run (10 seeds x 4000 episodes)
python scripts/train_eval.py --synthetic # force synthetic data (offline)
```

Requirements: pandas, numpy, and `quantkit` (this repository, for
validation + ccxt data fetching). Outputs:
`output/experiment_results.json` (verdict + all tables),
`output/oos_per_entry.csv` (per-entry IS).

## Limitations

1. **Sandbox-to-reality gap**: no limit order book, no queue-position
   slippage model, bar-total volume used as tradable volume; the impact
   model is a calibrated square-root approximation, not an estimated
   propagator.
2. **Very coarse state**: 3x3 vol/trend buckets capture little timing
   information; the execution window is 24h and cross-regime
   generalization is untested.
3. **Statistical power**: RSI<25 + BB-lower entries are sparse (~hundreds
   across train/OOS); wide bootstrap CIs are the norm — CAUTION is the
   expected baseline answer for this experiment class.
4. **Single signal family**: evaluated only on oversold-bounce entries;
   conclusions do not extrapolate to trend entries or sell-side
   execution.
5. Tabular Q-learning is a deliberate choice (sample size ~10^3; deep RL
   would overfit — consistent with the well-known RL-in-finance risk
   list and the 2024–2026 literature digest conclusion); gymnasium/torch,
   LOB data, and minute-frequency are requirements of a production
   version, not goals of this sandbox.

# Strategy Foundations (academic register)

**Revision:** 2026-08-05  
**Scope:** What a GALAHAD trading strategy is allowed to claim, and which
literatures gate promotion from toy signal → paper → live.  
Production ledgers stay private; this note is the public research spine.

Evidence for the citations below was gathered primarily via **local** host
Firecrawl (`scrape` of arXiv abs pages) and arXiv API — not paywalled harvest.

---

## 1. Positioning: strategy research vehicle, not a chat trader

GALAHAD answers a financial-engineering question:

> Which **numeric target positions**, under **hard risk caps** and
> **anti-overfitting validation**, survive paper execution on venue-native data?

It is **not** an LLM multi-agent that invents orders. Neighbor systems
(OpenAlice, TradingAgents, OpenClaw) may inform human research; they do not
substitute margin accounting, target→risk→fill separation, or statistical gates.

---

## 2. Strategy families (first principles)

### 2.1 Trend / time-series momentum (primary academic spine)

| Claim in literature | Implication for GALAHAD |
|---|---|
| Trend following is a long-horizon anomaly across futures asset classes with high t-stats over centuries of data (Lempérière et al., arXiv:1404.3274, *Two centuries of trend following*) | Dual-MA / TSMOM-style **targets** are a legitimate *null family* to start paper plumbing — not because they are “alpha sure,” but because the economic hypothesis is pre-specified and well studied |
| Shorter trends have withered more than long trends in modern samples (same paper) | Prefer slower signals for research defaults; treat fast dual-MA as **plumbing / stress**, not production alpha |
| Crypto markets share commodity-like trend characteristics; decade-scale evidence of TF in crypto (arXiv:2009.12155) | Crypto-first paper path is academically coherent for trend families |
| Network / multi-asset momentum extensions exist (e.g. arXiv:2501.07135) | Later research track — only after single-name dual-MA passes validation gates |

**Engineering mapping:** strategy emits **signed target leverage** (or weight), never orders.
The private paper book applies leverage, fees, MTM, and liquidation.

### 2.2 Perpetual futures microstructure (execution substrate)

| Claim | Implication |
|---|---|
| Perpetuals do not guarantee spot convergence; **funding** is the alignment mechanism (He, Teng, Zhang et al., arXiv:2212.06888, *Fundamentals of Perpetual Futures*) | Paper engines must expose a **funding hook**; ignoring funding overstates edge |
| Funding design can be studied as a control mechanism (arXiv:2506.08573) | Research path for funding-skew strategies is secondary to trend null |
| Empirical liquidations on high-leverage BTC perps are material; optimal margins are far above exchange marketing mins under fat tails (arXiv:2102.04591) | Default leverage must be **conservative** (single-digit); liquidation force-close is a first-class test, not an edge case |

### 2.3 Deliberately naive defaults

The first coded strategy (dual moving-average) is intentionally **naive**:

1. Proves the execution stack (targets → risk → fills → journal).
2. Matches the industry’s best-known overfit-prone toy (see §3 — dual-MA parameter search often yields **high PBO** in industry studies).
3. Must **fail** statistical gates when over-tuned; that failure is a feature.

---

## 3. Anti-overfitting gates (non-negotiable)

Implementation lives in the private stack’s validation library (purged k-fold +
embargo, walk-forward, Deflated Sharpe Ratio, CSCV/PBO, block bootstrap). Public
doctrine:

| Gate | Role | Pass heuristic (research policy) |
|---|---|---|
| **Purged CV + embargo** | Stop label leakage on overlapping returns | Prefer simple params under purged folds |
| **Walk-forward** | Month-aligned OOS blocks | No collapse of OOS equity vs IS |
| **Deflated Sharpe (DSR)** | Correct Sharpe for multiple testing / non-normality (Bailey & López de Prado 2014 lineage) | DSR > 0 as a weak necessary condition |
| **PBO (CSCV)** | Probability that the selected strategy is overfit (Bailey–Borwein–López de Prado–Zhu lineage; industry applications e.g. Huatai AI series #22) | Treat **PBO > 0.5** as research red flag; dual-MA *parameter grids* often sit here |
| **Block bootstrap CI** | Dependence-aware uncertainty | Report distributions, not single Sharpe points |

**Doctrine (Lopez de Prado-aligned):** prefer procedures that **determine rules
without exhaustive backtest search** where possible (cf. arXiv:1408.1159 on
optimal trading rules without backtesting). When search is used, inflate trials
into DSR/`n_trials` and report PBO.

### 3.1 What dual-MA paper equity is *not*

A single fixture path that ends with higher equity is **plumbing evidence**, not
a statistical claim of edge. Promotion criteria (paper → live research):

1. Multi-window walk-forward on venue history (not one synthetic series).
2. DSR under an honest trial count (all MA pairs / cost tiers tried).
3. PBO on the returns matrix of those variants.
4. ≥4 weeks unattended paper with explainable fill vs backtest gap.
5. Invalidation condition written *before* live capital.

---

## 4. Risk doctrine restated for strategies

1. Strategies emit **targets**; execution layer alone places fills (hard caps).
2. News / digests / agent chat are **attention**, never L3 inputs.
3. Leverage defaults respect liquidation literature (fat tails → lower leverage).
4. HALT / kill-switch fail-closed (ops plane may halt; never auto-resume live).
5. Funding and fees are first-class costs in any claim of edge.

---

## 5. Research agenda (trading-first)

| Priority | Workstream | Academic anchor |
|---|---|---|
| **P0** | Pre-specified **TSMOM** paper on crypto (venue-cache OHLCV) with funding + walk-forward OOS; dual-MA only as plumbing null | 1404.3274; 2009.12155; validation gates |
| **P1** | Parameter search discipline: if any grid is used → PBO/DSR report artifact (never claim from synthetic fixture) | CSCV/PBO + DSR |
| **P2** | Funding-aware residual / carry as second family | 2212.06888; 2506.08573 |
| **P3** | Multi-asset network momentum only after single-name gates pass | 2501.07135 |
| **P4** | Broker equity paper (Alpaca/IBKR) as parallel track | data-stack note |

---

## 6. Bibliography (primary)

| ID | Title | Use |
|---|---|---|
| arXiv:1404.3274 | Two centuries of trend following | TF anomaly backbone |
| arXiv:2009.12155 | A Decade of Evidence of Trend Following Investing in Cryptocurrencies | Crypto TF |
| arXiv:2212.06888 | Fundamentals of Perpetual Futures | Funding / pricing |
| arXiv:2506.08573 | Designing funding rates for perpetual futures… | Funding mechanism design |
| arXiv:2102.04591 | Liquidation, Leverage and Optimal Margin in Bitcoin Futures Markets | Leverage / margin policy |
| arXiv:1408.1159 | Determining Optimal Trading Rules without Backtesting | Anti-search discipline |
| arXiv:2501.07135 | Follow the Leader: … Network Momentum | Later multi-asset track |
| Bailey & López de Prado (2014) | Deflated Sharpe Ratio (SSRN lineage) | Multiple-testing correction |
| Bailey et al. | Probability of Backtest Overfitting / CSCV | Selection bias |

Host Firecrawl scrape logs (private machine, 2026-08-05) covered the arXiv abs
pages above successfully; SSRN abstract pages were not required for doctrine.

---

## 7. One-sentence strategy thesis

**Start with a century-tested trend hypothesis on crypto perpetuals, execute only
through a margin-correct paper book with fail-closed risk, and refuse any live
claim that does not clear purged walk-forward, DSR, and PBO under honest trial
counts — treating dual-MA paper equity as a systems test until those gates pass.**

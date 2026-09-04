# Top-Tier Quantitative Firms: Market Structure and Execution Benchmarks

**Revision:** 2026-09-04
**Scope:** How the firms commonly ranked in the top quant tiers actually make
money — strategy families, execution mechanisms, and infrastructure — and which
of their practices transfer to GALAHAD. This is a benchmarking note, not a
hero catalogue: the point is to identify which disciplines are worth importing
and which games we structurally cannot and should not play.

Sources are listed in §6. Claims that rest on a single or secondhand source
are marked. Company-internal details are, by nature, only partially observable.

---

## 1. One ranking, two industries

The familiar tier charts mix two different businesses. The split that matters
is not talent or AUM; it is **who provides liquidity and who consumes it**.

| | Market makers (prop) | Asset managers |
|---|---|---|
| Firms | Jane Street, Citadel Securities, HRT, Jump, XTX, Optiver, IMC, SIG | Citadel (the fund), D.E. Shaw, Millennium, Point72 |
| Role | Continuously stand as counterparty | Intermittently convert views into positions |
| Optimizes | Quotes, inventory, adverse selection, queue position | Signal quality, portfolio construction, impact cost |
| Execution is | The product itself | A cost center protecting alpha |
| Capital | Own balance sheet | LP capital + leverage |
| Failure mode | 100 ns slow = stale quotes get picked off | Sloppy execution erodes, but does not void, the strategy |

A D.E. Shaw ranked "tier 1" and an Optiver ranked "tier 2" reflects brand and
capital, not fiber and FPGAs — they are playing different games. GALAHAD's
game is unambiguously the right-hand column.

## 2. Strategy families by firm (compressed)

- **Jane Street** — ETF arbitrage at industrial scale. As one of the largest
  Authorized Participants, it creates/redeems ETF baskets in the primary
  market while making markets in the secondary; roughly 41% of 2024 bond-ETF
  volume, ~$230B average daily traded. Bonds, options, ADR/cross-market
  arb alongside. Pure prop; OCaml (and Hardcaml FPGA) for correctness-critical
  systems, with GPU inference wired toward the trading path.
- **Citadel (fund)** — multi-strategy asset manager (~$65B): equities,
  commodities, fixed income/macro, Global Quantitative Strategies, credit.
  Institutional algorithmic execution; no market-making infrastructure.
- **Citadel Securities** — the largest US electronic market maker. Retail
  internalization via payment for order flow (SEC-cited ~47% of US listed
  retail volume in 2021; ~35% by more recent counts), NYSE DMM, ~30%+ of
  US listed options volume, growing FICC client franchise; acquired Morgan
  Stanley's exchange options market-making business in 2025.
- **HRT** — HFT market making ("Classic", ~half of profits) plus a
  longer-horizon book ("Prism": ETF arb, index rebalance, stat-arb,
  quant macro). C++/Python research, Verilog FPGA tick-to-trade with a
  simulation-first verification pipeline. Prop.
- **D.E. Shaw** — the original stat-arb fund, now a systematic↔discretionary
  continuum; >$100B invested and committed capital as of 2026-06. Buy-side
  execution: optimizers, implementation-shortfall management. Spun out
  Arcesium for post-trade. Asset manager.
- **XTX Markets** — fully automated ML market maker, no human traders.
  ~53,000 instruments, ~$250B/day, FX-rooted. Famously "0 microwave links":
  it exited the nanosecond race and spends on prediction instead —
  25,000+ GPUs, a self-built ~250MW data center in Finland, ~65% passive
  fills, OTC streaming so it need not win exchange queues.
- **Jump Trading** — Chicago HFT; owns microwave tower routes (e.g.
  London–Frankfurt via a purchased NATO tower), FPGA feed handling,
  CME/LSE colocation; Jump Crypto on the same capital base. Prop.
- **Five Rings** — small automated prop shop; virtually no reliable primary
  disclosure. Treat any detailed claim about it as unverified.
- **Optiver / IMC** — Amsterdam-school options market makers. Pricing, risk,
  execution as three engineering pillars; FPGA fleets repricing thousands of
  listed contracts across 100+ venues; IMC adds an institutional off-screen
  liquidity desk and 150+ ETF lead-market-maker mandates. Prop.
- **SIG (Susquehanna)** — options market maker with a poker/decision-training
  culture; top-six US retail wholesaler by Rule 605 data, >$10B/day ETF
  volume (self-reported). Jane Street's founders came from SIG. Prop.
- **Millennium / Point72** — multi-manager pod platforms. Hundreds of
  semi-independent pods (fundamental L/S, arb, quant, macro) under a hard
  central risk envelope (Millennium's ~5% de-risk / ~7.5% termination lines
  are widely reported, not official). Execution is buy-side: central desk +
  broker algos + TCA; Millennium prints 13M+ tickets/day without ever being
  anyone's quoted counterparty. Point72's systematic arm is Cubist. Asset
  managers.

## 3. Execution mechanisms, as a taxonomy

| Mechanism | Who | What it actually is |
|---|---|---|
| Internalization / PFOF | Citadel Securities, SIG, (JS, HRT in wholesale) | Retail market orders never reach an exchange; matched against the firm's own inventory at NBBO-or-better. Latency becomes internal-bus speed, and the flow is statistically benign. A market-structure privilege, not faster C++. |
| AP channel | Jane Street | ETF primary-market create/redeem as a private liquidity valve; secondary quotes hedged through the basket. |
| Latency complex | Jump, HRT, CS, Optiver, IMC | Owned microwave/millimetre-wave links, hollow-core fibre, FPGA/ASIC feed handlers and order paths, exchange colocation. |
| Prediction over speed | XTX (the outlier) | Skip the queue war; hold minutes, price with ML, stream OTC. Proof that top-tier execution has split into two profitable routes: shortest physical path vs best statistical price. |
| Buy-side execution | Citadel fund, D.E. Shaw, Millennium, Point72 | Central desk, broker algorithms (VWAP/TWAP/IS), TCA loops, pod information barriers. Alpha lives in signal and risk allocation, not tick-to-trade. |

## 4. What transfers to GALAHAD

GALAHAD is a research-driven vehicle in the asset-manager column: targets →
risk gate → fills, with an evidence trail. The correct benchmark set is
D.E. Shaw / Point72 discipline plus XTX's "prediction over speed", with
selected market-maker hygiene where it is free to copy.

### 4.1 Already aligned (keep, cite as doctrine)

| Professional practice | GALAHAD status |
|---|---|
| Target positions, never raw orders, out of the strategy layer | `decision.py` emits signed targets; engines own fills |
| Hard, pre-execution, non-bypassable risk gates; kill switch | `risk.py` RiskGate + unconditional live block; STAMMTISCH fail-closed adapters |
| Research-to-live parity: same decision stream on every engine | paper↔nautilus parity (`run_parity.py`); paper↔testnet shadow (`run_shadow.py`) |
| Anti-overfitting gates | DSR/PBO (`run_statistics.py`), walk-forward, embargoed CV in quantkit |
| Funding as a first-class P&L term | perpetual funding modeled in both engines |
| Reproducible evidence | STAMMTISCH receipts, digests, offline-verifiable bundles |

### 4.2 Worth importing (roadmap candidates)

1. **Transaction-cost analysis as a first-class report.** Every professional
   buy-side desk closes the loop with TCA: arrival price vs fill price,
   spread and impact decomposition. GALAHAD's paper book currently charges
   taker fees; it should additionally model spread + impact per fill and emit
   an implementation-shortfall block in the session summary, so backtest and
   (testnet) venue fills are comparable on the same cost basis. This is the
   single highest-value upgrade for "deviation between fills and backtest
   assumptions is explainable" (roadmap P2).
2. **Graduated de-risking instead of binary gates.** Millennium's tiered
   response (halve at one drawdown line, flatten at a deeper one) is more
   capital-efficient than block/force-flat alone. Add de-risk tiers to
   `risk.py`: drawdown ladder → leverage multiplier schedule, with the
   existing force-flat as the terminal rung. Gate semantics stay fail-closed.
3. **Event-window strategies as a research track.** Millennium's
   index-rebalance pods ($3.7B across two pods in one month, Bloomberg
   2026-07) show structural, calendar-bound events carry real capacity. The
   crypto-native analogues: funding-timestamp windows, quarterly futures
   basis rolls, large unlock/airdrop calendars. Defined windows make these
   strategies easy to gate statistically and to evidence.
4. **Cross-venue basis monitoring.** The retail-scale echo of Jane Street's
   AP arb: spot–perp basis and funding term structure across venues from
   public APIs. Not an execution strategy at our latency — a research signal
   and a regime indicator for the decision layer.
5. **Kill-switch and degradation drills in CI.** Market makers rehearse their
   kill paths. Add a standing test that the flatten path works (simulated
   venue outage, partial fill, node death mid-session) — the testnet backend
   already treats node death as an error rather than a fake "ok"; extend the
   drill to the full session level.
6. **Shadow parity as a standing promotion gate.** No engine or strategy
   change promotes without a fresh parity/shadow report archived as evidence
   (the HRT verification-pipeline habit, at our scale).

### 4.3 Explicitly not applicable (do not chase)

- **Latency competition.** No queue-priority or co-location-sensitive
  strategies; we will never be fastest and do not need to be (XTX lesson,
  amplified).
- **Internalization / PFOF economics.** Unavailable at retail scale.
- **Order-book microstructure alpha** that decays below our realistic
  decision latency. Strategy horizon stays at minutes and above.

## 5. Confidence notes

- Five Rings: no primary strategy/execution disclosure; omitted from detail.
- Citadel Securities "custom silicon" is a company claim via IFR; treat as
  plausible, not verified.
- All market-share and revenue figures are bound to their source's date and
  definition (volume vs order count, retail vs all, OCC vs total options).
- Millennium's 5%/7.5% risk lines are press-reported, not official.

## 6. Sources

Primary / official: janestreet.com (client offering, performance
engineering), citadel.com (GQS), citadelsecurities.com (options),
hudsonrivertrading.com (HRTBeat engineering and FPGA-verification posts),
deshaw.com (what-we-do), xtxmarkets.com (incl. global eFX trading-practices
disclosure), optiver.com (three-pillars engineering blog), imc.com
(what-we-do, hardware engineering), sig.com (quantitative trading),
mlp.com (approach), point72.com (what-we-do, Cubist).

Regulatory: Fidelity Rule 606 reports; SEC Form ADV brochures (Citadel,
Point72).

Press: Financial Times (Jane Street ETF scale; XTX/Gerko profile and Finland
data center; D.E. Shaw discretionary share; Millennium external allocations),
Bloomberg (2025 multi-strategy gains; Citadel Securities–Morgan Stanley
options deal, 2025-07-10; Millennium index-rebalance pods, 2026-07-06;
HFT microwave networks, 2014), Business Insider (HRT revenue mix, 2025-03;
Citadel Securities stack rebuild, 2024-10; Cubist leadership, 2025-09),
IFR (Citadel Securities flow market-maker profile), Wall Street Journal
(XTX/Gerko). Market-share history: Wikipedia (Jane Street, XTX).

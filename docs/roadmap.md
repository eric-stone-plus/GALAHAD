# Roadmap: From Sector Deep-Dives to Automated Execution (2026-08-04)

## 1. Architecture: research → selection → trading

Four decoupled layers with explicit I/O contracts. News lives only at the top;
the execution layer never sees a headline.

```
L0  Attention    Daily digest pipeline (scrape → machine-readable snapshots)
                 ↓ catalyst/theme extraction — which sectors keep igniting
L1  Research     Sector deep-dives: industry pages, broker research archive
                 Fundamentals: akshare/tushare (A-shares), SEC EDGAR/yfinance (US),
                 venue APIs (crypto)
                 ↓ sector universe + fundamentals scoring
L2  Signals      Factor screening + timing: local factor library, statistically
                 rigorous evaluation (purged k-fold with embargo, walk-forward,
                 DSR/PBO, block bootstrap)
                 ↓ target positions / add-rules
                 (e.g., price X% below MA AND fundamentals score ≥ Y → add Z)
L3  Execution    paper: exchange testnet / broker paper → small live capital
                 Risk controls up front: per-order and daily limits, portfolio
                 drawdown breaker, kill switch (manual, one action)
```

The sector deep-dive loop, concretely: a theme recurs in the digest → L1 builds
the sector universe with financials and valuation data → industry structure
research on top → L2 runs factor scoring and backtests over that universe →
a watchlist falls out. A human approval gate sits between the watchlist and L3;
rule-based pass-through is only considered at P3.

The bounded skill surface and the division between first-party research
procedures, deterministic Python, and optional MCP/provider I/O are specified
in the private workspace documentation (not part of this public repository).

## 2. Staged plan

| Phase | Scope | Acceptance criteria |
|---|---|---|
| **P0 Data foundation** | Venue WS feed + REST backfill + persistence (klines/ticker → parquet); equity dailies reuse existing pipelines behind the OHLCV consensus gate below | 7 consecutive unattended days; self-healing reconnects; message-latency p99 within budget; no research artifact consumes quarantined bars |
| **P0b Strategy doctrine** | Academic spine for strategy families + gates (see `strategy_foundations.md`) | Dual-MA/TSMOM as null; funding + liquidation first-class; DSR/PBO policy written |
| **P1 Signals & backtests** | One deliberately naive strategy to start — dual-MA / trend-filtered rules; full statistical evaluation | DSR > 0; PBO within bounds (policy: flag if >0.5); no out-of-sample collapse |
| **P2 Paper trading** | Run P1 on perp/spot paper; full telemetry (fills, slippage, disconnects, funding) | ≥4 weeks; deviation between fills and backtest assumptions is explainable |
| **P3 Small live capital** | Real money at a lose-it-all-tolerant size; kill switch validated first | Manual acceptance checklist, line by line; limits hard-coded |

**P1 acceptance status (2026-08-18):** DSR/PBO evaluation landed in
`galahad-futures/scripts/run_statistics.py` (DSR = deflated Sharpe,
Bailey & LdP 2014; PBO = CSCV, Bailey et al. 2017 — mirrors of
`quantkit.validation`, scipy-free so it runs in the futures venv). The
report emits `dsr_pass = DSR > 0` and `pbo_flag = PBO > 0.5`; these are
advisory flags, not promotion gates — they are evidence for human review.

US equities run as a parallel track after P2: Alpaca paper → IBKR. A-share
execution stays semi-automated (signal → human) unless a sanctioned channel
becomes available.

**Factor-library iteration (2026-08-22):** the L2 signal layer gains an exact
pandas port of QLib's Alpha158 factor set (`quantkit.alpha158`; factor
definitions attributed to microsoft/qlib, MIT-licensed) together with a
per-factor evaluation table (`quantkit.factor_eval.factor_ic_table`: blockwise
IC/RankIC mean, std, ICIR, observation count against forward returns). This
supersedes the approximate 62-factor `ml_pipeline.Alpha158` for research
screening; the port is lookahead-free by construction (every factor at row t
uses only rows ≤ t) and covered by hand-computed and mutation-based
no-lookahead tests.

## 3. Risk doctrine (precedes any strategy code)

1. Every execution path defaults to OFF; enabling one is an explicit human act
2. Strategies emit **target positions**, never orders; the execution layer is
   the only component allowed to place orders, and it is hard-capped
3. Every strategy ships with a pre-written invalidation condition — the
   circumstances under which it must stop — and stops when it trips
4. Live position limits are hard-coded in execution-layer configuration;
   changing them is a code change with an audit trail
5. News/digest layers never trigger trades (no L0→L3 edge)
6. Daily-loss breach force-flattens the book — target 0 is the only
   allowed action — and new risk stays blocked until equity recovers
   past the floor plus a hysteresis band; a losing position is never
   frozen

## 4. Execution tracks

The futures/USDT-M paper substrate ships in this repo (`galahad-futures/`):
signed positions, leverage, maintenance margin, liquidation, and a risk gate
that is the only path from strategy **targets** to fills. Complementary
execution tracks outside this repo:

- **Spot / broker paper** — venue testnet and broker paper APIs (P2).
- **Live execution** — never default-on: paper ≥4 weeks before any live
  capital, and strategies never emit orders.

**Dual-engine upgrade (2026-08).** The paper substrate gains a second
execution backend: a pinned NautilusTrader (`nautilus_trader==1.231.0`)
event-driven backtest engine runs the *same* decision stream — risk gate
included — against synthetic order books derived from the OHLC bars. The
hand-rolled book stays the default reference engine. Acceptance for this
phase: (a) every engine path remains paper-only and kill-switch-guarded;
(b) `run_parity.py` produces a reconciliation report (equity curves, fills,
funding, liquidation) for the delivery platform; (c) divergences between
engines are explainable from documented execution-mechanics differences
(close-path fills, margin accounting, funding timing), not from decision
drift. The parity report is a research artifact, not a promotion gate: the
reference book remains the arbiter until P2 paper trading validates fill
assumptions against venue behavior. The decision layer is shared by
every execution backend (and future live executors): a tested state
machine (ACTIVE / LOSS_HALTED / INVALIDATED / LIVE_BLOCKED /
LIQUIDATED) with per-decision audit records (seq, phase, boundary
headroom). Parity reports flag ``boundary_crossing`` — sessions where
execution-accounting differences land the engines on opposite sides of
a trip line — and threshold sensitivity scans around the configured
bands; a boundary crossing is a threshold-robustness signal, not an
engine bug.

Agent workspaces and LLM multi-agent research systems (e.g. TradingAgents) may
inform human judgment; they do not replace deterministic execution and margin
accounting.

### Provider-neutral OHLCV consensus gate

Daily equity research must not silently trust one vendor or combine incompatible
series.  The offline consensus gate therefore accepts already captured provider
frames only; it performs no network I/O.  Every input declares its provider,
independence group, price basis, and volume unit.  Raw, split-adjusted, and
total-return-adjusted prices are separate identities, as are share, lot, and
contract volumes.

The gate aligns observed sessions without forward filling, validates finite
OHLC values, price geometry, and non-negative volume, and requires observations
from at least two independent groups.  It accepts daily bars only and binds an
explicit, sorted venue-session calendar; agreeing weekend or holiday carry rows
are still quarantined.  Each source repeats the security identity and binds both
the caller-supplied immutable input-artifact SHA-256 and a computed canonical
frame SHA-256.  A supported two-of-three cluster may
exclude one outlier; a disagreement between the only two independent groups is
quarantined.  Pass, warning, and quarantine bands are explicit configuration,
not provider-specific exceptions.

Accepted bars and diagnostics are separate outputs.  A canonical JSON manifest
binds the research identity, accepted records, per-session and per-source
decisions, original-frame hashes, and accepted-output hash.  Empty or wholly
quarantined inputs never become an accepted research series.  This gate is a
data-integrity control only: it neither establishes causal attribution nor
authorizes a position or order.

The public manifest schema is ``quantkit.ohlcv-consensus.v1``.  It contains the
shared identity, actual-session calendar and its digest, structured tolerance
and voting policy, accepted bars plus ``bars_sha256``, sorted source records,
per-session diagnostics, and ``output_sha256`` over the manifest without that
final field.  Hashes use canonical UTF-8 JSON and explicit domain prefixes:
``quantkit.ohlcv-source-frame.v1``, ``quantkit.ohlcv-accepted-bars.v1``,
``quantkit.ohlcv-session-calendar.v1``, and
``quantkit.ohlcv-consensus-manifest.v1``.

### Semiconductor lead-lag calculation kernel

The first downstream calculation is an offline, provider-neutral local-
projection kernel for semiconductor-sector research.  It accepts only return
series already admitted by the OHLCV consensus boundary.  It has no provider,
network, cache, recommendation, ranking, or order path.  Each series carries
four distinct upstream bindings: the consensus semantic ``output_sha256``, the
SHA-256 of the exact manifest bytes, ``bars_sha256``, and the pinned calendar-
session digest.  The kernel recomputes the calendar and return-vector digests
and binds every series, calendar, session link, parameter, and result into
canonical domain-separated hashes.

The calculation requires a verified session map rebuilt from explicit UTC
regular-session schedules.  It maps a completed source close to the next
actual target open and binds both schedules, every retained link, terminal
unmatched closes, and holiday-collision exclusions.  When several source
closes collide on one target open, only the latest completed close survives.
The kernel never infers a weekday, fills a missing observation, reuses a source
close, or converts a holiday gap into a synthetic bar.  A target supplies
separate opening-gap and open-to-close returns; cumulative total return is
their deterministic sum.  Controls are named before calculation and declare
whether they were available at the linked source close or at the target's
previous close.  They enter every design together with the target's own
previously completed total return.  The fixed family is the latest 63, 126,
and 252 target sessions, horizons one through five, and cumulative gap,
intraday, and total outcomes.  The output scope is
``conditional_lead_lag_association``.

Every family member reports the raw and standardized driver coefficient, a
stationary-block-bootstrap 95% interval, a null-residual block-bootstrap
probability, the family-wide Benjamini--Hochberg adjusted value, a purged and
embargoed expanding-window out-of-sample comparison with the
controls-plus-own-lag null, the same specification in a preregistered adjacent
window, and a fixed-split stationary-block-bootstrap break diagnostic.
The historical ``statistical_status`` can be accepted only when all numerical
gates pass, the usable sample is at least the configured floor and ten
observations per free parameter, and the standardized association is at least
the preregistered magnitude.  The conventional ``status`` and
``publication_status`` remain ``ABSTAIN`` throughout this calibration phase,
including when a historical member passes those numerical gates.
Nested outputs use the same explicit boundary: ``historical_primary_status``,
``not_expired_at_historical_evaluation``, and
``summary.statistical_counts`` describe only the frozen backtest.  The summary
also repeats ``publication_status: ABSTAIN`` so generic consumers cannot infer
eligibility from a historical count.

Expiry is an actual target-session label, never a wall-clock duration.  It is
the earliest of the next five-session recalibration, twice the marginal
horizon-response half-life, and five target sessions.  Driver persistence is
not a proxy for effect decay.  Future expiry labels come from the target venue
schedule, not from observed bars.  If the half-life or required future schedule
label cannot be computed, the member is quarantined; evaluation strictly after
the bound expiry session yields ``expired``.  A statistical non-pass is
``descriptive_only`` and its publication status stays ``ABSTAIN``.

Six method families have explicit, non-voting roles.  The statistical-
econometric local projection is the only primary estimator.  Factor/risk
inputs are controls only; point-in-time event evidence supplies labels and
invalidation only; fundamental supply-chain and technical-regime artifacts may
corroborate or falsify; microstructure is diagnostic only when validated
intraday spread/order-flow coverage satisfies its declared minimum.  No
auxiliary family may promote a failed primary result, average a coefficient, or
turn ``ABSTAIN`` into publication eligibility.  Every supplied auxiliary
diagnostic binds both its semantic artifact digest and exact-byte digest.

The emitted ``quantkit.semiconductor-lead-lag.v1`` document binds method,
configuration, code-runtime, and environment digests, full consensus-manifest
byte lineage, verified schedule mapping, deterministic diagnostics, and
``output_sha256``.  It proves calculation integrity relative to accepted
upstream manifests; it does not independently prove the upstream snapshots or
provider independence.  It is a calculation payload for later artifact
wrapping; ResearchKit schema extension is deliberately out of scope for this
phase.

Historical calculation artifacts are accepted at this boundary only after a
strict structural and relational replay that is independent of the outer
seal.  The validator rejects unknown fields at every level, non-canonical
identifiers, sessions or digests, non-finite numerics, an incomplete or
duplicated 45-member Cartesian family, and drift in domain-separated method,
configuration, runtime, session-map, invalidation or diagnostic hashes.  It
recomputes every relation available from the emitted payload: driver/target
economic-identity separation, timing-domain-specific observation bounds,
same-design ``total = gap + intraday`` coefficient identities, standardized
coefficient scales, structural/adjacent nullability, and OOS count and
residual-norm bounds; family-wide Benjamini--Hochberg values; marginal-response
half-lives and schedule-bound expiry; primary and contextual gates;
deterministic reason codes; member statuses; summary counts; and the root
historical status.
Source-close inputs are bounded by their verified close-to-open links rather
than by comparing a source-market session label with the target-market
estimation label.  It deliberately does not rerun the estimator or treat a
caller signature as
evidence for floating-point estimates.  ``output_sha256`` is checked only
after those independent invariants, so changing a field and resealing the
document cannot convert an internally inconsistent historical payload into an
accepted ``CalculationArtifact``.  This is an integrity checksum and
relational schema boundary, not a producer-authentication primitive: a party
that changes an estimate and every dependent field before recomputing the
checksum can create a different self-consistent payload.  Authenticating or
independently reproducing floating-point estimates requires separately
retained, hash-bound inputs and the execution environment; neither a caller
assertion nor the checksum substitutes for that replay.

This phase emits a historical calibration only, always with publication status
``ABSTAIN``.  It does not apply the latest completed US shock to a current or
future A-share session, because doing so safely requires a separate frozen-
calibration application artifact, an externally time-stamped preregistration,
a global search-family adjustment across every chain edge and proxy, verified
exchange-calendar and provenance-audit artifacts, and a point-in-time event
review pack.  Later application work must never append a current shock to the
calibration sample or reinterpret per-edge FDR as study-wide control.

## 5. Open questions (next research round)

- P0 persistence format and directory conventions
- 24h disconnect profile of a WS long-connection over the local egress
  gateway — this sets the feasibility ceiling for the crypto real-time link
- Invalidation conditions for the trend-filtered DCA strategy
- Alpaca vs. IBKR: account opening and funding feasibility (user-side action)
- Funding-rate realism and multi-symbol margin portfolio for the futures paper track

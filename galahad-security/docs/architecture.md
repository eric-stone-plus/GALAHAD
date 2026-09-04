# galahad-security — architecture (v0.1)

US-equities **paper-trading** substrate: the sibling of `galahad-futures/`
for the security-daily domain. Same doctrine — strategies emit targets,
a risk gate is the only path from target to order, evidence trails
everywhere — on deliberately different account mechanics.

```
┌─────────────┐   target weights (0..1 of equity, per symbol)
│  Strategy   │ ───────────────┐
│  dual_ma    │                ▼
└─────────────┘        ┌──────────────┐   allowed weight   ┌──────────────────────────────┐
┌─────────────┐  daily │  Decision    │ ─────────────────▶ │  Execution backend           │
│  Data       │ ─────▶ │  driver      │                    │  paper        (reference,    │
│ fixture/CSV │        │  RiskGate    │                    │   offline cash book)         │
│ cache/venue │        └──────────────┘                    │  alpaca_paper (venue paper   │
└─────────────┘                                            │   REST, gated, one-shot)     │
                                                           └──────────────┬───────────────┘
                                                                          ▼
                                                                   result/journal
```

## Account model: cash, and why it differs from futures

`galahad-futures` is a margin book: signed positions, leverage, per-bar
funding, maintenance-margin liquidation. `galahad-security` is a **cash
account** by deliberate design, and none of those concepts exist here:

- **Long-only, no leverage.** Positions are non-negative integer share
  counts; there is no short side, no borrowing, no margin ratio. The
  binding constraint is cash: a buy can never spend more than the cash
  on hand. (A-share 100-share lots are a documented future variant of
  the integer-share rule — lot size 100 instead of 1 — not v1.)
- **No funding, no liquidation.** Cash equities accrue no perpetual
  funding and cannot be force-closed by a margin call; the worst case is
  the position's value going to zero. The risk layer's drawdown
  force-flat is the only forced exit.
- **Daily bars.** Equities decisions here are daily (interval `1d`),
  matching the security-daily pipeline cadence — not the futures 1h
  loop.
- **Corporate actions are out of scope for v1.** Splits and dividends
  change share counts and cost basis; v1 neither models nor adjusts for
  them. Prices are used as fetched (Alpaca data API default adjustment
  for venue runs; raw synthetic fixture otherwise). This is documented
  scope, not an oversight.

Why not reuse the margin book with leverage=1? Because "margin book with
the knobs turned down" still *admits* leverage, shorts, funding, and
liquidation as reachable states. A cash book makes those states
unrepresentable — the same fail-closed philosophy as the futures
package's unconditional mainnet block, applied to the account model.

## Engines

| Engine | What it is | Dependencies | Gate |
|---|---|---|---|
| `paper` (default) | Offline reference cash book over daily bars; deterministic; the arbiter | pandas/numpy/pyyaml only | none needed (always allowed in mode=paper) |
| `alpaca_paper` | Alpaca **paper-trading** REST API: one-shot execution of today's decision | stdlib `urllib` only, lazy | env creds + `enable_alpaca_paper: true` + kill switch off |

There is **no `live` engine value and none can be constructed**: the
only venue wired is Alpaca's paper endpoint (`paper-api.alpaca.markets`),
and the summary's `mode` is always `"paper"` — fake money everywhere;
the venue runs differ only in *where* the fills are recorded. A missing
credential or a closed venue gate is a hard `RuntimeError` before any
HTTP — never a silent fallback to the offline book.

`alpaca_paper` is **not a long-running session** (contrast with the
futures testnet backend's bounded `TradingNode`). `run_venue.py`
executes today's decision once: load latest daily bars → strategy
targets → risk gate → submit market orders → fetch resulting positions →
emit the same summary JSON + journal. Bounded, idempotent-safe to re-run
(market orders for the same target delta are a no-op when the book is
already there), and honest about unfilled orders: outside market hours
they count as submitted, not filled, and `position_mismatch` reports the
truth.

## Strategy interface

Strategies emit **target weights** — fraction of equity per symbol, in
`[0, 1]`, long-only — never orders. Multi-symbol from day one
(`config.symbols: [...]`): equities are a portfolio game. The shipped
null is `dual_ma` on daily closes (fast MA > slow MA → target weight,
else 0), evaluated per symbol independently — the same
deliberately-naive doctrine as the futures nulls: a plumbing benchmark,
not an edge claim.

## Risk gate

Ported from futures `risk.py`, adapted to cash weights:

- **Per-decision caps**: `max_weight` per symbol, `max_order_notional`
  per rebalance. Oversized targets are clipped, not silently passed.
- **Daily-loss halt with hysteresis**: equity below the day-start minus
  `max_daily_loss` force-flattens (target 0 the only allowed action) and
  blocks new risk until recovery past floor + `daily_loss_hysteresis`.
- **Drawdown invalidation**: session peak-to-trough `max_drawdown_pct`
  → terminal force-flat (`ok_invalidated`).
- **Graduated de-risking ladder**: identical config shape and semantics
  to futures — `risk.derisk_ladder` tiers `{drawdown, leverage_multiplier}`
  ascending in drawdown scale targets down; the deepest breached tier
  wins; multiplier 0.0 is the ladder's own flatten rung; existing gates
  still fire at their own thresholds (the ladder is strictly
  earlier/softer); malformed ladders are a hard `ValueError` at gate
  construction. Default `[]` = OFF, pre-ladder behavior exact.
- **Venue gate (kill switch)**: venue decisions pass only when
  `enable_alpaca_paper: true` AND `kill_switch: false`. `mode=paper` is
  never blocked. There is no live mode to block — the venue *is* paper.

No leverage concepts appear anywhere in the gate: weights, notionals,
and drawdown only.

## TCA

Identical cost model and block to futures (same doctrine, same field
names, so backtest and venue fills are comparable on one cost basis):

- **Arrival-price convention**: the decision for day *t* is evaluated at
  day *t*'s close; the order fills at that close ("close-path fills").
  The close is the arrival price. Sizing uses arrival; fill economics
  use the execution price = arrival × (1 ± (spread_bps/2 +
  impact_bps)/10⁴), buys pay up, sells receive less. Commission
  (`fee_bps`, default 0.0 — commission-free paper venue realism) stays
  separate.
- `costs.spread_bps` / `costs.impact_bps` default **0.0 = OFF and
  bit-identical** to the zero-cost book; suggested starting points are
  documented in `config.yaml`. Invalid values are a hard error.
- Summary `tca` block: `arrival_notional`, `filled_notional`,
  `implementation_shortfall_usdt`, `implementation_shortfall_bps`,
  `spread_cost_usdt`, `impact_cost_usdt`, `fee_cost_usdt`, `n_fills`.
  Per-fill arrival/exec/cost-split detail lands in the journal. On venue
  runs the split fields are `null` (only the total shortfall is
  observable from venue fills).

## Summary/journal contract

Mirrors the futures shape so a STAMMTISCH adapter can consume both
components with one parser:

`run_id`, `mode: "paper"` (always), `engine`, `engine_version`,
`strategy`, `strategy_kwargs`, `symbols` (list), `interval: "1d"`,
`bars`, `source_used`, `sample_kind`, `n_fills`, `n_risk_rejects`,
`invalidated`, `invalidation_reason`, `peak_equity`, `max_drawdown`,
`initial_equity`, `final_equity`, `equity_curve_len`, `status` (`"ok"` /
`"no-trade but risk-idle OK"` / `"ok_invalidated"`), `journal_path`,
`tca`, `derisk`. Venue runs add `venue: "ALPACA"` and the three-field
`reconciliation` (`orders_submitted`, `orders_filled`,
`position_mismatch`; venue state unknown ⇒ `true`, fail closed).
Absent on purpose vs futures: `liquidated` and every funding field —
unrepresentable states in a cash book.

CLI: `scripts/run_paper.py --engine {paper,alpaca_paper} --json`
(same conventions as futures), `scripts/run_venue.py` as the venue
one-shot launcher.

## Data tiers

`data.py` resolves per-symbol daily bars, offline-first:

1. **CSV cache** — `data/cache/{symbol}_1d.csv` (written by venue
   fetches; `sample_kind: venue`).
2. **synthetic fixture** — `data/fixtures/{symbol}_1d.csv`, deterministic
   (seeded generator, `scripts/gen_fixture.py`, committed output);
   always available, always labeled `synthetic_fixture`, never presented
   as venue data.

`source: auto` = cache → fixture; `fixture` and `cache` force a tier.
The Alpaca data API is read **only in venue mode** (the venue path
fetches and write-caches); `data.py` itself stays pure-offline and
`source="venue"` raises with a pointer at the venue engine.

## Parity doctrine for equities

The futures component reconciles two *backtest* engines on identical
bars. For equities the parity axis is different by design: the offline
book and the Alpaca paper venue should agree on **orders and positions**
for the same decision inputs, with divergences explainable from
execution mechanics (venue fill prices vs close-path assumptions,
market-hours fill timing, venue-side rounding). The reconciliation block
on venue runs is the v1 instrument of that doctrine; a standing
paper↔venue shadow report (the futures `run_shadow.py` pattern) is the
next slice once venue runs accumulate.

## What v1 deliberately lacks

- Shorting, leverage, margin, funding, liquidation (account model).
- Corporate actions (splits/dividends) and adjusted-price handling.
- Intraday bars, limit/stop orders, partial-fill modeling.
- A-share lot rules (100-share lots) and the A-share execution path.
- A standing shadow/parity report for the venue (reconciliation is the
  v1 instrument).

## A-shares

A-share execution stays **out of this substrate** per the roadmap:
signal → human (semi-automated), unless a sanctioned channel becomes
available. This component targets the US-equities parallel track
(Alpaca paper first); the A-share lane consumes research signals from
the shared stack, not orders from this package.

# Architecture (v0.2)

```
┌─────────────┐   targets (signed leverage)
│  Strategy   │ ───────────────┐
│  tsmom etc. │                ▼
└─────────────┘        ┌──────────────┐   allowed target   ┌──────────────────────────────┐
┌─────────────┐  bars  │  Decision    │ ─────────────────▶ │  Execution backend           │
│  Data       │ ─────▶ │  driver      │                    │  paper book  (reference)     │
│ REST/cache/ │        │  RiskGate    │                    │  nautilus    (event engine)  │
│ fixture     │        └──────────────┘                    └──────────────┬───────────────┘
└─────────────┘        └──────────────┘                    └──────────────┬───────────────┘
                                                                          ▼
                                                                   result/journal
```

- **Strategy** never imports a book or engine; it emits per-bar target
  signed leverage only.
- **RiskGate** is the only path from a strategy target to an execution
  intent. It runs inside the decision driver, identically for every
  backend — both engines therefore see the same allowed/clipped/rejected
  decision stream. Divergence between engines is attributable to
  execution mechanics, which is exactly what the parity tool measures.
- **Decision driver** owns the per-bar loop: pre-trade equity snapshot →
  `gate.update_equity` → `gate.filter_target` → `backend.rebalance_to`
  → `backend.settle_bar` (MTM + funding + liquidation check, semantics
  owned by the backend) → post-bar equity snapshot → halt on liquidation.
- **Execution backends** implement a narrow surface
  (`equity`, `position`, `rebalance_to`, `settle_bar`, `collect`).
  - `paper` — the reference backend: `FuturesPaperBook`, deterministic
    pure accounting, fills at bar close, per-bar funding, margin-capped
    adds, maintenance-margin liquidation. Default engine; zero extra
    dependencies.
  - `nautilus` — NautilusTrader `BacktestEngine` backend: the same
    decisions are submitted as orders against synthetic L1 books derived
    from the OHLC bars (close-path fills, taker fees, per-bar funding via
    injected `FundingRateUpdate`s, Nautilus's own margin and liquidation
    machinery). Pinned `nautilus_trader==1.231.0` (the final stable 1.x
    line; see below).
- **Engine selection**: `run_paper.py --engine paper|nautilus` (default
  `paper`). `nautilus` is an optional dependency; a missing package is a
  clear usage error — never a silent fallback to the paper book.
- **Parity**: `scripts/run_parity.py` runs both backends on the same bars
  and writes a reconciliation report — equity curves by timestamp, fills,
  funding totals, liquidation events, and divergence statistics — as an
  evidence artifact for the delivery platform.
- **Summary/journal shapes are unchanged**, plus two additive fields
  (`engine`, `engine_version`) that existing consumers may ignore.
- **mode=paper** default; live requires kill_switch off + enable_live.
  The nautilus backend inherits the same policy and never enables live
  paths.

## Decision layer (v2)

The decision layer (`decision.py` + `risk.py`) is the single authority
for *what* a position should be; executors translate decisions into
orders. Contract:

- **Pure and side-effect-free** — no I/O, never places orders.
- **Deterministic** — same (config, bar stream, executor-reported
  equity/position) in ⇒ same decision stream out; the audit spine for
  automated trading.
- **Terminal force-flats first** — invalidation and the daily-loss halt
  force target 0 (the only allowed action) and block all new risk.
  Reducing/flattening is never blocked.

Session phases and transitions:

| Phase | Meaning |
|---|---|
| `ACTIVE` | trading allowed |
| `LOSS_HALTED` | daily-loss floor breached; force flat until equity recovers past floor + `daily_loss_hysteresis` |
| `INVALIDATED` | drawdown trip (terminal for the session) |
| `LIVE_BLOCKED` | live mode with kill switch / `enable_live` off |
| `LIQUIDATED` | executor-reported liquidation (terminal) |

Legal transitions: `ACTIVE ↔ LOSS_HALTED`; `ACTIVE → INVALIDATED`;
`LOSS_HALTED → INVALIDATED`; any → `LIQUIDATED`. Anything else raises
(fail closed). Decisions after a reported liquidation raise.

Every decision record carries a monotonic `seq`, `phase_before`/
`phase_after`, and boundary headroom instrumentation — `dd_headroom`
(distance to the invalidation trip line) and `loss_headroom` (distance
above the daily-loss floor). The parity tool uses these to flag
`boundary_crossing` — sessions where the engines land on opposite sides
of a trip line — and reports threshold sensitivity scans around the
configured bands.

## Data layer (P0 parquet slice)

`galahad_futures/data.py` resolves bars through four tiers, in order:

1. **venue REST** — Binance vision spot klines (primary), optional fapi
   USDT-M `rest_url_template`, fapi last resort. On success the pull is
   written to both cache tiers.
2. **CSV cache** — `data/cache/{symbol}_{interval}.csv` plus a
   `{symbol}_{interval}.meta.json` sidecar (`save_venue_cache`).
3. **parquet cache** — `data/cache/{symbol}_{interval}.parquet`, a
   self-describing single-file tier: metadata (symbol, interval, row
   count, venue, ts span, sample_kind) is embedded in the parquet schema
   key-value metadata, so the file round-trips alone. Written alongside
   the CSV on every venue pull and via `save_parquet_cache` for any
   normalized bars frame (rest, cache, or fixture source). The `ts`
   column round-trips as ISO-8601 strings — existing consumers parse
   strings.
4. **synthetic fixture** — `data/fixtures/btcusdt_1h.csv`, always
   available but labeled `synthetic_fixture`; never presented as venue
   data.

`load_bars` `source` values: `auto` (rest → cache → parquet → fixture),
`venue` (rest → cache → parquet; raises rather than ever falling back to
the synthetic fixture), `rest`, `cache`, `parquet`, `fixture`.
`source="parquet"` runs fully offline and fails closed: a missing file
raises `FileNotFoundError` naming the expected path (never a silent
fixture fallback), and a corrupt file raises loudly instead of being
skipped. `sample_kind` is `venue` for every cache tier — CSV or parquet
— and `synthetic_fixture` for the fixture.

parquet I/O requires `pyarrow` — a declared project dependency since
this slice, imported lazily so the rest of the data layer imports
without it.

> **Scope note.** This is the P0 parquet slice: a durable, offline-
> readable bar cache for the futures paper substrate. Venue WS streaming
> and 7-day unattended operation remain the next milestone; nothing here
> claims them.

## NautilusTrader pinning rationale

`nautilus_trader` releases move fast and deliberately break APIs between
versions. The pinned line (`1.231.0`) is the final Cython 1.x stable
release; the successor (`2.0.0rc*`) is a Rust-native rewrite still in
release-candidate state. This package pins the exact stable version and
treats any upgrade as a deliberate dependency review with a parity run
before and after.

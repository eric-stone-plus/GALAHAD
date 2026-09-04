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
    from the OHLC bars (close-path fills, taker fees, Nautilus's own
    margin and liquidation machinery). The v1.231.0 backtest engine has
    no funding-settlement path, so per-bar funding is applied in the
    harness itself — payment = qty × mark × rate after each bar's
    rebalance, exactly the paper book's timing — and the funding-adjusted
    equity feeds the shared risk gate. Pinned
    `nautilus_trader==1.231.0` (the final stable 1.x line; see below and
    `docs/nautilus-v2-migration.md`).
- **Engine selection**: `run_paper.py --engine paper|nautilus|nautilus_live`
  (default `paper`). `nautilus` and `nautilus_live` are optional
  dependencies; a missing package or missing testnet credentials is a
  clear usage error — never a silent fallback to the paper book.
- **Parity**: `scripts/run_parity.py` runs both backends on the same bars
  and writes a reconciliation report — equity curves by timestamp, fills,
  funding totals, liquidation events, and divergence statistics — as an
  evidence artifact for the delivery platform.
- **Summary/journal shapes are unchanged**, plus two additive fields
  (`engine`, `engine_version`) that existing consumers may ignore.
- **mode=paper** default; testnet (`nautilus_live`) requires
  kill_switch off + enable_testnet on and is still bounded to the
  Binance futures testnet. mode=live (mainnet) is blocked
  unconditionally — see the testnet backend section below.

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

## Binance USDT-M futures testnet backend (`nautilus_live`)

The `nautilus_live` engine is the live-path rehearsal: the *same* shared
decision stream, executed as real orders against the **Binance USDT-M
futures testnet** via the NautilusTrader `TradingNode` live stack
(bundled in the pinned `nautilus_trader==1.231.0` wheel — no new
dependency). Mainnet remains impossible by construction.

### Design

```
historical bars (warmup) ──▶ strategy history seed
testnet kline stream (EXTERNAL time bars)
        │  per completed bar
        ▼
Decision driver (SessionRisk, shared verbatim with paper/nautilus)
        │  allowed targets only
        ▼
market-order delta → TradingNode → Binance USDT-M futures TESTNET
        │
        ▼
bounded stop (testnet.max_minutes) → reconcile → summary + journal
```

- **Decision identity.** The live strategy seeds its indicator history
  with the session's warmup bars, then appends each completed testnet
  bar and re-evaluates `strategy.targets(...)`, feeding every target
  through `decision.SessionRisk` exactly as the paper engine does. Only
  gate-passed targets become orders; order deltas are the same
  `target * equity / mark − current_qty` projection the nautilus
  backtest backend uses.
- **Execution mechanics.** Market orders only, netting OMS
  (`OmsType.NETTING`), venue-loaded instrument (real tick/step
  precisions and minima from the testnet exchange info). Venue leverage
  is pinned from `default_leverage` via the exec client's
  `futures_leverages`.
- **Bounded session.** The node runs on a worker thread and is stopped
  from the controlling thread after `testnet.max_minutes` (default 30)
  via a thread-safe loop callback, or earlier on liquidation/halt. A
  session that receives no completed bar inside its bound is a valid
  no-trade session, not an error.

### Credential discipline (fail closed)

Credentials are read **only** from the testnet-specific env vars
`BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET`. The mainnet
key names (`BINANCE_API_KEY` / `BINANCE_API_SECRET`) are never read.
Missing or empty testnet credentials raise a hard `RuntimeError`
before any client is constructed — no fallback, no anonymous mode.

### Gate policy

`risk.enable_testnet: true` **and** `risk.kill_switch: false` are both
required for testnet decisions to pass the gate; the engine also
refuses to connect while the gate is closed. `mode: live` (mainnet) is
`LIVE_BLOCKED` **unconditionally** — no configuration in this package
can enable a mainnet path. The engine forces `mode: "testnet"` into
the effective session config, so the summary's `mode` can never read
`"live"`.

### Reconciliation contract

The summary keeps every existing field and adds:

- `mode: "testnet"` (never `"live"`),
- `engine: "nautilus_live"`, `engine_version: "nautilus_trader-1.231.0"`,
- `venue: "BINANCE"`,
- `reconciliation: {"orders_submitted": int, "orders_filled": int,
  "position_mismatch": bool}` — the decision layer's expected final net
  quantity vs the venue-reported net position. A missing venue state
  counts as a mismatch (fail closed).

Status strings follow the paper engine (`ok`, `no-trade …` variants,
`ok_invalidated`); an executor-detected liquidation sets
`liquidated: true`. The journal file matches the paper journal shape.

### Fail-closed rules (summary)

1. Missing testnet credentials → `RuntimeError` before any network I/O.
2. Missing `nautilus_trader` → `RuntimeError` naming the extra.
3. Gate closed (`kill_switch` / `!enable_testnet`) → refuse to connect.
4. `mode: live` anywhere → blocked at the gate, unconditionally.
5. Venue position unavailable at reconcile → `position_mismatch: true`.
6. Real testnet runs are additionally env-gated
   (`GALAHAD_TESTNET_IT=1`) in `scripts/run_shadow.py` and the
   integration test skeleton.

## NautilusTrader pinning rationale

`nautilus_trader` releases move fast and deliberately break APIs between
versions. The pinned line (`1.231.0`) is the final Cython 1.x stable
release; the successor (`2.0.0rc*`) is a Rust-native rewrite still in
release-candidate state. This package pins the exact stable version and
treats any upgrade as a deliberate dependency review with a parity run
before and after.

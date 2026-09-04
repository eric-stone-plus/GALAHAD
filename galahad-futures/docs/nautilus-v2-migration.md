# NautilusTrader v2 migration preparation

Status: preparation only. The package stays pinned to
`nautilus_trader==1.231.0` until the triggers in
[Migration triggers](#migration-triggers) fire. This document inventories
every NautilusTrader v1 API touchpoint in
`galahad_futures/nautilus_backend.py`, maps each to the v2 (Rust-native)
surface, and records the risk notes that the migration changeset must
resolve. Symbol names are the stable reference; line numbers reflect the
file as reviewed and may drift as the module evolves.

## Upstream state of play

- v2 is a Rust-native rewrite of the core with PyO3 Python bindings; the
  legacy Cython v1 core is superseded on `develop`. Both packages install
  and import as `nautilus_trader`, so environments must never mix them
  ([MIGRATION_V2.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/MIGRATION_V2.md)).
- `1.231.0` is the final v1 release line. The upstream transition notice
  moves the legacy core to a `develop_v1` branch accepting only critical
  security backports for approximately three months after the v2 cutover
  ([releases](https://github.com/nautechsystems/nautilus_trader/releases)).
- v2 is at the release-candidate stage (`2.0.0rc4`, 2026-09-02). Upstream
  does not recommend release candidates in production environments
  ([README](https://github.com/nautechsystems/nautilus_trader),
  [installation](https://nautilustrader.io/docs/latest/getting_started/installation/)).
  Our use is paper-only backtesting, but the pin discipline below holds
  regardless: the nautilus backend feeds parity evidence, and evidence
  grade is production grade for this repository.
- Funding settlement is native in v2: backtest venues settle perpetual
  funding at funding boundaries from `FundingRateUpdate` data, emitting
  `FundingSettlement` / `PositionAdjusted` (funding) events through the
  account and portfolio
  ([accounts and margin](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/backtesting/accounts-and-margin.md)).
  The v1.231.0 backtest engine has no funding-settlement path (verified
  against the v1.231.0 source; see the `nautilus_backend.py` module
  docstring).

## Pin policy

`1.231.0` stays until the triggers fire:

1. It is the final stable release of the Cython 1.x line; the only
   post-cutover changes on `develop_v1` are critical security backports.
2. Any version move is a deliberate dependency review: bump the pin,
   run the dual-engine parity check before and after
   (`scripts/run_parity.py`), and archive the parity report as evidence
   (`docs/evidence/`).
3. Release candidates are excluded by policy, matching upstream's own
   guidance. The migration target is `2.0.0` stable.

## v1 API touchpoint inventory and v2 mapping

All touchpoints live in `galahad_futures/nautilus_backend.py`. The
upstream references for every mapping are
[MIGRATION_V2.md](https://github.com/nautechsystems/nautilus_trader/blob/develop/MIGRATION_V2.md)
and the v2 concept docs; risk notes flag surfaces still moving at RC.

| # | Touchpoint (v1) | v2 mapping / expectation | Risk |
|---|---|---|---|
| 1 | Engine construction: `BacktestEngine(BacktestEngineConfig(trader_id=..., run_analysis=False, logging=LoggingConfig(log_level="ERROR")))`; import `nautilus_trader.backtest.engine.BacktestEngine` | Import moves to `nautilus_trader.backtest.BacktestEngine`. `BacktestEngineConfig` keeps its name under `nautilus_trader.config`; `LoggingConfig` is renamed `LoggerConfig`. The low-level `BacktestEngine` API remains a supported API level. | Low for the class move (documented); medium for config-field detail — re-check every constructor kwarg against the generated type stubs, the declared supported Python contract. |
| 2 | Venue config: `engine.add_venue(venue, OmsType.NETTING, account_type=AccountType.MARGIN, base_currency=..., starting_balances=[Money.from_str(...)], default_leverage=...)` | Same shape in v2 (`venue`, `oms_type`, `account_type`, `starting_balances` appear verbatim in the v2 accounts-and-margin guide); `OmsType.NETTING` / `AccountType.MARGIN` persist. | Behavior change: an *omitted* `default_leverage` now selects 10x for margin accounts (1x for cash). We pass leverage explicitly — keep it explicit; never rely on either default. |
| 3 | Instrument construction: `CryptoPerpetual(...)` with `InstrumentId(Symbol(...), Venue("BINANCE"))`, `Currency.from_str`, `Price/Quantity/Money.from_str`, `margin_init`/`margin_maint`/`maker_fee`/`taker_fee` as `Decimal`, `tick_scheme_name=None`, `ts_event`/`ts_init` | `CryptoPerpetual` persists as an economic instrument type with a documented read-only inspection surface (currencies, fees, margins, limits, `multiplier`, `tick_scheme`). Imports consolidate to the flat `nautilus_trader.model` package root. Inspection rename: `tick_scheme_name` → `tick_scheme`. | Medium: v2 constructors are validated by the PyO3 base; exact kwarg names and accepted value types must be re-verified against the stubs (the migration guide flags the inspection rename, not the constructor spelling). |
| 4 | Bar data feeding: `BarSpecification(1, BarAggregation.HOUR, PriceType.LAST)`, `BarType(instrument_id, spec)`, `Bar(bar_type, Price..., Quantity..., ts_event, ts_init)` from `nautilus_trader.model.data` / `model.enums`; `engine.add_data(list_of_bars)` | `Bar`, `BarType`, `BarSpecification` persist (v2 historical delivery is typed: `on_historical_bars(Sequence[Bar])`). `add_data` now requires supported model objects — duck typing is removed, which matches how we already call it. `Bar.is_revision` is removed (unused here). | Low. Timestamps stay UNIX nanoseconds. |
| 5 | Strategy subclassing: `DecisionDrivenStrategy(Strategy)` from `nautilus_trader.trading.strategy`, constructed with `StrategyConfig(strategy_id=..., oms_type=OmsType.NETTING)` | `from nautilus_trader.trading import Strategy` (package root); `StrategyConfig` stays at `nautilus_trader.config.StrategyConfig`. `Strategy.id` → `Strategy.strategy_id`; runtime properties read via `Strategy.config`. Custom annotated config fields do not carry over (we define none; per-instance attributes set after `super().__init__` remain plain Python). | Low–medium: confirm `oms_type` remains a `StrategyConfig` field and that `subscribe_bars` keeps its spelling (bars were not in the rename list; the tick/book subscription methods were renamed, e.g. `subscribe_quote_ticks` → `subscribe_quotes`, which we do not use). |
| 6 | `order_factory.market(instrument_id, side, Quantity(...))` + `submit_order(...)`; `OrderSide.BUY`/`SELL` | `OrderFactory` remains, including `trader_id`/`strategy_id` readback; market-order construction persists. `OrderSide.BUY`/`SELL` unchanged (only the `NO_*` absence variants change, to `None`). | Low. Confirm the `Quantity(value, precision)` constructor spelling against the stubs. |
| 7 | Reports: `ReportProvider.generate_fills_report(orders)` and `generate_positions_report(pos_list, pos_list)` from `nautilus_trader.analysis.reporter`; `engine.cache.orders(instrument_id=...)`; report columns `ts_event`, `instrument_id`, `order_side`, `last_qty`, `last_px`, `commission`, `quantity`, `avg_px` | High-level reporting moves onto the node: `node.generate_fills_report(config.id)`, `generate_orders_report`, `generate_order_fills_report`, `generate_positions_report`, `generate_account_report` (run-config-ID scoped, requiring `dispose_on_completion=False`). `Portfolio.analyzer` is replaced by `statistics()`, `snapshots()`, and `nautilus_trader.analysis`. | High: the migration guide does not pin down the low-level-engine report path or report column names; expect dataframe schema drift. `Order.avg_px`/`slippage` are `Decimal` in v2, not `float` — our `_money_float` string parsing already tolerates this, but report-cell types must be re-checked. |
| 8 | Margin account events / liquidation detection: `on_event` scanning for `"Liquidation"` in the event class name; post-run `engine.portfolio.account(venue)`, iteration of `account.events`, `account.positions.get(...)`, `account.balance_total(...)`, `account.leverage(...)` | The generic `on_event` hook is **removed** in v2; replace with typed hooks (`on_order_event`, `on_position_event`, `on_time_event`) or post-run inspection. `portfolio.account(...)` returns a detached account value; `Account.type` → `account_type`; collection properties on orders/positions become methods (e.g. `Order.events` → `Order.events()`). v2 also adds deterministic liquidation simulation for margin backtest venues via `BacktestVenueConfig` fields. | High: class-name sniffing of liquidation events is the most fragile touchpoint and must be redesigned against typed v2 events (or the deterministic-liquidation venue config). `account.events` / `account.positions` access shapes need verification against the stubs. |
| 9 | Logging silencing: `_silence_nautilus_logging()` iterates the Python `logging.Logger.manager.loggerDict` and sets `nautilus*` loggers to `CRITICAL` | v2's core emits from Rust; Python `logging` silencing does not reach it. Log control moves to `LoggerConfig` (already `log_level="ERROR"` at engine config). | Medium: verify that the Rust-side logger honors the config level for all engine output so the CLI JSON contract stays clean; the Python-side sweep likely becomes dead code. |
| 10 | Dynamic import guard: `_import_nautilus()` raising a usage error naming the pin and the `nautilus` extra | Unchanged mechanics — v1 and v2 both install as `nautilus_trader` (upstream: never install both into one environment). The error string and the extra's pin update with the version bump. | Low. |
| 11 | Harness funding patch (module docstring; `on_bar` post-rebalance block): payment = qty × mark × rate applied per bar in the strategy callback, funding-adjusted equity fed to the shared risk gate | Native v2 settlement: feed `FundingRateUpdate` data (with `next_funding_ns` boundaries) through `add_data`; the simulated exchange settles open positions at funding boundaries and emits `FundingSettlement`/`PositionAdjusted` events. **The harness patch is deleted at migration.** | High on semantics, not API: native settlement pays positions open at the funding boundary using the stored rate and the engine's mark; our patch pays the post-rebalance quantity at bar close each bar. Timing conventions differ — the parity run must reconcile funding totals and per-bar equity under both conventions before the patch is deleted. |
| 12 | Engine run loop and post-run state: `engine.add_instrument(...)` / `add_data(...)` / `add_strategy(...)` / `engine.run()`; `engine.cache`, `engine.portfolio` | The low-level `BacktestEngine` API level persists in v2 with cache and portfolio exposed as read-only properties. | Low–medium: confirm `add_strategy`/`run` spellings and whether engine state survives `run()` by default (the node-level default disposes; low-level behavior to be verified). |

## Funding migration plan (the one semantic change)

1. Emit one `FundingRateUpdate` per funding boundary from the session
   config's `funding_rate_per_bar` series and feed it with the bars.
2. Let the engine settle funding natively; read funding from position
   adjustments / account states rather than a harness accumulator.
3. Delete the harness funding block, the `funding_events` accumulator in
   the strategy, and the funding paragraphs of the module docstring; the
   result schema keeps `total_funding` / `funding_events` (now sourced
   from engine events) so downstream consumers are unaffected.
4. Re-run parity (below). The reference paper book remains the arbiter;
   any funding-timing divergence must appear in `known_divergences` with
   numbers, not prose.

## Migration triggers

All three must hold before the pin moves:

1. **Stable release**: `2.0.0` final on PyPI. Release candidates are
   excluded per upstream guidance and repository policy.
2. **Mandatory dual-engine parity**: `scripts/run_parity.py` before the
   migration (v1 pin, archived baseline) and after (v2), on the same
   bars and configuration; decision streams identical, divergences only
   in documented execution mechanics.
3. **Evidence archived**: both parity reports committed under
   `docs/evidence/` with the migration changeset.

## References

- Migration guide (normative for this document):
  <https://github.com/nautechsystems/nautilus_trader/blob/develop/MIGRATION_V2.md>
- Release notes and the v1.231.0 transition notice:
  <https://github.com/nautechsystems/nautilus_trader/releases>
- v2 backtest accounts, margin, and funding settlement:
  <https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/concepts/backtesting/accounts-and-margin.md>
- Backtesting concepts (API levels):
  <https://nautilustrader.io/docs/latest/concepts/backtesting/>
- Installation / release-candidate guidance:
  <https://nautilustrader.io/docs/latest/getting_started/installation/>
- Upstream repository:
  <https://github.com/nautechsystems/nautilus_trader>

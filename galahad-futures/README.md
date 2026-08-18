# futures — GALAHAD Futures (paper substrate)

Crypto **USDT-M perpetual/futures paper trading** substrate for the GALAHAD stack.
Public research notebook: [GALAHAD](https://github.com/eric-stone-plus/GALAHAD) (English docs only).

Strategies emit **target signed leverage** only. A separate **risk layer** is the sole place that turns targets into paper fills. Live paths default OFF (kill switch).

## Engines

One decision layer, two execution backends (`run_paper.py --engine`):

- `paper` (default) — the reference accounting book (close-price fills,
  per-bar funding, margin caps, liquidation). The arbiter.
- `nautilus` — NautilusTrader 1.231.0 event-driven backtest engine
  (pinned optional extra; funding applied per the reference convention
  in the harness — the v1.231 backtest has no funding settlement path).

`scripts/run_parity.py` reconciles both engines on the same bars
(schema `galahad.parity.v1`): decision diffs, equity/fill/funding diffs,
boundary crossings, and threshold sensitivity.

## Layout

```
futures/
  config.yaml              # paper defaults, risk caps, strategy params
  .env.example             # no secrets in-repo
  galahad_futures/         # book, risk, strategy, data, engine, cli
  galahad_futures/decision.py       # shared decision layer (state machine)
  galahad_futures/nautilus_backend.py  # NautilusTrader backend
  galahad_futures/report.py          # shared summary/journal assembly
  data/fixtures/           # offline OHLCV (paper always runnable)
  output/                  # journals + equity curves (gitignored content)
  tests/                   # book/risk/decision/engine + parity tests
  docs/strategy_research.md # strategy research notes
  docs/architecture.md      # component architecture
  scripts/run_paper.py     # CLI launcher (--engine paper|nautilus)
  scripts/run_parity.py    # dual-engine reconciliation report
```

## Quick start (paper + perception)

```bash
# from this directory — use shared quant venv
quant-python scripts/gen_fixture.py          # if fixture missing
quant-python scripts/market_perception.py --offline   # prices snapshot (or REST)
quant-python scripts/fetch_venue_bars.py
quant-python scripts/run_paper.py --source cache --strategy tsmom
quant-python scripts/run_paper.py --source cache --strategy tsmom_long   # 168×1h
quant-python scripts/run_paper.py --source cache --symbol ETHUSDT
quant-python scripts/compare_strategies.py --source cache
quant-python scripts/validate_walkforward.py --source cache --strategy tsmom
quant-python scripts/run_cycle.py --source fixture
quant-python -m pytest tests/ -q
```

Default: **TSMOM** short; **tsmom_long** = fixed 7d lookback; session **max_drawdown_pct** force-flat.  
Self-contained cycle ops: `scripts/auto_cycle.py` · `scripts/halt.py` · `scripts/walkforward_runner.py` — state in `state/`  
Design: `docs/perception_and_ops.md` · strategy research: `docs/strategy_research.md`

Or:

```bash
cd /path/to/GALAHAD/galahad-futures
quant-python -m galahad_futures.cli --source fixture --json
```

Primary artifacts: `output/paper_journal_*.json`, `output/paper_last_summary.json`, `output/equity_curve_*.csv`.

## Design rules

1. **Paper is default** — no API keys required.
2. **Targets ≠ orders** — `DualMAStrategy.targets` → `RiskGate.filter_target` → `FuturesPaperBook.apply_target`.
3. **Futures accounting** — long/short, leverage, MTM equity, maintenance margin, forced liquidation.
4. **Hard caps** — max order/position notional, max daily loss, max leverage; oversized targets clipped or rejected *before* fill.
5. **Data** — `source=fixture` offline; `source=parquet` offline from the parquet cache; `auto` tries venue REST, then CSV cache, then parquet cache, then fixture (required when network is blocked).
6. **Ops automation** — in-repo cycle scripts only; **must not place orders** (see `docs/evaluation.md`).

## Non-goals (v0.1)

- Live capital and signed-order automation (scaffolded OFF only)
- Tick-level exchange simulator / HFT matching
- LLM or news-driven order generation
- Replacing LLM research agents

## Related

| Path | Role |
|---|---|
| `../quantkit/` | Validation & research library (DSR/PBO, walk-forward, conformal) |
| `../quant-desk/` | Equity lifecycle pipeline (selection → trade → review → gates) |
| `../docs/` | Roadmap & risk doctrine |

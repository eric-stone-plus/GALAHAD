# galahad-security — US-equities paper-trading substrate

Cash-account, long-only **equities paper trading** for the GALAHAD stack —
the sibling of `galahad-futures/` (margin crypto perps) for the
security-daily domain. Public research notebook:
[GALAHAD](https://github.com/eric-stone-plus/GALAHAD) (English docs only).

Strategies emit **target weights** (fraction of equity per symbol,
long-only) only. A separate **risk layer** is the sole place that turns
targets into fills or orders. Venue paths default OFF (kill switch +
explicit enable flag + env credentials).

Design doc: `docs/architecture.md` (account-model rationale vs futures,
engine table, risk/TCA semantics, contract, venue gate, parity doctrine,
v1 scope cuts).

## Engines

One decision layer, two execution backends (`run_paper.py --engine`):

- `paper` (default) — offline reference cash book over daily bars
  (integer shares, close-path fills, optional spread/impact cost model).
  Zero dependencies beyond pandas/numpy/pyyaml. The arbiter.
- `alpaca_paper` — Alpaca **paper-trading** REST API (stdlib `urllib`,
  no new dependency). One-shot: today's decision → market orders →
  positions → same summary contract + reconciliation. Requires
  `ALPACA_PAPER_API_KEY` / `ALPACA_PAPER_API_SECRET` env vars AND
  `risk.enable_alpaca_paper: true` AND `risk.kill_switch: false`;
  missing any = hard error, never a silent fallback. There is no live
  engine value and none can be constructed.

## Layout

```
galahad-security/
  config.yaml              # paper defaults, risk caps, strategy params (no secrets)
  .env.example             # commented Alpaca paper placeholders
  galahad_security/        # book, risk, decision, strategy, data, engine, report, cli
  galahad_security/venue_alpaca.py   # Alpaca paper venue path (lazy HTTP boundary)
  data/fixtures/           # offline synthetic daily OHLCV (paper always runnable)
  data/cache/              # venue-fetched daily bars (write-through cache)
  output/                  # journals + equity curves (gitignored content)
  tests/                   # offline pytest suite
  docs/architecture.md     # component design doc
  scripts/run_paper.py     # CLI launcher (--engine paper|alpaca_paper)
  scripts/run_venue.py     # venue one-shot launcher (engine=alpaca_paper)
  scripts/gen_fixture.py   # deterministic fixture generator
```

## Quick start (offline, no credentials)

```bash
cd galahad-security
uv sync                                # or any python ≥3.12 with pandas/numpy/pyyaml/pytest
python -m pytest tests/ -q
python scripts/run_paper.py --json                     # fixture bars, paper book
python scripts/run_paper.py --source fixture --strategy dual_ma
```

Venue one-shot (Alpaca paper; refuses cleanly without preconditions):

```bash
python scripts/run_venue.py --json
# with preconditions:
ALPACA_PAPER_API_KEY=... ALPACA_PAPER_API_SECRET=... \
  python scripts/run_venue.py --json     # also needs risk.enable_alpaca_paper: true, kill_switch: false
```

Primary artifacts: `output/paper_journal_*.json`, `output/paper_last_summary.json`, `output/equity_curve_*.csv`.

## Design rules

1. **Paper money everywhere** — `mode` in the summary is always `"paper"`;
   the only venue attached is Alpaca's paper endpoint.
2. **Targets ≠ orders** — strategy weights → `RiskGate.filter_weight` →
   `CashEquityBook.apply_target_weight` (or venue order plan).
3. **Cash accounting** — long-only integer shares, MTM equity, cash
   constraint on buys; no leverage, funding, margin, or liquidation.
4. **Hard caps** — max weight per symbol, max order notional, max daily
   loss (hysteresis), drawdown force-flat; oversized targets clipped or
   rejected *before* fill. Optional **graduated de-risking ladder**
   (`risk.derisk_ladder`, default OFF) scales targets down in drawdown
   tiers ahead of the terminal force-flat.
5. **Realistic costs** — optional spread + impact on paper fills
   (`costs.spread_bps`/`impact_bps`, default 0 = bit-identical);
   every summary carries a TCA implementation-shortfall block.
6. **Offline-first data** — `source=auto` resolves CSV cache → synthetic
   fixture; venue data only in venue mode.

## Related

| Path | Role |
|---|---|
| `../galahad-futures/` | USDT-M futures paper substrate (margin book, parity/testnet engines) |
| `../quantkit/` | Validation & research library (DSR/PBO, walk-forward, conformal) |
| `../quant-desk/` | Equity lifecycle pipeline (selection → trade → review → gates) |
| `../docs/` | Roadmap & risk doctrine |

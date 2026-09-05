# Exchange integration

Venue and data-source coverage for the crypto stack, and the
discipline every integration must follow. Adapter availability below
reflects the NautilusTrader 1.231.0 adapter set; verify against vendor
documentation before wiring a new venue.

## Research data (no credentials)

| Source | Use | Notes |
|---|---|---|
| ccxt public REST | OHLCV/quote history for backtests | Public endpoints only, no API key; `galahad-crypto` conventions follow Binance/OKX symbol formats |
| Pinned parquet cache | Offline reproducibility | `galahad-crypto/research/execution-rl/data/cache/` — committed public data (BTC/USDT, ETH/USDT 4h from 2022-01-01) so experiments rerun with zero network access |
| Databento / Tardis (Nautilus adapters) | Institutional-grade historical data | Licensed feeds; not used by the shipped offline examples |

## Execution venues (NautilusTrader adapters)

| Adapter | Modalities | Official testnet/demo |
|---|---|---|
| Binance | spot, margin, USDT-M & COIN-M futures, options | yes — spot & futures testnet (used by `galahad-futures`) |
| Bybit | spot, linear/inverse perps, options | yes |
| OKX | spot, margin, perps, dated futures, options | yes (demo trading) |
| Deribit | options, perps | yes |
| Hyperliquid | on-chain perps | yes |
| dYdX | on-chain perps | yes (staging) |
| BitMEX | perps | yes |
| Kraken | spot, margin | no official public testnet — use the Nautilus `sandbox` adapter for rehearsals |
| Interactive Brokers | equities, options, futures | paper account |
| Polymarket / Betfair | prediction markets | n/a (sandbox for rehearsals) |

Selection rationale for the shipped substrate: **Binance USDT-M
testnet** — the deepest liquidity conventions in crypto derivatives,
a stable public testnet, and first-class NautilusTrader support
(`BinanceLiveDataClient` / `BinanceInstrumentProvider` with
`use_testnet=True`).

## Integration discipline

1. **Testnet/paper only.** Shipped configs point at testnet endpoints
   (`enable_testnet: true`); there is no live-key path in the repo.
2. **Credentials are environment-only.** Keys live in a gitignored
   `.env` (mode 600) or exported env vars consumed by the launcher;
   pipeline configs reference them by `token_env` name, never by
   value. Nothing key-shaped is ever committed or logged.
3. **Reconcile or fail.** Every session compares expected vs. venue
   positions and equity; a mismatch fails the session closed
   (`position_mismatch`) instead of trading on.
4. **Kill switch and force-flat.** The futures substrate carries a
   kill switch and drawdown force-flat; rehearsal configs may disable
   the kill switch only for bounded testnet sessions.
5. **Receipts.** Venue interactions in orchestrated runs go through
   STAMMTISCH adapters, which record content-addressed receipts into
   the run's evidence bundle.

## Adding a new venue

- Prefer an existing Nautilus adapter with an official testnet (rows
  above); otherwise rehearse against the `sandbox` adapter.
- Add the instrument provider + data/client factory wiring in
  `galahad-futures`, behind the same `enable_testnet` discipline.
- Ship offline tests first (fixtures, no network); env-gated
  integration tests stay out of default CI.
- Update this matrix in the same change (`docs/` moves before code).

# galahad-crypto — crypto research workbench (backtest-only)

The crypto market-research layer of GALAHAD: strategy backtesting and
execution-layer RL experiments. **Every tool here is backtest-only and
never places an order.** Live-path execution research runs on
`galahad-futures/` (NautilusTrader, testnet venue); orchestration and
review live in the STAMMTISCH / QUINTE / HIGHBALL stack — see
[`docs/trading-system-architecture.md`](../docs/trading-system-architecture.md)
and [`docs/exchange-integration.md`](../docs/exchange-integration.md).

## Layout

| Path | What it is |
|---|---|
| `tools/crypto_backtest.py` | Strategy backtester (SMA cross, RSI, Bollinger, momentum). Multi-exchange-ready conventions (Binance/OKX via ccxt when wired to live data); ships a deterministic sample-data generator, so the default run is fully offline. Writes `data/backtest_summary.json`. |
| `research/execution-rl/` | Execution-layer RL sandbox: given an already-decided parent order, can RL schedule execution over six 4h bars at lower implementation shortfall than execute-all-now / TWAP? Pre-registered verdict discipline; see its [README](research/execution-rl/README.md). |
| `data/backtest_summary.json` | Recorded results of the shipped backtest round. |

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas pyarrow pytest
.venv/bin/pip install -e ../quantkit        # validation stats for execution-rl

.venv/bin/python tools/crypto_backtest.py   # offline, deterministic sample data

cd research/execution-rl
../../.venv/bin/pytest tests -q             # offline test suite
../../.venv/bin/python scripts/train_eval.py --fast --synthetic
```

`train_eval.py` without `--synthetic` uses the pinned parquet cache
below; `--fast` reduces seeds/episodes for a smoke run.

## Data policy

- `research/execution-rl/data/cache/*.parquet` — pinned public OHLCV
  (ccxt public endpoints; BTC/USDT and ETH/USDT, 4h bars from
  2022-01-01). Committed on purpose so every experiment is reproducible
  offline, with no exchange account or network access.
- `research/execution-rl/output/` — recorded experiment evidence,
  including negative results. Kept verbatim; numbers are never edited.

## Recorded results (honest headline)

- Strategy backtest round (`data/backtest_summary.json`) — generated on
  the tool's deterministic **synthetic** sample data; these are plumbing
  benchmarks, not market-edge claims:
  RSI +48.2% (max DD 8.8%, 232 trades), SMA cross +46.8% (9.2%, 242),
  Bollinger +43.0% (15.3%, 226).
- Execution-RL experiment (`research/execution-rl/output/`) — verdict
  **FAIL** under the pre-registered bootstrap criterion: the RL
  ensemble did not beat the mechanical baselines out-of-sample. Kept as
  negative-result evidence; the full criterion and numbers are in the
  experiment README and `experiment_results.json`.

## Discipline

- No credentials anywhere in this component; research data comes from
  public endpoints only.
- Paper-first: the only order-placing code in GALAHAD lives in
  `galahad-futures/` / `galahad-security/` behind testnet/paper venues
  and explicit enablement flags.

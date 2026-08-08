# GALAHAD

**Quantitative trading research notebook + reusable toolkit** — data-stack evaluation,
staged roadmap, anti-overfitting validation discipline, and paper-execution substrates.

The repository pairs research notes with the code that implements them, so every
claim is reproducible: install the toolkit, run the tests, replay the backtests.

## Contents

| Path | What it is |
|---|---|
| `docs/` | Research notebook: roadmap, data-stack evaluation, academic strategy foundations |
| `quantkit/` | Shared toolkit: factors, portfolio optimizer, conformal sizing, validation gates (purged walk-forward, DSR/PBO, block bootstrap), sentiment factors, backtest scripts |
| `quant-desk/` | Full lifecycle pipeline — selection → optimizer → trade → review → gates (US equities and A-share variants) |
| `galahad-futures/` | USDT-M futures paper substrate: margin book, funding, drawdown force-flat, TSMOM/RSI/Bollinger strategies, walk-forward runner |

## Quickstart

```bash
# toolkit (Python ≥3.12)
pip install -e ./quantkit
python -m pytest quantkit/tests -q

# futures paper engine (offline fixtures, no API keys)
cd galahad-futures
python -m pytest tests -q
python scripts/run_paper.py --source fixture
python scripts/run_cycle.py --source fixture
```

A-share data fetches route through `QUANT_DESK_PROXY` when set
(e.g. `http://127.0.0.1:PORT`); otherwise they go direct.

## Design doctrine

- **Targets ≠ orders** — strategies emit target positions; a risk gate clips or
  rejects them under hard caps *before* any fill.
- **Validation before belief** — purged walk-forward is the final arbiter;
  CPCV/PBO/DSR are diagnostics, block bootstrap for uncertainty.
- **Paper is default** — no live capital path ships enabled.

## License

MIT (see `LICENSE`).

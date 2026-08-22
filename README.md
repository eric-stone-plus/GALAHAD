# GALAHAD

**Quantitative trading research notebook + reusable toolkit** — data-stack evaluation,
staged roadmap, anti-overfitting validation discipline, and paper-execution substrates.

The repository pairs research notes with the code that implements them, so every
claim is reproducible: install the toolkit, run the tests, replay the backtests.

## Positioning

GALAHAD is the analysis identity of the
[STAMMTISCH](https://github.com/eric-stone-plus/STAMMTISCH) quant workstation —
its Jarvis. End users see two surfaces: they open STAMMTISCH to read data and
run workbenches, and they talk to GALAHAD for analysis. The adversarial review
orchestrator (QUINTE) and the delivery rules plane (HIGHBALL) are internal
mechanisms behind that surface: they run inside STAMMTISCH pipelines and are
never presented to users as separate concepts.

## Contents

| Path | What it is |
|---|---|
| `.agents/skills/` | Six installed Wave A public-markets research skills; each owns one bounded procedure and emits auditable artifacts |
| `docs/` | Research notebook: roadmap, data stack, strategy foundations, and the professional-finance skill architecture |
| `researchkit/` | Offline v2 artifact-graph validation plus deterministic SEC normalization, financial-statement, DCF, and trading-comps kernels |
| `quantkit/` | Shared toolkit: factors, portfolio optimizer, conformal sizing, validation gates (purged walk-forward, DSR/PBO, block bootstrap), sentiment factors, backtest scripts |
| `quant-desk/` | Full lifecycle pipeline — selection → optimizer → trade → review → gates (US equities and A-share variants) |
| `galahad-futures/` | USDT-M futures paper substrate: margin book, funding, drawdown force-flat, TSMOM/RSI/Bollinger strategies, walk-forward runner |

## Quickstart

```bash
# finance research substrate (no credentials or network)
cd researchkit
uv run --with 'pytest>=8.3,<10' python -m pytest -q
cd ..

# toolkit (Python ≥3.11; full research env: quantkit/requirements.txt)
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

## Finance skills

The research skills are not part of this public repository.

Design doctrine

- **Targets ≠ orders** — strategies emit target positions; a risk gate clips or
  rejects them under hard caps *before* any fill.
- **Validation before belief** — purged walk-forward is the final arbiter;
  CPCV/PBO/DSR are diagnostics, block bootstrap for uncertainty.
- **Factors are ore, not ammunition** — candidate factor libraries are ported
  from public references (e.g. QLib's Alpha158, MIT) and every candidate is
  screened before it may influence a decision: formula-fidelity tests against
  the reference definitions, no-lookahead property tests, IC/RankIC/ICIR
  ranking on the desk's own universes, then purged walk-forward with
  PSR/DSR/PBO. A candidate that fails any stage is discarded, not debated.
- **Post-training rewards must be unfakeable** — model post-training is
  rewarded only by deterministically verifiable signals: settlement-matched
  calibration of pre-registered forecasts (frozen at issuance, timestamped,
  settled against the realized target), penalties from the validation
  machinery, and the paper ledger as the long-horizon evaluator. Backtest
  return is never a reward — it teaches overfitting — and the adversarial
  review stays outside the reward loop: an auditor inside the objective gets
  optimized, not consulted.
- **Paper is default** — no live capital path ships enabled.
- **Skills own procedure, Python owns arithmetic** — MCP is an optional,
  replaceable boundary for external data, never the source of finance doctrine.

## License

Apache-2.0 (see `LICENSE`).

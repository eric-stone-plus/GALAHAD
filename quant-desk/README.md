# quant_desk

**Full investment-research / paper-trading / review workbench**: selection → trading → performance evaluation → review.

```text
Selection (selection) → Trading (paper book) → Evaluation (performance) → Review (review HTML)
```

## Layout

```text
quant_desk/
  config.yaml
  scripts/
    01_select.py          # Cross-sectional stock selection
    02_trade.py           # Paper rebalance to target weights
    03_evaluate.py        # Return/risk evaluation
    04_review.py          # Review HTML report
    05_gates.py           # Gate go/no-go evaluation (quantkit.gates, fail-closed)
    run_lifecycle.py      # Historical simulation: full chain in one run (gates wired at exit)
    run_today.py          # Today: selection + rebalance + snapshot + gates
  state/                  # Paper ledger JSON (never commit secrets)
  data/                   # Market-data cache
  output/                 # Reports and tables
  tests/                  # pytest: gate wiring (fixtures/gate_metrics_go.json = GO reproduction example)
```

## Quick Start

```bash
cd /path/to/GALAHAD/quant-desk

# A) Historical full chain (recommended to run this first)
quant-python scripts/run_lifecycle.py

# B) Step by step
quant-python scripts/01_select.py
quant-python scripts/02_trade.py
quant-python scripts/03_evaluate.py
quant-python scripts/04_review.py
quant-python scripts/05_gates.py

# C) Today's routine
quant-python scripts/run_today.py

# Tests
quant-python -m pytest tests/ -q
```

## Gates (go/no-go, fail-closed)

Both the lifecycle exit (`run_lifecycle.py`) and the daily routine (`run_today.py` → `05_gates.py`) call
`quantkit.gates.evaluate_gates` and write **`output/gate_report.json`** (verdict + per-gate
missing/failures + input snapshot). Rules:

- This pipeline can only **measure** gate0's three legs itself: `turnover_annual` (single-side traded
  notional from fills / average equity / years — deliberately left unmeasured when the span is <28 days),
  `aum_scale` (= `initial_cash`), and `friction_cost`/`cost_tier` (= config `cost_tier`, see below).
  The remaining gates (pbo/dsr/crowding/paper/live) have no local evidence →
  **unmeasured means NO-GO, with each gate named**; evidence is never fabricated.
- External evidence (walk-forward statistics, weekly crowding reviews, paper-tracking months …) can be
  merged in as JSON at `output/gate_metrics.json` (or via `05_gates.py --metrics PATH`) for re-evaluation;
  **measured keys take priority** over same-named keys in the evidence file.
- GO reproduction example (evidence fixture + measured gate0):

  ```bash
  quant-python scripts/05_gates.py --metrics tests/fixtures/gate_metrics_go.json
  ```

## Cost discipline

Config uses `cost_tier: low|mid|high` (COST_TIERS full-caliber two-sided 0.2%/0.4%/0.5%);
**the 0.05% default is banned**. Paper-ledger single-side fee = tier/2 (`scripts/_common.resolve_fee_bps`).
Note: fees already persisted in `state/paper_book.json` stay with the ledger; the new tier applies after
`02_trade.py --reset`.

Main reports:

- `output/lifecycle_report.html` — Historical lifecycle review
- `output/review_latest.html` — Latest review
- `output/gate_report.json` — Gate go/no-go (fail-closed on missing evidence)

## Capability boundary

| Available | Not yet (extensible) |
|----|----------------|
| Cross-sectional universe scoring & selection | Whole-market real-time order scanning |
| Factor filters + Top-N weighting | Fundamental three-statement deep factors |
| Paper-trade ledger | Real broker order placement (see vnpy_cloud_bridge) |
| Return/drawdown/contribution/review HTML | Compliance audit trail |

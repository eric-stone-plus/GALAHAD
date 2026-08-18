# GALAHAD Futures — strategy research

**2026-08-05.** Complements `GALAHAD/docs/strategy_foundations.md`.

## Canon used (arXiv API + abs-page scrapes)

Papers reviewed:

| Paper | Takeaway for this repo |
|---|---|
| arXiv:1404.3274 Two centuries of trend following | Dual-MA/TSMOM as null family is academically legitimate |
| arXiv:2009.12155 Crypto TF decade | Crypto-first paper coherent |
| arXiv:2212.06888 Perp fundamentals | Funding hook required for claims of edge |
| arXiv:2102.04591 BTC liq/margin | Keep leverage low; test liquidation path |
| arXiv:2506.08573 Funding rate design | Later funding strategies |
| arXiv:1408.1159 OTR without backtest | Prefer pre-specified rules; when grid-searching, inflate DSR trials |

Validation doctrine (DSR/PBO/purged CV) is implemented in `quantkit.validation`
(Huatai (华泰) AI series + AFML doctrine).

## Strategy status

| Layer | Status |
|---|---|
| Default strategy | **TSMOM** lookback=48; **tsmom_long** = lookback 168 (7d); dual_ma plumbing |
| Invalidation | `risk.max_drawdown_pct` session peak-to-trough → force flat |
| Multi-symbol | `--symbol ETHUSDT` + `data/cache/ETHUSDT_1h.csv` |
| Comparison | `scripts/compare_strategies.py` → `output/strategy_compare.json` |
| Plumbing paper | Green — targets → risk → book + funding |
| Venue history | Green — `data/cache/BTCUSDT_1h.csv`; `--source auto|cache` |
| Walk-forward | `scripts/validate_walkforward.py` — bar expanding folds + DSR/PBO flags |
| Edge claim | **Not claimed** unless gates pass on venue OOS (expect dual-MA PBO red) |

## Commands

```bash
cd /path/to/GALAHAD/galahad-futures
quant-python scripts/fetch_venue_bars.py
quant-python scripts/run_paper.py --source cache --strategy tsmom
quant-python scripts/validate_walkforward.py --source cache --strategy tsmom
quant-python scripts/validate_paper.py --source cache   # dual_ma grid comparison
quant-python scripts/run_cycle.py --source fixture
```

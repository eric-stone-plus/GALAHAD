# quantkit

Shared quant research toolkit for A-share / US / HK / crypto.

## Modules

| Module | Description |
|--------|-------------|
| `data` | Multi-market OHLCV with local cache |
| `indicators` | SMA/EMA/RSI/MACD/BB/ATR |
| `factors` | Factor engineering, style factors (6 Fama-style), GP mining, LightGBM ranking |
| `selection` | Cross-sectional stock selection (universe → top-N) |
| `backtest` | Vectorized single-asset backtest |
| `portfolio` | Multi-asset weights + rebalance + **conformal weight policy** |
| `conformal` | Split-conformal + ACI + DtACI prediction intervals |
| `optimizer` | Index-enhanced weights + lambda sweep + alpha attribution |
| `gates` | Six-gate fail-closed strategy evaluation |
| `book` | Paper trading book |
| `review` | Performance evaluation + HTML reports |
| `report` | Tearsheets (quantstats) |
| `cn_market` | A-share limit/suspend/tradeable masks |
| `sentiment` | News keyword sentiment + OOS accumulation |
| `sweep` | Vectorized parameter sweeps (vectorbt; optional dependency) |
| `paths` | Project-root helpers |

### Sweep exposure convention

`quantkit.sweep` runs vectorbt portfolios at **1x exposure**: cash-constrained,
no margin, no leverage. All sweep statistics (total return, CAGR, Sharpe,
drawdown, ...) are per 1x unit of notional. To compare against levered
strategies, scale the return series linearly by the strategy's leverage
(e.g. multiply per-bar returns by 2 for a 2x position) before computing
stats. Funding and margin effects are **not** in the sweep — they are
settled inside the futures engines (`galahad-futures` paper book / nautilus
backend), which trade at configured leverage (e.g. 1.5x/3x) and net those
costs per bar.

## Quick start

```bash
python -m pip install -e .          # core runtime deps
python -c "import quantkit; print(quantkit.__version__)"
python -m pytest tests/ -q

# full research environment (data providers, vectorbt sweeps, ...)
python -m pip install -r requirements.txt
python -m pip install -e '.[sweep,ml]'
```

## Key features

### Style factors
```python
from quantkit.factors import style_factors
factors = style_factors(ohlcv)  # value, momentum, quality, size, low_vol, liquidity
```

### Conformal weight policy
```python
from quantkit.portfolio import conformal_weight_policy
scaled = conformal_weight_policy(target_weights, returns, alpha=0.10)
```

### GP factor mining
```python
from quantkit.factors import gp_mine_factors
gp_factors = gp_mine_factors(features, labels, n_factors=5)
```

### Sentiment
```python
from quantkit.sentiment import keyword_sentiment, build_sentiment_factor
score = keyword_sentiment("bullish rise exceeds expectations")
```

## Testing

```bash
python -m pytest tests/ --durations=10 -q
# 189 tests, ~8 min (2 skipped without lightgbm)
```

## Version

0.4.0 — style factors, conformal weight policy, GP mining, sentiment, 10x test speedup

## Backtest Results (Real Data)

### A-share (15 blue chips, 2024-07 to 2026-08)
| Strategy | CAGR | Sharpe | Max DD |
|----------|------|--------|--------|
| Factor-tilt (value+quality+low_vol) | -4.79% | -0.30 | -17.30% |
| Pure momentum optimizer | -21.05% | -0.37 | -43.96% |
| Equal-weight benchmark | -5.06% | - | - |

**Finding**: In range-bound A-share markets, factor tilt beats pure momentum — momentum signals chase rallies and sell into dips when the market is choppy.

### US Stocks (20 large caps, 2020-2026)
| Strategy | CAGR | Sharpe | Max DD |
|----------|------|--------|--------|
| Factor-tilt | 30.30% | 1.28 | -29.00% |
| Pure momentum optimizer | 27.10% | 0.70 | -66.41% |

**Finding**: In US growth stocks, momentum is effective but requires risk control (conformal weight policy).

### Rebalance Frequency (A-share)
| Frequency | CAGR | Sharpe | Max DD |
|-----------|------|--------|--------|
| 5d | -5.03% | -0.32 | -17.48% |
| 10d | -5.18% | -0.33 | -17.58% |
| 21d | -4.79% | -0.30 | -17.30% |
| 63d | -4.90% | -0.31 | -17.00% |

**Finding**: Rebalancing frequency has little impact on the results.

"""quantkit — shared quantitative-research toolkit.

Modules:
  data        multi-market OHLCV (US / CN / HK / crypto) with local cache
  indicators  technical indicators
  factors     factor engineering + style factors + GP mining + simple ML ranking
  selection   cross-sectional stock selection (universe → top-N)
  backtest    vectorized single-asset helpers
  portfolio   multi-asset weights + rebalance engine + conformal weight policy
  book        paper trading book (cash / fills / MTM)
  review      performance evaluation + HTML review reports
  cn_market   A-share limit / suspend / tradeable masks
  report      performance tearsheets (quantstats)
  paths       project-root helpers
  conformal   split-conformal + ACI + DtACI prediction intervals
  gates       six-gate fail-closed strategy evaluation
  optimizer   index-enhanced weights + lambda sweep + alpha attribution
  sentiment   news keyword sentiment + OOS accumulation
"""

from __future__ import annotations

__version__ = "0.4.0"

from quantkit.paths import project_root, ensure_dirs

__all__ = ["__version__", "project_root", "ensure_dirs"]

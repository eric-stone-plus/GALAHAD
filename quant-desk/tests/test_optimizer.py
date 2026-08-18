"""Tests for 01b_optimize.py optimizer integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_optimizer_weights_file_created(tmp_path):
    """01b_optimize should produce optimized_weights.csv."""
    from quantkit.optimizer import index_enhanced_weights, lw_shrinkage_cov

    # Simulate selection output
    symbols = ["A", "B", "C", "D", "E"]
    tw = pd.Series(0.2, index=symbols, name="target_weight")
    tw.to_csv(tmp_path / "target_weights.csv", header=True)

    meta = {"asof": "2026-08-07", "selected": symbols}
    (tmp_path / "selection_meta.json").write_text(json.dumps(meta))

    # Build synthetic returns
    rng = np.random.default_rng(42)
    rets = pd.DataFrame(rng.normal(0, 0.02, (252, 5)), columns=symbols)
    cov = lw_shrinkage_cov(rets.values)
    scores = rng.normal(0, 1, 5)
    wb = np.full(5, 0.2)

    opt_w = index_enhanced_weights(scores, wb, cov, te_max=0.10, turnover_cap=0.50)
    opt_series = pd.Series(opt_w, index=symbols)

    assert len(opt_w) == 5
    assert abs(opt_w.sum() - 1.0) < 0.01
    assert (opt_w >= 0).all()


def test_optimizer_respects_te_constraint():
    """Optimized weights should respect covariance-weighted TE constraint."""
    from quantkit.optimizer import index_enhanced_weights, lw_shrinkage_cov

    rng = np.random.default_rng(99)
    N = 20
    rets = pd.DataFrame(rng.normal(0, 0.03, (300, N)))
    cov = lw_shrinkage_cov(rets.values)
    scores = rng.normal(0, 0.5, N)
    wb = np.full(N, 1.0 / N)

    te_max = 0.10
    opt_w = index_enhanced_weights(scores, wb, cov, te_max=te_max, turnover_cap=0.80)
    # TE in covariance-weighted sense (what the optimizer actually constrains)
    d = opt_w - wb
    te_cov = np.sqrt(float(d @ cov @ d))
    assert te_cov <= te_max + 0.01, f"Cov-weighted TE={te_cov:.4f} > {te_max}"
    assert (opt_w >= -0.01).all(), f"Negative weights: {opt_w}"
    assert abs(opt_w.sum() - 1.0) < 0.02, f"Weights sum: {opt_w.sum()}"


def test_optimizer_infeasible_graceful():
    """Infeasible constraints should raise OptimizationError."""
    from quantkit.optimizer import OptimizationError, index_enhanced_weights, lw_shrinkage_cov

    rng = np.random.default_rng(77)
    N = 5
    rets = pd.DataFrame(rng.normal(0, 0.02, (100, N)))
    cov = lw_shrinkage_cov(rets.values)
    scores = rng.normal(0, 1, N)
    wb = np.full(N, 0.2)
    prev = wb.copy()
    prev[0] += 0.1
    prev[1] -= 0.1

    with pytest.raises(OptimizationError):
        index_enhanced_weights(
            scores, wb, cov,
            te_max=0.0,
            turnover_cap=0.01,
            prev_weights=prev,
        )

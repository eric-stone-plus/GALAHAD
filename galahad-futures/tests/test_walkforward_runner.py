"""Walk-forward aggregate metrics — folds chain on returns, never equity levels.

Regression: each fold restarts at ``initial_equity`` after its own warmup, so
splicing per-fold equity curves by level injected a fake jump at every fold
boundary (into Sharpe / MaxDD / win rate) and the aggregate total return kept
only the last fold's P&L.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from walkforward_runner import chain_oos_equity_curves, compute_metrics, synthetic_fixture_seed


def test_synthetic_fixture_seed_is_process_stable():
    # same symbol → same seed on every call and across interpreter restarts
    # (crc32, not the per-process randomized builtin hash()); different
    # symbols must not collide onto the same sequence
    assert synthetic_fixture_seed("BTCUSDT") == synthetic_fixture_seed("BTCUSDT")
    assert synthetic_fixture_seed("btcusdt") == synthetic_fixture_seed("BTCUSDT")
    seeds = {synthetic_fixture_seed(s) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")}
    assert len(seeds) == 3


def test_synthetic_fallback_is_deterministic_and_leaves_fixtures_alone(tmp_path, monkeypatch):
    """REST+cache failure → identical synthetic bars every run, written
    under output/, never over the committed data/fixtures files."""
    import walkforward_runner as wr

    real_load_bars = wr.load_bars

    def _fail_auto(**kwargs):
        if kwargs.get("source") == "auto":
            raise RuntimeError("simulated cache+REST failure")
        return real_load_bars(**kwargs)

    monkeypatch.setattr(wr, "load_bars", _fail_auto)
    monkeypatch.setattr(wr, "OUTPUT_DIR", tmp_path)

    fixtures_before = {
        p.name: p.read_bytes()
        for p in (wr.PROJECT_ROOT / "data" / "fixtures").glob("*.csv")
    }

    bars_a, source_a = wr.load_symbol_data("BTCUSDT")
    first = (tmp_path / "synthetic_btcusdt_1h.csv").read_bytes()
    bars_b, source_b = wr.load_symbol_data("BTCUSDT")

    assert source_a == source_b == "fixture_synthetic"
    assert len(bars_a) == 500 and len(bars_b) == 500
    # byte-identical re-generation: the fallback is reproducible
    assert first == (tmp_path / "synthetic_btcusdt_1h.csv").read_bytes()
    pd.testing.assert_frame_equal(bars_a, bars_b)
    # committed fixtures untouched
    fixtures_after = {
        p.name: p.read_bytes()
        for p in (wr.PROJECT_ROOT / "data" / "fixtures").glob("*.csv")
    }
    assert fixtures_after == fixtures_before


def test_chain_compounds_across_folds_without_boundary_jump():
    # fold 1: 10000 → 12000 (+20%); fold 2 restarts at 9900 and is flat
    fold1 = [10000.0, 11000.0, 12000.0]
    fold2 = [9900.0, 9900.0]
    chained = chain_oos_equity_curves([fold1, fold2], 10000.0)
    # the fold-2 reset must not leak in as a -17.5% jump bar
    assert chained == pytest.approx([10000.0, 11000.0, 12000.0, 12000.0])


def test_aggregate_total_return_is_chained_compound_return():
    # fold 1 +50%, fold 2 +10% → chained +65%, not the last fold's +10%;
    # long flat prefixes keep compute_metrics' annualization in range
    fold1 = [10000.0] * 50 + [15000.0]
    fold2 = [10000.0] * 50 + [11000.0]
    agg = compute_metrics(chain_oos_equity_curves([fold1, fold2], 10000.0), 10000.0)
    assert agg["total_return_pct"] == pytest.approx(65.0)
    # the old level-splice aggregation for comparison (documents the bug)
    spliced = compute_metrics(fold1 + fold2, 10000.0)
    assert spliced["total_return_pct"] == pytest.approx(10.0)


def test_aggregate_maxdd_ignores_fold_boundary_reset():
    # fold 1 rises to 12000, fold 2 is flat from its own reset at 9900:
    # level splicing manufactured a -17.5% drawdown bar that never happened
    fold1 = [10000.0, 11000.0, 12000.0]
    fold2 = [9900.0, 9900.0]
    chained = chain_oos_equity_curves([fold1, fold2], 10000.0)
    agg = compute_metrics(chained, 10000.0)
    assert agg["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-9)
    spliced = compute_metrics(fold1 + fold2, 10000.0)
    assert spliced["max_drawdown_pct"] == pytest.approx(17.5, abs=0.1)


def test_chain_preserves_per_fold_percentage_paths():
    # two folds with identical +10% paths chain to +21%, level 12100
    fold = [5000.0, 5500.0]
    chained = chain_oos_equity_curves([fold, fold], 10000.0)
    assert chained == pytest.approx([10000.0, 11000.0, 12100.0])


def test_chain_handles_dict_entries_and_empty():
    fold = [{"equity": 10000.0}, {"equity": 10100.0}, {"eq": 10201.0}]
    chained = chain_oos_equity_curves([fold], 10000.0)
    assert chained == pytest.approx([10000.0, 10100.0, 10201.0])
    # no successful folds → degenerate single-point curve
    assert chain_oos_equity_curves([], 10000.0) == [10000.0]

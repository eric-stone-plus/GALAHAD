"""DSR/PBO statistics tests — synthetic paths and fixture integration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.statistics import (  # noqa: E402
    _moments,
    _norm_cdf,
    _norm_ppf,
    deflated_sharpe_ratio,
    evaluate_oos_statistics,
    prob_backtest_overfitting,
    probabilistic_sharpe_ratio,
)


# ---------------------------------------------------------------------------
# scipy-free normal CDF / PPF mirror sanity
# ---------------------------------------------------------------------------


def test_norm_cdf_known_values():
    assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert _norm_cdf(1.96) == pytest.approx(0.9750021048517796, abs=1e-9)
    assert _norm_cdf(-1.96) == pytest.approx(0.0249978951482204, abs=1e-9)


def test_norm_ppf_known_values():
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-6)
    assert _norm_ppf(1.0 - 1.0 / 3.0) == pytest.approx(0.4307272992954575, abs=1e-6)


def test_norm_ppf_invalid_p():
    with pytest.raises(ValueError):
        _norm_ppf(0.0)
    with pytest.raises(ValueError):
        _norm_ppf(1.0)


def test_norm_ppf_roundtrip_cdf():
    for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-9)


# ---------------------------------------------------------------------------
# DSR (mirror of quantkit.validation.deflated_sharpe_ratio)
# ---------------------------------------------------------------------------


def test_dsr_no_selection_reduces_to_psr():
    rng = np.random.default_rng(3)
    r = rng.normal(0.002, 0.01, 500)
    mu, sd, sk, ku = _moments(r)
    psr = probabilistic_sharpe_ratio(mu / sd, 0.0, len(r), skew=sk, kurt=ku)
    dsr = deflated_sharpe_ratio(r, n_trials=1, sr_std=0.0)
    assert dsr == pytest.approx(psr, abs=1e-12)
    assert dsr > 0.5  # genuinely positive-Sharpe series


def test_dsr_deflates_with_more_trials():
    rng = np.random.default_rng(11)
    r = rng.normal(0.002, 0.01, 500)
    base = deflated_sharpe_ratio(r, n_trials=2, sr_std=0.05)
    deflated = deflated_sharpe_ratio(r, n_trials=20, sr_std=0.2)
    assert deflated < base
    assert deflated > 0.0


def test_dsr_identical_trials_no_deflation():
    # sr_std = 0 (all candidates identical) -> PSR vs 0, per quantkit branch.
    # Mirror fidelity note: numpy 2.x std of a constant float array is ~2e-19,
    # not exactly 0, so the sd==0 guard is skipped and the mirror reproduces
    # quantkit's own result for the same input (verified: quantkit returns 1.0
    # on numpy 2.4.6; this module returns the same).
    r = np.full(100, 0.001)
    assert deflated_sharpe_ratio(r, n_trials=3, sr_std=0.0) == pytest.approx(1.0)
    # Exactly-zero-variance input (all zeros) does hit the sd==0 guard.
    assert deflated_sharpe_ratio(np.zeros(100), n_trials=3, sr_std=0.0) == 0.0


def test_dsr_fails_closed():
    with pytest.raises(ValueError, match="3 observations"):
        deflated_sharpe_ratio([0.1, 0.2], n_trials=1, sr_std=0.0)
    with pytest.raises(ValueError, match="sr_std"):
        deflated_sharpe_ratio([0.1, 0.2, 0.3], n_trials=1, sr_std=None)


# ---------------------------------------------------------------------------
# PBO (mirror of quantkit.validation.prob_backtest_overfitting)
# ---------------------------------------------------------------------------


def test_pbo_constant_strategy_low():
    # Identical windows (constant strategy) -> no selection -> PBO == 0
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 0.01, 64)
    m = np.column_stack([base, base, base])
    assert prob_backtest_overfitting(m, n_blocks=16) == pytest.approx(0.0)


def test_pbo_reversed_fortunes_high():
    # Two mirror-image variants: the train-best is the test-worst except on
    # perfectly balanced combos -> PBO far above the 0.5 flag line.
    n, half = 64, 32
    a = np.concatenate([np.full(half, 0.02), np.full(n - half, -0.02)])
    b = np.concatenate([np.full(half, -0.02), np.full(n - half, 0.02)])
    m = np.column_stack([a, b])
    pbo = prob_backtest_overfitting(m, n_blocks=16)
    assert pbo > 0.5


def test_pbo_fails_closed():
    with pytest.raises(ValueError, match="2 strategy columns"):
        prob_backtest_overfitting(np.zeros((16, 1)), n_blocks=4)
    with pytest.raises(ValueError, match="even"):
        prob_backtest_overfitting(np.zeros((16, 2)), n_blocks=5)
    with pytest.raises(ValueError, match="rows per block"):
        prob_backtest_overfitting(np.zeros((3, 2)), n_blocks=16)


def test_pbo_high_low_matches_evaluate_flag():
    # end-to-end: evaluate_oos_statistics flag agrees with raw PBO level
    rng = np.random.default_rng(5)
    base = rng.normal(0.0, 0.01, 64)
    windows = [list(base) for _ in range(3)]
    stats = evaluate_oos_statistics(windows, periods_per_year=1.0)
    assert stats["pbo"] == pytest.approx(0.0)
    assert stats["pbo_flag"] is False

    n, half = 64, 32
    a = np.concatenate([np.full(half, 0.02), np.full(n - half, -0.02)])
    b = np.concatenate([np.full(half, -0.02), np.full(n - half, 0.02)])
    stats = evaluate_oos_statistics([list(a), list(b)], periods_per_year=1.0)
    assert stats["pbo"] > 0.5
    assert stats["pbo_flag"] is True


# ---------------------------------------------------------------------------
# evaluate_oos_statistics — report shape and fail-closed behavior
# ---------------------------------------------------------------------------


def test_evaluate_report_shape():
    rng = np.random.default_rng(9)
    windows = [rng.normal(0.0005, 0.01, 64) for _ in range(3)]
    stats = evaluate_oos_statistics(windows, periods_per_year=8760.0)
    for key in ("dsr", "pbo", "oos_sharpe", "n_windows", "dsr_pass", "pbo_flag"):
        assert key in stats
    assert stats["n_windows"] == 3
    assert isinstance(stats["dsr_pass"], bool)
    assert isinstance(stats["pbo_flag"], bool)
    assert stats["oos_sharpe"] > 0.0  # positive-drift series annualizes positive
    assert stats["n_blocks_used"] == 16


def test_evaluate_unequal_window_lengths_padded():
    # Shorter windows are NaN-padded and zero-filled for CSCV (quantkit convention)
    rng = np.random.default_rng(21)
    stats = evaluate_oos_statistics(
        [rng.normal(0.001, 0.01, 64), rng.normal(0.001, 0.01, 40)],
        periods_per_year=1.0,
    )
    assert stats["n_windows"] == 2
    assert stats["oos_obs"] == 104


def test_evaluate_fails_closed_empty():
    with pytest.raises(ValueError, match="no usable OOS"):
        evaluate_oos_statistics([])
    with pytest.raises(ValueError, match="no usable OOS"):
        evaluate_oos_statistics([[0.5, 0.5], [0.1, 0.2]])  # < 3 obs per window
    with pytest.raises(ValueError, match="CSCV"):
        evaluate_oos_statistics([[0.1, 0.2, 0.3, 0.4]])


# ---------------------------------------------------------------------------
# Fixture integration: walk-forward OOS sweep -> statistics report
# ---------------------------------------------------------------------------

from galahad_futures.engine import load_config  # noqa: E402
from galahad_futures.data import load_bars  # noqa: E402
from scripts.run_statistics import build_statistics_report  # noqa: E402


def _fixture_inputs():
    bars, source_used, _note = load_bars(
        source="fixture",
        fixture_path="data/fixtures/btcusdt_1h.csv",
        rest_url=None,
        rest_timeout=12.0,
        project_root=ROOT,
        symbol="BTCUSDT",
        interval="1h",
        limit=500,
        rest_url_template=None,
    )
    return load_config(), bars.iloc[-120:].reset_index(drop=True), source_used


def test_fixture_statistics_report_shape_and_flags():
    cfg, bars, source_used = _fixture_inputs()
    report = build_statistics_report(
        cfg,
        bars,
        strategy_name="dual_ma",
        n_folds=3,
        min_train=60,
        test_size=20,
        purge=5,
        warmup=40,
        n_blocks=8,
        source_used=source_used,
    )
    assert report["schema"] == "galahad.statistics.v1"
    s = report["statistics"]
    assert set(s) == {
        "dsr", "pbo", "oos_sharpe", "n_windows", "oos_obs", "dsr_pass", "pbo_flag",
    }
    assert s["n_windows"] == 3
    assert isinstance(s["dsr_pass"], bool)
    assert isinstance(s["pbo_flag"], bool)
    assert 0.0 <= s["dsr"] <= 1.0
    assert 0.0 <= s["pbo"] <= 1.0
    assert len(report["windows"]) == 3
    for w in report["windows"]:
        assert w["test_bars"] == 20

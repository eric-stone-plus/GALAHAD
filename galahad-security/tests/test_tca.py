"""TCA tests — zero-cost identity, hand-computed costs, summary block."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_security.book import tca_from_fills
from galahad_security.data import load_bars
from galahad_security.engine import load_config, run_paper_on_bars, run_paper_session


def _bars():
    bars, _, _ = load_bars(
        source="fixture", project_root=ROOT, symbols=["AAPL", "MSFT"], limit=250
    )
    return bars


def test_zero_costs_bit_identical():
    cfg = load_config()
    cfg_zero = dict(cfg)
    cfg_zero["costs"] = {"spread_bps": 0.0, "impact_bps": 0.0}
    bars = _bars()
    a = run_paper_on_bars(bars, cfg, symbols=["AAPL", "MSFT"], strategy_name="dual_ma")
    b = run_paper_on_bars(bars, cfg_zero, symbols=["AAPL", "MSFT"], strategy_name="dual_ma")
    for key in ("fills", "equity_curve", "positions"):
        assert a[key] == b[key], key
    assert a["final_equity"] == b["final_equity"]
    assert all(f["price"] == f["arrival_price"] for f in b["fills"])


def test_costs_reduce_equity_and_show_in_tca(tmp_path):
    cfg = load_config()
    cfg["costs"] = {"spread_bps": 1.0, "impact_bps": 1.0}
    bars = _bars()
    base = run_paper_on_bars(bars, load_config(), symbols=["AAPL", "MSFT"], strategy_name="dual_ma")
    costed = run_paper_on_bars(bars, cfg, symbols=["AAPL", "MSFT"], strategy_name="dual_ma")
    tca = costed["tca"]
    # 0.5 half-spread + 1.0 impact = 1.5 bps of arrival notional
    assert tca["implementation_shortfall_bps"] == pytest.approx(1.5)
    assert tca["implementation_shortfall_usdt"] > 0.0
    assert tca["spread_cost_usdt"] == pytest.approx(tca["arrival_notional"] * 0.5 / 10_000.0)
    assert tca["impact_cost_usdt"] == pytest.approx(tca["arrival_notional"] * 1.0 / 10_000.0)
    # costs come out of the book: strictly worse final equity on the same stream
    assert costed["final_equity"] < base["final_equity"]
    # buys pay up / sells receive less on every fill
    for f in costed["fills"]:
        if f["side"] == "BUY":
            assert f["price"] > f["arrival_price"]
        else:
            assert f["price"] < f["arrival_price"]


def test_tca_from_fills_edges():
    assert tca_from_fills([])["n_fills"] == 0
    tca = tca_from_fills([{"ts": "t", "side": "BUY", "qty": 1, "price": 100.0}])
    assert tca["n_fills"] == 0  # no arrival price → excluded
    # venue-style fill: arrival present, no spread/impact split → nulls
    tca = tca_from_fills(
        [{"ts": "t", "side": "SELL", "qty": 2, "price": 99.9, "arrival_price": 100.0}]
    )
    assert tca["implementation_shortfall_usdt"] == pytest.approx(0.2)
    assert tca["spread_cost_usdt"] is None
    assert tca["impact_cost_usdt"] is None


def test_cost_config_validation_errors():
    bars = _bars()
    for bad in (
        {"spread_bps": -1.0},
        {"impact_bps": -0.5},
        {"spread_bps": float("nan")},
        {"impact_bps": float("inf")},
        {"spread_bps": "abc"},
    ):
        cfg = dict(load_config())
        cfg["costs"] = {**{"spread_bps": 0.0, "impact_bps": 0.0}, **bad}
        with pytest.raises(ValueError, match="costs\\."):
            run_paper_on_bars(bars, cfg, symbols=["AAPL"], strategy_name="dual_ma")


def test_summary_carries_tca_block(tmp_path):
    summary = run_paper_session(
        config=load_config(), force_source="fixture", output_dir=tmp_path
    )
    tca = summary["tca"]
    for key in (
        "arrival_notional",
        "filled_notional",
        "implementation_shortfall_usdt",
        "implementation_shortfall_bps",
        "spread_cost_usdt",
        "impact_cost_usdt",
        "fee_cost_usdt",
        "n_fills",
    ):
        assert key in tca, key

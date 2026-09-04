"""TCA / implementation-shortfall tests — cost model, backward compat, summary block."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.book import FuturesPaperBook, tca_from_fills
from galahad_futures.engine import load_config, run_paper_on_bars, run_paper_session


def _book(**over):
    kw = dict(
        wallet=10_000.0,
        fee_bps=0.0,
        maintenance_margin_rate=0.005,
        funding_rate_per_bar=0.0,
        default_leverage=5.0,
        max_leverage=5.0,
    )
    kw.update(over)
    return FuturesPaperBook(**kw)


# --- cost math (hand-computed) ----------------------------------------------


def test_buy_pays_up_hand_computed():
    book = _book(spread_bps=2.0, impact_bps=4.0)
    fill = book.market_order("BTCUSDT", 1.0, 100.0, ts="t0")
    # half-spread 1 bps + impact 4 bps = 5 bps on arrival 100
    assert fill.price == pytest.approx(100.05)
    assert fill.arrival_price == pytest.approx(100.0)
    assert fill.spread_cost == pytest.approx(0.01)
    assert fill.impact_cost == pytest.approx(0.04)
    assert fill.fee == 0.0
    assert book.position("BTCUSDT").entry_price == pytest.approx(100.05)


def test_sell_receives_less_hand_computed():
    book = _book(spread_bps=2.0, impact_bps=4.0)
    book.market_order("BTCUSDT", 1.0, 100.0, ts="t0")
    fill = book.market_order("BTCUSDT", -1.0, 100.0, ts="t1")
    assert fill.price == pytest.approx(99.95)
    # round-trip realized loss = spread + impact both legs = 0.10 USDT
    assert fill.realized_pnl == pytest.approx(-0.10)
    assert book.wallet == pytest.approx(9_999.90)


def test_tca_block_aggregation_hand_computed():
    book = _book(spread_bps=2.0, impact_bps=4.0)
    book.market_order("BTCUSDT", 1.0, 100.0, ts="t0")
    book.market_order("BTCUSDT", -1.0, 100.0, ts="t1")
    tca = tca_from_fills([asdict(f) for f in book.fills])
    assert tca["arrival_notional"] == pytest.approx(200.0)
    assert tca["filled_notional"] == pytest.approx(200.0)
    assert tca["implementation_shortfall_usdt"] == pytest.approx(0.10)
    assert tca["implementation_shortfall_bps"] == pytest.approx(5.0)
    assert tca["spread_cost_usdt"] == pytest.approx(0.02)
    assert tca["impact_cost_usdt"] == pytest.approx(0.08)
    assert tca["fee_cost_usdt"] == 0.0
    assert tca["n_fills"] == 2


def test_fees_stay_separate_from_shortfall():
    book = _book(fee_bps=4.0)  # fees only, no spread/impact
    book.market_order("BTCUSDT", 1.0, 100.0, ts="t0")
    tca = tca_from_fills([asdict(f) for f in book.fills])
    assert tca["implementation_shortfall_usdt"] == 0.0
    assert tca["fee_cost_usdt"] == pytest.approx(0.04)  # 4 bps on 100


def test_tca_from_fills_edges():
    assert tca_from_fills([])["n_fills"] == 0
    # fills without arrival_price are excluded
    tca = tca_from_fills([{"ts": "t", "side": "BUY", "qty": 1.0, "price": 100.0}])
    assert tca["n_fills"] == 0
    # venue-style fill: arrival present, no spread/impact split → nulls
    tca = tca_from_fills(
        [{"ts": "t", "side": "SELL", "qty": 2.0, "price": 99.9, "arrival_price": 100.0}]
    )
    assert tca["implementation_shortfall_usdt"] == pytest.approx(0.2)
    assert tca["spread_cost_usdt"] is None
    assert tca["impact_cost_usdt"] is None


# --- zero-cost backward compatibility ----------------------------------------


def test_zero_costs_bit_identical_to_pre_tca():
    cfg = load_config()
    bars_cfg = dict(cfg)
    bars_cfg["costs"] = {"spread_bps": 0.0, "impact_bps": 0.0}
    import galahad_futures.data as data

    bars, _, _ = data.load_bars(
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
    bars = bars.iloc[-120:].reset_index(drop=True)
    a = run_paper_on_bars(
        bars, cfg, symbol="BTCUSDT", strategy_name="tsmom", strategy_kwargs={"lookback": 48}
    )
    b = run_paper_on_bars(
        bars, bars_cfg, symbol="BTCUSDT", strategy_name="tsmom", strategy_kwargs={"lookback": 48}
    )
    for key in ("fills", "equity_curve", "positions", "funding_events", "liquidation_events"):
        assert a[key] == b[key], key
    assert a["final_equity"] == b["final_equity"]
    # and zero-cost fills execute exactly at arrival
    assert all(f["price"] == f["arrival_price"] for f in b["fills"])
    assert b["tca"]["implementation_shortfall_usdt"] == 0.0


def test_direct_book_default_costs_are_zero():
    book = _book()  # no cost args: default off
    fill = book.market_order("BTCUSDT", 1.0, 100.0, ts="t0")
    assert fill.price == 100.0
    assert fill.spread_cost == 0.0
    assert fill.impact_cost == 0.0


# --- summary / journal wiring -------------------------------------------------


def test_summary_and_journal_carry_tca_block(tmp_path):
    cfg = load_config()
    cfg["costs"] = {"spread_bps": 2.0, "impact_bps": 4.0}
    summary = run_paper_session(
        config=cfg,
        force_source="fixture",
        force_strategy="dual_ma",
        output_dir=tmp_path,
    )
    tca = summary.get("tca")
    assert tca is not None
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
    assert tca["n_fills"] == summary["n_fills"] > 0
    assert tca["implementation_shortfall_usdt"] > 0.0
    # bps consistent with usdt per arrival notional
    assert tca["implementation_shortfall_bps"] == pytest.approx(
        tca["implementation_shortfall_usdt"] / tca["arrival_notional"] * 10_000.0
    )
    # per-fill detail in the journal
    journal = json.loads(Path(summary["journal_path"]).read_text(encoding="utf-8"))
    assert journal["tca"] == tca
    f0 = journal["fills"][0]
    assert f0["arrival_price"] > 0
    assert f0["spread_cost"] >= 0.0
    assert f0["impact_cost"] >= 0.0
    # buys pay up, sells receive less
    for f in journal["fills"]:
        if f["side"] == "BUY":
            assert f["price"] > f["arrival_price"]
        else:
            assert f["price"] < f["arrival_price"]


def test_summary_tca_zero_when_costs_off(tmp_path):
    summary = run_paper_session(
        config=load_config(), force_source="fixture", force_strategy="dual_ma",
        output_dir=tmp_path,
    )
    assert summary["tca"]["implementation_shortfall_usdt"] == 0.0
    assert summary["tca"]["fee_cost_usdt"] > 0.0  # taker fee still separate


# --- config validation (fail closed) ------------------------------------------


def test_cost_config_validation_errors():
    bars = None  # validation happens at session construction, before any fill
    import galahad_futures.data as data

    bars, _, _ = data.load_bars(
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
    bars = bars.iloc[-10:].reset_index(drop=True)
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
            run_paper_on_bars(
                bars, cfg, symbol="BTCUSDT", strategy_name="dual_ma", strategy_kwargs={}
            )

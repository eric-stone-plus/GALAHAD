"""Cash book tests — hand-computed accounting, integer shares, long-only caps."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_security.book import CashEquityBook


def _book(**over):
    kw = dict(cash=100_000.0, fee_bps=0.0, spread_bps=0.0, impact_bps=0.0)
    kw.update(over)
    return CashEquityBook(**kw)


def test_buy_sell_cash_accounting_hand_computed():
    book = _book()
    f1 = book.market_order("AAPL", 10, 50.0, ts="t0")
    assert f1 is not None and f1.qty == 10 and f1.price == 50.0
    assert book.cash == pytest.approx(99_500.0)
    pos = book.position("AAPL")
    assert pos.qty == 10 and pos.avg_cost == 50.0

    f2 = book.market_order("AAPL", -4, 55.0, ts="t1")
    assert f2.realized_pnl == pytest.approx(20.0)  # (55 - 50) × 4
    assert book.cash == pytest.approx(99_720.0)  # +220 proceeds
    assert book.position("AAPL").qty == 6
    # equity = cash + shares × mark
    assert book.equity({"AAPL": 55.0}) == pytest.approx(100_050.0)


def test_integer_shares_floored():
    book = _book()
    assert book.market_order("AAPL", 0.7, 50.0, ts="t0") is None  # sub-share dust
    fill = book.market_order("AAPL", 10.9, 50.0, ts="t1")
    assert fill.qty == 10  # floored, never rounded up


def test_buy_capped_by_cash():
    book = _book(cash=100.0)
    fill = book.market_order("AAPL", 10, 50.0, ts="t0")
    assert fill.qty == 2  # affordable = floor(100 / 50)
    assert book.cash == pytest.approx(0.0)
    # no cash left → no more fills
    assert book.market_order("AAPL", 1, 50.0, ts="t1") is None


def test_sell_capped_by_holdings_long_only():
    book = _book()
    book.market_order("AAPL", 6, 50.0, ts="t0")
    fill = book.market_order("AAPL", -99, 55.0, ts="t1")
    assert fill.qty == 6  # never short
    assert book.position("AAPL").qty == 0
    assert book.position("AAPL").side == "flat"


def test_weight_to_share_conversion_rounding():
    book = _book()
    # weight 0.2 × 100k equity / mark 50 → exactly 400 shares
    delta = book.target_weight_to_qty("AAPL", 0.2, 50.0, equity=100_000.0)
    assert delta == 400.0
    # 0.2 × 100k / 51 → floor(392.15) = 392
    assert book.target_weight_to_qty("AAPL", 0.2, 51.0, equity=100_000.0) == 392.0
    # negative weight clamps to flat (long-only)
    assert book.target_weight_to_qty("AAPL", -0.5, 50.0, equity=100_000.0) == 0.0


def test_apply_target_weight_rebalance():
    book = _book()
    fill = book.apply_target_weight("AAPL", 0.2, 50.0, ts="t0")
    assert fill is not None and fill.qty == 400
    # Re-target same weight at same mark → sub-share dust → no fill
    assert book.apply_target_weight("AAPL", 0.2, 50.0, ts="t1") is None
    # Target 0 → sell everything
    flat = book.apply_target_weight("AAPL", 0.0, 55.0, ts="t2")
    assert flat is not None and flat.side == "SELL" and flat.qty == 400
    assert book.position("AAPL").qty == 0


def test_cost_model_hand_computed():
    book = _book(spread_bps=2.0, impact_bps=4.0)
    buy = book.market_order("AAPL", 10, 100.0, ts="t0")
    # half-spread 1 bps + impact 4 bps = 5 bps on arrival 100
    assert buy.price == pytest.approx(100.05)
    assert buy.arrival_price == pytest.approx(100.0)
    assert buy.spread_cost == pytest.approx(0.10)
    assert buy.impact_cost == pytest.approx(0.40)
    assert book.cash == pytest.approx(100_000.0 - 1_000.50)
    sell = book.market_order("AAPL", -10, 100.0, ts="t1")
    assert sell.price == pytest.approx(99.95)
    # round-trip realized loss = spread + impact both legs = 1.00 USD
    assert sell.realized_pnl == pytest.approx(-1.00)


def test_fee_separate_from_costs():
    book = _book(fee_bps=10.0)  # 10 bps commission, no spread/impact
    fill = book.market_order("AAPL", 10, 100.0, ts="t0")
    assert fill.price == 100.0  # arrival untouched
    assert fill.fee == pytest.approx(1.0)  # 10 bps on 1000
    assert book.cash == pytest.approx(100_000.0 - 1_001.0)


def test_mark_to_market_snapshots():
    book = _book()
    book.market_order("AAPL", 10, 50.0, ts="t0")
    snap = book.mark_to_market({"AAPL": 55.0}, ts="t1")
    assert snap["equity"] == pytest.approx(99_500.0 + 550.0)
    assert snap["positions"]["AAPL"]["qty"] == 10
    assert len(book.equity_curve) == 1

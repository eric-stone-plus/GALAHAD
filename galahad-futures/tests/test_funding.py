"""Funding accounting — long pays / short receives when rate > 0."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.book import FuturesPaperBook


def test_long_pays_funding_when_rate_positive():
    book = FuturesPaperBook(wallet=10_000.0, fee_bps=0.0, default_leverage=2.0)
    book.set_leverage("BTCUSDT", 2.0)
    book.market_order("BTCUSDT", 1.0, 100.0, ts="t0")  # long 1 @ 100
    w0 = book.wallet
    marks = {"BTCUSDT": 100.0}
    paid = book.apply_funding(marks, rate=0.01, ts="t1")  # 1% of notional
    assert paid == pytest.approx(1.0)  # 1 * 100 * 0.01
    assert book.wallet == pytest.approx(w0 - 1.0)
    assert book.total_funding == pytest.approx(1.0)
    assert len(book.funding_events) == 1
    # MTM equity = wallet + uPnL (0) 
    assert book.equity(marks) == pytest.approx(book.wallet)


def test_short_receives_funding_when_rate_positive():
    book = FuturesPaperBook(wallet=10_000.0, fee_bps=0.0, default_leverage=2.0)
    book.set_leverage("BTCUSDT", 2.0)
    book.market_order("BTCUSDT", -2.0, 50.0, ts="t0")  # short 2 @ 50
    w0 = book.wallet
    marks = {"BTCUSDT": 50.0}
    # payment = qty * mark * rate = -2 * 50 * 0.01 = -1 → wallet -= (-1) → +1
    paid = book.apply_funding(marks, rate=0.01, ts="t1")
    assert paid == pytest.approx(-1.0)
    assert book.wallet == pytest.approx(w0 + 1.0)
    assert book.total_funding == pytest.approx(-1.0)


def test_zero_rate_leaves_wallet_unchanged():
    book = FuturesPaperBook(wallet=10_000.0, fee_bps=0.0, funding_rate_per_bar=0.0)
    book.set_leverage("BTCUSDT", 2.0)
    book.market_order("BTCUSDT", 1.0, 100.0, ts="t0")
    w0 = book.wallet
    book.mark_to_market({"BTCUSDT": 100.0}, ts="t1")
    assert book.wallet == pytest.approx(w0)
    assert book.total_funding == pytest.approx(0.0)
    assert book.funding_events == []


def test_mark_to_market_accumulates_funding_with_default_rate():
    book = FuturesPaperBook(
        wallet=10_000.0,
        fee_bps=0.0,
        funding_rate_per_bar=0.001,
        default_leverage=2.0,
    )
    book.set_leverage("BTCUSDT", 2.0)
    book.market_order("BTCUSDT", 1.0, 1000.0, ts="t0")
    marks = {"BTCUSDT": 1000.0}
    book.mark_to_market(marks, ts="t1")
    book.mark_to_market(marks, ts="t2")
    # each bar: 1 * 1000 * 0.001 = 1
    assert book.total_funding == pytest.approx(2.0)
    assert len(book.funding_events) == 2
    assert book.equity_curve[-1]["total_funding"] == pytest.approx(2.0)

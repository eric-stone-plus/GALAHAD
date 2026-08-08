"""Unit tests for FuturesPaperBook — fixed price paths, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.book import FuturesPaperBook


def test_long_open_close_updates_margin_and_mtm():
    book = FuturesPaperBook(wallet=10_000.0, fee_bps=0.0, default_leverage=2.0, max_leverage=5.0)
    book.set_leverage("BTCUSDT", 2.0)

    # Open long 1 BTC @ 100
    fill = book.market_order("BTCUSDT", 1.0, 100.0, ts="t0", note="open")
    assert fill is not None
    assert fill.side == "BUY"
    pos = book.position("BTCUSDT")
    assert pos.qty == pytest.approx(1.0)
    assert pos.entry_price == pytest.approx(100.0)

    marks = {"BTCUSDT": 100.0}
    # notional 100, lev 2 → margin 50
    assert book.margin_used(marks) == pytest.approx(50.0)
    assert book.equity(marks) == pytest.approx(10_000.0)

    # Mark up to 110 → uPnL +10
    marks = {"BTCUSDT": 110.0}
    assert book.unrealized_pnl(marks) == pytest.approx(10.0)
    assert book.equity(marks) == pytest.approx(10_010.0)
    snap = book.mark_to_market(marks, ts="t1")
    assert snap["equity"] == pytest.approx(10_010.0)

    # Close long @ 110
    fill2 = book.market_order("BTCUSDT", -1.0, 110.0, ts="t2", note="close")
    assert fill2 is not None
    assert fill2.realized_pnl == pytest.approx(10.0)
    assert abs(book.position("BTCUSDT").qty) < 1e-12
    assert book.wallet == pytest.approx(10_010.0)
    assert book.equity({"BTCUSDT": 110.0}) == pytest.approx(10_010.0)


def test_short_open_close_mtm():
    book = FuturesPaperBook(wallet=10_000.0, fee_bps=0.0, default_leverage=2.0)
    book.set_leverage("BTCUSDT", 2.0)

    fill = book.market_order("BTCUSDT", -2.0, 200.0, ts="t0")  # short 2 @ 200
    assert fill is not None
    assert book.position("BTCUSDT").side == "short"
    marks = {"BTCUSDT": 200.0}
    assert book.margin_used(marks) == pytest.approx(200.0)  # notional 400 / lev 2

    # Price drops to 180 → short profit 2 * 20 = 40
    marks = {"BTCUSDT": 180.0}
    assert book.unrealized_pnl(marks) == pytest.approx(40.0)
    assert book.equity(marks) == pytest.approx(10_040.0)

    fill2 = book.market_order("BTCUSDT", 2.0, 180.0, ts="t1")
    assert fill2 is not None
    assert fill2.realized_pnl == pytest.approx(40.0)
    assert book.wallet == pytest.approx(10_040.0)


def test_leverage_bounds():
    book = FuturesPaperBook(wallet=1_000.0, max_leverage=5.0)
    with pytest.raises(ValueError):
        book.set_leverage("BTCUSDT", 0.5)
    with pytest.raises(ValueError):
        book.set_leverage("BTCUSDT", 10.0)
    book.set_leverage("BTCUSDT", 5.0)
    assert book.position("BTCUSDT").leverage == pytest.approx(5.0)


def test_liquidation_force_close_when_margin_fails():
    # High leverage long, adverse move → equity < maintenance
    book = FuturesPaperBook(
        wallet=1_000.0,
        fee_bps=0.0,
        default_leverage=10.0,
        max_leverage=20.0,
        maintenance_margin_rate=0.05,  # 5% MM for easy trip
    )
    book.set_leverage("BTCUSDT", 10.0)
    # Open long notional ~9000 with 1000 wallet (9x used margin ~900)
    fill = book.market_order("BTCUSDT", 9.0, 1000.0, ts="t0")
    assert fill is not None
    assert not book.liquidated

    # Crash: mark 100 — huge loss, equity goes negative vs MM
    marks = {"BTCUSDT": 100.0}
    # uPnL = 9 * (100 - 1000) = -8100; equity = 1000 - 8100 = -7100
    assert book.equity(marks) < 0
    tripped = book.check_liquidation(marks, ts="t_liq")
    assert tripped is True
    assert book.liquidated is True
    assert abs(book.position("BTCUSDT").qty) < 1e-12
    assert len(book.liquidation_events) == 1
    # Further orders rejected
    assert book.market_order("BTCUSDT", 1.0, 100.0, ts="t_after") is None


def test_target_to_delta_and_apply_target():
    book = FuturesPaperBook(wallet=10_000.0, fee_bps=0.0, default_leverage=2.0)
    book.set_leverage("BTCUSDT", 2.0)
    mark = 50_000.0
    # target +1x leverage → notional 10k → qty = 10000/50000 = 0.2
    delta = book.target_to_delta_qty("BTCUSDT", 1.0, mark)
    assert delta == pytest.approx(0.2)
    fill = book.apply_target("BTCUSDT", 1.0, mark, ts="t0")
    assert fill is not None
    assert book.position("BTCUSDT").qty == pytest.approx(0.2)
    # Flatten
    fill2 = book.apply_target("BTCUSDT", 0.0, mark, ts="t1")
    assert fill2 is not None
    assert abs(book.position("BTCUSDT").qty) < 1e-9


def test_fees_reduce_wallet():
    book = FuturesPaperBook(wallet=10_000.0, fee_bps=10.0, default_leverage=1.0)  # 10 bps
    book.set_leverage("BTCUSDT", 1.0)
    book.market_order("BTCUSDT", 1.0, 1000.0, ts="t0")
    # fee = 1000 * 0.001 = 1
    assert book.wallet == pytest.approx(10_000.0 - 1.0)


def test_same_side_add_is_margin_capped():
    """Skeptic repro: small open then oversized same-side add must not blow margin.

    wallet=1000 lev=2 open 0.1@100 then +100@100 must downsize/reject so that
    available_margin >= 0 and margin_used <= equity + eps.
    """
    book = FuturesPaperBook(
        wallet=1_000.0,
        fee_bps=0.0,
        default_leverage=2.0,
        max_leverage=5.0,
    )
    book.set_leverage("BTCUSDT", 2.0)
    f1 = book.market_order("BTCUSDT", 0.1, 100.0, ts="t0")
    assert f1 is not None
    assert book.position("BTCUSDT").qty == pytest.approx(0.1)

    f2 = book.market_order("BTCUSDT", 100.0, 100.0, ts="t1")
    # Must not accept full +100 (would need margin ~5000)
    assert f2 is not None  # downsized fill is OK
    assert f2.qty < 100.0 - 1e-6
    marks = {"BTCUSDT": 100.0}
    eq = book.equity(marks)
    mu = book.margin_used(marks)
    avail = book.available_margin(marks)
    assert avail >= -1e-6, f"available_margin={avail}"
    assert mu <= eq + 1e-6, f"margin_used={mu} equity={eq} qty={book.position('BTCUSDT').qty}"
    # Position still modest relative to equity*lev
    notional = abs(book.position("BTCUSDT").qty) * 100.0
    assert notional <= eq * 2.0 + 1e-3


def test_same_side_add_rejects_when_no_margin_left():
    book = FuturesPaperBook(wallet=100.0, fee_bps=0.0, default_leverage=2.0, max_leverage=5.0)
    book.set_leverage("BTCUSDT", 2.0)
    # Use almost all margin: notional 200 → margin 100
    f1 = book.market_order("BTCUSDT", 2.0, 100.0, ts="t0")
    assert f1 is not None
    marks = {"BTCUSDT": 100.0}
    assert book.available_margin(marks) < 1e-6
    f2 = book.market_order("BTCUSDT", 1.0, 100.0, ts="t1")
    assert f2 is None
    assert book.position("BTCUSDT").qty == pytest.approx(2.0)


def test_flip_fees_charged_once_per_leg():
    """Flip long 1 → short 1: close fee + open fee only (not full-order then open again).

    fee_bps=10 → 10 bps. open 1@1000 fee=1; flip -2@1000 → close 1 fee=1 + open 1 fee=1
    total fees = 3 across both fills; wallet after flat-ish accounting:
      after open: 10000-1=9999
      flip close: realized 0, -close_fee 1 → 9998; open fee 1 → 9997
    """
    book = FuturesPaperBook(wallet=10_000.0, fee_bps=10.0, default_leverage=5.0, max_leverage=10.0)
    book.set_leverage("BTCUSDT", 5.0)
    f1 = book.market_order("BTCUSDT", 1.0, 1000.0, ts="t0")
    assert f1 is not None
    assert f1.fee == pytest.approx(1.0)
    assert book.wallet == pytest.approx(9_999.0)

    f2 = book.market_order("BTCUSDT", -2.0, 1000.0, ts="t1")
    assert f2 is not None
    # close 1 + open 1 → fee 1+1=2 (not 3 from double-charging full order)
    assert f2.fee == pytest.approx(2.0)
    assert book.wallet == pytest.approx(9_997.0)
    assert book.position("BTCUSDT").qty == pytest.approx(-1.0)
    total_fees = sum(f.fee for f in book.fills)
    assert total_fees == pytest.approx(3.0)

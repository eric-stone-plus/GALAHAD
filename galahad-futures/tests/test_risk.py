"""Risk gate tests — reject oversized targets before any fill."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.book import FuturesPaperBook
from galahad_futures.risk import RiskConfig, RiskGate


def test_rejects_oversized_position_by_clipping():
    cfg = RiskConfig(
        max_order_notional=50_000.0,
        max_position_notional=5_000.0,
        max_daily_loss=1_000.0,
        max_leverage=5.0,
        mode="paper",
        kill_switch=True,
        enable_live=False,
    )
    gate = RiskGate(config=cfg, day_start_equity=10_000.0)
    d = gate.filter_target(
        symbol="BTCUSDT",
        target_signed_leverage=5.0,  # wants 50k notional
        mark=50_000.0,
        equity=10_000.0,
        current_qty=0.0,
        leverage=5.0,
        ts="t0",
    )
    assert d.allowed
    assert d.clipped
    # position notional cap → |t| * equity <= 5000 → |t| <= 0.5
    assert abs(d.target_signed_leverage) == pytest.approx(0.5)


def test_daily_loss_halt_forces_flat():
    """Daily-loss breach force-flattens: target 0 is the only allowed
    action — never a frozen position. The halt state is driven by
    update_equity (the decision layer feeds pre-trade equity there)."""
    cfg = RiskConfig(max_daily_loss=100.0, mode="paper")
    gate = RiskGate(config=cfg, day_start_equity=10_000.0)
    gate.update_equity(9_850.0, ts="t0")  # lost 150 > 100
    assert gate.loss_halted
    d = gate.filter_target(
        symbol="BTCUSDT",
        target_signed_leverage=1.0,
        mark=100.0,
        equity=9_850.0,
        current_qty=-0.2,  # holding a losing position — flatten is allowed
        leverage=2.0,
        ts="t0",
    )
    assert d.allowed
    assert d.target_signed_leverage == 0.0
    assert d.reason == "daily_loss_force_flat"
    # new risk is logged as a reject for the audit trail
    assert len(gate.rejects) == 1


def test_daily_loss_halt_recovers_with_hysteresis():
    cfg = RiskConfig(max_daily_loss=100.0, daily_loss_hysteresis=20.0, mode="paper")
    gate = RiskGate(config=cfg, day_start_equity=10_000.0)
    gate.update_equity(9_850.0, ts="t0")
    assert gate.loss_halted
    # Inside the hysteresis band: still halted.
    gate.update_equity(9_910.0, ts="t1")
    assert gate.loss_halted
    # Past floor + band (9_900 + 20): cleared.
    gate.update_equity(9_921.0, ts="t2")
    assert not gate.loss_halted
    assert [e["event"] for e in gate.loss_halt_events] == ["halted", "cleared"]


def test_daily_loss_zero_hysteresis_matches_legacy_floor():
    cfg = RiskConfig(max_daily_loss=100.0, mode="paper")
    gate = RiskGate(config=cfg, day_start_equity=10_000.0)
    gate.update_equity(9_899.0, ts="t0")
    assert gate.loss_halted
    gate.update_equity(9_900.0, ts="t1")
    assert not gate.loss_halted


def test_live_blocked_by_kill_switch():
    cfg = RiskConfig(mode="live", kill_switch=True, enable_live=True)
    gate = RiskGate(config=cfg, day_start_equity=10_000.0)
    d = gate.filter_target(
        symbol="BTCUSDT",
        target_signed_leverage=1.0,
        mark=100.0,
        equity=10_000.0,
        current_qty=0.0,
        leverage=2.0,
    )
    assert not d.allowed
    assert "live_blocked" in d.reason


def test_live_blocked_when_enable_live_false():
    cfg = RiskConfig(mode="live", kill_switch=False, enable_live=False)
    gate = RiskGate(config=cfg, day_start_equity=10_000.0)
    d = gate.filter_target(
        symbol="BTCUSDT",
        target_signed_leverage=1.0,
        mark=100.0,
        equity=10_000.0,
        current_qty=0.0,
        leverage=2.0,
    )
    assert not d.allowed


def test_order_notional_cap_clips_delta():
    cfg = RiskConfig(
        max_order_notional=1_000.0,
        max_position_notional=100_000.0,
        max_leverage=10.0,
        mode="paper",
    )
    gate = RiskGate(config=cfg, day_start_equity=10_000.0)
    # target 1x on 10k equity @ mark 100 → desired qty 100, order notional 10k → clip to 1000
    d = gate.filter_target(
        symbol="BTCUSDT",
        target_signed_leverage=1.0,
        mark=100.0,
        equity=10_000.0,
        current_qty=0.0,
        leverage=5.0,
        ts="t0",
    )
    assert d.allowed
    assert d.clipped
    # max order 1000 / 100 = 10 qty → target lev = 10*100/10000 = 0.1
    assert d.target_signed_leverage == pytest.approx(0.1)


def test_risk_then_book_never_sees_raw_oversized_target():
    """Integration: risk clips before book.apply_target."""
    book = FuturesPaperBook(wallet=10_000.0, fee_bps=0.0, default_leverage=2.0, max_leverage=5.0)
    book.set_leverage("BTCUSDT", 2.0)
    cfg = RiskConfig(max_position_notional=2_000.0, max_order_notional=50_000.0, mode="paper")
    gate = RiskGate(config=cfg, day_start_equity=10_000.0)
    raw = 5.0
    mark = 1_000.0
    d = gate.filter_target(
        symbol="BTCUSDT",
        target_signed_leverage=raw,
        mark=mark,
        equity=book.equity({"BTCUSDT": mark}),
        current_qty=0.0,
        leverage=2.0,
        ts="t0",
    )
    assert d.allowed
    book.apply_target("BTCUSDT", d.target_signed_leverage, mark, ts="t0")
    notional = abs(book.position("BTCUSDT").qty) * mark
    assert notional <= 2_000.0 + 1e-6

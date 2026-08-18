"""Decision-layer state machine tests.

Contracts:
  - phase derivation matches the documented precedence
    (LIQUIDATED > INVALIDATED > LOSS_HALTED > LIVE_BLOCKED > ACTIVE)
  - terminal phases never revert (illegal transitions fail closed)
  - decision records carry monotonic seq, phase_before/phase_after, and
    boundary headroom instrumentation (dd_headroom, loss_headroom)
  - liquidation reported by an executor makes the session terminal
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.decision import SessionRisk

CFG_PAPER = {
    "mode": "paper",
    "max_leverage": 5.0,
    "risk": {
        "max_order_notional": 50_000.0,
        "max_position_notional": 15_000.0,
        "max_daily_loss": 500.0,
        "max_drawdown_pct": 0.15,
        "daily_loss_hysteresis": 50.0,
        "kill_switch": True,
        "enable_live": False,
    },
}


def _evaluate(session: SessionRisk, equity: float, target: float = 1.0, ts: str = "t") -> None:
    session.update_equity(equity, ts=ts)
    session.evaluate_target(
        symbol="BTCUSDT",
        raw_target=target,
        mark=100.0,
        pre_trade_equity=equity,
        current_qty=0.0,
        leverage=3.0,
        ts=ts,
    )


def test_phases_derive_in_documented_order():
    s = SessionRisk.from_config(CFG_PAPER, start_equity=10_000.0)
    assert s.phase() == "ACTIVE"
    s.update_equity(9_000.0, ts="t0")  # below daily-loss floor 9_500
    assert s.phase() == "LOSS_HALTED"
    # drawdown trip: 10_000 → 6_000 is 40% > 15%
    s.update_equity(6_000.0, ts="t1")
    assert s.phase() == "INVALIDATED"  # terminal outranks halt
    s.note_liquidation(ts="t2")
    assert s.phase() == "LIQUIDATED"


def test_active_to_halt_and_recovery_with_hysteresis():
    s = SessionRisk.from_config(CFG_PAPER, start_equity=10_000.0)
    _evaluate(s, 10_000.0, ts="t0")
    assert s.risk_decisions[-1]["phase_before"] == "ACTIVE"
    _evaluate(s, 9_400.0, ts="t1")  # breach floor
    assert s.risk_decisions[-1]["phase_before"] == "ACTIVE"
    assert s.risk_decisions[-1]["phase_after"] == "LOSS_HALTED"
    _evaluate(s, 9_480.0, ts="t2")  # above floor, inside band (9_500+50)
    assert s.risk_decisions[-1]["phase_before"] == "LOSS_HALTED"
    assert s.risk_decisions[-1]["phase_after"] == "LOSS_HALTED"
    _evaluate(s, 9_560.0, ts="t3")  # past band
    assert s.risk_decisions[-1]["phase_before"] == "LOSS_HALTED"
    assert s.risk_decisions[-1]["phase_after"] == "ACTIVE"


def test_records_carry_seq_and_headroom():
    s = SessionRisk.from_config(CFG_PAPER, start_equity=10_000.0)
    _evaluate(s, 10_000.0, ts="t0")
    _evaluate(s, 9_800.0, ts="t1")
    r0, r1 = s.risk_decisions[-2], s.risk_decisions[-1]
    assert r0["seq"] == 1 and r1["seq"] == 2
    assert r0["phase_before"] == "ACTIVE" and r0["phase_after"] == "ACTIVE"
    assert r0["dd_headroom"] == pytest.approx(0.15)
    assert r0["loss_headroom"] == pytest.approx(500.0)
    assert r1["dd_headroom"] == pytest.approx(0.13)
    assert r1["loss_headroom"] == pytest.approx(300.0)


def test_terminal_phase_transition_fails_closed():
    s = SessionRisk.from_config(CFG_PAPER, start_equity=10_000.0)
    _evaluate(s, 10_000.0, ts="t0")
    _evaluate(s, 6_000.0, ts="t1")  # INVALIDATED (terminal)
    assert s.phase() == "INVALIDATED"
    # A forged recovery of the gate flags (corrupted state) is caught by
    # the transition checker: INVALIDATED never reverts.
    s.gate.invalidated = False
    s.gate.loss_halted = False
    with pytest.raises(ValueError, match="terminal"):
        s.update_equity(20_000.0, ts="t2")
        s.evaluate_target(
            symbol="BTCUSDT",
            raw_target=0.0,
            mark=100.0,
            pre_trade_equity=20_000.0,
            current_qty=0.0,
            leverage=3.0,
            ts="t2",
        )


def test_liquidation_note_is_terminal():
    s = SessionRisk.from_config(CFG_PAPER, start_equity=10_000.0)
    _evaluate(s, 10_000.0, ts="t0")
    s.note_liquidation(ts="t1")
    assert s.phase() == "LIQUIDATED"
    # No further decisions are legal once liquidated.
    with pytest.raises(ValueError, match="liquidation"):
        _evaluate(s, 9_900.0, ts="t2")


def test_daily_loss_force_flat_allows_flatten_only():
    s = SessionRisk.from_config(CFG_PAPER, start_equity=10_000.0)
    s.update_equity(9_400.0, ts="t0")
    d = s.evaluate_target(
        symbol="BTCUSDT",
        raw_target=1.0,
        mark=100.0,
        pre_trade_equity=9_400.0,
        current_qty=-0.5,
        leverage=3.0,
        ts="t0",
    )
    assert d.allowed
    assert d.target_signed_leverage == 0.0
    assert d.reason == "daily_loss_force_flat"

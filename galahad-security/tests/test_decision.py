"""Decision layer tests — phases, transitions, per-decision records."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_security.decision import SessionRisk


def _session(**risk_over):
    risk = {"max_daily_loss": 100.0, "max_drawdown_pct": 0.15}
    risk.update(risk_over)
    return SessionRisk.from_config({"mode": "paper", "risk": risk}, start_equity=100_000.0)


def _eval(session, weight=0.2, equity=100_000.0, ts="t0"):
    return session.evaluate_weight(
        symbol="AAPL",
        raw_weight=weight,
        mark=100.0,
        pre_trade_equity=equity,
        current_qty=0.0,
        ts=ts,
    )


def test_records_carry_seq_phases_and_headroom():
    s = _session()
    _eval(s)
    s.update_equity(99_850.0, ts="t1")  # daily-loss halt
    _eval(s, equity=99_850.0, ts="t1")
    r0, r1 = s.risk_decisions
    assert r0["seq"] == 1 and r1["seq"] == 2
    assert r0["phase_after"] == "ACTIVE"
    assert r1["phase_before"] == "ACTIVE" and r1["phase_after"] == "LOSS_HALTED"
    assert "dd_headroom" in r1 and "loss_headroom" in r1
    assert r1["derisk_multiplier"] == 1.0
    assert r1["symbol"] == "AAPL"


def test_invalidation_is_terminal():
    s = _session()
    s.update_equity(100_000.0, ts="t0")
    s.update_equity(84_000.0, ts="t1")  # dd 16% > 15% trip
    d = _eval(s, equity=84_000.0, ts="t1")
    assert d.reason == "invalidation_force_flat"
    assert s.phase() == "INVALIDATED"
    # recovery equity cannot leave the terminal state
    s.update_equity(120_000.0, ts="t2")
    assert s.phase() == "INVALIDATED"
    d2 = _eval(s, equity=120_000.0, ts="t2")
    assert d2.target_weight == 0.0


def test_venue_blocked_phase():
    s = SessionRisk.from_config(
        {"mode": "venue", "risk": {"kill_switch": True, "enable_alpaca_paper": True}},
        start_equity=100_000.0,
    )
    assert s.phase() == "VENUE_BLOCKED"
    d = _eval(s)
    assert not d.allowed
    assert "venue_blocked" in d.reason

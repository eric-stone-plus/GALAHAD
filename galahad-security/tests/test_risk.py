"""Risk gate tests — weight caps, daily-loss halt, invalidation, ladder, venue gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_security.decision import SessionRisk
from galahad_security.risk import RiskConfig, RiskGate, validate_derisk_ladder

LADDER = [
    {"drawdown": 0.05, "leverage_multiplier": 0.5},
    {"drawdown": 0.075, "leverage_multiplier": 0.0},
]


def _gate(ladder=None, **cfg_over):
    kw = dict(
        max_weight=1.0,
        max_order_notional=1e9,
        max_daily_loss=1e9,  # parked unless under test
        max_drawdown_pct=0.5,  # parked unless under test
        mode="paper",
        derisk_ladder=ladder,
    )
    kw.update(cfg_over)
    return RiskGate(config=RiskConfig(**kw), day_start_equity=100_000.0)


def _filter(gate, weight=0.2, equity=100_000.0, qty=0.0, mark=100.0):
    return gate.filter_weight(
        symbol="AAPL",
        target_weight=weight,
        mark=mark,
        equity=equity,
        current_qty=qty,
        ts="t0",
    )


# --- caps ---------------------------------------------------------------------


def test_weight_cap_clips():
    gate = _gate(max_weight=0.25)
    d = _filter(gate, weight=0.5)
    assert d.allowed and d.clipped
    assert d.target_weight == pytest.approx(0.25)


def test_weight_capped_at_one_for_cash_account():
    gate = _gate(max_weight=5.0)  # misconfigured high cap still cannot exceed 1.0
    d = _filter(gate, weight=3.0)
    assert d.allowed and d.clipped
    assert d.target_weight == pytest.approx(1.0)


def test_negative_weight_rejected():
    gate = _gate()
    d = _filter(gate, weight=-0.1)
    assert not d.allowed
    assert d.reason == "negative_weight_long_only"


def test_order_notional_cap_clips_delta():
    gate = _gate(max_weight=0.25, max_order_notional=1_000.0)
    d = _filter(gate, weight=0.25, equity=100_000.0, mark=100.0)
    assert d.allowed and d.clipped
    # max order 1000 / 100 = 10 shares → weight = 10×100/100k = 0.01
    assert d.target_weight == pytest.approx(0.01)


# --- daily-loss halt + hysteresis ----------------------------------------------


def test_daily_loss_halt_forces_flat():
    gate = _gate(max_daily_loss=100.0)
    gate.update_equity(99_850.0, ts="t0")  # lost 150 > 100
    assert gate.loss_halted
    d = _filter(gate, weight=0.2, equity=99_850.0, qty=10.0)
    assert d.allowed
    assert d.target_weight == 0.0
    assert d.reason == "daily_loss_force_flat"
    assert len(gate.rejects) == 1  # new risk logged for the audit trail


def test_daily_loss_halt_recovers_with_hysteresis():
    gate = _gate(max_daily_loss=100.0, daily_loss_hysteresis=20.0)
    gate.update_equity(99_850.0, ts="t0")
    assert gate.loss_halted
    gate.update_equity(99_910.0, ts="t1")  # inside the band: still halted
    assert gate.loss_halted
    gate.update_equity(99_921.0, ts="t2")  # past floor + band: cleared
    assert not gate.loss_halted
    assert [e["event"] for e in gate.loss_halt_events] == ["halted", "cleared"]


# --- drawdown invalidation ------------------------------------------------------


def test_drawdown_invalidation_force_flat_terminal():
    gate = _gate(max_drawdown_pct=0.15)
    gate.update_equity(100_000.0, ts="t0")
    gate.update_equity(84_000.0, ts="t1")  # dd 16% > 15%
    assert gate.invalidated
    d = _filter(gate, weight=0.2, equity=84_000.0)
    assert d.allowed and d.target_weight == 0.0
    assert d.reason == "invalidation_force_flat"


# --- de-risking ladder ------------------------------------------------------------


def test_ladder_disabled_by_default():
    gate = _gate()
    gate.update_equity(100_000.0, ts="t0")
    assert gate.derisk_multiplier(1.0) == 1.0  # even at total loss
    d = _filter(gate)
    assert d.allowed and d.target_weight == pytest.approx(0.2)
    assert d.derisk_multiplier == 1.0
    assert gate.derisk_summary() == {
        "ladder_enabled": False,
        "tiers_triggered": 0,
        "min_multiplier": 1.0,
    }


def test_tier_selection_boundaries():
    gate = _gate(ladder=LADDER)
    gate.update_equity(100_000.0, ts="t0")
    assert gate.derisk_multiplier(95_001.0) == 1.0   # dd 4.999%: below first tier
    assert gate.derisk_multiplier(95_000.0) == 0.5   # dd exactly 5%: breached
    assert gate.derisk_multiplier(92_600.0) == 0.5   # dd 7.4%: between tiers
    assert gate.derisk_multiplier(92_500.0) == 0.0   # dd exactly 7.5%: deepest tier
    assert gate.derisk_multiplier(90_000.0) == 0.0   # beyond


def test_ladder_scales_target():
    gate = _gate(ladder=LADDER)
    gate.update_equity(100_000.0, ts="t0")
    gate.update_equity(95_000.0, ts="t1")  # dd 5% → ×0.5
    d = _filter(gate, weight=0.2, equity=95_000.0)
    assert d.allowed
    assert d.target_weight == pytest.approx(0.1)
    assert d.derisk_multiplier == 0.5
    assert d.reason == "ok_clipped"


def test_multiplier_zero_flattens():
    gate = _gate(ladder=LADDER)
    gate.update_equity(100_000.0, ts="t0")
    gate.update_equity(92_500.0, ts="t1")  # dd 7.5% → ×0.0
    d = _filter(gate, weight=0.2, equity=92_500.0, qty=47)
    assert d.allowed and d.target_weight == 0.0
    assert d.reason == "derisk_force_flat"
    assert gate.rejects[-1]["reason"] == "derisk_block_new_risk"


def test_existing_gates_still_fire_with_ladder():
    gate = _gate(ladder=LADDER, max_drawdown_pct=0.15)
    gate.update_equity(100_000.0, ts="t0")
    gate.update_equity(84_000.0, ts="t1")
    assert gate.invalidated
    d = _filter(gate, weight=0.2, equity=84_000.0)
    assert d.reason == "invalidation_force_flat"  # terminal gate, not the ladder


def test_malformed_ladders_raise():
    for bad in (
        [{"drawdown": 0.075, "leverage_multiplier": 0.5},
         {"drawdown": 0.05, "leverage_multiplier": 0.0}],   # non-ascending
        [{"drawdown": 0.05, "leverage_multiplier": 1.5}],   # multiplier > 1
        [{"drawdown": 0.05, "leverage_multiplier": -0.1}],  # multiplier < 0
        [{"drawdown": 0.0, "leverage_multiplier": 0.5}],    # drawdown 0
        [{"drawdown": 1.5, "leverage_multiplier": 0.5}],    # drawdown > 1
        [{"drawdown": 0.05}],                               # missing key
        {"drawdown": 0.05, "leverage_multiplier": 0.5},     # not a list
        "abc",
    ):
        with pytest.raises(ValueError, match="derisk_ladder"):
            _gate(ladder=bad)


def test_validate_derisk_ladder_idempotent():
    once = validate_derisk_ladder(LADDER)
    assert validate_derisk_ladder(once) == once


def test_shared_session_constructor_validates():
    """Every engine builds its gate through SessionRisk.from_config."""
    cfg = {"mode": "paper", "risk": {"derisk_ladder": LADDER}}
    session = SessionRisk.from_config(cfg, start_equity=100_000.0)
    assert session.gate.config.derisk_ladder == ((0.05, 0.5), (0.075, 0.0))
    bad = {"mode": "paper",
           "risk": {"derisk_ladder": [{"drawdown": 0.05, "leverage_multiplier": 2.0}]}}
    with pytest.raises(ValueError, match="derisk_ladder"):
        SessionRisk.from_config(bad, start_equity=100_000.0)


# --- venue gate --------------------------------------------------------------------


def test_venue_blocked_by_kill_switch():
    gate = _gate(mode="venue", kill_switch=True, enable_alpaca_paper=True)
    d = _filter(gate)
    assert not d.allowed
    assert "venue_blocked" in d.reason


def test_venue_blocked_when_not_enabled():
    gate = _gate(mode="venue", kill_switch=False, enable_alpaca_paper=False)
    assert not _filter(gate).allowed


def test_venue_allowed_when_enabled_and_kill_switch_off():
    gate = _gate(mode="venue", kill_switch=False, enable_alpaca_paper=True)
    assert _filter(gate).allowed


def test_paper_never_venue_blocked():
    gate = _gate(mode="paper", kill_switch=True, enable_alpaca_paper=False)
    assert _filter(gate).allowed

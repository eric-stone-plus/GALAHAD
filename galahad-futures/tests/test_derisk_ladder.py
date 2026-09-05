"""Graduated de-risking ladder tests — tiers, boundaries, gates, validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.decision import SessionRisk
from galahad_futures.engine import load_config, run_paper_on_bars, run_paper_session
from galahad_futures.risk import RiskConfig, RiskGate, validate_derisk_ladder

LADDER = [
    {"drawdown": 0.05, "leverage_multiplier": 0.5},
    {"drawdown": 0.075, "leverage_multiplier": 0.0},
]


def _gate(ladder=None, **cfg_over):
    kw = dict(
        max_order_notional=1e9,
        max_position_notional=1e9,
        max_daily_loss=1e9,  # parked: ladder exercised in isolation
        max_leverage=10.0,
        mode="paper",
        derisk_ladder=ladder,
    )
    kw.update(cfg_over)
    return RiskGate(config=RiskConfig(**kw), day_start_equity=10_000.0)


def _filter(gate, target=1.0, equity=10_000.0, qty=0.0):
    return gate.filter_target(
        symbol="BTCUSDT",
        target_signed_leverage=target,
        mark=100.0,
        equity=equity,
        current_qty=qty,
        leverage=5.0,
        ts="t0",
    )


# --- default OFF = current behavior -------------------------------------------


def test_ladder_disabled_by_default():
    gate = _gate()
    gate.update_equity(10_000.0, ts="t0")
    assert gate.derisk_multiplier(1.0) == 1.0  # even at total loss
    d = _filter(gate)
    assert d.allowed
    assert d.target_signed_leverage == pytest.approx(1.0)
    assert d.derisk_multiplier == 1.0
    assert gate.derisk_summary() == {
        "ladder_enabled": False,
        "tiers_triggered": 0,
        "min_multiplier": 1.0,
    }


def test_existing_clip_behavior_unchanged_without_ladder():
    """The pre-ladder reference case: position-notional cap clips to 0.5."""
    cfg = RiskConfig(
        max_order_notional=50_000.0,
        max_position_notional=5_000.0,
        max_daily_loss=1_000.0,
        max_leverage=5.0,
        mode="paper",
    )
    gate = RiskGate(config=cfg, day_start_equity=10_000.0)
    d = gate.filter_target(
        symbol="BTCUSDT",
        target_signed_leverage=5.0,
        mark=50_000.0,
        equity=10_000.0,
        current_qty=0.0,
        leverage=5.0,
        ts="t0",
    )
    assert d.allowed and d.clipped
    assert abs(d.target_signed_leverage) == pytest.approx(0.5)


# --- tier selection at boundaries ----------------------------------------------


def test_tier_selection_boundaries():
    gate = _gate(ladder=LADDER)
    gate.update_equity(10_000.0, ts="t0")  # peak
    assert gate.derisk_multiplier(9_501.0) == 1.0  # dd 4.99%: below first tier
    assert gate.derisk_multiplier(9_500.0) == 0.5  # dd exactly 5%: breached
    assert gate.derisk_multiplier(9_260.0) == 0.5  # dd 7.4%: between tiers
    assert gate.derisk_multiplier(9_250.0) == 0.0  # dd exactly 7.5%: deepest tier
    assert gate.derisk_multiplier(9_000.0) == 0.0  # beyond


def test_ladder_scales_target():
    gate = _gate(ladder=LADDER)
    gate.update_equity(10_000.0, ts="t0")
    gate.update_equity(9_500.0, ts="t1")  # dd 5% → ×0.5
    d = _filter(gate, target=1.0, equity=9_500.0)
    assert d.allowed
    assert d.target_signed_leverage == pytest.approx(0.5)
    assert d.derisk_multiplier == 0.5
    assert d.reason == "ok_clipped"


def test_multiplier_zero_flattens():
    gate = _gate(ladder=LADDER)
    gate.update_equity(10_000.0, ts="t0")
    gate.update_equity(9_250.0, ts="t1")  # dd 7.5% → ×0.0
    d = _filter(gate, target=1.0, equity=9_250.0, qty=47.5)
    assert d.allowed
    assert d.target_signed_leverage == 0.0
    assert d.reason == "derisk_force_flat"
    assert d.derisk_multiplier == 0.0
    assert gate.rejects[-1]["reason"] == "derisk_block_new_risk"
    # an already-flat target stays a clean no-op (no reject logged)
    n_rejects = len(gate.rejects)
    d0 = _filter(gate, target=0.0, equity=9_250.0, qty=0.0)
    assert d0.allowed and d0.target_signed_leverage == 0.0
    assert len(gate.rejects) == n_rejects


# --- existing gates still fire at their own thresholds -------------------------


def test_invalidation_still_fires_with_ladder():
    gate = _gate(ladder=LADDER, max_drawdown_pct=0.15)
    gate.update_equity(10_000.0, ts="t0")
    gate.update_equity(8_400.0, ts="t1")  # dd 16% > 15% trip
    assert gate.invalidated
    d = _filter(gate, target=1.0, equity=8_400.0)
    assert d.allowed
    assert d.target_signed_leverage == 0.0
    assert d.reason == "invalidation_force_flat"  # terminal gate, not the ladder


def test_daily_loss_halt_still_fires_with_ladder():
    gate = _gate(ladder=LADDER, max_daily_loss=100.0, max_drawdown_pct=0.5)
    gate.update_equity(10_000.0, ts="t0")
    gate.update_equity(9_850.0, ts="t1")  # loss 150 > 100 (dd 1.5%: no tier)
    assert gate.loss_halted
    d = _filter(gate, target=1.0, equity=9_850.0, qty=-0.2)
    assert d.allowed
    assert d.target_signed_leverage == 0.0
    assert d.reason == "daily_loss_force_flat"


# --- fail-closed config validation ---------------------------------------------


def test_malformed_ladders_raise():
    for bad in (
        [{"drawdown": 0.075, "leverage_multiplier": 0.5},  # non-ascending
         {"drawdown": 0.05, "leverage_multiplier": 0.0}],
        [{"drawdown": 0.05, "leverage_multiplier": 1.5}],   # multiplier > 1
        [{"drawdown": 0.05, "leverage_multiplier": -0.1}],  # multiplier < 0
        [{"drawdown": 0.0, "leverage_multiplier": 0.5}],    # drawdown 0
        [{"drawdown": 1.5, "leverage_multiplier": 0.5}],    # drawdown > 1
        [{"drawdown": 0.05}],                               # missing key
        [{"leverage_multiplier": 0.5}],                     # missing key
        {"drawdown": 0.05, "leverage_multiplier": 0.5},     # not a list
        "abc",                                              # not a list
        [{"drawdown": 0.05, "leverage_multiplier": 0.5},    # multiplier rises
         {"drawdown": 0.10, "leverage_multiplier": 0.8}],   # with depth
    ):
        with pytest.raises(ValueError, match="derisk_ladder"):
            _gate(ladder=bad)


def test_multiplier_must_be_non_increasing_with_depth():
    """A deeper rung may never re-leverage; equal multipliers are allowed."""
    with pytest.raises(ValueError, match="non-increasing"):
        _gate(
            ladder=[
                {"drawdown": 0.05, "leverage_multiplier": 0.2},
                {"drawdown": 0.10, "leverage_multiplier": 0.6},
            ]
        )
    gate = _gate(
        ladder=[
            {"drawdown": 0.05, "leverage_multiplier": 0.5},
            {"drawdown": 0.10, "leverage_multiplier": 0.5},
            {"drawdown": 0.15, "leverage_multiplier": 0.0},
        ]
    )
    assert gate.config.derisk_ladder == ((0.05, 0.5), (0.10, 0.5), (0.15, 0.0))
    # idempotent re-validation of normalized pairs enforces it too
    with pytest.raises(ValueError, match="non-increasing"):
        validate_derisk_ladder(((0.05, 0.5), (0.10, 0.7)))


def test_validation_via_shared_session_constructor():
    """Every engine builds its gate through SessionRisk.from_config — the
    shared path validates identically (parity by construction)."""
    cfg = {"mode": "paper", "risk": {"derisk_ladder": LADDER}}
    session = SessionRisk.from_config(cfg, start_equity=10_000.0)
    assert session.gate.config.derisk_ladder == ((0.05, 0.5), (0.075, 0.0))

    bad = {"mode": "paper", "risk": {"derisk_ladder": [{"drawdown": 0.05, "leverage_multiplier": 2.0}]}}
    with pytest.raises(ValueError, match="derisk_ladder"):
        SessionRisk.from_config(bad, start_equity=10_000.0)


def test_validate_derisk_ladder_idempotent():
    once = validate_derisk_ladder(LADDER)
    assert validate_derisk_ladder(once) == once


# --- engine integration: shared decision path, summary/journal evidence -------


def _synthetic_bars() -> pd.DataFrame:
    """60 up bars (+0.5/bar from 100), then 20 down bars (−1.0/bar)."""
    closes = [100.0 + 0.5 * i for i in range(60)]
    closes += [closes[-1] - 1.0 * j for j in range(1, 21)]
    rows = []
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    for i, c in enumerate(closes):
        rows.append(
            {
                "ts": str(base + pd.Timedelta(hours=i)),
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _ladder_cfg(with_ladder: bool = True):
    """Config with invalidation/daily-loss parked so only the ladder acts."""
    cfg = load_config()
    cfg["fee_bps"] = 0.0
    cfg["funding_rate_per_bar"] = 0.0
    cfg["risk"] = {
        **dict(cfg.get("risk") or {}),
        "max_drawdown_pct": 0.5,  # invalidation parked above the ladder
        "max_daily_loss": 1e9,    # daily-loss halt parked
        "max_order_notional": 1e9,
        "max_position_notional": 1e9,
    }
    if with_ladder:
        cfg["risk"]["derisk_ladder"] = [
            {"drawdown": 0.05, "leverage_multiplier": 0.5},
            {"drawdown": 0.10, "leverage_multiplier": 0.0},
        ]
    return cfg


def test_ladder_applies_in_paper_engine():
    bars = _synthetic_bars()
    kw = {"lookback": 48}
    base = run_paper_on_bars(
        bars, _ladder_cfg(with_ladder=False), symbol="BTCUSDT",
        strategy_name="tsmom", strategy_kwargs=kw,
    )
    # sanity: no-ladder run stays long through the decline
    base_qty = base["positions"]["BTCUSDT"]["qty"]
    assert base_qty != 0.0

    res = run_paper_on_bars(
        bars, _ladder_cfg(), symbol="BTCUSDT", strategy_name="tsmom", strategy_kwargs=kw
    )
    assert not res["invalidated"]
    derisk = res["derisk"]
    assert derisk == {"ladder_enabled": True, "tiers_triggered": 2, "min_multiplier": 0.0}
    mults = {d["derisk_multiplier"] for d in res["risk_decisions"]}
    assert 0.5 in mults and 0.0 in mults and 1.0 in mults
    # deepest tier flattened the book; the no-ladder run ends long
    assert res["positions"].get("BTCUSDT", {}).get("qty", 0.0) == 0.0
    # force-flat was reported as risk evidence
    assert any(r["reason"] == "derisk_block_new_risk" for r in res["risk_rejects"])


def test_summary_carries_derisk_block(tmp_path):
    summary = run_paper_session(
        config=_ladder_cfg(),
        force_source="fixture",
        force_strategy="dual_ma",
        output_dir=tmp_path,
    )
    assert summary["derisk"]["ladder_enabled"] is True
    assert summary["derisk"]["tiers_triggered"] >= 0
    journal = json.loads(Path(summary["journal_path"]).read_text(encoding="utf-8"))
    assert journal["derisk"] == summary["derisk"]
    assert "derisk_multiplier" in journal["risk_decisions_tail"][-1]


def test_summary_derisk_disabled_block(tmp_path):
    summary = run_paper_session(
        config=load_config(), force_source="fixture", force_strategy="dual_ma",
        output_dir=tmp_path,
    )
    assert summary["derisk"] == {
        "ladder_enabled": False,
        "tiers_triggered": 0,
        "min_multiplier": 1.0,
    }

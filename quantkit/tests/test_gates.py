"""Tests for quantkit.gates — R3 + quiz2 round3 main-path constants.

Contracts:
  - every gate is fail-closed: missing metric → NO-GO naming the gap
  - boundary values use >= / <= semantics as written in the thresholds
  - gate0 enforces turnover ≤30, AUM ≤5e6, friction in [0.002, 0.005]
  - cost_tier must match COST_TIERS and friction_cost when both present
  - gate4 calibration window is an interval check (60..120 days)
  - evaluate_gates runs ALL gates and aggregates verdict
"""

from __future__ import annotations

import json

import pytest

from quantkit.gates import (
    COST_TIERS,
    GATES_CONFIG,
    MAIN_PATH,
    check_gate,
    evaluate_gates,
    report_to_dict,
)


GOOD = {
    # gate0 main path (quiz2 r3)
    "turnover_annual": 25.0,
    "aum_scale": 4_500_000.0,
    "friction_cost": 0.002,
    "cost_tier": "low",
    # gate1
    "pc": 0.30, "isolation_days": 400,
    # gate2
    "dsr": 0.97, "pbo": 0.03, "zero_shot_r2": 0.05, "ebh_fdr": 0.02,
    # gate3
    "n_seeds": 20, "worst_path_alpha": 0.01, "pov": 0.08,
    # gate4
    "z_comp": 1.5, "basis_pct": 0.20, "calib_window": 90,
    # gate5
    "paper_months": 6, "paper_trades": 80, "deviation": 0.10, "win_rate": 0.56,
    # gate6
    "live_months": 3, "sharpe_decay": 0.12, "daily_orders": 15000,
}


def test_main_path_constants_match_r3_and_cost_tiers():
    assert MAIN_PATH["turnover_annual_max"] == 30.0
    assert MAIN_PATH["aum_scale_max"] == 5_000_000.0
    assert MAIN_PATH["friction_cost_min"] == COST_TIERS["low"] == 0.002
    assert MAIN_PATH["friction_cost_max"] == COST_TIERS["high"] == 0.005
    assert GATES_CONFIG["gate2_statistics"]["pbo"] == 0.05
    assert GATES_CONFIG["gate2_statistics"]["dsr"] == 0.95


def test_all_gates_pass_on_good_metrics():
    out = evaluate_gates(GOOD)
    assert out["verdict"] == "GO"
    assert all(r.passed for r in out["gates"])
    assert len(out["gates"]) == 7


def test_evaluate_runs_all_gates_and_aggregates():
    bad = dict(GOOD, dsr=0.50, paper_trades=10)
    out = evaluate_gates(bad)
    assert out["verdict"] == "NO-GO"
    failed = {r.gate for r in out["gates"] if not r.passed}
    assert failed == {"gate2_statistics", "gate5_paper"}


def test_pbo_above_005_fails_closed():
    bad = dict(GOOD, pbo=0.06)
    out = evaluate_gates(bad)
    assert out["verdict"] == "NO-GO"
    g2 = next(r for r in out["gates"] if r.gate == "gate2_statistics")
    assert not g2.passed
    assert any("PBO" in f for f in g2.failures)


def test_dsr_below_095_fails_closed():
    bad = dict(GOOD, dsr=0.94)
    r = check_gate("gate2_statistics", bad)
    assert not r.passed
    assert any("DSR" in f for f in r.failures)


@pytest.mark.parametrize("missing_key, gate", [
    ("turnover_annual", "gate0_main_path"),
    ("aum_scale", "gate0_main_path"),
    ("friction_cost", "gate0_main_path"),
    ("pc", "gate1_contamination"),
    ("dsr", "gate2_statistics"),
    ("pbo", "gate2_statistics"),
    ("n_seeds", "gate3_capacity"),
    ("z_comp", "gate4_crowding"),
    ("calib_window", "gate4_crowding"),
    ("paper_trades", "gate5_paper"),
    ("daily_orders", "gate6_live"),
])
def test_missing_metric_fails_closed(missing_key, gate):
    m = {k: v for k, v in GOOD.items() if k != missing_key}
    r = check_gate(gate, m)
    assert not r.passed
    assert missing_key in r.missing


def test_main_path_turnover_and_aum_limits():
    base = {
        "turnover_annual": 30.0,
        "aum_scale": 5_000_000.0,
        "friction_cost": 0.002,
    }
    assert check_gate("gate0_main_path", base).passed
    assert not check_gate(
        "gate0_main_path", {**base, "turnover_annual": 30.01}
    ).passed
    assert not check_gate(
        "gate0_main_path", {**base, "aum_scale": 5_000_000.01}
    ).passed


def test_friction_cost_interval_and_cost_tier_alignment():
    base = {
        "turnover_annual": 20.0,
        "aum_scale": 1_000_000.0,
        "friction_cost": 0.004,
        "cost_tier": "mid",
    }
    assert check_gate("gate0_main_path", base).passed
    # out of band friction
    assert not check_gate(
        "gate0_main_path", {**base, "friction_cost": 0.001, "cost_tier": "low"}
    ).passed
    assert not check_gate(
        "gate0_main_path", {**base, "friction_cost": 0.01}
    ).passed
    # tier / friction mismatch
    r = check_gate(
        "gate0_main_path",
        {**base, "friction_cost": 0.002, "cost_tier": "high"},
    )
    assert not r.passed
    assert any("不一致" in f or "cost_tier" in f for f in r.failures)
    # unknown tier
    r2 = check_gate(
        "gate0_main_path",
        {**base, "cost_tier": "ultra"},
    )
    assert not r2.passed


def test_boundary_values_are_inclusive():
    c2 = GATES_CONFIG["gate2_statistics"]
    m = {"dsr": c2["dsr"], "pbo": c2["pbo"],
         "zero_shot_r2": c2["zero_shot_r2"], "ebh_fdr": c2["ebh_fdr"]}
    assert check_gate("gate2_statistics", m).passed


def test_gate4_calib_window_interval():
    base = {"z_comp": 1.0, "basis_pct": 0.5}
    assert check_gate("gate4_crowding", {**base, "calib_window": 60}).passed
    assert check_gate("gate4_crowding", {**base, "calib_window": 120}).passed
    assert not check_gate("gate4_crowding", {**base, "calib_window": 30}).passed
    assert not check_gate("gate4_crowding", {**base, "calib_window": 252}).passed


def test_gate1_pc_redline():
    ok = {"pc": 0.50, "isolation_days": 365}
    assert check_gate("gate1_contamination", ok).passed
    assert not check_gate("gate1_contamination", {**ok, "pc": 0.51}).passed


def test_gate6_compliance_warn_line():
    m = {"live_months": 3, "sharpe_decay": 0.10, "daily_orders": 18000}
    assert check_gate("gate6_live", m).passed
    assert not check_gate("gate6_live", {**m, "daily_orders": 20000}).passed


def test_unknown_gate_raises():
    with pytest.raises(KeyError):
        check_gate("gate9_nope", GOOD)


def test_str_shows_failures():
    r = check_gate("gate2_statistics", {"dsr": 0.1})
    s = str(r)
    assert "NO-GO" in s and "DSR" in s


def test_empty_metrics_all_fail_closed():
    out = evaluate_gates({})
    assert out["verdict"] == "NO-GO"
    assert all(not r.passed for r in out["gates"])
    assert any(r.missing for r in out["gates"])


def test_report_to_dict_go_shape_and_json_roundtrip():
    d = report_to_dict(evaluate_gates(GOOD))
    assert d["verdict"] == "GO"
    assert d["missing"] == []
    assert len(d["gates"]) == 7
    assert all(g["passed"] and not g["missing"] and not g["failures"] for g in d["gates"])
    # gate0 echoes the measured metrics incl. resolved cost tier rate
    g0 = next(g for g in d["gates"] if g["gate"] == "gate0_main_path")
    assert g0["metrics"]["friction_cost"] == pytest.approx(0.002)
    assert g0["metrics"]["cost_tier_rate"] == pytest.approx(COST_TIERS["low"])
    # JSON round-trip must be lossless (gate_report.json persistence path)
    assert json.loads(json.dumps(d, ensure_ascii=False)) == d


def test_report_to_dict_nogo_names_missing_and_failures():
    bad = {k: v for k, v in GOOD.items() if k not in ("dsr", "pbo")}
    bad["paper_trades"] = 10
    d = report_to_dict(evaluate_gates(bad))
    assert d["verdict"] == "NO-GO"
    assert d["missing"] == ["dsr", "pbo"]
    g2 = next(g for g in d["gates"] if g["gate"] == "gate2_statistics")
    assert not g2["passed"]
    assert g2["missing"] == ["dsr", "pbo"]
    g5 = next(g for g in d["gates"] if g["gate"] == "gate5_paper")
    assert any("MinTRL" in f for f in g5["failures"])
    assert json.loads(json.dumps(d, ensure_ascii=False)) == d

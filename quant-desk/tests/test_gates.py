"""quant_desk gate-wiring tests (scripts/_gates.py + _common.resolve_fee_bps).

Contracts:
  - once config declares cost_tier, the paper single-side fee = tier/2; the legacy "万五" (0.05%) default is banned; legacy fee_bps still readable
  - collect_metrics only emits pipeline-measurable keys (turnover/aum/friction/cost_tier); it never fabricates evidence
  - missing fills/equity or too-short span → turnover unmeasured → gate0 fail-closed
  - run_gate_report writes gate_report.json; measured keys take priority over same-named external-evidence keys
  - with external evidence filled in, the verdict can be GO (fixtures/gate_metrics_go.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _common import resolve_fee_bps
from _gates import collect_metrics, load_extra_metrics, run_gate_report

FIXTURE_GO = Path(__file__).parent / "fixtures" / "gate_metrics_go.json"


def _write_artifacts(out_dir: Path, *, n_fills: float = 40.0, span_days: int = 365) -> None:
    """Synthetic fills + equity: turnover = n_fills×1e5 / 1e6 / (span/365.25)."""
    rows = ["ts,symbol,side,qty,price,fee,note"]
    for i in range(int(n_fills)):
        rows.append(f"2024-01-{i % 28 + 1:02d},AAPL,BUY,500,200.0,10.0,rebalance")
    (out_dir / "fills.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    eq = ["date,equity", f"2023-01-01,1000000.0", f"2023-01-02,1000000.0"]
    import datetime as _dt

    end = _dt.date(2023, 1, 1) + _dt.timedelta(days=span_days)
    eq.append(f"{end.isoformat()},1000000.0")
    (out_dir / "lifecycle_equity.csv").write_text("\n".join(eq) + "\n", encoding="utf-8")


CFG = {"initial_cash": 1_000_000, "cost_tier": "mid"}


def test_resolve_fee_bps_cost_tier_half_per_side():
    fee, tier = resolve_fee_bps(CFG)
    assert tier == "mid"
    assert fee == pytest.approx(20.0)  # 0.4% double-sided → single-sided 20 bps


def test_resolve_fee_bps_legacy_fallback_and_unknown_tier():
    assert resolve_fee_bps({"fee_bps": 7.5}) == (7.5, None)
    with pytest.raises(ValueError, match="unknown cost_tier"):
        resolve_fee_bps({"cost_tier": "ultra"})


def test_collect_metrics_measured_keys(tmp_path):
    _write_artifacts(tmp_path)  # turnover = 40×1e5/1e6/≈1y ≈ 4.0
    m = collect_metrics(CFG, tmp_path)
    assert m["aum_scale"] == 1_000_000.0
    assert m["cost_tier"] == "mid"
    assert m["friction_cost"] == pytest.approx(0.004)
    assert m["turnover_annual"] == pytest.approx(4.0, rel=0.02)
    # Do not forge evidence that cannot be obtained locally.
    assert "pbo" not in m and "paper_months" not in m


def test_collect_metrics_legacy_fee_maps_to_two_sided(tmp_path):
    _write_artifacts(tmp_path)
    m = collect_metrics({"initial_cash": 1e6, "fee_bps": 5.0}, tmp_path)
    assert "cost_tier" not in m
    assert m["friction_cost"] == pytest.approx(0.001)  # 0.05% single-sided → double-sided 0.1% (lower than bandwidth, gate0 will reject)


def test_turnover_missing_without_artifacts_or_short_span(tmp_path):
    assert "turnover_annual" not in collect_metrics(CFG, tmp_path)  # empty directory
    _write_artifacts(tmp_path, span_days=7)  # span too short → missing data rather than giving noisy numbers
    assert "turnover_annual" not in collect_metrics(CFG, tmp_path)


def test_run_gate_report_fail_closed_and_persists(tmp_path):
    _write_artifacts(tmp_path)
    report = run_gate_report(CFG, tmp_path)
    assert report["verdict"] == "NO-GO"
    assert {"dsr", "pbo", "paper_months"} <= set(report["missing"])
    path = tmp_path / "gate_report.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["verdict"] == "NO-GO"
    assert saved["metrics"]["turnover_annual"] == pytest.approx(4.0, rel=0.02)
    assert saved["inputs"]["extra_metrics"] is None
    assert "turnover_annual" in saved["inputs"]["measured_keys"]


def test_run_gate_report_go_with_extra_evidence(tmp_path):
    _write_artifacts(tmp_path)
    report = run_gate_report(CFG, tmp_path, extra_path=FIXTURE_GO)
    assert report["verdict"] == "GO"
    assert report["missing"] == []
    assert report["inputs"]["extra_metrics"] == str(FIXTURE_GO)


def test_measured_keys_win_over_extra(tmp_path):
    _write_artifacts(tmp_path)
    extra = tmp_path / "extra.json"
    extra.write_text(
        json.dumps({"friction_cost": 0.002, "cost_tier": "low", "aum_scale": 1.0}),
        encoding="utf-8",
    )
    report = run_gate_report(CFG, tmp_path, extra_path=extra)
    # Empirical test: mid/0.004 and fills deduction are not softened by stale evidence.
    assert report["metrics"]["friction_cost"] == pytest.approx(0.004)
    assert report["metrics"]["cost_tier"] == "mid"
    assert report["metrics"]["aum_scale"] == 1_000_000.0


def test_explicit_extra_path_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_gate_report(CFG, tmp_path, extra_path=tmp_path / "nope.json")


def test_load_extra_metrics_rejects_non_object(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_extra_metrics(bad)

"""Parity report tests — skipped when the optional nautilus dep is absent."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("nautilus_trader")

from galahad_futures.data import load_bars  # noqa: E402
from galahad_futures.engine import load_config  # noqa: E402
from scripts.run_parity import build_parity_report  # noqa: E402


def _inputs():
    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    cfg["engine_nautilus"] = {"price_precision": 6, "size_precision": 6}
    bars, source_used, data_note = load_bars(
        source="fixture",
        fixture_path="data/fixtures/btcusdt_1h.csv",
        rest_url=None,
        rest_timeout=12.0,
        project_root=root,
        symbol="BTCUSDT",
        interval="1h",
        limit=500,
        rest_url_template=None,
    )
    return cfg, {
        "bars": bars.iloc[-120:].reset_index(drop=True),
        "symbol": "BTCUSDT",
        "interval": "1h",
        "source_used": source_used,
        "sample_kind": "fixture",
        "data_note": data_note,
    }


def test_parity_report_writes_stable_pointer():
    import json as _json
    import subprocess as _sp
    from galahad_futures.engine import load_config as _lc

    cfg = _lc()
    root = Path(__file__).resolve().parent.parent
    out = root / "output" / "test_parity_stable"
    r = _sp.run(
        [
            sys.executable,
            str(root / "scripts" / "run_parity.py"),
            "--source",
            "fixture",
            "--strategy",
            "tsmom",
            "--parity-precision",
            "6",
            "--output-dir",
            str(out),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    report = _json.loads(r.stdout)
    stable = Path(report["stable_path"])
    assert stable.is_file()
    assert _json.loads(stable.read_text(encoding="utf-8"))["schema"] == "galahad.parity.v1"


def test_parity_report_structure_and_identity():
    cfg, inputs = _inputs()
    report = build_parity_report(cfg, inputs, force_strategy="tsmom", force_lookback=48)
    assert report["schema"] == "galahad.parity.v1"
    assert report["inputs"]["bars"] == 120
    assert set(report["engines"]) == {"paper", "nautilus"}
    fills = report["diffs"]["fills"]
    fill_gap = abs(fills["n_fills_paper"] - fills["n_fills_nautilus"])
    assert fill_gap <= 2
    # Any fill-count gap must be explained by a detected boundary crossing.
    if fill_gap > 0:
        assert report["boundary_crossing"]["detected"], report["boundary_crossing"]
    assert report["engines"]["nautilus"]["orders_submitted"] == report["engines"]["nautilus"]["orders_filled"]
    assert report["diffs"]["equity"]["overlap"] == 120
    assert report["diffs"]["decisions"]["n_decisions"] == 120
    assert report["engines"]["paper"]["decision_phase_final"]
    assert report["engines"]["nautilus"]["decision_phase_final"]
    assert len(report["known_divergences"]) >= 1
    # Sensitivity sections present with the documented threshold bands.
    assert len(report["sensitivity"]["max_drawdown_pct"]) == 5
    assert len(report["sensitivity"]["max_daily_loss"]) == 5


def test_parity_report_divergences_bounded():
    """Mechanics-level divergences are small on the fixture session.

    The daily-loss floor on this fixture sits near the session's actual
    adverse excursion, so the engines may legitimately diverge by a bar
    or two of fills around the boundary; fill prices must stay tightly
    aligned regardless, and the crossing must be flagged.
    """
    cfg, inputs = _inputs()
    report = build_parity_report(cfg, inputs, force_strategy="tsmom", force_lookback=48)
    fills = report["diffs"]["fills"]
    assert abs(fills["n_fills_paper"] - fills["n_fills_nautilus"]) <= 2
    assert fills["max_price_diff"] < 1e-3
    # Decision drift, not execution mechanics, would show as many field
    # diffs; execution-driven drift shows as a minority.
    dec = report["diffs"]["decisions"]
    assert dec["decision_field_diffs"] <= dec["decision_fields_compared"] // 2

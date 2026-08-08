"""Perception unit tests — fixture offline path; no network required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.perception import (
    build_snapshot,
    run_perception,
    write_snapshot,
)


def test_offline_snapshot_has_ts_and_price(tmp_path):
    fixture = ROOT / "data" / "fixtures" / "btcusdt_1h.csv"
    assert fixture.is_file()
    snap = build_snapshot(force_offline=True, fixture_path=fixture, reports_root=None)
    assert snap.ts.endswith("Z") or "T" in snap.ts
    assert snap.symbols
    assert all(isinstance(v, float) and v > 0 for v in snap.symbols.values())
    assert snap.source == "fixture"
    paths = write_snapshot(snap, [tmp_path / "p.json"])
    data = json.loads(Path(paths[0]).read_text())
    assert data["ts"] == snap.ts
    assert data["symbols"]


def test_run_perception_offline_writes_last(tmp_path, monkeypatch):
    # write into project output is fine; also verify return shape
    summary = run_perception(project_root=ROOT, ops_state=tmp_path, force_offline=True)
    assert summary["status"] in ("ok", "degraded", "fetch_failed")
    assert summary["n_symbols"] >= 1
    assert (tmp_path / "last_perception.json").is_file()
    last = json.loads((tmp_path / "last_perception.json").read_text())
    assert last.get("ts")
    assert last.get("symbols") or last.get("fetch_error")


def test_fetch_error_field_when_no_source(tmp_path):
    snap = build_snapshot(
        force_offline=True,
        fixture_path=tmp_path / "missing.csv",
        reports_root=None,
    )
    assert snap.fetch_error
    assert snap.status == "fetch_failed"
    assert snap.symbols == {}

"""Engine / data tests — fixture path drives real paper session."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.data import load_bars, load_fixture, write_synthetic_fixture
from galahad_futures.engine import run_paper_session


@pytest.fixture(scope="module")
def fixture_csv(tmp_path_module_factory=None):
    path = ROOT / "data" / "fixtures" / "btcusdt_1h.csv"
    if not path.is_file():
        write_synthetic_fixture(path, n=120, start_price=40_000.0, seed=42)
    return path


def test_fixture_loads_ohlcv(fixture_csv):
    bars = load_fixture(fixture_csv)
    assert len(bars) >= 50
    for c in ("ts", "open", "high", "low", "close", "volume"):
        assert c in bars.columns
    assert (bars["close"] > 0).all()


def test_load_bars_fixture_source(fixture_csv):
    bars, src, note = load_bars(
        source="fixture",
        fixture_path=fixture_csv,
        project_root=ROOT,
    )
    assert src == "fixture"
    assert note is None
    assert len(bars) > 0


def test_paper_session_fixture_produces_fills_and_equity(fixture_csv, tmp_path):
    summary = run_paper_session(
        force_source="fixture",
        force_strategy="dual_ma",
        output_dir=tmp_path,
    )
    assert summary["status"] in ("ok", "ok_invalidated", "no-trade but risk-idle OK")
    assert summary["equity_curve_len"] >= summary["bars"] or summary["equity_curve_len"] >= 1
    assert isinstance(summary["final_equity"], (int, float))
    assert summary["bars"] > 0
    # Synthetic dual-trend fixture + dual MA → at least one fill
    assert summary["n_fills"] >= 1, summary
    journal = Path(summary["journal_path"])
    assert journal.is_file()
    data = json.loads(journal.read_text(encoding="utf-8"))
    assert len(data["fills"]) == summary["n_fills"]
    assert len(data["equity_curve"]) == summary["equity_curve_len"]

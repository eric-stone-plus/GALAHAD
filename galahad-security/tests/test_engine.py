"""Engine / contract tests — fixture sessions drive the real paper path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_security.data import load_bars
from galahad_security.engine import load_config, run_paper_on_bars, run_paper_session

CONTRACT_KEYS = (
    "run_id",
    "mode",
    "engine",
    "engine_version",
    "strategy",
    "strategy_kwargs",
    "symbols",
    "interval",
    "bars",
    "source_used",
    "sample_kind",
    "n_fills",
    "n_risk_rejects",
    "invalidated",
    "invalidation_reason",
    "peak_equity",
    "max_drawdown",
    "initial_equity",
    "final_equity",
    "equity_curve_len",
    "status",
    "journal_path",
    "tca",
    "derisk",
)


def test_fixture_session_summary_contract(tmp_path):
    summary = run_paper_session(
        force_source="fixture", force_strategy="dual_ma", output_dir=tmp_path
    )
    for key in CONTRACT_KEYS:
        assert key in summary, f"missing contract field {key}"
    assert summary["mode"] == "paper"
    assert summary["engine"] == "paper"
    assert summary["engine_version"] == "galahad-security.book.v1"
    assert summary["symbols"] == ["AAPL", "MSFT"]
    assert summary["interval"] == "1d"
    assert summary["status"] in ("ok", "ok_invalidated", "no-trade but risk-idle OK")
    assert summary["n_fills"] >= 1
    assert isinstance(summary["final_equity"], (int, float))
    # never any liquidation/funding fields in a cash account
    assert "liquidated" not in summary
    assert "total_funding" not in summary
    # tca / derisk blocks
    assert summary["tca"]["n_fills"] == summary["n_fills"]
    assert summary["tca"]["implementation_shortfall_usdt"] == 0.0  # costs off
    assert summary["derisk"]["ladder_enabled"] is False
    # journal on disk
    journal = Path(summary["journal_path"])
    assert journal.is_file()
    data = json.loads(journal.read_text(encoding="utf-8"))
    assert len(data["fills"]) == summary["n_fills"]
    assert len(data["equity_curve"]) == summary["equity_curve_len"]
    assert "derisk_multiplier" in data["risk_decisions_tail"][-1]


def test_no_trade_status_when_strategy_flat(tmp_path):
    cfg = load_config()
    cfg["strategy"] = {"name": "dual_ma", "target_weight": 0.0}
    summary = run_paper_session(
        config=cfg, force_source="fixture", output_dir=tmp_path
    )
    assert summary["status"] == "no-trade but risk-idle OK"
    assert summary["n_fills"] == 0


def test_symbols_override(tmp_path):
    summary = run_paper_session(
        force_source="fixture", force_symbols=["AAPL"], output_dir=tmp_path
    )
    assert summary["symbols"] == ["AAPL"]


def test_unknown_engine_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown engine"):
        run_paper_session(
            config=load_config(), force_source="fixture",
            engine="not-an-engine", output_dir=tmp_path,
        )


def test_disjoint_calendars_fail_closed():
    def bars(dates):
        return pd.DataFrame(
            {
                "ts": dates,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            }
        )

    a = bars(["2026-01-05", "2026-01-06"])
    b = bars(["2026-02-02", "2026-02-03"])
    with pytest.raises(ValueError, match="no common trading days"):
        run_paper_on_bars({"AAPL": a, "MSFT": b}, load_config(), symbols=["AAPL", "MSFT"])


def test_load_bars_fixture_and_cache_tiers(tmp_path):
    bars, src, note = load_bars(
        source="fixture", project_root=ROOT, symbols=["AAPL", "MSFT"], limit=250
    )
    assert src == "fixture"
    assert set(bars) == {"AAPL", "MSFT"}
    assert all(len(df) == 250 for df in bars.values())
    # cache source without cache files fails closed
    with pytest.raises(FileNotFoundError):
        load_bars(source="cache", project_root=tmp_path, symbols=["AAPL"], limit=250)
    # venue source refuses in the offline data module
    with pytest.raises(RuntimeError, match="venue"):
        load_bars(source="venue", project_root=ROOT, symbols=["AAPL"], limit=250)


def test_fixture_regeneration_is_deterministic(tmp_path):
    a, _, _ = load_bars(source="fixture", project_root=tmp_path, symbols=["AAPL"], limit=250)
    b, _, _ = load_bars(source="fixture", project_root=tmp_path, symbols=["AAPL"], limit=250)
    pd.testing.assert_frame_equal(a["AAPL"], b["AAPL"])

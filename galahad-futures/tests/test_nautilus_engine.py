"""NautilusTrader backend tests — skipped when the optional dep is absent."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

nautilus = pytest.importorskip("nautilus_trader")

from galahad_futures.data import load_bars  # noqa: E402
from galahad_futures.engine import load_config, run_paper_session  # noqa: E402
from galahad_futures.nautilus_backend import ENGINE_NAME, run_nautilus_on_bars  # noqa: E402


def _fixture_bars_and_cfg():
    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    bars, _, _ = load_bars(
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
    return bars.iloc[-120:].reset_index(drop=True), cfg


def test_nautilus_result_shape_matches_reference():
    bars, cfg = _fixture_bars_and_cfg()
    result = run_nautilus_on_bars(
        bars, cfg, symbol="BTCUSDT", strategy_name="tsmom", strategy_kwargs={"lookback": 48}
    )
    for key in (
        "strategy",
        "strategy_kwargs",
        "symbol",
        "bars",
        "n_fills",
        "n_risk_rejects",
        "liquidated",
        "invalidated",
        "peak_equity",
        "max_drawdown",
        "initial_equity",
        "final_equity",
        "equity_curve",
        "total_funding",
        "n_funding_events",
        "fills",
        "risk_rejects",
        "positions",
    ):
        assert key in result, f"missing key {key}"
    assert result["engine"] == ENGINE_NAME
    assert result["bars"] == 120
    assert result["equity_curve_len"] == 120
    assert result["n_fills"] > 0
    assert result["n_funding_events"] == 120
    assert result["orders_submitted"] == result["orders_filled"]


def test_nautilus_session_via_run_paper_session():
    bars, cfg = _fixture_bars_and_cfg()
    cfg["engine_nautilus"] = {"price_precision": 6, "size_precision": 6}
    summary = run_paper_session(
        config=cfg,
        force_source="fixture",
        output_dir=ROOT / "output" / "test_nautilus",
        force_strategy="tsmom",
        engine="nautilus",
    )
    assert summary["engine"] == "nautilus"
    assert summary["engine_version"].startswith("nautilus_trader-")
    assert summary["status"] in ("ok", "ok_invalidated", "no-trade but risk-idle OK")
    assert summary["journal_path"].endswith(".json")


def test_cli_engine_flag_and_json_contract():
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "run_paper.py"
    venv_py = sys.executable
    out = subprocess.run(
        [str(venv_py), str(script), "--source", "fixture", "--engine", "nautilus", "--json"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    summary = json.loads(out.stdout)
    assert summary["engine"] == "nautilus"

    # Default remains the reference paper book.
    out2 = subprocess.run(
        [str(venv_py), str(script), "--source", "fixture", "--json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out2.returncode == 0, out2.stderr[-2000:]
    assert json.loads(out2.stdout)["engine"] == "paper"


def test_unknown_engine_raises():
    cfg = load_config()
    with pytest.raises(ValueError, match="unknown engine"):
        run_paper_session(config=cfg, force_source="fixture", engine="not-an-engine")

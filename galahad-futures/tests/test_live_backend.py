"""nautilus_live (Binance USDT-M futures TESTNET) backend tests.

All unit tests in this module run WITHOUT nautilus_trader installed and
WITHOUT network access: the TradingNode wiring is behind lazy imports,
and the decision/reconciliation logic is factored into pure functions.
The real-testnet integration skeleton at the bottom is skipped unless
GALAHAD_TESTNET_IT=1.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures import live_backend
from galahad_futures.decision import SessionRisk
from galahad_futures.engine import load_config, run_paper_session
from galahad_futures.report import build_summary
from galahad_futures.risk import RiskConfig, RiskGate


def _filter(gate: RiskGate):
    return gate.filter_target(
        symbol="BTCUSDT",
        target_signed_leverage=1.0,
        mark=100.0,
        equity=10_000.0,
        current_qty=0.0,
        leverage=2.0,
        ts="t0",
    )


# --- gate branch: testnet allowed/refused, live unconditionally blocked ----


def test_testnet_allowed_when_enabled_and_kill_switch_off():
    gate = RiskGate(
        config=RiskConfig(mode="testnet", kill_switch=False, enable_testnet=True),
        day_start_equity=10_000.0,
    )
    d = _filter(gate)
    assert d.allowed
    # default max_order_notional clips the delta — exposure stays positive
    assert d.target_signed_leverage > 0.0


def test_testnet_blocked_when_enable_testnet_false():
    gate = RiskGate(
        config=RiskConfig(mode="testnet", kill_switch=False, enable_testnet=False),
        day_start_equity=10_000.0,
    )
    d = _filter(gate)
    assert not d.allowed
    assert "live_blocked" in d.reason


def test_testnet_blocked_by_kill_switch():
    gate = RiskGate(
        config=RiskConfig(mode="testnet", kill_switch=True, enable_testnet=True),
        day_start_equity=10_000.0,
    )
    d = _filter(gate)
    assert not d.allowed
    assert "live_blocked" in d.reason


def test_live_blocked_unconditionally():
    """mode=live (mainnet) stays LIVE_BLOCKED even with every flag enabled."""
    for kill_switch in (True, False):
        gate = RiskGate(
            config=RiskConfig(
                mode="live",
                kill_switch=kill_switch,
                enable_live=True,
                enable_testnet=True,
            ),
            day_start_equity=10_000.0,
        )
        assert gate.live_blocked()
        assert not _filter(gate).allowed


def test_paper_unaffected_by_testnet_flags():
    gate = RiskGate(
        config=RiskConfig(mode="paper", kill_switch=True, enable_testnet=False),
        day_start_equity=10_000.0,
    )
    assert _filter(gate).allowed


def test_session_risk_reads_enable_testnet_from_config():
    cfg = {"mode": "testnet", "risk": {"kill_switch": False, "enable_testnet": True}}
    session = SessionRisk.from_config(cfg, start_equity=10_000.0)
    assert not session.gate.live_blocked()
    assert session.phase() == "ACTIVE"

    cfg = {"mode": "testnet", "risk": {"kill_switch": False, "enable_testnet": False}}
    session = SessionRisk.from_config(cfg, start_equity=10_000.0)
    assert session.gate.live_blocked()
    assert session.phase() == "LIVE_BLOCKED"


# --- credential discipline -------------------------------------------------


def test_credentials_missing_both_raises():
    with pytest.raises(RuntimeError, match="BINANCE_TESTNET_API_KEY"):
        live_backend.testnet_credentials({})


def test_credentials_missing_one_raises_and_names_it():
    with pytest.raises(RuntimeError, match="BINANCE_TESTNET_API_SECRET"):
        live_backend.testnet_credentials({"BINANCE_TESTNET_API_KEY": "k"})
    with pytest.raises(RuntimeError, match="BINANCE_TESTNET_API_KEY"):
        live_backend.testnet_credentials({"BINANCE_TESTNET_API_SECRET": "s"})


def test_credentials_never_read_mainnet_names():
    env = {"BINANCE_API_KEY": "mainnet-key", "BINANCE_API_SECRET": "mainnet-secret"}
    with pytest.raises(RuntimeError, match="BINANCE_TESTNET_API_KEY"):
        live_backend.testnet_credentials(env)


def test_credentials_ok():
    key, secret = live_backend.testnet_credentials(
        {"BINANCE_TESTNET_API_KEY": " k ", "BINANCE_TESTNET_API_SECRET": "s"}
    )
    assert (key, secret) == ("k", "s")


# --- pure helpers ------------------------------------------------------------


def test_order_delta_for_decision():
    d = live_backend.order_delta_for_decision(
        target_signed_leverage=1.0, equity=10_000.0, mark=100.0, current_qty=0.0
    )
    assert d == pytest.approx(100.0)
    # dust delta dropped
    assert (
        live_backend.order_delta_for_decision(
            target_signed_leverage=0.0, equity=10_000.0, mark=100.0, current_qty=1e-9
        )
        == 0.0
    )
    # invalid inputs fail closed to no order
    assert (
        live_backend.order_delta_for_decision(
            target_signed_leverage=1.0, equity=0.0, mark=100.0, current_qty=0.0
        )
        == 0.0
    )


def test_quantize_delta_rounds_toward_zero():
    assert live_backend.quantize_delta(0.1239, 3) == pytest.approx(0.123)
    assert live_backend.quantize_delta(-0.1239, 3) == pytest.approx(-0.123)
    assert live_backend.quantize_delta(0.0009, 3) == 0.0
    # never rounds up into more exposure than the gate passed
    assert live_backend.quantize_delta(0.12399999, 3) <= 0.12399999


def test_infer_external_flatten():
    assert live_backend.infer_external_flatten(
        expected_qty=0.5, venue_qty=0.0, open_orders=0
    )
    assert not live_backend.infer_external_flatten(
        expected_qty=0.5, venue_qty=0.0, open_orders=1
    )
    assert not live_backend.infer_external_flatten(
        expected_qty=0.0, venue_qty=0.0, open_orders=0
    )
    assert not live_backend.infer_external_flatten(
        expected_qty=0.5, venue_qty=0.5, open_orders=0
    )


def test_bar_type_str():
    assert (
        live_backend.bar_type_str("BTCUSDT", "1h")
        == "BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL"
    )
    assert (
        live_backend.bar_type_str("ETHUSDT", "5m")
        == "ETHUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL"
    )
    assert (
        live_backend.bar_type_str("BTCUSDT", "1d")
        == "BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL"
    )
    for bad in ("1w", "0h", "x", "1H2"):
        with pytest.raises(ValueError):
            live_backend.bar_type_str("BTCUSDT", bad)


def test_testnet_max_minutes():
    assert live_backend.testnet_max_minutes({}) == 30.0
    assert live_backend.testnet_max_minutes({"testnet": {"max_minutes": 5}}) == 5.0
    with pytest.raises(ValueError):
        live_backend.testnet_max_minutes({"testnet": {"max_minutes": 0}})


def test_precheck_gate():
    with pytest.raises(RuntimeError, match="testnet gate closed"):
        live_backend.precheck_gate(
            {"mode": "testnet", "risk": {"kill_switch": True, "enable_testnet": True}}
        )
    with pytest.raises(RuntimeError, match="testnet gate closed"):
        live_backend.precheck_gate(
            {"mode": "testnet", "risk": {"kill_switch": False, "enable_testnet": False}}
        )
    live_backend.precheck_gate(
        {"mode": "testnet", "risk": {"kill_switch": False, "enable_testnet": True}}
    )


# --- reconciliation logic (fakes) -------------------------------------------


def test_reconcile_clean():
    r = live_backend.reconcile_execution(
        orders_submitted=3, orders_filled=3, expected_qty=0.5, venue_qty=0.5
    )
    assert r == {"orders_submitted": 3, "orders_filled": 3, "position_mismatch": False}


def test_reconcile_qty_divergence_flags_mismatch():
    r = live_backend.reconcile_execution(
        orders_submitted=3, orders_filled=2, expected_qty=0.5, venue_qty=0.2
    )
    assert r["position_mismatch"] is True
    assert r["orders_filled"] == 2


def test_reconcile_unknown_venue_state_fails_closed():
    r = live_backend.reconcile_execution(
        orders_submitted=0, orders_filled=0, expected_qty=0.0, venue_qty=None
    )
    assert r["position_mismatch"] is True


def test_reconcile_flat_vs_flat_is_clean():
    r = live_backend.reconcile_execution(
        orders_submitted=0, orders_filled=0, expected_qty=0.0, venue_qty=0.0
    )
    assert r["position_mismatch"] is False


# --- summary contract shape --------------------------------------------------


def _fake_live_result(cfg):
    session = SessionRisk.from_config(cfg, start_equity=10_000.0)
    return live_backend.build_live_result(
        cfg=cfg,
        symbol="BTCUSDT",
        strategy_name="tsmom",
        strategy_kwargs={"lookback": 48},
        session=session,
        fills=[],
        equity_curve=[{"ts": "2026-01-01T00:00:00+00:00", "equity": 10_000.0}],
        account_curve=[{"ts": "2026-01-01T00:00:00+00:00", "equity": 10_000.0}],
        funding_events=[],
        liquidation_events=[],
        positions={},
        orders_submitted=2,
        orders_filled=2,
        warmup_bars=400,
        live_bars=3,
        initial_equity=10_000.0,
        final_equity=10_001.0,
        expected_qty=0.25,
        venue_qty=0.25,
        session_seconds=65.0,
        max_minutes=30.0,
        equity_source="venue",
    )


def test_live_summary_contract_shape(tmp_path):
    cfg = {**load_config(), "mode": "testnet"}
    result = _fake_live_result(cfg)
    summary = build_summary(
        result,
        cfg=cfg,
        symbol="BTCUSDT",
        interval="1h",
        source_used="venue",
        sample_kind="venue",
        data_note=None,
        out_dir=tmp_path,
        engine=live_backend.ENGINE_NAME,
        engine_version=live_backend.ENGINE_VERSION,
        strategy_name="tsmom",
    )
    # New contract fields
    assert summary["mode"] == "testnet"
    assert summary["mode"] != "live"
    assert summary["engine"] == "nautilus_live"
    assert summary["engine_version"] == "nautilus_trader-1.231.0"
    assert summary["venue"] == "BINANCE"
    assert summary["reconciliation"] == {
        "orders_submitted": 2,
        "orders_filled": 2,
        "position_mismatch": False,
    }
    # Existing fields and semantics intact
    for key in (
        "run_id",
        "symbol",
        "interval",
        "status",
        "liquidated",
        "invalidated",
        "n_fills",
        "n_risk_rejects",
        "initial_equity",
        "final_equity",
        "equity_curve_len",
        "total_funding",
        "source_used",
        "sample_kind",
        "strategy",
    ):
        assert key in summary, f"missing contract field {key}"
    assert summary["liquidated"] is False
    assert summary["status"] == "no-trade but risk-idle OK"
    assert summary["final_equity"] == pytest.approx(10_001.0)


def test_paper_summary_does_not_gain_live_fields(tmp_path):
    """Existing consumers see a byte-compatible paper summary shape."""
    summary = run_paper_session(
        force_source="fixture", force_strategy="dual_ma", output_dir=tmp_path
    )
    assert "venue" not in summary
    assert "reconciliation" not in summary
    assert summary["mode"] == "paper"
    assert summary["engine"] == "paper"


# --- engine dispatch / CLI fail-closed (offline) -----------------------------


def test_dispatch_missing_credentials_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_TESTNET_API_SECRET", raising=False)
    cfg = load_config()
    with pytest.raises(RuntimeError, match="BINANCE_TESTNET_API_KEY"):
        run_paper_session(
            config=cfg,
            force_source="fixture",
            engine="nautilus_live",
            output_dir=tmp_path,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("nautilus_trader") is not None,
    reason="nautilus_trader installed — missing-dependency path not exercisable",
)
def test_dispatch_missing_nautilus_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "fake")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "fake")
    cfg = load_config()
    with pytest.raises(RuntimeError, match="nautilus_trader==1.231.0"):
        run_paper_session(
            config=cfg,
            force_source="fixture",
            engine="nautilus_live",
            output_dir=tmp_path,
        )


def test_cli_nautilus_live_without_credentials_fails_clean():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("BINANCE_") and k != "GALAHAD_TESTNET_IT"
    }
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_paper.py"),
            "--source",
            "fixture",
            "--engine",
            "nautilus_live",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert out.returncode != 0
    assert "BINANCE_TESTNET_API_KEY" in out.stderr
    assert "Traceback" not in out.stderr
    assert "error:" in out.stderr


def test_run_shadow_requires_env_gate():
    env = {k: v for k, v in os.environ.items() if k != "GALAHAD_TESTNET_IT"}
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_shadow.py"), "--source", "fixture"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert out.returncode != 0
    assert "GALAHAD_TESTNET_IT" in out.stderr


# --- real-testnet integration skeleton (env-gated) ---------------------------


@pytest.mark.skipif(
    os.environ.get("GALAHAD_TESTNET_IT") != "1",
    reason="real Binance futures testnet run disabled; set GALAHAD_TESTNET_IT=1 "
    "with BINANCE_TESTNET_API_KEY/BINANCE_TESTNET_API_SECRET and the "
    "nautilus extra installed",
)
def test_live_testnet_session_smoke(tmp_path):
    pytest.importorskip("nautilus_trader")
    cfg = load_config()
    cfg = {
        **cfg,
        "mode": "testnet",
        "interval": "1m",
        "bar_limit": 120,
        "testnet": {"max_minutes": 3},
    }
    cfg["risk"] = {
        **dict(cfg.get("risk") or {}),
        "kill_switch": False,
        "enable_testnet": True,
    }
    summary = run_paper_session(
        config=cfg,
        force_source="fixture",
        engine="nautilus_live",
        output_dir=tmp_path,
    )
    assert summary["mode"] == "testnet"
    assert summary["engine"] == "nautilus_live"
    assert summary["engine_version"] == "nautilus_trader-1.231.0"
    assert summary["venue"] == "BINANCE"
    assert set(summary["reconciliation"]) == {
        "orders_submitted",
        "orders_filled",
        "position_mismatch",
    }
    assert summary["status"] in ("ok", "ok_invalidated", "no-trade but risk-idle OK")
    journal = json.loads(Path(summary["journal_path"]).read_text(encoding="utf-8"))
    assert journal["reconciliation"] == summary["reconciliation"]

def test_url_overrides_absent_by_default():
    assert live_backend.testnet_url_overrides({}) == {}
    assert live_backend.testnet_url_overrides({"testnet": {"max_minutes": 30}}) == {}


def test_url_overrides_rest_and_ws():
    cfg = {
        "testnet": {
            "rest_url": "https://testnet.binancefuture.com/",
            "ws_url": "wss://stream.binancefuture.com",
        }
    }
    out = live_backend.testnet_url_overrides(cfg)
    assert out == {
        "base_url_http": "https://testnet.binancefuture.com",
        "base_url_ws": "wss://stream.binancefuture.com",
    }


def test_url_overrides_fail_closed_on_bad_scheme():
    with pytest.raises(ValueError, match="https"):
        live_backend.testnet_url_overrides({"testnet": {"rest_url": "http://x"}})
    with pytest.raises(ValueError, match="wss"):
        live_backend.testnet_url_overrides({"testnet": {"ws_url": "https://x"}})


def test_live_result_denial_fields_default_zero():
    cfg = {**load_config(), "mode": "testnet"}
    result = _fake_live_result(cfg)
    assert result["instrument_missing_skips"] == 0
    assert result["orders_denied"] == 0
    assert result["denials"] == []


def test_live_result_denial_fields_passthrough():
    cfg = {**load_config(), "mode": "testnet"}
    session = SessionRisk.from_config(cfg, start_equity=10_000.0)
    result = live_backend.build_live_result(
        cfg=cfg,
        symbol="BTCUSDT",
        strategy_name="tsmom",
        strategy_kwargs={"lookback": 48},
        session=session,
        fills=[],
        equity_curve=[{"ts": "2026-01-01T00:00:00+00:00", "equity": 10_000.0}],
        account_curve=[{"ts": "2026-01-01T00:00:00+00:00", "equity": 10_000.0}],
        funding_events=[],
        liquidation_events=[],
        positions={},
        orders_submitted=1,
        orders_filled=0,
        instrument_missing_skips=2,
        orders_denied=1,
        denials=[{"ts": "2026-01-01T00:01:00+00:00", "reason": "instrument not found"}],
        warmup_bars=400,
        live_bars=3,
        initial_equity=10_000.0,
        final_equity=10_000.0,
        expected_qty=0.0,
        venue_qty=0.0,
        session_seconds=65.0,
        max_minutes=30.0,
        equity_source="venue",
    )
    assert result["instrument_missing_skips"] == 2
    assert result["orders_denied"] == 1
    assert result["denials"][0]["reason"] == "instrument not found"


def test_summary_passes_denial_counters_through(tmp_path):
    cfg = {**load_config(), "mode": "testnet"}
    result = _fake_live_result(cfg)
    result["orders_denied"] = 1
    result["instrument_missing_skips"] = 3
    result["denials"] = [{"ts": "t", "reason": "r"}]
    summary = build_summary(
        result,
        cfg=cfg,
        symbol="BTCUSDT",
        interval="1h",
        source_used="venue",
        sample_kind="venue",
        data_note=None,
        out_dir=tmp_path,
        engine=result["engine"],
        engine_version=result["engine_version"],
        strategy_name="tsmom",
    )
    assert summary["orders_denied"] == 1
    assert summary["instrument_missing_skips"] == 3
    assert summary["denials"][0]["reason"] == "r"

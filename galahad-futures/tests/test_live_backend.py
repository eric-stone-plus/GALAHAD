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
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_shadow

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


def test_passes_min_notional_gate():
    assert live_backend.passes_min_notional(0.001, 80_000.0, 50.0) is True   # $80
    assert live_backend.passes_min_notional(0.0005, 80_000.0, 50.0) is False  # $40
    assert live_backend.passes_min_notional(0.0, 80_000.0, 50.0) is False
    assert live_backend.passes_min_notional(-0.001, 80_000.0, 0.0) is True


def test_live_result_rejection_fields_default_and_passthrough():
    cfg = {**load_config(), "mode": "testnet"}
    assert _fake_live_result(cfg)["orders_rejected"] == 0
    assert _fake_live_result(cfg)["dust_skips"] == 0


# --- equity-outage resolution (fail-closed, never the session peak) ----------


def test_resolve_live_equity_prefers_venue_then_last_good():
    assert live_backend.resolve_live_equity(10_100.0, 9_900.0) == 10_100.0
    # outage with a prior good reading: last-known-good, never peak
    assert live_backend.resolve_live_equity(None, 9_900.0) == 9_900.0
    # outage from session start: no trustworthy snapshot → None (fail closed)
    assert live_backend.resolve_live_equity(None, None) is None


def test_outage_last_good_fallback_keeps_loss_halt_engaged():
    """Regression: feeding peak_equity as CURRENT equity during a venue
    outage cleared an active LOSS_HALTED and re-enabled submissions."""
    gate = RiskGate(
        config=RiskConfig(mode="testnet", max_daily_loss=500.0),
        day_start_equity=10_000.0,
    )
    gate.update_equity(10_000.0, ts="t0")  # peak; daily-loss floor = 9_500
    gate.update_equity(9_400.0, ts="t1")  # below floor → LOSS_HALTED
    assert gate.loss_halted
    # Venue unreadable; last known good is 9_400 → the halt must persist.
    gate.update_equity(live_backend.resolve_live_equity(None, 9_400.0), ts="t2")
    assert gate.loss_halted
    assert gate.max_drawdown_seen == pytest.approx(0.06)


# --- external-flatten streak gate ----------------------------------------------


def test_external_flatten_requires_consecutive_quiet_bars():
    streak, confirmed = live_backend.external_flatten_confirmed(0, observed_flat=True)
    assert (streak, confirmed) == (1, False)  # single observation never fires
    streak, confirmed = live_backend.external_flatten_confirmed(streak, observed_flat=True)
    assert (streak, confirmed) == (2, True)  # second consecutive → confirmed
    # submission bar / venue not flat / working orders → streak resets
    streak, confirmed = live_backend.external_flatten_confirmed(2, observed_flat=False)
    assert (streak, confirmed) == (0, False)
    streak, confirmed = live_backend.external_flatten_confirmed(streak, observed_flat=True)
    assert (streak, confirmed) == (1, False)  # back to one observation


# --- _money_float ----------------------------------------------------------------


def test_money_float_parses_moneyish_values():
    assert live_backend._money_float(None) == 0.0
    assert live_backend._money_float(0.005) == pytest.approx(0.005)
    assert live_backend._money_float("0.005 USDT") == pytest.approx(0.005)


def test_money_float_unparseable_raises_valueerror_not_indexerror():
    with pytest.raises(ValueError, match="unparseable money value"):
        live_backend._money_float("")
    with pytest.raises(ValueError, match="unparseable money value"):
        live_backend._money_float("   ")
    with pytest.raises(ValueError):
        live_backend._money_float("USDT")


# --- _run_node session driver via a fake nautilus_trader stack ------------------
#
# nautilus_trader is an optional dependency (not installed in CI). These tests
# fake exactly the import surface _run_node touches, so the session driver
# (equity-outage handling, flatten streak, crash detection, shutdown order)
# is exercised offline.


class _FakeQuantity:
    def __init__(self, value, precision):
        self.value = float(value)
        self.precision = precision


class _FakeAccount:
    def __init__(self, equity):
        self._equity = equity

    def balance_total(self, currency):
        return self._equity

    def leverage(self, instrument_id):
        return 3.0


class _FakePortfolio:
    """Venue state knob: ``equity = None`` simulates an account-read outage."""

    def __init__(self):
        self.equity: float | None = 10_000.0
        self.qty = 0.0

    def net_position(self, instrument_id):
        return self.qty

    def account(self, venue):
        return None if self.equity is None else _FakeAccount(self.equity)

    def unrealized_pnl(self, instrument_id):
        return 0.0


class _FakeCache:
    def orders_open(self, instrument_id=None):
        return []

    def instrument(self, instrument_id):
        return SimpleNamespace(size_precision=3, size_increment=0.001, min_notional=None)

    def positions(self, instrument_id=None):
        return []


class _FakeStrategyBase:
    """Stands in for nautilus_trader.trading.strategy.Strategy."""

    def __init__(self, config):
        self.config = config
        self.portfolio = _FakePortfolio()
        self.cache = _FakeCache()
        self.order_factory = SimpleNamespace(
            market=lambda *a, **kw: SimpleNamespace(args=a, kwargs=kw)
        )

    def subscribe_instrument(self, instrument_id):
        pass

    def subscribe_bars(self, bar_type):
        pass

    def submit_order(self, order):
        pass


class _FakeTrader:
    def __init__(self):
        self.strategies: list = []

    def add_strategy(self, strategy):
        self.strategies.append(strategy)


class _FakeLoop:
    def __init__(self, node):
        self._node = node
        self.halted_when_stop_scheduled: bool | None = None

    def call_soon_threadsafe(self, fn):
        strategies = self._node.trader.strategies
        self.halted_when_stop_scheduled = bool(strategies and strategies[0].halted)
        fn()


class _FakeTradingNode:
    instances: list = []
    bars: list = []  # per-install knobs (a fresh subclass per test)
    equity_script: list = []

    def __init__(self, config):
        self.config = config
        self.trader = _FakeTrader()
        self.kernel = SimpleNamespace(loop=_FakeLoop(self))
        self.portfolio = _FakePortfolio()
        self.cache = _FakeCache()
        self._stopped = False
        _FakeTradingNode.instances.append(self)

    def add_data_client_factory(self, venue, factory):
        pass

    def add_exec_client_factory(self, venue, factory):
        pass

    def build(self):
        pass

    def stop(self):
        self._stopped = True

    def dispose(self):
        pass

    def run(self, raise_exception=False):
        strat = self.trader.strategies[0]
        strat.on_start()
        for i, bar in enumerate(type(self).bars):
            if i < len(type(self).equity_script):
                strat.portfolio.equity = type(self).equity_script[i]
            try:
                strat.on_bar(bar)
            except Exception:
                if raise_exception:
                    raise
        # The real node blocks until stopped; mirror that so the deadline /
        # halt paths in _run_node are what terminates the session.
        while not self._stopped:
            time.sleep(0.005)


def _fake_bar(i: int, close: float = 100.5, ts_init=None):
    return SimpleNamespace(
        ts_init=1_700_000_000_000_000_000 + i * 3_600_000_000_000 if ts_init is None else ts_init,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def _bar_ts(i: int) -> str:
    return datetime.fromtimestamp(1_700_000_000 + i * 3600, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )


class _StubDecisionModel:
    """Deterministic strategy model: every bar targets +1.0 leverage."""

    def targets(self, df):
        return pd.Series([1.0] * len(df))


def _install_fake_nautilus(monkeypatch, *, bars, equities):
    """Register fake nautilus_trader modules; return the fake node class."""
    _FakeTradingNode.instances.clear()
    node_cls = type(
        "_FakeTradingNode",
        (_FakeTradingNode,),
        {"bars": list(bars), "equity_script": list(equities)},
    )

    class _Cfg:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    def _mod(name, **attrs):
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    _mod("nautilus_trader")
    _mod("nautilus_trader.adapters")
    _mod("nautilus_trader.adapters.binance")
    _mod("nautilus_trader.adapters.binance.common")
    _mod(
        "nautilus_trader.adapters.binance.common.enums",
        BinanceAccountType=SimpleNamespace(USDT_FUTURES="USDT_FUTURES"),
        BinanceEnvironment=SimpleNamespace(TESTNET="TESTNET", MAINNET="MAINNET"),
    )
    _mod("nautilus_trader.adapters.binance.common.symbol", BinanceSymbol=lambda s: s)
    _mod(
        "nautilus_trader.adapters.binance.config",
        BinanceDataClientConfig=_Cfg,
        BinanceExecClientConfig=_Cfg,
        BinanceInstrumentProviderConfig=_Cfg,
    )
    _mod(
        "nautilus_trader.adapters.binance.factories",
        BinanceLiveDataClientFactory=object,
        BinanceLiveExecClientFactory=object,
    )
    _mod("nautilus_trader.common", Environment=SimpleNamespace(SANDBOX="SANDBOX"))
    _mod(
        "nautilus_trader.config",
        LoggingConfig=_Cfg,
        StrategyConfig=_Cfg,
        TradingNodeConfig=_Cfg,
    )
    _mod("nautilus_trader.live")
    _mod("nautilus_trader.live.node", TradingNode=node_cls)
    _mod("nautilus_trader.model")
    _mod("nautilus_trader.model.currencies", Currency=SimpleNamespace(from_str=lambda s: s))
    _mod("nautilus_trader.model.data", BarType=SimpleNamespace(from_str=lambda s: s))
    _mod(
        "nautilus_trader.model.enums",
        OmsType=SimpleNamespace(NETTING="NETTING"),
        OrderSide=SimpleNamespace(BUY="BUY", SELL="SELL"),
    )
    _mod(
        "nautilus_trader.model.identifiers",
        InstrumentId=SimpleNamespace(from_str=lambda s: s),
        Venue=lambda s: s,
    )
    _mod("nautilus_trader.model.objects", Quantity=_FakeQuantity)
    _mod("nautilus_trader.trading")
    _mod("nautilus_trader.trading.strategy", Strategy=_FakeStrategyBase)
    monkeypatch.setattr(
        live_backend, "build_strategy", lambda name, **kw: _StubDecisionModel()
    )
    return node_cls


def _run_fake_session(monkeypatch, *, bars, equities, max_minutes=0.005):
    node_cls = _install_fake_nautilus(monkeypatch, bars=bars, equities=equities)
    warmup = pd.DataFrame(
        {
            "ts": ["2026-01-01T00:00:00+00:00"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1.0],
        }
    )
    cfg = {
        "mode": "testnet",
        "interval": "1h",
        "initial_equity": 10_000.0,
        "default_leverage": 3.0,
        "max_leverage": 5.0,
        "risk": {
            "kill_switch": False,
            "enable_testnet": True,
            "max_daily_loss": 500.0,
            "daily_loss_hysteresis": 0.0,
            "max_drawdown_pct": 0.15,
            "max_order_notional": 5000.0,
            "max_position_notional": 15_000.0,
        },
    }
    result = live_backend._run_node(
        warmup=warmup,
        cfg=cfg,
        symbol="BTCUSDT",
        strategy_name="dual_ma",
        strategy_kwargs={},
        api_key="k",
        api_secret="s",
        max_minutes=max_minutes,
    )
    return result, node_cls


def test_session_outage_with_prior_good_equity_uses_last_good(monkeypatch):
    """Mid-session venue-equity outage: the last known-good reading carries
    the session — an active LOSS_HALTED must NOT clear on the peak."""
    bars = [_fake_bar(i) for i in range(4)]
    result, _ = _run_fake_session(
        monkeypatch,
        bars=bars,
        equities=[10_000.0, 9_400.0, None, 9_900.0],  # None = outage bar
    )
    # floor = 10_000 - 500 = 9_500: bar1 halts (9_400), the outage bar must
    # keep the halt (old behavior fed peak 10_000 → spurious clear), and the
    # halt clears only when the venue genuinely reads 9_900 again.
    events = result["loss_halt_events"]
    assert [e["event"] for e in events] == ["halted", "cleared"]
    assert events[0]["ts"] == _bar_ts(1) and events[0]["equity"] == pytest.approx(9_400.0)
    assert events[1]["ts"] == _bar_ts(3) and events[1]["equity"] == pytest.approx(9_900.0)
    # outage bar recorded the last-good equity, never the session peak
    assert result["account_curve"][2]["equity"] == pytest.approx(9_400.0)
    assert result["equity_curve"][2]["equity"] == pytest.approx(9_400.0)
    assert result["risk_decisions"][2]["pre_trade_equity"] == pytest.approx(9_400.0)
    assert result["peak_equity"] == pytest.approx(10_000.0)
    # trading continued on the stale snapshot: submit on bar0 and bar3 only
    assert result["orders_submitted"] == 2
    assert result["equity_source"] == "venue"


def test_session_outage_from_start_fails_closed(monkeypatch):
    """No good equity snapshot yet: decide nothing, submit nothing, record
    nothing — and resume once the venue account becomes readable."""
    bars = [_fake_bar(i) for i in range(3)]
    result, _ = _run_fake_session(
        monkeypatch,
        bars=bars,
        equities=[None, None, 9_900.0],
    )
    assert result["bars"] == 3
    # no decisions/submissions/curve entries while equity was unreadable
    assert len(result["risk_decisions"]) == 1  # bar2 only
    assert result["orders_submitted"] == 1
    assert result["equity_curve_len"] == 1
    assert len(result["account_curve"]) == 1
    assert result["liquidated"] is False
    assert result["invalidated"] is False
    assert result["loss_halt_events"] == []
    # the trace honestly reports the venue was never readable at session init
    assert result["equity_source"] == "config_fallback"
    assert result["initial_equity"] == pytest.approx(10_000.0)


# the crash test intentionally kills the runner thread (that traceback is the
# failure signal the production code surfaces as a RuntimeError)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_runner_thread_crash_marks_session_errored(monkeypatch):
    """A strategy exception in on_bar kills the runner thread; the session
    must surface as an error, never as a normal status:ok result."""
    bars = [_fake_bar(0, ts_init="not-a-number")]
    node_cls = _install_fake_nautilus(monkeypatch, bars=bars, equities=[10_000.0])
    warmup = pd.DataFrame(
        {
            "ts": ["2026-01-01T00:00:00+00:00"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1.0],
        }
    )
    cfg = {
        "mode": "testnet",
        "risk": {"kill_switch": False, "enable_testnet": True},
    }
    with pytest.raises(RuntimeError, match="died mid-session"):
        live_backend._run_node(
            warmup=warmup,
            cfg=cfg,
            symbol="BTCUSDT",
            strategy_name="dual_ma",
            strategy_kwargs={},
            api_key="k",
            api_secret="s",
            max_minutes=1.0,  # deadline far away: the crash is what ends it
        )
    node = _FakeTradingNode.instances[-1]
    assert node.trader.strategies[0].started  # mid-session crash, not startup
    assert node._stopped  # node still wound down cleanly


def test_external_flatten_fires_only_on_second_quiet_bar(monkeypatch):
    """Stale-flat venue reads one bar past submission no longer trip the
    liquidation heuristic; two consecutive quiet-bar observations do."""
    # bar0 submits (expected_qty > 0, venue stays flat — no fill simulation);
    # bars 1-2 have close 0.0 so the gate rejects (invalid mark) and nothing
    # resubmits: quiet bars with a stale-flat venue read and expected qty.
    bars = [_fake_bar(0), _fake_bar(1, close=0.0), _fake_bar(2, close=0.0)]
    result, _ = _run_fake_session(
        monkeypatch, bars=bars, equities=[10_000.0] * 3, max_minutes=1.0
    )
    assert result["orders_submitted"] == 1
    # exactly one inference event, on the SECOND quiet bar (bar2, not bar1)
    assert result["liquidation_events"] == [{"ts": _bar_ts(2), "detector": "external_flatten"}]
    assert result["liquidated"] is True
    assert result["decision_phase_final"] == "LIQUIDATED"


def test_deadline_path_halts_strategy_before_scheduling_stop(monkeypatch):
    """Until node.stop actually runs, bars keep arriving on the daemon node;
    the strategy must be halted before the stop is scheduled."""
    result, node_cls = _run_fake_session(
        monkeypatch, bars=[_fake_bar(0)], equities=[10_000.0], max_minutes=0.005
    )
    assert result["orders_submitted"] == 1  # clean session sanity check
    node = _FakeTradingNode.instances[-1]
    assert node._stopped
    assert node.kernel.loop.halted_when_stop_scheduled is True


# --- run_shadow --json stdout contract ------------------------------------------


def test_run_shadow_json_diverts_raw_fd1_writes(monkeypatch, capfd, tmp_path):
    """run_shadow runs the same live TradingNode whose Rust logger writes fd 1
    directly; under --json that channel must be diverted so stdout parses as
    exactly one JSON document (same contract as cli.py)."""

    def fake_build(cfg, bars, inputs, *, force_strategy, force_lookback):
        # raw fd-1 write = the Rust logger path; print = Python-level logging
        os.write(1, b"rust-fd1 shadow noise\n")
        print("python-level session log")
        return {
            "schema": "galahad.shadow.v1",
            "run_id": "20260905T000000Z",
            "inputs": {},
            "engines": {},
            "reconciliation": {},
            "known_divergences": [],
        }

    monkeypatch.setattr(run_shadow, "_preflight", lambda cfg: [])
    monkeypatch.setattr(run_shadow, "build_shadow_report", fake_build)
    rc = run_shadow.main(["--source", "fixture", "--json", "--output-dir", str(tmp_path)])
    out, err = capfd.readouterr()
    assert rc == 0
    # stdout is exactly one JSON document
    doc, end = json.JSONDecoder().raw_decode(out)
    assert out[end:].strip() == ""
    assert doc["schema"] == "galahad.shadow.v1"
    # diverted session noise is preserved on stderr, not dropped
    assert "rust-fd1 shadow noise" not in out
    assert "rust-fd1 shadow noise" in err
    assert "python-level session log" in err

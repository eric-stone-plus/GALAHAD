"""Venue (alpaca_paper) tests — fail-closed gates, fakes above the HTTP boundary."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.parse
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_security import venue_alpaca
from galahad_security.engine import load_config, run_paper_session


def _rising_bars(n=30, start=100.0, step=0.5) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-06-01")
    for i in range(n):
        c = start + step * i
        rows.append(
            {
                "ts": str(base + pd.Timedelta(days=i)),
                "open": c,
                "high": c * 1.001,
                "low": c * 0.999,
                "close": c,
                "volume": 1_000_000.0,
            }
        )
    return pd.DataFrame(rows)


class FakeClient:
    """In-memory Alpaca paper stand-in. apply_fills=False simulates orders
    resting unfilled (e.g. submitted outside market hours)."""

    def __init__(self, *, apply_fills: bool = True, fill_slippage_bps: float = 0.0):
        self.bars = {s: _rising_bars() for s in ("AAPL", "MSFT")}
        self.account = {"equity": "100000.00", "cash": "100000.00"}
        self.positions: list[dict] = []
        self.orders: list[dict] = []
        self.apply_fills = apply_fills
        self.fill_slippage_bps = fill_slippage_bps

    def get_daily_bars(self, symbols, *, limit):
        return {s: self.bars[s] for s in symbols}

    def get_account(self):
        # Recompute a venue-style equity from cash + positions at last close
        pos_val = sum(
            int(p["qty"]) * float(self.bars[p["symbol"]].iloc[-1]["close"])
            for p in self.positions
        )
        cash = float(self.account["cash"])
        return {"equity": f"{cash + pos_val:.2f}", "cash": f"{cash:.2f}"}

    def get_positions(self):
        return [dict(p) for p in self.positions]

    def submit_market_order(self, symbol, qty, side):
        oid = f"order-{len(self.orders)}"
        self.orders.append({"id": oid, "symbol": symbol, "qty": qty, "side": side})
        if self.apply_fills:
            mark = float(self.bars[symbol].iloc[-1]["close"])
            slip = self.fill_slippage_bps / 10_000.0
            px = mark * (1 + slip) if side == "buy" else mark * (1 - slip)
            self.account["cash"] = f"{float(self.account['cash']) - (qty * px if side == 'buy' else -qty * px):.2f}"
            held = next((p for p in self.positions if p["symbol"] == symbol), None)
            if side == "buy":
                if held is None:
                    self.positions.append(
                        {"symbol": symbol, "qty": str(qty), "avg_entry_price": f"{px:.4f}"}
                    )
                else:
                    held["qty"] = str(int(held["qty"]) + qty)
            elif held is not None:
                held["qty"] = str(int(held["qty"]) - qty)
                if int(held["qty"]) <= 0:
                    self.positions.remove(held)
        return {"id": oid}

    def get_order(self, oid):
        order = next(o for o in self.orders if o["id"] == oid)
        if not self.apply_fills:
            return {"status": "accepted", "filled_qty": "0", "filled_avg_price": None}
        mark = float(self.bars[order["symbol"]].iloc[-1]["close"])
        slip = self.fill_slippage_bps / 10_000.0
        px = mark * (1 + slip) if order["side"] == "buy" else mark * (1 - slip)
        return {"status": "filled", "filled_qty": str(order["qty"]), "filled_avg_price": f"{px:.4f}"}


def _venue_cfg():
    cfg = load_config()
    cfg["risk"] = {
        **dict(cfg.get("risk") or {}),
        "kill_switch": False,
        "enable_alpaca_paper": True,
    }
    return cfg


# --- fail-closed gates ---------------------------------------------------------


def test_credentials_missing_raise():
    with pytest.raises(RuntimeError, match="ALPACA_PAPER_API_KEY"):
        venue_alpaca.venue_credentials({})


def test_credentials_missing_one_names_it():
    with pytest.raises(RuntimeError, match="ALPACA_PAPER_API_SECRET"):
        venue_alpaca.venue_credentials({"ALPACA_PAPER_API_KEY": "k"})


def test_credentials_ok():
    assert venue_alpaca.venue_credentials(
        {"ALPACA_PAPER_API_KEY": " k ", "ALPACA_PAPER_API_SECRET": "s"}
    ) == ("k", "s")


def test_venue_gate_closed_raises():
    cfg = load_config()  # kill_switch: true, enable_alpaca_paper: false
    with pytest.raises(RuntimeError, match="venue gate closed"):
        venue_alpaca.precheck_venue_gate({**cfg, "mode": "venue"})


def test_non_paper_base_url_refused():
    with pytest.raises(RuntimeError, match="paper endpoint only"):
        venue_alpaca._AlpacaPaperClient("k", "s", base_url="https://api.alpaca.markets")


def test_get_daily_bars_anchors_explicit_start_window():
    # Without ``start`` the Alpaca bars endpoint returns only the current-day
    # bar; the client must anchor a window wide enough for ``limit`` trades.
    client = venue_alpaca._AlpacaPaperClient("k", "s")
    seen: dict[str, str] = {}

    def fake_request(method: str, url: str):
        seen["url"] = url
        return {
            "bars": {
                "AAPL": [
                    {"t": "2026-09-03T04:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 100.0},
                    {"t": "2026-09-04T04:00:00Z", "o": 1.5, "h": 2.5, "l": 1.0, "c": 2.0, "v": 200.0},
                ]
            }
        }

    client._request = fake_request
    out = client.get_daily_bars(["AAPL"], limit=250)
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(seen["url"]).query)
    assert qs["timeframe"] == ["1Day"]
    # explicit page size: the default page silently truncates to the oldest
    # bars of the window (pagination covered by the multi-page test below)
    assert qs["limit"] == [str(venue_alpaca._BARS_PAGE_SIZE)]
    assert qs["feed"] == ["iex"]
    start = date.fromisoformat(qs["start"][0])
    days = (date.today() - start).days
    assert 400 <= days <= 700
    assert len(out["AAPL"]) == 2
    assert out["AAPL"].iloc[-1]["close"] == 2.0


def _bar(ts_day: str, close: float) -> dict:
    return {"t": f"2026-09-{ts_day}T04:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": close, "v": 100.0}


def test_get_daily_bars_follows_next_page_token():
    # Multi-page window: page 1 (oldest bars) + token -> page 2. The
    # assembled window must hold the newest ``limit`` bars ACROSS pages —
    # reading page 1 only would silently keep a stale window.
    client = venue_alpaca._AlpacaPaperClient("k", "s")
    pages = {
        None: {
            "bars": {"AAPL": [_bar("01", 1.0), _bar("02", 2.0)]},
            "next_page_token": "p2",
        },
        "p2": {
            "bars": {"AAPL": [_bar("03", 3.0), _bar("04", 4.0)]},
            "next_page_token": None,
        },
    }
    seen_tokens: list[str | None] = []

    def fake_request(method: str, url: str):
        token = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("page_token", [None])[0]
        seen_tokens.append(token)
        return pages[token]

    client._request = fake_request
    out = client.get_daily_bars(["AAPL"], limit=3)
    assert seen_tokens == [None, "p2"]
    assert [c for c in out["AAPL"]["close"]] == [2.0, 3.0, 4.0]
    assert out["AAPL"].iloc[-1]["ts"] == "2026-09-04"


def test_get_daily_bars_page_cap_fails_loud(monkeypatch):
    # A venue that keeps handing out page tokens must not loop forever:
    # bounded pages, then a hard error (never a silently truncated window).
    monkeypatch.setattr(venue_alpaca, "_BARS_MAX_PAGES", 3)
    client = venue_alpaca._AlpacaPaperClient("k", "s")
    client._request = lambda method, url: {
        "bars": {"AAPL": [_bar("04", 1.5)]},
        "next_page_token": "never-ending",
    }
    with pytest.raises(RuntimeError, match="pagination"):
        client.get_daily_bars(["AAPL"], limit=5)


def test_get_daily_bars_rejects_limit_below_1():
    # rows[-0:] would return the ENTIRE window — hard error at the boundary.
    client = venue_alpaca._AlpacaPaperClient("k", "s")
    for bad in (0, -5):
        with pytest.raises(ValueError, match=">= 1"):
            client.get_daily_bars(["AAPL"], limit=bad)


def test_get_daily_bars_keeps_newest_limit_rows():
    # The window may hold more than ``limit`` bars; keep the newest tail.
    client = venue_alpaca._AlpacaPaperClient("k", "s")
    rows = [
        {"t": f"2026-09-{d:02d}T04:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": float(d), "v": 100.0}
        for d in range(1, 6)
    ]
    client._request = lambda method, url: {"bars": {"AAPL": rows}}
    out = client.get_daily_bars(["AAPL"], limit=3)
    assert len(out["AAPL"]) == 3
    assert [c for c in out["AAPL"]["close"]] == [3.0, 4.0, 5.0]


def test_get_daily_bars_fetches_one_symbol_per_request():
    # Multi-symbol bar requests silently drop symbols on some plans; the
    # client must issue one request per symbol.
    client = venue_alpaca._AlpacaPaperClient("k", "s")
    urls: list[str] = []

    def fake_request(method: str, url: str):
        urls.append(url)
        sym = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["symbols"][0]
        return {"bars": {sym: [{"t": "2026-09-04T04:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 100.0}]}}

    client._request = fake_request
    out = client.get_daily_bars(["AAPL", "MSFT"], limit=30)
    assert len(urls) == 2
    assert [urllib.parse.parse_qs(urllib.parse.urlparse(u).query)["symbols"][0] for u in urls] == ["AAPL", "MSFT"]
    assert set(out) == {"AAPL", "MSFT"}


def test_cli_venue_without_credentials_fails_clean():
    env = {k: v for k, v in os.environ.items() if not k.startswith("ALPACA_")}
    for script in ("run_paper.py", "run_venue.py"):
        argv = [sys.executable, str(ROOT / "scripts" / script), "--json"]
        if script == "run_paper.py":
            argv += ["--engine", "alpaca_paper"]
        out = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env)
        assert out.returncode != 0, script
        assert "ALPACA_PAPER_API_KEY" in out.stderr
        assert "Traceback" not in out.stderr


def test_cli_venue_gate_closed_fails_clean(tmp_path):
    env = dict(os.environ)
    env["ALPACA_PAPER_API_KEY"] = "fake"
    env["ALPACA_PAPER_API_SECRET"] = "fake"
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_venue.py"), "--json"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert out.returncode != 0
    assert "venue gate closed" in out.stderr
    assert "Traceback" not in out.stderr


# --- pure order plan / reconciliation -------------------------------------------


def test_order_plan_math():
    plan = venue_alpaca.venue_order_plan(
        target_weights={"AAPL": 0.2, "MSFT": 0.0},
        marks={"AAPL": 110.0, "MSFT": 50.0},
        equity=100_000.0,
        current_qty={"MSFT": 40},
    )
    # AAPL: floor(20_000/110)=181 buy; MSFT: sell all 40; zero deltas absent
    assert {tuple(sorted(o.items())) for o in plan} == {
        tuple(sorted({"symbol": "AAPL", "side": "buy", "qty": 181}.items())),
        tuple(sorted({"symbol": "MSFT", "side": "sell", "qty": 40}.items())),
    }


def test_order_plan_sell_capped_at_holdings():
    plan = venue_alpaca.venue_order_plan(
        target_weights={"AAPL": 0.0},
        marks={"AAPL": 100.0},
        equity=100_000.0,
        current_qty={"AAPL": 10},
    )
    assert plan == [{"symbol": "AAPL", "side": "sell", "qty": 10}]


def test_reconcile_venue():
    clean = venue_alpaca.reconcile_venue(
        orders_submitted=2, orders_filled=2,
        expected_positions={"AAPL": 5}, venue_positions={"AAPL": 5},
    )
    assert clean == {"orders_submitted": 2, "orders_filled": 2, "position_mismatch": False}
    mismatch = venue_alpaca.reconcile_venue(
        orders_submitted=2, orders_filled=1,
        expected_positions={"AAPL": 5}, venue_positions={"AAPL": 3},
    )
    assert mismatch["position_mismatch"] is True
    unknown = venue_alpaca.reconcile_venue(
        orders_submitted=1, orders_filled=0,
        expected_positions={"AAPL": 5}, venue_positions=None,
    )
    assert unknown["position_mismatch"] is True  # fail closed


# --- venue account equity: fail closed on a broken venue response -------------


def test_venue_account_equity_missing_or_empty_raises():
    for account in ({}, {"equity": None}, {"equity": ""}, {"equity": "   "}):
        with pytest.raises(RuntimeError, match="equity"):
            venue_alpaca._venue_account_equity(account, context="test")


def test_venue_account_equity_numeric_zero_is_legitimate():
    assert venue_alpaca._venue_account_equity({"equity": 0}, context="t") == 0.0
    assert venue_alpaca._venue_account_equity({"equity": "0.00"}, context="t") == 0.0


# --- one-shot venue session with fakes -------------------------------------------


def test_run_venue_once_filled(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "fake")
    client = FakeClient(apply_fills=True)
    result = venue_alpaca.run_venue_once(
        _venue_cfg(), symbols=["AAPL", "MSFT"], client=client
    )
    assert result["engine"] == "alpaca_paper"
    assert result["engine_version"] == "alpaca-paper-api-v2"
    assert result["venue"] == "ALPACA"
    assert result["orders_submitted"] == 2  # both symbols long at the rising tail
    assert result["orders_filled"] == 2
    assert result["reconciliation"]["position_mismatch"] is False
    assert result["positions"]["AAPL"]["qty"] > 0
    # venue TCA: arrival = decision close; no spread/impact decomposition
    assert result["tca"]["n_fills"] == 2
    assert result["tca"]["implementation_shortfall_usdt"] == pytest.approx(0.0)
    assert result["tca"]["spread_cost_usdt"] is None
    assert result["derisk"]["ladder_enabled"] is False


def test_run_venue_once_unfilled_orders_flag_mismatch(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "fake")
    client = FakeClient(apply_fills=False)  # orders rest (outside market hours)
    result = venue_alpaca.run_venue_once(
        _venue_cfg(), symbols=["AAPL", "MSFT"], client=client
    )
    assert result["orders_submitted"] == 2
    assert result["orders_filled"] == 0
    assert result["reconciliation"]["position_mismatch"] is True


def test_run_venue_once_summary_via_engine(monkeypatch, tmp_path):
    """Full engine dispatch with an injected client — summary contract shape."""
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "fake")
    client = FakeClient(apply_fills=True, fill_slippage_bps=3.0)

    import galahad_security.engine as eng
    import galahad_security.venue_alpaca as va

    cfg = _venue_cfg()
    # venue bars write through to {project_root}/data/cache — redirect the
    # component root so the test does not touch the real data/ tree
    monkeypatch.setattr(eng, "project_root", lambda: tmp_path)
    monkeypatch.setattr(va, "_AlpacaPaperClient", lambda *a, **k: client)
    summary = run_paper_session(
        config=cfg, engine="alpaca_paper", output_dir=tmp_path / "out"
    )
    assert summary["mode"] == "paper"
    assert summary["engine"] == "alpaca_paper"
    assert summary["venue"] == "ALPACA"
    assert set(summary["reconciliation"]) == {
        "orders_submitted", "orders_filled", "position_mismatch",
    }
    assert summary["status"] in ("ok", "ok_invalidated", "no-trade but risk-idle OK")
    # 3 bps slippage against the close shows up as implementation shortfall
    # (fake venue prices are 4dp-rounded, so allow rounding tolerance)
    assert summary["tca"]["implementation_shortfall_bps"] == pytest.approx(3.0, abs=0.01)
    assert Path(summary["journal_path"]).is_file()


def test_run_venue_once_via_engine_populates_bar_cache(monkeypatch, tmp_path):
    """Docs contract: the shipped venue path write-caches fetched bars to
    data/cache/{symbol}_1d.csv so later offline runs resolve the venue tier."""
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "fake")
    client = FakeClient(apply_fills=True)

    import galahad_security.engine as eng
    import galahad_security.venue_alpaca as va

    cfg = _venue_cfg()
    monkeypatch.setattr(eng, "project_root", lambda: tmp_path)
    monkeypatch.setattr(va, "_AlpacaPaperClient", lambda *a, **k: client)
    summary = run_paper_session(
        config=cfg, engine="alpaca_paper", output_dir=tmp_path / "out"
    )
    for sym in summary["symbols"]:
        cache = tmp_path / "data" / "cache" / f"{sym}_1d.csv"
        assert cache.is_file(), f"venue cache not written for {sym}"
        df = pd.read_csv(cache)
        assert len(df) == 30  # FakeClient serves 30 rising bars


def test_run_venue_once_missing_venue_equity_fails_closed(monkeypatch):
    """A venue account response without a usable equity field is a broken
    venue — raise, never fall back to the config default."""
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "fake")

    class NoEquityClient(FakeClient):
        def get_account(self):
            return {"cash": "100000.00"}

    with pytest.raises(RuntimeError, match="equity"):
        venue_alpaca.run_venue_once(_venue_cfg(), symbols=["AAPL"], client=NoEquityClient())

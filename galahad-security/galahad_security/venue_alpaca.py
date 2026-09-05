"""Alpaca paper venue backend (``alpaca_paper``): one-shot daily execution.

NOT a long-running session (contrast the futures testnet backend's bounded
TradingNode): one ``run_venue_once`` call executes *today's* decision —
fetch latest daily bars from the Alpaca data API → strategy targets →
shared risk gate → market orders on the Alpaca **paper** endpoint →
resulting positions → the same summary/journal contract as the offline
reference book, plus ``venue``/``reconciliation``.

Fail-closed order: credentials → venue gate → network.

- Credentials come only from ``ALPACA_PAPER_API_KEY`` /
  ``ALPACA_PAPER_API_SECRET`` (paper-trading keys; never live-trading
  key names). Missing/empty → ``RuntimeError`` before anything else.
- The venue gate requires ``risk.enable_alpaca_paper: true`` AND
  ``risk.kill_switch: false``; a closed gate raises before any HTTP.
- Only the Alpaca *paper* endpoint is wired (paper-api.alpaca.markets);
  no live-money client can be constructed from this module.

HTTP is stdlib ``urllib`` only (no new dependency), isolated behind
``_AlpacaPaperClient``; everything above the HTTP boundary (order plan,
reconciliation, session assembly) is pure and unit-tested with fakes.
Market-hours caveat: market orders submitted outside RTH may rest
unfilled — they count as submitted, not filled, and the reconciliation
reports the truth (``position_mismatch``).
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import Any, Mapping

import pandas as pd

from galahad_security.book import tca_from_fills
from galahad_security.data import cache_path_for, save_cache
from galahad_security.decision import SessionRisk
from galahad_security.strategy import build_strategy, strategy_kwargs_from_config

ENGINE_NAME = "alpaca_paper"
ENGINE_VERSION = "alpaca-paper-api-v2"
VENUE = "ALPACA"

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"
API_KEY_ENV = "ALPACA_PAPER_API_KEY"
API_SECRET_ENV = "ALPACA_PAPER_API_SECRET"
_HTTP_TIMEOUT_SEC = 15.0
# Bars pagination: the endpoint's default page (~1000 bars) silently
# truncates the window to its OLDEST page, so pass an explicit page size
# and follow next_page_token until the window is complete (bounded).
_BARS_PAGE_SIZE = 10_000
_BARS_MAX_PAGES = 50


# --- pure helpers (no network) ----------------------------------------------


def venue_credentials(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Read Alpaca *paper* credentials from the environment; hard error."""
    env = os.environ if env is None else env
    key = (env.get(API_KEY_ENV) or "").strip()
    secret = (env.get(API_SECRET_ENV) or "").strip()
    missing = [
        name
        for name, value in ((API_KEY_ENV, key), (API_SECRET_ENV, secret))
        if not value
    ]
    if missing:
        raise RuntimeError(
            "engine=alpaca_paper: missing Alpaca PAPER credentials: "
            f"{', '.join(missing)}. Set both env vars with paper-trading keys "
            "from the Alpaca dashboard (paper account). This package has no "
            "live-money path and never reads live-trading key names."
        )
    return key, secret


def precheck_venue_gate(cfg: dict[str, Any]) -> None:
    """Fail closed before any network I/O: the venue gate must be open.

    The caller passes the effective session config (mode already forced to
    ``venue`` by the engine dispatch). A closed gate (kill switch on, or
    ``enable_alpaca_paper`` off) raises ``RuntimeError``.
    """
    session = SessionRisk.from_config(cfg, start_equity=float(cfg.get("initial_equity", 100_000)))
    if session.gate.venue_blocked():
        raise RuntimeError(
            "engine=alpaca_paper: venue gate closed — set risk.enable_alpaca_paper: true "
            "and risk.kill_switch: false in config.yaml to run against the Alpaca "
            "paper endpoint (the offline paper engine is unaffected)"
        )


def venue_order_plan(
    *,
    target_weights: dict[str, float],
    marks: dict[str, float],
    equity: float,
    current_qty: dict[str, int],
) -> list[dict[str, Any]]:
    """Gate-passed target weights → share-delta market orders (long-only).

    Integer shares (floor), sells capped at the shares held; zero deltas
    produce no order. Pure: this is the whole order-planning decision
    above the HTTP boundary.
    """
    orders: list[dict[str, Any]] = []
    for symbol, weight in target_weights.items():
        mark = float(marks[symbol])
        if mark <= 0 or equity <= 0:
            continue
        desired = math.floor(max(0.0, float(weight)) * float(equity) / mark + 1e-9)
        delta = desired - int(current_qty.get(symbol, 0))
        if delta > 0:
            orders.append({"symbol": symbol, "side": "buy", "qty": int(delta)})
        elif delta < 0:
            sell = min(-delta, int(current_qty.get(symbol, 0)))
            if sell > 0:
                orders.append({"symbol": symbol, "side": "sell", "qty": int(sell)})
    return orders


def reconcile_venue(
    *,
    orders_submitted: int,
    orders_filled: int,
    expected_positions: dict[str, int],
    venue_positions: dict[str, int] | None,
) -> dict[str, Any]:
    """Venue reconciliation contract (summary field ``reconciliation``).

    ``position_mismatch`` compares planned post-session positions with the
    venue-reported ones (all symbols in either set must agree). An
    unavailable venue state fails closed (mismatch=True).
    """
    if venue_positions is None:
        mismatch = True
    else:
        keys = set(expected_positions) | set(venue_positions)
        mismatch = any(
            int(expected_positions.get(k, 0)) != int(venue_positions.get(k, 0))
            for k in keys
        )
    return {
        "orders_submitted": int(orders_submitted),
        "orders_filled": int(orders_filled),
        "position_mismatch": bool(mismatch),
    }


def _venue_account_equity(account: Mapping[str, Any], *, context: str) -> float:
    """Venue-reported account equity; hard error on a missing/empty field.

    ``account.get("equity") or <default>`` would mask a broken venue
    response exactly where reconciliation is the tripwire. A legitimate
    numeric zero (``0`` / ``"0.00"``) is a valid equity and passes.
    """
    raw = account.get("equity")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise RuntimeError(
            f"alpaca_paper: venue account {context} has no usable 'equity' "
            f"field (got {raw!r}) — fail closed, never fall back to config "
            "defaults for venue-reported state"
        )
    return float(raw)


# --- HTTP boundary (lazy; the only network code in the component) ------------


class _AlpacaPaperClient:
    """Minimal Alpaca paper REST client (urllib only).

    Hard ``RuntimeError`` on HTTP/transport errors — never a silent
    empty result. Only the paper base URL is reachable here.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = PAPER_BASE_URL,
        data_url: str = DATA_BASE_URL,
        timeout: float = _HTTP_TIMEOUT_SEC,
    ) -> None:
        if base_url != PAPER_BASE_URL:
            raise RuntimeError(
                f"engine=alpaca_paper: refusing non-paper base URL {base_url!r} "
                "(this package wires the Alpaca paper endpoint only)"
            )
        self.base_url = base_url
        self.data_url = data_url
        self.timeout = float(timeout)
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, url: str, payload: dict | None = None) -> Any:
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=body, headers=self._headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(
                f"alpaca_paper HTTP {exc.code} for {url}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"alpaca_paper transport error for {url}: {exc}") from exc
        return json.loads(raw.decode("utf-8")) if raw else {}

    # Trading API (paper endpoint)
    def get_account(self) -> dict[str, Any]:
        return self._request("GET", f"{self.base_url}/v2/account")

    def get_positions(self) -> list[dict[str, Any]]:
        return self._request("GET", f"{self.base_url}/v2/positions")

    def submit_market_order(self, symbol: str, qty: int, side: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{self.base_url}/v2/orders",
            {
                "symbol": symbol,
                "qty": str(int(qty)),
                "side": side,
                "type": "market",
                "time_in_force": "day",
            },
        )

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._request("GET", f"{self.base_url}/v2/orders/{order_id}")

    # Market data API (daily bars; write-cached by the caller)
    def get_daily_bars(self, symbols: list[str], *, limit: int) -> dict[str, pd.DataFrame]:
        # The bars endpoint without ``start`` returns only the current-day bar,
        # so anchor an explicit window wide enough to hold ``limit`` trading
        # days (~252/yr, 1.5x safety). Pages come back oldest-first, so an
        # explicit page size + next_page_token loop assembles the FULL window
        # (the default page would silently truncate to the oldest bars) and
        # the newest ``limit`` rows are kept client-side. Multi-symbol
        # requests silently drop symbols on some plans, so fetch one symbol
        # per request.
        if int(limit) < 1:
            raise ValueError(f"alpaca_paper: bar limit must be >= 1 (got {limit!r})")
        window_days = int(limit * 365 / 252 * 1.5) + 10
        start = (date.today() - timedelta(days=window_days)).isoformat()
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            rows: list[dict[str, Any]] = []
            page_token: str | None = None
            for _ in range(_BARS_MAX_PAGES):
                params = {
                    "symbols": symbol,
                    "timeframe": "1Day",
                    "start": start,
                    "feed": "iex",
                    # v1 models no corporate actions — raw prices, documented.
                    "adjustment": "raw",
                    "limit": _BARS_PAGE_SIZE,
                }
                if page_token:
                    params["page_token"] = page_token
                query = urllib.parse.urlencode(params)
                payload = self._request("GET", f"{self.data_url}/v2/stocks/bars?{query}")
                bars = payload.get("bars") or {}
                rows.extend(bars.get(symbol) or [])
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
            else:
                raise RuntimeError(
                    f"alpaca_paper: bars pagination for {symbol} exceeded "
                    f"{_BARS_MAX_PAGES} pages (page size {_BARS_PAGE_SIZE}) — "
                    "fail closed rather than trade on a truncated window"
                )
            if not rows:
                raise RuntimeError(
                    f"alpaca_paper: no daily bars returned for {symbol} "
                    "(fail closed — the venue session needs real history)"
                )
            rows = rows[-int(limit):]
            out[symbol] = pd.DataFrame(
                [
                    {
                        "ts": str(r["t"])[:10],
                        "open": float(r["o"]),
                        "high": float(r["h"]),
                        "low": float(r["l"]),
                        "close": float(r["c"]),
                        "volume": float(r["v"]),
                    }
                    for r in rows
                ]
            )
        return out


# --- one-shot venue session ---------------------------------------------------


def run_venue_once(
    cfg: dict[str, Any],
    *,
    symbols: list[str],
    strategy_name: str | None = None,
    strategy_kwargs: dict[str, Any] | None = None,
    bar_limit: int = 250,
    client: Any | None = None,
    project_root: Any | None = None,
) -> dict[str, Any]:
    """Execute today's decision once against the Alpaca paper endpoint.

    Returns the same result dict shape as ``engine.run_paper_on_bars``
    (plus venue/reconciliation), so the shared report module renders both
    engines identically. ``client`` injects a fake for offline tests.
    """
    api_key, api_secret = venue_credentials()
    cfg = {**cfg, "mode": "venue"}
    precheck_venue_gate(cfg)

    symbols = list(symbols)
    strat_cfg = dict(cfg.get("strategy") or {})
    name = strategy_name or str(strat_cfg.get("name", "dual_ma"))
    kw = strategy_kwargs if strategy_kwargs is not None else strategy_kwargs_from_config(strat_cfg)
    strategy = build_strategy(name, **kw)

    if client is None:
        client = _AlpacaPaperClient(api_key, api_secret)

    # Latest daily bars from the venue data API; write through to cache.
    bars_by_symbol = client.get_daily_bars(symbols, limit=bar_limit)
    if project_root is not None:
        for sym, df in bars_by_symbol.items():
            save_cache(cache_path_for(project_root, sym), df)

    account = client.get_account()
    equity = _venue_account_equity(account, context="before the session")
    venue_qty_before = {
        p["symbol"]: int(float(p["qty"])) for p in (client.get_positions() or [])
    }

    session = SessionRisk.from_config(cfg, start_equity=equity)
    gate = session.gate

    marks: dict[str, float] = {}
    latest_ts = ""
    target_weights: dict[str, float] = {}
    for sym in symbols:
        df = bars_by_symbol[sym]
        targets = strategy.targets(df)
        last = df.iloc[-1]
        ts = str(last["ts"])
        latest_ts = max(latest_ts, ts)
        mark = float(last["close"])
        marks[sym] = mark
        session.update_equity(equity, ts=ts)
        decision = session.evaluate_weight(
            symbol=sym,
            raw_weight=float(targets.iloc[-1]) if len(targets) else 0.0,
            mark=mark,
            pre_trade_equity=equity,
            current_qty=float(venue_qty_before.get(sym, 0)),
            ts=ts,
        )
        if decision.allowed:
            target_weights[sym] = decision.target_weight

    plan = venue_order_plan(
        target_weights=target_weights,
        marks=marks,
        equity=equity,
        current_qty=venue_qty_before,
    )
    # Expected post-session positions = venue positions before + planned deltas.
    expected_positions: dict[str, int] = dict(venue_qty_before)
    for sym in symbols:
        expected_positions.setdefault(sym, 0)
    for order in plan:
        expected_positions[order["symbol"]] = venue_qty_before.get(order["symbol"], 0) + (
            order["qty"] if order["side"] == "buy" else -order["qty"]
        )

    # Submit, then read back each order (market orders outside RTH may rest).
    submitted_ids: list[str] = []
    for order in plan:
        resp = client.submit_market_order(order["symbol"], order["qty"], order["side"])
        submitted_ids.append(str(resp.get("id", "")))

    fills: list[dict[str, Any]] = []
    for oid, order in zip(submitted_ids, plan):
        status = client.get_order(oid) if oid else {}
        filled_qty = int(float(status.get("filled_qty") or 0))
        avg_px = status.get("filled_avg_price")
        if filled_qty > 0 and avg_px:
            fills.append(
                {
                    "ts": latest_ts,
                    "symbol": order["symbol"],
                    "side": order["side"].upper(),
                    "qty": filled_qty,
                    "price": float(avg_px),
                    "fee": 0.0,  # commission-free paper venue
                    "realized_pnl": 0.0,
                    "note": "alpaca_paper",
                    "arrival_price": marks[order["symbol"]],
                }
            )

    venue_after_raw = client.get_positions()
    venue_positions = (
        {p["symbol"]: int(float(p["qty"])) for p in venue_after_raw}
        if venue_after_raw is not None
        else None
    )
    account_after = client.get_account()
    final_equity = _venue_account_equity(account_after, context="after the session")

    positions = {
        p["symbol"]: {
            "qty": int(float(p["qty"])),
            "avg_cost": float(p.get("avg_entry_price") or 0.0),
            "side": "long" if float(p["qty"]) > 0 else "flat",
        }
        for p in (venue_after_raw or [])
    }

    equity_curve = [
        {"ts": latest_ts, "cash": float(account.get("cash") or equity), "positions_value": equity - float(account.get("cash") or equity), "equity": equity, "positions": {}},
        {"ts": latest_ts, "cash": float(account_after.get("cash") or final_equity), "positions_value": final_equity - float(account_after.get("cash") or final_equity), "equity": final_equity, "positions": positions},
    ]

    reconciliation = reconcile_venue(
        orders_submitted=len(plan),
        orders_filled=len(fills),
        expected_positions=expected_positions,
        venue_positions=venue_positions,
    )

    return {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "venue": VENUE,
        "strategy": name,
        "strategy_kwargs": dict(kw),
        "symbols": symbols,
        "bars": min(len(bars_by_symbol[s]) for s in symbols),
        "n_fills": len(fills),
        "n_risk_rejects": len(gate.rejects),
        "invalidated": gate.invalidated,
        "invalidation_reason": gate.invalidation_reason or None,
        "invalidation_events": list(gate.invalidation_events),
        "loss_halt_events": list(gate.loss_halt_events),
        "decision_phase_final": session.phase(),
        "peak_equity": float(gate.peak_equity),
        "max_drawdown": float(gate.max_drawdown_seen),
        "initial_equity": float(equity),
        "final_equity": float(final_equity),
        "equity_curve": equity_curve,
        "equity_curve_len": len(equity_curve),
        "returns_oos": [],
        "fills": fills,
        "tca": tca_from_fills(fills),
        "derisk": gate.derisk_summary(),
        "risk_rejects": gate.rejects,
        "risk_decisions": session.risk_decisions,
        "risk_decisions_tail": session.risk_decisions[-20:],
        "positions": positions,
        "orders_submitted": reconciliation["orders_submitted"],
        "orders_filled": reconciliation["orders_filled"],
        "reconciliation": reconciliation,
        "expected_positions": {k: int(v) for k, v in expected_positions.items()},
        "equity_source": "venue",
    }

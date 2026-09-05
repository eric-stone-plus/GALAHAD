"""Binance USDT-M futures TESTNET execution backend (``nautilus_live``).

Runs the SAME decision stream as the reference paper book against the
Binance USDT-M futures **testnet** via NautilusTrader's ``TradingNode``
live stack (bundled in the pinned ``nautilus_trader==1.231.0`` wheel —
the ``nautilus`` extra, no new dependency). This is the live-path
rehearsal: real order placement mechanics against a sandbox venue.
**Mainnet is impossible by construction** — the Binance clients are
hardcoded to ``BinanceEnvironment.TESTNET``, credentials come only from
the testnet-specific env vars, and ``mode=live`` is blocked
unconditionally in ``risk.RiskGate``.

Fail-closed order: credentials → optional dependency → gate → session.

- Credentials are read only from ``BINANCE_TESTNET_API_KEY`` /
  ``BINANCE_TESTNET_API_SECRET``; the mainnet key names are never read.
  Missing/empty values raise ``RuntimeError`` before any import or I/O.
- A missing ``nautilus_trader`` package raises ``RuntimeError`` naming
  the extra — never a silent fallback to the paper book.
- The session config's mode is forced to ``"testnet"`` here; the risk
  gate must be open (``risk.enable_testnet: true`` AND
  ``risk.kill_switch: false``) before any client is constructed.
- The session is bounded by ``testnet.max_minutes`` (default 30): the
  node is stopped from the controlling thread, then fills and positions
  are reconciled (decision-expected net qty vs venue-reported net
  position; unknown venue state counts as a mismatch).

Everything above the ``_run_node`` boundary is pure and unit-testable
without ``nautilus_trader`` installed and without network access; all
TradingNode wiring is isolated behind lazy imports inside
``_run_node``.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

import numpy as np
import pandas as pd

from galahad_futures.book import tca_from_fills
from galahad_futures.decision import SessionRisk
from galahad_futures.strategy import build_strategy, strategy_kwargs_from_config

ENGINE_NAME = "nautilus_live"
ENGINE_VERSION = "nautilus_trader-1.231.0"
VENUE = "BINANCE"
MODE = "testnet"  # the only mode this backend can ever produce

TESTNET_API_KEY_ENV = "BINANCE_TESTNET_API_KEY"
TESTNET_API_SECRET_ENV = "BINANCE_TESTNET_API_SECRET"
DEFAULT_MAX_MINUTES = 30.0

_INTERVAL_UNITS = {"m": "MINUTE", "h": "HOUR", "d": "DAY"}


# --- pure helpers (no nautilus, no network) -------------------------------


def testnet_credentials(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Read testnet credentials from the environment; hard error if absent.

    Only the testnet-specific variable names are consulted — the mainnet
    names (``BINANCE_API_KEY`` / ``BINANCE_API_SECRET``) are never read.
    """
    env = os.environ if env is None else env
    key = (env.get(TESTNET_API_KEY_ENV) or "").strip()
    secret = (env.get(TESTNET_API_SECRET_ENV) or "").strip()
    missing = [
        name
        for name, value in ((TESTNET_API_KEY_ENV, key), (TESTNET_API_SECRET_ENV, secret))
        if not value
    ]
    if missing:
        raise RuntimeError(
            "engine=nautilus_live: missing Binance USDT-M futures TESTNET "
            f"credentials: {', '.join(missing)}. Set both env vars with keys "
            "from the futures testnet (testnet.binancefuture.com). The mainnet "
            "key names BINANCE_API_KEY / BINANCE_API_SECRET are never read by "
            "this backend."
        )
    return key, secret


def testnet_max_minutes(cfg: dict[str, Any]) -> float:
    """Session bound in minutes (``testnet.max_minutes``, default 30)."""
    opts = dict(cfg.get("testnet") or {})
    minutes = float(opts.get("max_minutes", DEFAULT_MAX_MINUTES))
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError(f"testnet.max_minutes must be a positive number (got {minutes!r})")
    return minutes


def testnet_url_overrides(cfg: dict[str, Any]) -> dict[str, str]:
    """Optional endpoint overrides for the testnet clients.

    ``testnet.rest_url`` (https) and ``testnet.ws_url`` (wss) remap the
    environment's default endpoints — e.g. when the default
    ``demo-fapi.binance.com`` is unreachable but the equivalent
    ``testnet.binancefuture.com`` host serves the same API. Fail closed on
    malformed values; absent keys mean the environment defaults apply.
    """
    opts = dict(cfg.get("testnet") or {})
    out: dict[str, str] = {}
    rest = opts.get("rest_url")
    if rest is not None:
        rest = str(rest)
        if not rest.startswith("https://"):
            raise ValueError(f"testnet.rest_url must be an https URL (got {rest!r})")
        out["base_url_http"] = rest.rstrip("/")
    ws = opts.get("ws_url")
    if ws is not None:
        ws = str(ws)
        if not ws.startswith("wss://"):
            raise ValueError(f"testnet.ws_url must be a wss URL (got {ws!r})")
        out["base_url_ws"] = ws.rstrip("/")
    return out


def bar_type_str(symbol: str, interval: str) -> str:
    """Map (BTCUSDT, 1h) → 'BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL'.

    The Binance data client only aggregates time bars externally (kline
    websocket); minute/hour/day intervals are supported.
    """
    s = str(interval).strip().lower()
    unit = s[-1:] if s else ""
    if unit not in _INTERVAL_UNITS or not s[:-1].isdigit() or int(s[:-1]) <= 0:
        raise ValueError(
            f"unsupported interval {interval!r} for live bars "
            "(expected a positive count of m/h/d, e.g. 1m, 1h, 4h, 1d)"
        )
    return f"{symbol}-PERP.{VENUE}-{int(s[:-1])}-{_INTERVAL_UNITS[unit]}-LAST-EXTERNAL"


def order_delta_for_decision(
    *,
    target_signed_leverage: float,
    equity: float,
    mark: float,
    current_qty: float,
    min_order_notional: float = 1e-6,
) -> float:
    """Signed quantity delta implied by a gate-passed decision.

    Same projection as the backtest backends: desired net qty =
    target * equity / mark; deltas below the dust threshold are dropped.
    """
    equity = float(equity)
    mark = float(mark)
    if equity <= 0 or mark <= 0:
        return 0.0
    desired_qty = float(target_signed_leverage) * equity / mark
    delta = desired_qty - float(current_qty)
    if abs(delta) * mark < float(min_order_notional):
        return 0.0
    return delta


def quantize_delta(delta: float, size_precision: int) -> float:
    """Round a delta toward zero onto the instrument size grid.

    Toward zero (not half-even) so quantization never increases exposure
    beyond the risk-gate-passed target.
    """
    factor = 10.0 ** int(size_precision)
    q = math.floor(abs(float(delta)) * factor + 1e-12) / factor
    return math.copysign(q, float(delta)) if q > 0 else 0.0


def passes_min_notional(delta_q: float, mark: float, min_notional: float) -> bool:
    """Venue minimum-notional gate (Binance rejects smaller orders, -4164)."""
    return abs(float(delta_q)) * float(mark) >= float(min_notional)


def infer_external_flatten(
    *,
    expected_qty: float,
    venue_qty: float,
    open_orders: int,
    qty_tolerance: float = 1e-8,
) -> bool:
    """Heuristic liquidation/ADL detector for live sessions.

    The venue reports flat while the decision layer expects exposure and
    no orders are working — the position was closed outside our order
    flow (liquidation, ADL, or manual interference on the account).
    """
    return (
        abs(float(expected_qty)) > float(qty_tolerance)
        and abs(float(venue_qty)) <= float(qty_tolerance)
        and int(open_orders) == 0
    )


def resolve_live_equity(
    venue_equity: float | None, last_good_equity: float | None
) -> float | None:
    """Fail-closed equity resolution for a live bar.

    The venue reading wins; on outage fall back to the last known-good
    venue reading — never the session peak (peak as current equity masks
    drawdown, blocks invalidation, and can clear an active LOSS_HALTED
    mid-outage). ``None`` means no trustworthy snapshot exists yet:
    decide nothing, submit nothing.
    """
    if venue_equity is not None:
        return float(venue_equity)
    return last_good_equity


def external_flatten_confirmed(
    streak: int,
    *,
    observed_flat: bool,
    required: int = 2,
) -> tuple[int, bool]:
    """Streak gate for the external-flatten heuristic.

    A venue fill/user-stream update can lag one bar past submission, so a
    single quiet-bar flat reading false-trips the heuristic. It only fires
    after ``required`` consecutive quiet-bar observations. Returns the new
    streak and whether the observation is confirmed.
    """
    new_streak = int(streak) + 1 if observed_flat else 0
    return new_streak, new_streak >= int(required)


def reconcile_execution(
    *,
    orders_submitted: int,
    orders_filled: int,
    expected_qty: float,
    venue_qty: float | None,
    qty_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Session reconciliation contract (summary field ``reconciliation``).

    ``position_mismatch`` compares the decision layer's expected final
    net quantity with the venue-reported net position. An unavailable
    venue state fails closed (mismatch=True).
    """
    if venue_qty is None:
        mismatch = True
    else:
        mismatch = abs(float(expected_qty) - float(venue_qty)) > float(qty_tolerance)
    return {
        "orders_submitted": int(orders_submitted),
        "orders_filled": int(orders_filled),
        "position_mismatch": bool(mismatch),
    }


def precheck_gate(cfg: dict[str, Any]) -> None:
    """Fail closed before any network I/O: the testnet gate must be open.

    Callers pass the effective session config (mode already forced to
    ``testnet`` by this module / the engine). A closed gate (kill switch
    on, or ``enable_testnet`` off) raises ``RuntimeError`` before any
    client is constructed.
    """
    session = SessionRisk.from_config(cfg, start_equity=float(cfg.get("initial_equity", 10_000)))
    if session.gate.live_blocked():
        raise RuntimeError(
            "engine=nautilus_live: testnet gate closed — set risk.enable_testnet: true "
            "and risk.kill_switch: false in config.yaml to rehearse the live path "
            "(paper/nautilus engines are unaffected)"
        )


def build_live_result(
    *,
    cfg: dict[str, Any],
    symbol: str,
    strategy_name: str,
    strategy_kwargs: dict[str, Any],
    session: SessionRisk,
    fills: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
    account_curve: list[dict[str, Any]],
    funding_events: list[dict[str, Any]],
    liquidation_events: list[dict[str, Any]],
    positions: dict[str, Any],
    orders_submitted: int,
    orders_filled: int,
    instrument_missing_skips: int = 0,
    orders_denied: int = 0,
    orders_rejected: int = 0,
    dust_skips: int = 0,
    denials: list[dict[str, Any]] | None = None,
    warmup_bars: int,
    live_bars: int,
    initial_equity: float,
    final_equity: float,
    expected_qty: float,
    venue_qty: float | None,
    session_seconds: float,
    max_minutes: float,
    equity_source: str,
) -> dict[str, Any]:
    """Assemble the live-session result dict (same shape as the other backends).

    Pure: all inputs are already-collected session artifacts, so the
    summary contract is unit-testable without nautilus or network.
    """
    gate = session.gate
    reconciliation = reconcile_execution(
        orders_submitted=orders_submitted,
        orders_filled=orders_filled,
        expected_qty=expected_qty,
        venue_qty=venue_qty,
    )
    eq_series = [float(s["equity"]) for s in equity_curve]
    rets: list[float] = []
    if len(eq_series) >= 2:
        a = np.asarray(eq_series, dtype=float)
        rets = (np.diff(a) / np.maximum(a[:-1], 1e-9)).tolist()
    return {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "venue": VENUE,
        "strategy": strategy_name,
        "strategy_kwargs": dict(strategy_kwargs),
        "symbol": symbol,
        "bars": int(live_bars),
        "warmup_bars": int(warmup_bars),
        "evaluate_from": 0,
        "n_fills": len(fills),
        "n_fills_oos": len(fills),
        "n_risk_rejects": len(gate.rejects),
        "liquidated": bool(liquidation_events),
        "invalidated": gate.invalidated,
        "invalidation_reason": gate.invalidation_reason or None,
        "invalidation_events": list(gate.invalidation_events),
        "loss_halt_events": list(gate.loss_halt_events),
        "decision_phase_final": session.phase(),
        "peak_equity": float(gate.peak_equity),
        "max_drawdown": float(gate.max_drawdown_seen),
        "initial_equity": float(initial_equity),
        "final_equity": float(final_equity),
        "equity_curve": list(equity_curve),
        "equity_curve_len": len(equity_curve),
        "account_curve": list(account_curve),
        "oos_equity_curve_len": len(equity_curve),
        "returns_oos": rets,
        # Live funding is settled by the venue inside the account balance;
        # nothing is injected by the harness on the live path.
        "total_funding": 0.0,
        "n_funding_events": len(funding_events),
        "funding_events": list(funding_events),
        "fills": list(fills),
        # Venue fills vs decision-bar close (arrival): total shortfall is
        # observable; the spread/impact split is None for venue fills.
        "tca": tca_from_fills(list(fills)),
        "derisk": gate.derisk_summary(),
        "risk_rejects": gate.rejects,
        "risk_decisions": session.risk_decisions,
        "risk_decisions_tail": session.risk_decisions[-20:],
        "liquidation_events": list(liquidation_events),
        "positions": dict(positions),
        "orders_submitted": int(orders_submitted),
        "orders_filled": int(orders_filled),
        "instrument_missing_skips": int(instrument_missing_skips),
        "orders_denied": int(orders_denied),
        "orders_rejected": int(orders_rejected),
        "dust_skips": int(dust_skips),
        "denials": list(denials or []),
        "reconciliation": reconciliation,
        "expected_final_qty": float(expected_qty),
        "venue_final_qty": None if venue_qty is None else float(venue_qty),
        "session_seconds": float(session_seconds),
        "testnet_max_minutes": float(max_minutes),
        "equity_source": equity_source,
    }


def _import_nautilus() -> None:
    try:
        import nautilus_trader  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via CLI test
        raise RuntimeError(
            "engine=nautilus_live requires the optional dependency "
            "nautilus_trader==1.231.0 (package extra 'nautilus'); the "
            "reference paper engine has no extra dependencies"
        ) from exc


# --- live session entry point ---------------------------------------------


def run_live_testnet_session(
    bars: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    symbol: str | None = None,
    strategy_name: str | None = None,
    strategy_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one bounded testnet session. ``bars`` is the warmup history.

    The warmup bars seed the strategy's indicator history (identical to
    the paper/nautilus engines' inputs); live testnet bars then drive the
    shared decision driver in real time. Returns the result dict consumed
    by ``report.build_summary`` / ``report.write_journal``.
    """
    cfg = {**cfg, "mode": MODE}  # forced: this backend can never run "live"
    api_key, api_secret = testnet_credentials()
    _import_nautilus()
    precheck_gate(cfg)
    max_minutes = testnet_max_minutes(cfg)

    symbol = symbol or str(cfg.get("symbol", "BTCUSDT"))
    strat_cfg = dict(cfg.get("strategy") or {})
    name = strategy_name or str(strat_cfg.get("name", "dual_ma"))
    kw = strategy_kwargs if strategy_kwargs is not None else strategy_kwargs_from_config(strat_cfg)

    return _run_node(
        warmup=bars,
        cfg=cfg,
        symbol=symbol,
        strategy_name=name,
        strategy_kwargs=kw,
        api_key=api_key,
        api_secret=api_secret,
        max_minutes=max_minutes,
    )


# --- TradingNode wiring (lazy: nautilus + network only below this line) ----


def _run_node(
    *,
    warmup: pd.DataFrame,
    cfg: dict[str, Any],
    symbol: str,
    strategy_name: str,
    strategy_kwargs: dict[str, Any],
    api_key: str,
    api_secret: str,
    max_minutes: float,
) -> dict[str, Any]:
    import threading
    import time
    from datetime import datetime, timezone

    from nautilus_trader.adapters.binance.common.enums import (
        BinanceAccountType,
        BinanceEnvironment,
    )
    from nautilus_trader.adapters.binance.common.symbol import BinanceSymbol
    from nautilus_trader.adapters.binance.config import (
        BinanceDataClientConfig,
        BinanceExecClientConfig,
        BinanceInstrumentProviderConfig,
    )
    from nautilus_trader.adapters.binance.factories import (
        BinanceLiveDataClientFactory,
        BinanceLiveExecClientFactory,
    )
    from nautilus_trader.common import Environment
    from nautilus_trader.config import LoggingConfig, StrategyConfig, TradingNodeConfig
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.model.currencies import Currency
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.enums import OmsType, OrderSide
    from nautilus_trader.model.identifiers import InstrumentId, Venue
    from nautilus_trader.model.objects import Quantity
    from nautilus_trader.trading.strategy import Strategy

    interval = str(cfg.get("interval", "1h"))
    default_leverage = float(cfg.get("default_leverage", 3.0))
    cfg_initial = float(cfg.get("initial_equity", 10_000))
    venue = Venue(VENUE)
    instrument_id = InstrumentId.from_str(f"{symbol}-PERP.{VENUE}")
    bar_type = BarType.from_str(bar_type_str(symbol, interval))
    usdt = Currency.from_str("USDT")
    strategy = build_strategy(strategy_name, **strategy_kwargs)

    def _iso(ts_ns: int) -> str:
        return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        )

    class TestnetDecisionStrategy(Strategy):
        """Replays the shared decision stream through live testnet execution.

        Indicator history is seeded with the session's warmup bars; every
        completed testnet bar is appended and re-evaluated through
        ``strategy.targets`` + ``SessionRisk`` — the identical decision
        driver the paper engine runs. Only gate-passed targets become
        market orders (netting OMS); the venue account is the equity and
        position source of truth.
        """

        def __init__(self, strat_cfg: StrategyConfig):
            super().__init__(strat_cfg)
            self.started = False
            self.halted = False
            self.session: SessionRisk | None = None
            self.initial_equity = cfg_initial
            self.equity_source = "config_fallback"
            self.expected_qty = 0.0
            self.submitted = 0
            self.instrument_missing_skips = 0
            self.orders_denied = 0
            self.orders_rejected = 0
            self.dust_skips = 0
            self.denials: list[dict[str, Any]] = []
            self.fills: list[dict[str, Any]] = []
            self.liquidated_events: list[dict[str, Any]] = []
            self.equity_curve: list[dict[str, Any]] = []
            self.account_curve: list[dict[str, Any]] = []
            self.bars_seen = 0
            # Last known-good venue equity reading; the outage fallback
            # (never the session peak — see resolve_live_equity).
            self._last_good_equity: float | None = None
            # Consecutive quiet-bar external-flatten observations; the
            # heuristic fires only once the streak confirms (stale-flat
            # venue reads lagging one bar past submission false-trip it).
            self._quiet_flat_streak = 0
            # Arrival price (decision bar close) of the most recently
            # submitted order — one order per bar, fills arrive before the
            # next bar, so a single slot suffices for the TCA record.
            self._pending_arrival: float | None = None
            self._rows: list[dict[str, Any]] = [
                {
                    "ts": str(row["ts"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                for _, row in warmup.iterrows()
            ]

        # -- venue state reads (None = unavailable, never guess) ---------

        def _venue_qty(self) -> float | None:
            try:
                return float(self.portfolio.net_position(instrument_id))
            except Exception:
                return None

        def _venue_equity(self) -> float | None:
            try:
                account = self.portfolio.account(venue)
                if account is None:
                    return None
                balance = float(account.balance_total(usdt))
                upnl = self.portfolio.unrealized_pnl(instrument_id)
                return balance + (float(upnl) if upnl is not None else 0.0)
            except Exception:
                return None

        def _open_orders(self) -> int:
            try:
                return len(self.cache.orders_open(instrument_id=instrument_id))
            except Exception:
                return 0

        def _leverage(self) -> float:
            try:
                return float(self.portfolio.account(venue).leverage(instrument_id))
            except Exception:
                return default_leverage

        # -- callbacks ----------------------------------------------------

        def on_start(self) -> None:
            self.started = True
            # Load the venue instrument into the cache before any order path
            # can touch it; submissions gate on its presence below.
            self.subscribe_instrument(instrument_id)
            self.subscribe_bars(bar_type)

        def on_event(self, event) -> None:
            cls = type(event).__name__
            if cls == "OrderFilled":
                self.fills.append(
                    {
                        "ts": _iso(int(getattr(event, "ts_init", 0))),
                        "symbol": symbol,
                        "side": str(getattr(event, "order_side", "")).split(".")[-1],
                        "qty": float(getattr(event, "last_qty", 0.0) or 0.0),
                        "price": float(getattr(event, "last_px", 0.0) or 0.0),
                        "fee": _money_float(getattr(event, "commission", None)),
                        "realized_pnl": 0.0,
                        "note": "nautilus_live",
                        "leverage": default_leverage,
                        # Decision bar close this order was submitted at;
                        # venue fills carry no spread/impact decomposition.
                        "arrival_price": self._pending_arrival,
                    }
                )
            elif cls == "OrderDenied":
                # A denied order breaks decision/venue parity; fail closed
                # by halting the session (kept separate from liquidations).
                self.orders_denied += 1
                self.denials.append(
                    {
                        "ts": _iso(int(getattr(event, "ts_init", 0))),
                        "reason": str(getattr(event, "reason", ""))[:200],
                    }
                )
                self.halted = True
            elif cls == "OrderRejected":
                # Venue-side rejection after submission. Dust deltas are
                # gated pre-trade (min qty / min notional), so a rejection
                # reaching us is genuinely abnormal — halt fail-closed.
                self.orders_rejected += 1
                self.denials.append(
                    {
                        "ts": _iso(int(getattr(event, "ts_init", 0))),
                        "reason": ("rejected: " + str(getattr(event, "reason", "")))[:200],
                    }
                )
                self.halted = True
            elif "Liquidation" in cls:
                self.liquidated_events.append(
                    {"ts": _iso(int(getattr(event, "ts_init", 0))), "detector": "venue_event"}
                )

        def on_bar(self, bar) -> None:
            if self.halted:
                return
            submitted_before = self.submitted
            ts_iso = _iso(bar.ts_init)
            self._rows.append(
                {
                    "ts": ts_iso,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
            )
            self.bars_seen += 1
            mark = float(bar.close)

            if self.session is None:
                eq0 = self._venue_equity()
                if eq0 is not None:
                    self.initial_equity = eq0
                    self.equity_source = "venue"
                self.session = SessionRisk.from_config(cfg, start_equity=self.initial_equity)
            session = self.session

            pre_eq = resolve_live_equity(self._venue_equity(), self._last_good_equity)
            if pre_eq is None:
                # No trustworthy equity snapshot yet this session: fail
                # closed — no state transitions, no decisions, no order
                # submissions. A venue-reported liquidation still halts.
                if self.liquidated_events:
                    session.note_liquidation(ts=ts_iso)
                    self.halted = True
                return
            self._last_good_equity = pre_eq
            session.update_equity(pre_eq, ts=ts_iso)

            targets = strategy.targets(pd.DataFrame(self._rows))
            raw_target = float(targets.iloc[-1]) if len(targets) else 0.0
            venue_qty = self._venue_qty()
            qty = venue_qty if venue_qty is not None else self.expected_qty
            decision = session.evaluate_target(
                symbol=symbol,
                raw_target=raw_target,
                mark=mark,
                pre_trade_equity=pre_eq,
                current_qty=qty,
                leverage=self._leverage(),
                ts=ts_iso,
            )
            if decision.allowed and not self.liquidated_events:
                instrument = self.cache.instrument(instrument_id)
                if instrument is None:
                    # Fail closed: never submit against an unknown instrument
                    # (the venue RiskEngine would deny it anyway, which reads
                    # as a spurious external-flatten downstream).
                    self.instrument_missing_skips += 1
                    self.expected_qty = qty
                else:
                    size_precision = int(instrument.size_precision)
                    min_qty = float(instrument.size_increment)
                    min_notional_attr = getattr(instrument, "min_notional", None)
                    min_notional = (
                        float(min_notional_attr) if min_notional_attr is not None else 0.0
                    )
                    delta = order_delta_for_decision(
                        target_signed_leverage=decision.target_signed_leverage,
                        equity=pre_eq,
                        mark=mark,
                        current_qty=qty,
                    )
                    delta_q = quantize_delta(delta, size_precision)
                    if abs(delta_q) >= min_qty and passes_min_notional(
                        delta_q, mark, min_notional
                    ):
                        side = OrderSide.BUY if delta_q > 0 else OrderSide.SELL
                        self.submit_order(
                            self.order_factory.market(
                                instrument_id, side, Quantity(abs(delta_q), size_precision)
                            )
                        )
                        self.submitted += 1
                        self.expected_qty = qty + delta_q
                        self._pending_arrival = mark
                    else:
                        # Dust delta (below min qty or min notional): skip the
                        # order and do NOT move expected_qty — the venue will
                        # not hold it, so the decision layer must not either.
                        self.dust_skips += 1
                        self.expected_qty = qty

            venue_qty_after = self._venue_qty()
            # The fill events for an order submitted on this bar arrive
            # asynchronously AFTER this callback returns; running the
            # flatten inference on the submission bar itself races and
            # reads as a false liquidation. Only infer on quiet bars — and
            # even there a fill/user-stream update can lag one bar past
            # submission, so a single quiet-bar flat reading is not enough:
            # the streak gate requires consecutive observations.
            self._quiet_flat_streak, confirmed_flat = external_flatten_confirmed(
                self._quiet_flat_streak,
                observed_flat=(
                    venue_qty_after is not None
                    and self.submitted == submitted_before
                    and infer_external_flatten(
                        expected_qty=self.expected_qty,
                        venue_qty=venue_qty_after,
                        open_orders=self._open_orders(),
                    )
                ),
            )
            if confirmed_flat:
                self.liquidated_events.append({"ts": ts_iso, "detector": "external_flatten"})
            if self.liquidated_events:
                session.note_liquidation(ts=ts_iso)
                self.halted = True

            # pre_eq is non-None past the outage guard above, so post_eq
            # always carries a real equity number into the curves.
            post_eq = resolve_live_equity(self._venue_equity(), pre_eq)
            self._last_good_equity = post_eq
            self.equity_curve.append({"ts": ts_iso, "equity": post_eq})
            self.account_curve.append({"ts": ts_iso, "equity": pre_eq})
            session.update_equity(post_eq, ts=ts_iso)

    url_overrides = testnet_url_overrides(cfg)
    # Load exactly the session instrument into the cache (the provider's
    # default loads nothing, and every order gates on instrument presence).
    instrument_provider = BinanceInstrumentProviderConfig(
        load_all=False,
        load_ids=frozenset({instrument_id}),
    )
    node_config = TradingNodeConfig(
        environment=Environment.SANDBOX,
        trader_id="GALAHAD-TESTNET-001",
        logging=LoggingConfig(log_level="INFO"),
        data_clients={
            VENUE: BinanceDataClientConfig(
                api_key=api_key,
                api_secret=api_secret,
                account_type=BinanceAccountType.USDT_FUTURES,
                environment=BinanceEnvironment.TESTNET,
                instrument_provider=instrument_provider,
                **url_overrides,
            ),
        },
        exec_clients={
            VENUE: BinanceExecClientConfig(
                api_key=api_key,
                api_secret=api_secret,
                account_type=BinanceAccountType.USDT_FUTURES,
                environment=BinanceEnvironment.TESTNET,
                instrument_provider=instrument_provider,
                futures_leverages={BinanceSymbol(symbol): max(1, int(round(default_leverage)))},
                # Netting (venue one-way mode): order events carry no hedge
                # position ids.
                use_position_ids=False,
                **url_overrides,
            ),
        },
        timeout_connection=30.0,
        timeout_disconnection=10.0,
    )
    # Structural guarantee: mainnet can never be constructed by this backend.
    # Explicit raise, not assert: the guarantee must survive `python -O`.
    for client_cfg in (*node_config.data_clients.values(), *node_config.exec_clients.values()):
        if client_cfg.environment is not BinanceEnvironment.TESTNET:
            raise RuntimeError(
                "engine=nautilus_live: internal invariant violated — a client "
                "was configured outside BinanceEnvironment.TESTNET; aborting "
                "before node construction"
            )

    node = TradingNode(config=node_config)
    node.add_data_client_factory(VENUE, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(VENUE, BinanceLiveExecClientFactory)
    node.build()
    strategy_obj = TestnetDecisionStrategy(
        StrategyConfig(strategy_id="GALAHAD-TESTNET-DECISION", oms_type=OmsType.NETTING)
    )
    node.trader.add_strategy(strategy_obj)

    started = time.monotonic()
    runner = threading.Thread(
        target=node.run,
        kwargs={"raise_exception": True},
        daemon=True,
        name="galahad-testnet-node",
    )
    runner.start()
    runner_died_early = False
    try:
        deadline = started + max_minutes * 60.0
        while runner.is_alive() and time.monotonic() < deadline:
            if strategy_obj.halted:
                break
            time.sleep(0.5)
        # A runner that died before the deadline without a risk halt means
        # the strategy/node thread crashed (raise_exception=True); the
        # session must not be reported as a clean run.
        runner_died_early = (
            not runner.is_alive()
            and time.monotonic() < deadline
            and not strategy_obj.halted
        )
    finally:
        # Halt new submissions before the stop is scheduled: until node.stop
        # actually runs, bars keep arriving on the daemon node and the
        # strategy would otherwise keep evaluating and submitting.
        strategy_obj.halted = True
        try:
            node.kernel.loop.call_soon_threadsafe(node.stop)
        except Exception:
            pass
        runner.join(timeout=float(node_config.timeout_disconnection) + 30.0)
    session_seconds = time.monotonic() - started

    if not strategy_obj.started:
        try:
            node.dispose()
        except Exception:
            pass
        raise RuntimeError(
            "engine=nautilus_live: trading node exited before the strategy started "
            "(connection or authentication failure — check testnet credentials and "
            "network reachability); no summary was produced"
        )

    if runner_died_early:
        try:
            node.dispose()
        except Exception:
            pass
        raise RuntimeError(
            "engine=nautilus_live: trading node thread died mid-session before "
            "the deadline without a risk halt (unhandled exception in the "
            "strategy/node thread — see the thread traceback on stderr); "
            "refusing to report a clean session result"
        )

    session = strategy_obj.session or SessionRisk.from_config(
        cfg, start_equity=strategy_obj.initial_equity
    )

    venue_qty: float | None = None
    positions: dict[str, Any] = {}
    try:
        venue_qty = float(node.portfolio.net_position(instrument_id))
        pos_list = list(node.cache.positions(instrument_id=instrument_id))
        if pos_list:
            pos = pos_list[-1]
            qty = float(pos.quantity)
            positions[str(instrument_id)] = {
                "qty": qty,
                "entry_price": float(pos.avg_px_open),
                "leverage": default_leverage,
                "side": "long" if qty > 0 else ("short" if qty < 0 else "flat"),
            }
    except Exception:
        venue_qty = None
        positions = {}

    final_equity = (
        strategy_obj.equity_curve[-1]["equity"]
        if strategy_obj.equity_curve
        else strategy_obj.initial_equity
    )
    result = build_live_result(
        cfg=cfg,
        symbol=symbol,
        strategy_name=strategy_name,
        strategy_kwargs=strategy_kwargs,
        session=session,
        fills=strategy_obj.fills,
        equity_curve=strategy_obj.equity_curve,
        account_curve=strategy_obj.account_curve,
        funding_events=[],
        liquidation_events=strategy_obj.liquidated_events,
        positions=positions,
        orders_submitted=strategy_obj.submitted,
        orders_filled=len(strategy_obj.fills),
        instrument_missing_skips=strategy_obj.instrument_missing_skips,
        orders_denied=strategy_obj.orders_denied,
        orders_rejected=strategy_obj.orders_rejected,
        dust_skips=strategy_obj.dust_skips,
        denials=strategy_obj.denials,
        warmup_bars=len(warmup),
        live_bars=strategy_obj.bars_seen,
        initial_equity=strategy_obj.initial_equity,
        final_equity=float(final_equity),
        expected_qty=strategy_obj.expected_qty,
        venue_qty=venue_qty,
        session_seconds=session_seconds,
        max_minutes=max_minutes,
        equity_source=strategy_obj.equity_source,
    )
    try:
        node.dispose()
    except Exception:
        pass
    return result


def _money_float(value: object) -> float:
    """Parse a Money/Money-ish value (e.g. '0.005 USDT'); None → 0.0."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    parts = str(value).split()
    if not parts:
        raise ValueError(f"unparseable money value: {value!r}")
    return float(parts[0])

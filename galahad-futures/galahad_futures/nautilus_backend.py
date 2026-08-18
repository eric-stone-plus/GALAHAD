"""NautilusTrader execution backend for GALAHAD paper sessions.

Runs the SAME decision stream as the reference paper book through
NautilusTrader's event-driven backtest engine (pinned ``1.231.0``):
synthetic L1 books derived from the OHLC bars, taker fees, and
Nautilus's own margin machinery.

Funding note: the v1.231.0 backtest engine has no funding-settlement
path (verified against the v1.231.0 source — the backtest engine never
processes ``FundingRateUpdate``; funding settlement arrived with the
v2 line). The backend therefore applies the reference per-bar funding
convention in the harness itself — payment = qty * mark * rate after
each bar's rebalance, exactly the paper book's timing — records the
events, and feeds the *funding-adjusted* equity into the shared risk
gate so decision identity with the paper engine holds.

Paper-only by construction: a ``BacktestEngine`` has no live path. The
reference book (``engine.py``) remains the arbiter in parity runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from galahad_futures.decision import SessionRisk
from galahad_futures.strategy import build_strategy, strategy_kwargs_from_config

ENGINE_NAME = "nautilus"
ENGINE_VERSION = "nautilus_trader-1.231.0"

_PRICE_PRECISION = 2
_SIZE_PRECISION = 3


def _iso_from_ns(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )


def _ns_from_ts(ts: object) -> int:
    return int(pd.Timestamp(str(ts)).value)


def _silence_nautilus_logging() -> None:
    """Nautilus INFO logs pollute the CLI JSON contract; quiet them."""
    for name, logger in logging.Logger.manager.loggerDict.items():
        if str(name).startswith("nautilus"):
            try:
                getattr(logger, "setLevel")(logging.CRITICAL)
            except Exception:
                pass


def _import_nautilus():
    try:
        import nautilus_trader  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via CLI test
        raise RuntimeError(
            "engine=nautilus requires the optional dependency "
            "nautilus_trader==1.231.0 (package extra 'nautilus'); the "
            "reference paper engine has no extra dependencies"
        ) from exc


def _round_price(value: float) -> str:
    return f"{float(value):.{_PRICE_PRECISION}f}"


def _round_qty(value: float) -> str:
    return f"{float(value):.{_SIZE_PRECISION}f}"


def _money_float(value: object) -> float:
    """Parse a Money/Money-ish column value (e.g. '2.00559379 USDT')."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(str(value).split()[0])


def _engine_precision(cfg: dict[str, Any]) -> tuple[int, int]:
    """Price/size precision for the nautilus instrument.

    Defaults match the venue (2dp price / 3dp size for BTCUSDT). Parity
    runs may raise both to isolate execution-mechanics divergence from
    venue quantization; the reference book is not quantized.
    """
    opts = dict(cfg.get("engine_nautilus") or {})
    price_precision = int(opts.get("price_precision", _PRICE_PRECISION))
    size_precision = int(opts.get("size_precision", _SIZE_PRECISION))
    return price_precision, size_precision


def _build_instrument(symbol: str, cfg: dict[str, Any]):
    """CryptoPerpetual instrument consistent with the reference book.

    margin_init = 1/leverage matches the book's margin accounting
    (notional / leverage); margin_maint and taker fee come straight from
    the session config. Precisions come from ``engine_nautilus`` config
    (venue defaults; parity runs override).
    """
    from nautilus_trader.model.currencies import Currency
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.instruments import CryptoPerpetual
    from nautilus_trader.model.objects import Money, Price, Quantity

    venue = Venue("BINANCE")
    raw = symbol.removesuffix("-PERP").removesuffix("PERP")
    raw = raw or symbol
    leverage = Decimal(str(float(cfg.get("default_leverage", 3.0))))
    fee_bps = float(cfg.get("fee_bps", 4.0))
    mm_rate = float(cfg.get("maintenance_margin_rate", 0.005))
    price_precision, size_precision = _engine_precision(cfg)
    price_inc = f"0.{'0' * (price_precision - 1)}1"
    size_inc = f"0.{'0' * (size_precision - 1)}1"
    max_qty = f"{10 ** 3:.{size_precision}f}"
    min_qty = size_inc
    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol(f"{symbol}-PERP"), venue),
        raw_symbol=Symbol(raw),
        base_currency=Currency.from_str(raw.removesuffix("USDT")),
        quote_currency=Currency.from_str("USDT"),
        settlement_currency=Currency.from_str("USDT"),
        is_inverse=False,
        price_precision=price_precision,
        size_precision=size_precision,
        price_increment=Price.from_str(price_inc),
        size_increment=Quantity.from_str(size_inc),
        multiplier=Quantity.from_str("1"),
        max_quantity=Quantity.from_str(max_qty),
        min_quantity=Quantity.from_str(min_qty),
        max_notional=None,
        min_notional=Money.from_str("5.00 USDT"),
        max_price=None,
        min_price=None,
        margin_init=Decimal(1) / leverage,
        margin_maint=Decimal(str(mm_rate)),
        maker_fee=Decimal("0"),
        taker_fee=Decimal(str(fee_bps / 10_000.0)),
        tick_scheme_name=None,
        ts_event=0,
        ts_init=0,
    ), venue


def run_nautilus_on_bars(
    bars: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    symbol: str | None = None,
    strategy_name: str | None = None,
    strategy_kwargs: dict[str, Any] | None = None,
    evaluate_from: int = 0,
) -> dict[str, Any]:
    """Run the decision stream through the NautilusTrader backtest engine.

    Produces the same result shape as ``engine.run_paper_on_bars`` (minus
    the in-memory ``book`` object) so the shared report module can render
    both engines' sessions identically.
    """
    _import_nautilus()
    _silence_nautilus_logging()

    from nautilus_trader.analysis.reporter import ReportProvider
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
    from nautilus_trader.model.currencies import Currency
    from nautilus_trader.model.data import Bar, BarSpecification, BarType
    from nautilus_trader.model.enums import (
        AccountType,
        BarAggregation,
        OmsType,
        OrderSide,
        PriceType,
    )
    from nautilus_trader.model.objects import Money, Price, Quantity
    from nautilus_trader.trading.strategy import Strategy

    symbol = symbol or str(cfg.get("symbol", "BTCUSDT"))
    strat_cfg = dict(cfg.get("strategy") or {})
    name = strategy_name or str(strat_cfg.get("name", "dual_ma"))
    kw = strategy_kwargs if strategy_kwargs is not None else strategy_kwargs_from_config(strat_cfg)
    strategy = build_strategy(name, **kw)
    targets = strategy.targets(bars)
    # Map target series by bar ts_init (ns) for the strategy callback.
    target_by_ns: dict[int, float] = {}
    for i, row in bars.iterrows():
        ts_ns = _ns_from_ts(row["ts"])
        target_by_ns[ts_ns] = float(targets.iloc[i]) if i < len(targets) else 0.0

    instrument, venue = _build_instrument(symbol, cfg)
    instrument_id = instrument.id
    base_currency = Currency.from_str("USDT")
    spec = BarSpecification(1, BarAggregation.HOUR, PriceType.LAST)
    bar_type = BarType(instrument_id, spec)

    session = SessionRisk.from_config(cfg, start_equity=float(cfg.get("initial_equity", 10_000)))
    funding_rate = float(cfg.get("funding_rate_per_bar", 0.0))
    price_precision, size_precision = _engine_precision(cfg)

    def rp(value: float) -> str:
        return f"{float(value):.{price_precision}f}"

    def rq(value: float) -> str:
        return f"{float(value):.{size_precision}f}"

    nautilus_bars: list[Bar] = []
    last_ts_ns = 0
    for _, row in bars.iterrows():
        ts_ns = _ns_from_ts(row["ts"])
        last_ts_ns = ts_ns
        nautilus_bars.append(
            Bar(
                bar_type,
                Price.from_str(rp(row["open"])),
                Price.from_str(rp(row["high"])),
                Price.from_str(rp(row["low"])),
                Price.from_str(rp(row["close"])),
                Quantity.from_str(rq(row["volume"])),
                ts_ns,
                ts_ns,
            )
        )

    default_leverage = Decimal(str(float(cfg.get("default_leverage", 3.0))))
    bar_ts_boundary = _ns_from_ts(bars.iloc[int(evaluate_from)]["ts"]) if len(bars) else 0

    class DecisionDrivenStrategy(Strategy):
        """Replays the shared decision stream through Nautilus execution.

        Order fills and account events are applied by the engine *after*
        the strategy callback returns, so in-callback reads of the
        account and portfolio are one bar stale. The strategy therefore
        keeps a paper-convention projection (position, fees, funding,
        mark-to-market) from its own submitted deltas for the shared
        risk gate and the parity equity curve; Nautilus remains the
        actual executor, and its raw account curve is exported alongside
        in the result for transparency.
        """

        def __init__(self, strat_cfg: StrategyConfig):
            super().__init__(strat_cfg)
            self.halted = False
            self.liquidated_events: list[dict[str, Any]] = []
            self.funding_events: list[dict[str, Any]] = []
            self.total_funding = 0.0
            self.equity_curve: list[dict[str, Any]] = []
            self.account_curve: list[dict[str, Any]] = []
            self.expected_qty = 0.0
            self.projected_equity = float(cfg.get("initial_equity", 10_000))
            self.submitted = 0
            self.prev_mark: float | None = None

        def on_start(self) -> None:
            self.subscribe_bars(bar_type)

        def _account_equity(self) -> float:
            return float(self.portfolio.account(venue).balance_total(base_currency))

        def _leverage(self) -> float:
            try:
                return float(self.portfolio.account(venue).leverage(instrument_id))
            except Exception:
                return float(default_leverage)

        def on_event(self, event) -> None:
            cls = type(event).__name__
            if "Liquidation" in cls:
                self.liquidated_events.append(
                    {"ts": _iso_from_ns(int(getattr(event, "ts_init", 0)))}
                )

        def on_bar(self, bar: Bar) -> None:
            if self.halted:
                return
            ts_ns = bar.ts_init
            ts_iso = _iso_from_ns(ts_ns)
            mark = float(bar.close)
            pre_eq = self.projected_equity
            session.update_equity(pre_eq, ts=ts_iso)
            raw_target = target_by_ns.get(ts_ns, 0.0)
            qty = self.expected_qty
            decision = session.evaluate_target(
                symbol=symbol,
                raw_target=raw_target,
                mark=mark,
                pre_trade_equity=pre_eq,
                current_qty=qty,
                leverage=self._leverage(),
                ts=ts_iso,
            )
            delta = 0.0
            if decision.allowed and not self.liquidated_events:
                desired_qty = decision.target_signed_leverage * max(pre_eq, 1e-9) / mark
                delta = desired_qty - qty
                if abs(delta) * mark >= 1e-6:
                    delta_r = round(delta, size_precision)
                    if abs(delta_r) >= 10 ** (-size_precision):
                        delta = delta_r
                        side = OrderSide.BUY if delta_r > 0 else OrderSide.SELL
                        self.submit_order(
                            self.order_factory.market(
                                instrument_id, side, Quantity(abs(delta_r), size_precision)
                            )
                        )
                        self.submitted += 1
            # Paper-convention projection for this bar: MTM of the prior
            # position at this bar's close, taker fee on the rebalance,
            # then per-bar funding on the post-rebalance position.
            if self.prev_mark is not None:
                self.projected_equity += qty * (mark - self.prev_mark)
            fee_delta = abs(delta) * mark * (float(cfg.get("fee_bps", 4.0)) / 10_000.0)
            self.projected_equity -= fee_delta
            qty_after = qty + delta
            self.expected_qty = qty_after
            if funding_rate != 0.0 and not self.liquidated_events:
                payment = qty_after * mark * funding_rate
                self.total_funding += payment
                self.projected_equity -= payment
                self.funding_events.append(
                    {
                        "ts": ts_iso,
                        "symbol": symbol,
                        "qty": qty_after,
                        "mark": mark,
                        "rate": funding_rate,
                        "payment": payment,
                    }
                )
            self.prev_mark = mark
            self.equity_curve.append({"ts": ts_iso, "equity": self.projected_equity})
            self.account_curve.append({"ts": ts_iso, "equity": self._account_equity()})
            session.update_equity(self.projected_equity, ts=ts_iso)
            if self.liquidated_events:
                session.note_liquidation(ts=ts_iso)
                self.halted = True

    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id="GALAHAD-001",
            run_analysis=False,
            logging=LoggingConfig(log_level="ERROR"),
        )
    )
    engine.add_venue(
        venue,
        OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=base_currency,
        starting_balances=[Money.from_str(f"{cfg.get('initial_equity', 10_000)} USDT")],
        default_leverage=default_leverage,
    )
    engine.add_instrument(instrument)
    engine.add_data(nautilus_bars)
    strategy_obj = DecisionDrivenStrategy(
        StrategyConfig(strategy_id="GALAHAD-DECISION", oms_type=OmsType.NETTING)
    )
    engine.add_strategy(strategy_obj)
    engine.run()

    # --- fills (one row per filled order) ---
    orders = list(engine.cache.orders(instrument_id=instrument_id))
    fills_report = ReportProvider.generate_fills_report(orders) if orders else None
    fills: list[dict[str, Any]] = []
    if fills_report is not None and len(fills_report):
        for _, row in fills_report.iterrows():
            ts_ns = _ns_from_ts(row.get("ts_event", 0))
            raw_instrument = str(row.get("instrument_id", symbol)).split(".")[0]
            raw_side = str(row.get("order_side", "")).split(".")[-1]
            fills.append(
                {
                    "ts": _iso_from_ns(ts_ns),
                    "symbol": raw_instrument.removesuffix("-PERP"),
                    "side": raw_side,
                    "qty": float(row.get("last_qty", 0.0) or 0.0),
                    "price": float(row.get("last_px", 0.0) or 0.0),
                    "fee": _money_float(row.get("commission", 0.0)),
                    "realized_pnl": 0.0,
                    "note": "nautilus",
                    "leverage": float(default_leverage),
                }
            )

    # --- equity curve (per-bar, funding-adjusted, recorded in-strategy) ---
    equity_curve = list(strategy_obj.equity_curve)
    if not equity_curve:
        equity_curve = [
            {
                "ts": _iso_from_ns(last_ts_ns),
                "equity": float(cfg.get("initial_equity", 10_000)),
            }
        ]
    final_equity = equity_curve[-1]["equity"]

    # --- liquidation events (Nautilus margin machinery) ---
    liquidation_events: list[dict[str, Any]] = []
    liquidated = False
    account = engine.portfolio.account(venue)
    for evt in account.events:
        if "Liquidation" in type(evt).__name__:
            liquidated = True
            liquidation_events.append(
                {"ts": _iso_from_ns(int(getattr(evt, "ts_init", 0)))}
            )
    if strategy_obj.liquidated_events and not liquidation_events:
        liquidation_events = list(strategy_obj.liquidated_events)
        liquidated = True

    # --- end positions ---
    positions: dict[str, dict[str, Any]] = {}
    try:
        pos_list = list(account.positions.get(instrument_id, []))
        pos_report = (
            ReportProvider.generate_positions_report(pos_list, pos_list)
            if pos_list
            else None
        )
        if pos_report is not None and len(pos_report):
            for _, row in pos_report.iterrows():
                qty = float(row.get("quantity", 0.0) or 0.0)
                side = "long" if qty > 0 else ("short" if qty < 0 else "flat")
                positions[str(row.get("instrument", symbol))] = {
                    "qty": qty,
                    "entry_price": float(row.get("avg_px", 0.0) or 0.0),
                    "leverage": float(default_leverage),
                    "side": side,
                }
    except Exception:
        positions = {}

    gate = session.gate
    n_fills_oos = sum(1 for f in fills if _ns_from_ts(f["ts"]) >= bar_ts_boundary)
    oos_curve = [s for s in equity_curve if _ns_from_ts(s["ts"]) >= bar_ts_boundary]
    eq_series = [float(s["equity"]) for s in (oos_curve or equity_curve)]
    rets: list[float] = []
    if len(eq_series) >= 2:
        a = np.asarray(eq_series, dtype=float)
        rets = (np.diff(a) / np.maximum(a[:-1], 1e-9)).tolist()

    return {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "strategy": name,
        "strategy_kwargs": dict(kw),
        "symbol": symbol,
        "bars": len(bars),
        "evaluate_from": int(evaluate_from),
        "n_fills": len(fills),
        "n_fills_oos": n_fills_oos,
        "n_risk_rejects": len(gate.rejects),
        "liquidated": liquidated,
        "invalidated": gate.invalidated,
        "invalidation_reason": gate.invalidation_reason or None,
        "invalidation_events": list(gate.invalidation_events),
        "loss_halt_events": list(gate.loss_halt_events),
        "decision_phase_final": session.phase(),
        "peak_equity": float(gate.peak_equity),
        "max_drawdown": float(gate.max_drawdown_seen),
        "initial_equity": float(cfg.get("initial_equity", 10_000)),
        "final_equity": float(final_equity),
        "equity_curve": equity_curve,
        "equity_curve_len": len(equity_curve),
        "account_curve": list(strategy_obj.account_curve),
        "orders_submitted": int(strategy_obj.submitted),
        "orders_filled": len(fills),
        "oos_equity_curve_len": len(oos_curve),
        "returns_oos": rets,
        "total_funding": float(strategy_obj.total_funding),
        "n_funding_events": len(strategy_obj.funding_events),
        "funding_events": list(strategy_obj.funding_events),
        "fills": fills,
        "risk_rejects": gate.rejects,
        "risk_decisions": session.risk_decisions,
        "risk_decisions_tail": session.risk_decisions[-20:],
        "liquidation_events": liquidation_events,
        "positions": positions,
    }

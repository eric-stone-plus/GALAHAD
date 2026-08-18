"""Futures paper book: long/short, leverage, margin, MTM equity, liquidation.

Deterministic pure accounting. No I/O. Unit-tested on fixed price paths.
Not an exchange matching engine — mid/low-frequency research substrate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Fill:
    ts: str
    symbol: str
    side: str  # BUY / SELL (increases long / increases short when reducing opposite)
    qty: float  # base asset quantity, always >= 0
    price: float
    fee: float
    realized_pnl: float = 0.0
    note: str = ""
    leverage: float = 1.0

    @property
    def notional(self) -> float:
        return abs(self.qty * self.price)


@dataclass
class Position:
    """Signed position: qty > 0 long, qty < 0 short."""

    symbol: str
    qty: float = 0.0  # signed base
    entry_price: float = 0.0
    leverage: float = 1.0

    @property
    def side(self) -> str:
        if self.qty > 1e-12:
            return "long"
        if self.qty < -1e-12:
            return "short"
        return "flat"

    def notional(self, mark: float) -> float:
        return abs(self.qty) * float(mark)

    def unrealized_pnl(self, mark: float) -> float:
        if abs(self.qty) < 1e-12:
            return 0.0
        # long: (mark - entry) * qty; short: qty negative → same formula works
        return float(self.qty) * (float(mark) - float(self.entry_price))

    def initial_margin(self, mark: float) -> float:
        lev = max(self.leverage, 1e-9)
        return self.notional(mark) / lev


@dataclass
class FuturesPaperBook:
    """Isolated-style futures paper wallet with multi-symbol positions.

    wallet: free collateral (USDT)
    equity = wallet + sum(unrealized_pnl)
    margin_used = sum(notional / leverage)
    available = equity - margin_used
    Liquidation when equity < maintenance_margin (sum notional * mm_rate).
    """

    wallet: float = 10_000.0
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    fee_bps: float = 4.0
    maintenance_margin_rate: float = 0.005
    funding_rate_per_bar: float = 0.0
    default_leverage: float = 3.0
    max_leverage: float = 5.0
    liquidated: bool = False
    liquidation_events: list[dict[str, Any]] = field(default_factory=list)
    funding_events: list[dict[str, Any]] = field(default_factory=list)
    total_funding: float = 0.0  # cumulative: positive = net paid by book (long-side bias)
    name: str = "futures-paper"

    def _fee(self, notional: float) -> float:
        return abs(notional) * (self.fee_bps / 10_000.0)

    def position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, leverage=self.default_leverage)
        return self.positions[symbol]

    def unrealized_pnl(self, marks: dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            mark = marks.get(sym)
            if mark is None:
                continue
            total += pos.unrealized_pnl(mark)
        return total

    def equity(self, marks: dict[str, float]) -> float:
        return float(self.wallet + self.unrealized_pnl(marks))

    def margin_used(self, marks: dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            if abs(pos.qty) < 1e-12:
                continue
            mark = marks.get(sym)
            if mark is None:
                continue
            total += pos.initial_margin(mark)
        return total

    def maintenance_margin(self, marks: dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            if abs(pos.qty) < 1e-12:
                continue
            mark = marks.get(sym)
            if mark is None:
                continue
            total += pos.notional(mark) * self.maintenance_margin_rate
        return total

    def available_margin(self, marks: dict[str, float]) -> float:
        return self.equity(marks) - self.margin_used(marks)

    def mark_to_market(
        self,
        marks: dict[str, float],
        *,
        ts: str,
        funding_rate: float | None = None,
    ) -> dict[str, Any]:
        """Snapshot equity and apply funding hook. Check liquidation after MTM.

        funding_rate: optional per-call override; else uses funding_rate_per_bar.
        Convention (perp): rate > 0 → longs pay shorts: payment = qty * mark * rate
        (qty > 0 long → wallet decreases; qty < 0 short → wallet increases).
        """
        rate = self.funding_rate_per_bar if funding_rate is None else float(funding_rate)
        funding_this_bar = 0.0
        if rate != 0.0 and not self.liquidated:
            funding_this_bar = self.apply_funding(marks, rate=rate, ts=ts)
        snap = {
            "ts": ts,
            "wallet": self.wallet,
            "unrealized_pnl": self.unrealized_pnl(marks),
            "equity": self.equity(marks),
            "margin_used": self.margin_used(marks),
            "maintenance_margin": self.maintenance_margin(marks),
            "funding_this_bar": funding_this_bar,
            "total_funding": self.total_funding,
            "liquidated": self.liquidated,
            "positions": {
                s: {"qty": p.qty, "entry": p.entry_price, "lev": p.leverage, "side": p.side}
                for s, p in self.positions.items()
                if abs(p.qty) > 1e-12
            },
        }
        self.equity_curve.append(snap)
        self.check_liquidation(marks, ts=ts)
        return snap

    def apply_funding(
        self,
        marks: dict[str, float],
        *,
        rate: float,
        ts: str = "",
    ) -> float:
        """Apply one funding settlement. Returns net payment (longs pay when rate>0)."""
        if rate == 0.0 or self.liquidated:
            return 0.0
        net = 0.0
        for sym, pos in self.positions.items():
            if abs(pos.qty) < 1e-12:
                continue
            mark = marks.get(sym)
            if mark is None:
                continue
            # long pays rate * notional; short receives (qty signed)
            payment = float(pos.qty) * float(mark) * float(rate)
            self.wallet -= payment
            net += payment
            self.funding_events.append(
                {
                    "ts": ts,
                    "symbol": sym,
                    "qty": pos.qty,
                    "mark": float(mark),
                    "rate": float(rate),
                    "payment": payment,
                }
            )
        self.total_funding += net
        return net

    def _apply_funding(self, marks: dict[str, float], *, ts: str) -> None:
        """Back-compat: apply default per-bar rate."""
        self.apply_funding(marks, rate=self.funding_rate_per_bar, ts=ts)

    def check_liquidation(self, marks: dict[str, float], *, ts: str) -> bool:
        """Force-close all positions if equity < maintenance margin."""
        if self.liquidated:
            return True
        eq = self.equity(marks)
        mm = self.maintenance_margin(marks)
        if mm <= 0:
            return False
        if eq + 1e-9 >= mm:
            return False
        # Liquidate: close everything at mark, wallet absorbs residual
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            if abs(pos.qty) < 1e-12:
                continue
            mark = marks.get(sym)
            if mark is None:
                continue
            self._close_position(sym, float(mark), ts=ts, note="liquidation")
        self.liquidated = True
        self.liquidation_events.append(
            {"ts": ts, "equity_before_close": eq, "maintenance_margin": mm, "wallet_after": self.wallet}
        )
        return True

    def set_leverage(self, symbol: str, leverage: float) -> None:
        lev = float(leverage)
        if lev < 1.0 or lev > self.max_leverage + 1e-12:
            raise ValueError(f"leverage {lev} outside [1, {self.max_leverage}]")
        self.position(symbol).leverage = lev

    def target_to_delta_qty(
        self,
        symbol: str,
        target_signed_leverage: float,
        mark: float,
        equity: float | None = None,
    ) -> float:
        """Map target signed leverage (e.g. +2 = 2x long notional of equity) to qty delta.

        target_signed_leverage in units of equity: desired_notional = target * equity
        desired_qty = desired_notional / mark (signed).
        """
        marks = {symbol: mark}
        eq = float(equity) if equity is not None else self.equity(marks)
        if mark <= 0 or eq <= 0:
            return 0.0
        desired_qty = (target_signed_leverage * eq) / mark
        current = self.position(symbol).qty
        return desired_qty - current

    def _marks_for(self, symbol: str, price: float) -> dict[str, float]:
        marks = {symbol: price}
        for s, p in self.positions.items():
            if s != symbol and abs(p.qty) > 1e-12:
                marks[s] = p.entry_price
        return marks

    def _max_add_qty(self, symbol: str, price: float, lev: float) -> float:
        """Max additional base qty fundable from available margin at mark=price."""
        marks = self._marks_for(symbol, price)
        # fee is charged on notional; solve: margin + fee <= avail
        # (q * price / lev) + (q * price * fee_bps/1e4) <= avail
        avail = self.available_margin(marks)
        if avail <= 1e-12:
            return 0.0
        fee_rate = self.fee_bps / 10_000.0
        cost_per_unit = price / max(lev, 1e-9) + price * fee_rate
        if cost_per_unit <= 0:
            return 0.0
        return max(0.0, avail / cost_per_unit)

    def market_order(
        self,
        symbol: str,
        qty: float,
        price: float,
        *,
        ts: str = "",
        note: str = "",
        leverage: float | None = None,
    ) -> Fill | None:
        """Signed qty: >0 buy/increase long, <0 sell/increase short.

        Margin enforced on any exposure increase (open from flat, same-side add,
        and flip residual open). Fees charged once per leg (close + open).
        """
        if self.liquidated:
            return None
        if abs(qty) < 1e-12 or price <= 0:
            return None

        pos = self.position(symbol)
        if leverage is not None:
            self.set_leverage(symbol, leverage)
        lev = pos.leverage

        signed_qty = float(qty)
        fill_qty = abs(signed_qty)

        # --- Opposite-side: reduce / close / flip ---
        if abs(pos.qty) > 1e-12 and (pos.qty * signed_qty) < 0:
            close_qty = min(fill_qty, abs(pos.qty))
            open_qty = fill_qty - close_qty  # residual that opens new side

            if pos.qty > 0:
                realized = (price - pos.entry_price) * close_qty
                pos.qty -= close_qty
            else:
                realized = (pos.entry_price - price) * close_qty
                pos.qty += close_qty

            close_fee = self._fee(close_qty * price)
            self.wallet += realized - close_fee

            if abs(pos.qty) < 1e-12:
                pos.qty = 0.0
                pos.entry_price = 0.0

            open_fee = 0.0
            opened = 0.0
            if open_qty > 1e-12:
                # Margin-cap residual open (same rules as a fresh open)
                max_open = self._max_add_qty(symbol, price, lev)
                opened = min(open_qty, max_open)
                if opened > 1e-12:
                    open_fee = self._fee(opened * price)
                    self.wallet -= open_fee
                    signed_open = opened if signed_qty > 0 else -opened
                    pos.qty = signed_open
                    pos.entry_price = price
                else:
                    opened = 0.0

            total_qty = close_qty + opened
            total_fee = close_fee + open_fee
            side = "BUY" if qty > 0 else "SELL"
            fill = Fill(
                ts=ts,
                symbol=symbol,
                side=side,
                qty=total_qty,
                price=float(price),
                fee=float(total_fee),
                realized_pnl=float(realized),
                note=note or ("flip" if opened > 1e-12 else "reduce_or_close"),
                leverage=lev,
            )
            self.fills.append(fill)
            return fill

        # --- Same-side add or open from flat: enforce margin on full increase ---
        max_add = self._max_add_qty(symbol, price, lev)
        if max_add < 1e-12:
            return None
        if fill_qty > max_add + 1e-12:
            fill_qty = max_add
            signed_qty = fill_qty if signed_qty > 0 else -fill_qty

        fee = self._fee(fill_qty * price)
        if abs(pos.qty) < 1e-12:
            pos.entry_price = price
            pos.qty = signed_qty
        else:
            new_abs = abs(pos.qty) + fill_qty
            pos.entry_price = (pos.entry_price * abs(pos.qty) + price * fill_qty) / new_abs
            pos.qty = pos.qty + signed_qty

        self.wallet -= fee
        side = "BUY" if signed_qty > 0 else "SELL"
        fill = Fill(
            ts=ts,
            symbol=symbol,
            side=side,
            qty=fill_qty,
            price=float(price),
            fee=float(fee),
            realized_pnl=0.0,
            note=note or "open_or_add",
            leverage=lev,
        )
        self.fills.append(fill)
        return fill

    def _close_position(self, symbol: str, price: float, *, ts: str, note: str) -> Fill | None:
        pos = self.position(symbol)
        if abs(pos.qty) < 1e-12:
            return None
        # Opposite order to flatten
        return self.market_order(symbol, -pos.qty, price, ts=ts, note=note)

    def apply_target(
        self,
        symbol: str,
        target_signed_leverage: float,
        mark: float,
        *,
        ts: str = "",
        note: str = "target",
    ) -> Fill | None:
        """Rebalance position to target signed leverage of equity."""
        if self.liquidated:
            return None
        marks = {symbol: mark}
        delta = self.target_to_delta_qty(symbol, target_signed_leverage, mark, equity=self.equity(marks))
        if abs(delta) * mark < 1e-6:  # dust
            return None
        return self.market_order(symbol, delta, mark, ts=ts, note=note)

    def to_dict(self, marks: dict[str, float] | None = None) -> dict[str, Any]:
        marks = marks or {}
        return {
            "wallet": self.wallet,
            "equity": self.equity(marks) if marks else self.wallet,
            "liquidated": self.liquidated,
            "total_funding": self.total_funding,
            "n_funding_events": len(self.funding_events),
            "fills": [asdict(f) for f in self.fills],
            "equity_curve": list(self.equity_curve),
            "liquidation_events": list(self.liquidation_events),
            "funding_events_tail": list(self.funding_events[-20:]),
            "positions": {
                s: {"qty": p.qty, "entry_price": p.entry_price, "leverage": p.leverage, "side": p.side}
                for s, p in self.positions.items()
            },
        }

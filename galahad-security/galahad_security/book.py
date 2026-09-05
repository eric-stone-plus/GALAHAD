"""Cash equity book: long-only integer shares, MTM equity, cash constraint.

Deterministic pure accounting. No I/O. Unit-tested on fixed price paths.

Deliberately NOT a margin book (contrast galahad-futures): positions are
non-negative integer share counts, there is no leverage, no funding, no
maintenance margin, and no liquidation — the binding constraint is cash.
Corporate actions (splits/dividends) are out of scope for v1.

Fill convention: the decision for day t is evaluated at day t's close
(the arrival price); orders fill at that close adjusted by the optional
spread/impact cost model (``costs.spread_bps`` / ``costs.impact_bps``,
both default 0 = execution price identical to arrival, bit-identical to
the zero-cost book). Per-fill arrival/execution detail feeds the session
TCA block (``tca_from_fills``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Fill:
    ts: str
    symbol: str
    side: str  # BUY / SELL
    qty: int  # shares, always > 0
    price: float  # execution price (arrival adjusted by the cost model)
    fee: float
    realized_pnl: float = 0.0
    note: str = ""
    # TCA detail: the decision (daily close) price this fill executed
    # against, and the USDT cost split of the execution-price adjustment.
    arrival_price: float | None = None
    spread_cost: float = 0.0
    impact_cost: float = 0.0

    @property
    def notional(self) -> float:
        return float(self.qty) * float(self.price)


@dataclass
class Position:
    """Long-only share position: qty >= 0 integer shares."""

    symbol: str
    qty: int = 0
    avg_cost: float = 0.0

    @property
    def side(self) -> str:
        return "long" if self.qty > 0 else "flat"

    def notional(self, mark: float) -> float:
        return float(self.qty) * float(mark)

    def unrealized_pnl(self, mark: float) -> float:
        if self.qty <= 0:
            return 0.0
        return float(self.qty) * (float(mark) - float(self.avg_cost))


@dataclass
class CashEquityBook:
    """Cash account: cash + long-only integer-share positions.

    equity = cash + sum(qty * mark). Buys can never spend more than the
    cash on hand; sells can never exceed the shares held.
    """

    cash: float = 100_000.0
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    fee_bps: float = 0.0  # commission bps; 0 = commission-free venue realism
    # Transaction-cost model (linear bps on notional): execution price =
    # arrival × (1 ± (spread_bps/2 + impact_bps)/10⁴), signed by side.
    spread_bps: float = 0.0
    impact_bps: float = 0.0
    name: str = "cash-equity-paper"

    def _fee(self, notional: float) -> float:
        return abs(notional) * (self.fee_bps / 10_000.0)

    def _exec_price(self, arrival: float, side_is_buy: bool) -> float:
        """Execution price: arrival adjusted by half-spread + linear impact.

        Buys pay up, sells receive less. With spread_bps = impact_bps = 0
        this is arrival × 1.0 exactly (backward-compatible identity).
        """
        rate = (self.spread_bps / 2.0 + self.impact_bps) / 10_000.0
        if rate == 0.0:
            return float(arrival)
        return float(arrival) * (1.0 + rate) if side_is_buy else float(arrival) * (1.0 - rate)

    def _cost_split(self, qty: int, arrival: float) -> tuple[float, float]:
        """(spread_cost, impact_cost) in USD for a fill of qty at arrival."""
        notional = abs(int(qty)) * float(arrival)
        return (
            notional * (self.spread_bps / 2.0) / 10_000.0,
            notional * self.impact_bps / 10_000.0,
        )

    def position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def equity(self, marks: dict[str, float]) -> float:
        """Cash + Σ qty × mark. Positions without a mark are valued at cost."""
        total = float(self.cash)
        for sym, pos in self.positions.items():
            if pos.qty <= 0:
                continue
            total += pos.notional(float(marks.get(sym, pos.avg_cost)))
        return total

    def mark_to_market(self, marks: dict[str, float], *, ts: str) -> dict[str, Any]:
        """Snapshot equity. No funding, no liquidation — cash account."""
        snap = {
            "ts": ts,
            "cash": self.cash,
            "positions_value": self.equity(marks) - self.cash,
            "equity": self.equity(marks),
            "positions": {
                s: {"qty": p.qty, "avg_cost": p.avg_cost, "side": p.side}
                for s, p in self.positions.items()
                if p.qty > 0
            },
        }
        self.equity_curve.append(snap)
        return snap

    def market_order(
        self,
        symbol: str,
        qty: float,
        arrival_price: float,
        *,
        ts: str = "",
        note: str = "",
    ) -> Fill | None:
        """Signed qty in shares: >0 buy, <0 sell. Rounded to integers.

        Sizing decisions happen upstream at the arrival price; all fill
        economics (proceeds, average cost, fees, cash affordability) use
        the execution price. Long-only: a sell is capped at the shares
        held; a buy is capped at what the cash on hand can pay for
        (price + fee). Both caps make the book fail-closed by shape.
        """
        if arrival_price <= 0 or abs(qty) < 0.5:
            return None
        pos = self.position(symbol)
        is_buy = qty > 0
        exec_price = self._exec_price(arrival_price, is_buy)

        if is_buy:
            # Cash constraint: qty * exec_price + fee <= cash
            fee_rate = self.fee_bps / 10_000.0
            per_share = exec_price * (1.0 + fee_rate)
            if per_share <= 0:
                return None
            affordable = math.floor(self.cash / per_share + 1e-9)
            n = min(math.floor(qty + 1e-9), affordable)
            if n < 1:
                return None
            cost = n * exec_price
            fee = self._fee(cost)
            self.cash -= cost + fee
            new_qty = pos.qty + n
            pos.avg_cost = (
                (pos.avg_cost * pos.qty + exec_price * n) / new_qty if new_qty else 0.0
            )
            pos.qty = new_qty
            realized = 0.0
        else:
            n = min(math.floor(abs(qty) + 1e-9), pos.qty)
            if n < 1:
                return None
            proceeds = n * exec_price
            fee = self._fee(proceeds)
            realized = (exec_price - pos.avg_cost) * n
            self.cash += proceeds - fee
            pos.qty -= n
            if pos.qty <= 0:
                pos.qty = 0
                pos.avg_cost = 0.0

        spread_cost, impact_cost = self._cost_split(n, arrival_price)
        fill = Fill(
            ts=ts,
            symbol=symbol,
            side="BUY" if is_buy else "SELL",
            qty=int(n),
            price=float(exec_price),
            fee=float(fee),
            realized_pnl=float(realized),
            note=note or ("open_or_add" if is_buy else "reduce_or_close"),
            arrival_price=float(arrival_price),
            spread_cost=float(spread_cost),
            impact_cost=float(impact_cost),
        )
        self.fills.append(fill)
        return fill

    def target_weight_to_qty(
        self,
        symbol: str,
        target_weight: float,
        mark: float,
        equity: float | None = None,
    ) -> float:
        """Signed share delta to reach target weight of equity at mark."""
        eq = float(equity) if equity is not None else self.equity({symbol: mark})
        if mark <= 0 or eq <= 0:
            return 0.0
        desired_qty = math.floor(max(0.0, float(target_weight)) * eq / mark + 1e-9)
        return float(desired_qty - self.position(symbol).qty)

    def apply_target_weight(
        self,
        symbol: str,
        target_weight: float,
        mark: float,
        *,
        ts: str = "",
        note: str = "target",
        equity: float | None = None,
    ) -> Fill | None:
        """Rebalance position to a target weight of equity (long-only).

        ``equity`` is the mark-to-market equity the risk gate approved the
        weight against; callers that have it (the session engine) must pass
        it. The fallback values other positions at avg_cost — fine for a
        single-position book, stale otherwise.
        """
        if equity is None:
            marks = {s: (mark if s == symbol else p.avg_cost) for s, p in self.positions.items()}
            marks[symbol] = mark
            equity = self.equity(marks)
        delta = self.target_weight_to_qty(
            symbol, target_weight, mark, equity=equity
        )
        if abs(delta) < 1.0:  # sub-share dust
            return None
        return self.market_order(symbol, delta, mark, ts=ts, note=note)

    def to_dict(self, marks: dict[str, float] | None = None) -> dict[str, Any]:
        marks = marks or {}
        return {
            "cash": self.cash,
            "equity": self.equity(marks) if marks else self.cash,
            "fills": [asdict(f) for f in self.fills],
            "equity_curve": list(self.equity_curve),
            "positions": {
                s: {"qty": p.qty, "avg_cost": p.avg_cost, "side": p.side}
                for s, p in self.positions.items()
            },
        }


def tca_from_fills(fills: list[dict[str, Any]]) -> dict[str, Any]:
    """Implementation-shortfall block over fill records (mappings).

    Per fill, implementation shortfall = (exec − arrival) × qty for BUY,
    (arrival − exec) × qty for SELL — the cost of crossing from the
    decision (arrival) price to the fill price. Fees are reported
    separately (``fee_cost_usdt``), not folded into the shortfall.
    Fills without an ``arrival_price`` are excluded. When any included
    fill lacks the spread/impact decomposition (e.g. venue fills, where
    only the total shortfall is observable), the split fields are None.
    """
    arrival_notional = 0.0
    filled_notional = 0.0
    is_usdt = 0.0
    fee_usdt = 0.0
    spread_usdt = 0.0
    impact_usdt = 0.0
    decomposed = True
    n = 0
    for f in fills:
        arrival = f.get("arrival_price")
        if arrival is None:
            continue
        arrival = float(arrival)
        qty = float(f.get("qty", 0.0))
        px = float(f.get("price", 0.0))
        sign = 1.0 if str(f.get("side", "")).upper() == "BUY" else -1.0
        arrival_notional += qty * arrival
        filled_notional += qty * px
        is_usdt += sign * (px - arrival) * qty
        fee_usdt += float(f.get("fee") or 0.0)
        sc, ic = f.get("spread_cost"), f.get("impact_cost")
        if sc is None or ic is None:
            decomposed = False
        else:
            spread_usdt += float(sc)
            impact_usdt += float(ic)
        n += 1
    return {
        "arrival_notional": float(arrival_notional),
        "filled_notional": float(filled_notional),
        "implementation_shortfall_usdt": float(is_usdt),
        "implementation_shortfall_bps": float(
            is_usdt / arrival_notional * 10_000.0 if arrival_notional > 0 else 0.0
        ),
        "spread_cost_usdt": float(spread_usdt) if decomposed else None,
        "impact_cost_usdt": float(impact_usdt) if decomposed else None,
        "fee_cost_usdt": float(fee_usdt),
        "n_fills": int(n),
    }

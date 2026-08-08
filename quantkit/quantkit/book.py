"""Paper trading book: cash, positions, fills, mark-to-market equity.

For mid/low-frequency research execution (not exchange matching engine).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Fill:
    ts: str
    symbol: str
    side: str  # BUY / SELL
    qty: float
    price: float
    fee: float
    note: str = ""

    @property
    def notional(self) -> float:
        return abs(self.qty * self.price)


@dataclass
class PaperBook:
    """Long-only paper book with cash + share quantities."""

    cash: float = 1_000_000.0
    positions: dict[str, float] = field(default_factory=dict)  # shares
    avg_cost: dict[str, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    currency: str = "USD"
    fee_bps: float = 5.0
    name: str = "paper"

    def position_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for sym, qty in self.positions.items():
            px = prices.get(sym)
            if px is not None and qty:
                total += qty * float(px)
        return total

    def equity(self, prices: dict[str, float]) -> float:
        return float(self.cash + self.position_value(prices))

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {s: 0.0 for s in self.positions}
        out = {}
        for sym, qty in self.positions.items():
            px = prices.get(sym)
            out[sym] = (qty * float(px) / eq) if px is not None else 0.0
        return out

    def _fee(self, notional: float) -> float:
        return abs(notional) * (self.fee_bps / 10_000.0)

    def market_order(
        self,
        symbol: str,
        qty: float,
        price: float,
        *,
        ts: str | None = None,
        note: str = "",
    ) -> Fill | None:
        """Buy qty>0, sell qty<0. Returns fill or None if rejected."""
        if qty == 0 or price <= 0:
            return None
        ts = ts or datetime.now(timezone.utc).isoformat()
        notional = abs(qty) * price
        fee = self._fee(notional)
        if qty > 0:  # buy
            cost = notional + fee
            if cost > self.cash + 1e-6:
                # downsize to available cash
                if self.cash <= fee:
                    return None
                qty = (self.cash - fee) / price
                if qty <= 0:
                    return None
                notional = qty * price
                fee = self._fee(notional)
                cost = notional + fee
            self.cash -= cost
            prev_q = self.positions.get(symbol, 0.0)
            prev_c = self.avg_cost.get(symbol, 0.0)
            new_q = prev_q + qty
            self.avg_cost[symbol] = (
                (prev_c * prev_q + price * qty) / new_q if new_q else 0.0
            )
            self.positions[symbol] = new_q
            side = "BUY"
        else:  # sell
            sell_q = min(-qty, self.positions.get(symbol, 0.0))
            if sell_q <= 0:
                return None
            qty = -sell_q
            notional = sell_q * price
            fee = self._fee(notional)
            self.cash += notional - fee
            self.positions[symbol] = self.positions.get(symbol, 0.0) - sell_q
            if self.positions[symbol] <= 1e-12:
                self.positions.pop(symbol, None)
                self.avg_cost.pop(symbol, None)
            side = "SELL"

        fill = Fill(
            ts=ts,
            symbol=symbol,
            side=side,
            qty=abs(qty),
            price=float(price),
            fee=float(fee),
            note=note,
        )
        self.fills.append(fill)
        return fill

    def rebalance_to_weights(
        self,
        target_weights: dict[str, float] | pd.Series,
        prices: dict[str, float],
        *,
        ts: str | None = None,
        band: float = 0.02,
    ) -> list[Fill]:
        """Trade toward target weights (long-only). Ignore names with no price."""
        if isinstance(target_weights, pd.Series):
            tw = target_weights.to_dict()
        else:
            tw = dict(target_weights)
        # normalize non-negative
        tw = {k: max(0.0, float(v)) for k, v in tw.items()}
        s = sum(tw.values())
        if s > 0:
            tw = {k: v / s for k, v in tw.items()}

        eq = self.equity(prices)
        fills: list[Fill] = []
        # sell first names not in target or overweight
        current = self.weights(prices)
        all_syms = set(current) | set(tw) | set(prices)
        # sells
        for sym in sorted(all_syms):
            px = prices.get(sym)
            if px is None or px <= 0:
                continue
            cur_w = current.get(sym, 0.0)
            tgt_w = tw.get(sym, 0.0)
            if cur_w - tgt_w > band:
                # sell down
                tgt_val = tgt_w * eq
                cur_val = self.positions.get(sym, 0.0) * px
                delta_val = tgt_val - cur_val
                qty = delta_val / px  # negative
                f = self.market_order(sym, qty, px, ts=ts, note="rebalance")
                if f:
                    fills.append(f)
        # refresh equity after sells
        eq = self.equity(prices)
        current = self.weights(prices)
        for sym in sorted(tw.keys()):
            px = prices.get(sym)
            if px is None or px <= 0:
                continue
            cur_w = current.get(sym, 0.0)
            tgt_w = tw[sym]
            if tgt_w - cur_w > band:
                tgt_val = tgt_w * eq
                cur_val = self.positions.get(sym, 0.0) * px
                delta_val = tgt_val - cur_val
                qty = delta_val / px
                f = self.market_order(sym, qty, px, ts=ts, note="rebalance")
                if f:
                    fills.append(f)
        return fills

    def mark(self, prices: dict[str, float], ts: str | None = None) -> dict[str, Any]:
        ts = ts or datetime.now(timezone.utc).isoformat()
        row = {
            "ts": ts,
            "cash": self.cash,
            "equity": self.equity(prices),
            "gross_exposure": self.position_value(prices),
            "n_positions": len(self.positions),
            "weights": self.weights(prices),
        }
        self.equity_curve.append(row)
        return row

    def holdings_table(self, prices: dict[str, float]) -> pd.DataFrame:
        rows = []
        for sym, qty in sorted(self.positions.items()):
            px = prices.get(sym)
            cost = self.avg_cost.get(sym, float("nan"))
            mkt = qty * px if px is not None else float("nan")
            pnl = (px - cost) * qty if px is not None else float("nan")
            rows.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "avg_cost": cost,
                    "price": px,
                    "market_value": mkt,
                    "unrealized_pnl": pnl,
                    "weight": self.weights(prices).get(sym, 0.0),
                }
            )
        return pd.DataFrame(rows)

    def fills_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(
                columns=["ts", "symbol", "side", "qty", "price", "fee", "note"]
            )
        return pd.DataFrame([asdict(f) for f in self.fills])

    def equity_frame(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame(columns=["ts", "cash", "equity", "gross_exposure", "n_positions"])
        rows = []
        for r in self.equity_curve:
            rows.append({k: v for k, v in r.items() if k != "weights"})
        return pd.DataFrame(rows)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": self.cash,
            "positions": self.positions,
            "avg_cost": self.avg_cost,
            "fills": [asdict(f) for f in self.fills],
            "equity_curve": self.equity_curve,
            "currency": self.currency,
            "fee_bps": self.fee_bps,
            "name": self.name,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "PaperBook":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        book = cls(
            cash=float(data.get("cash", 0)),
            positions={k: float(v) for k, v in (data.get("positions") or {}).items()},
            avg_cost={k: float(v) for k, v in (data.get("avg_cost") or {}).items()},
            currency=data.get("currency", "USD"),
            fee_bps=float(data.get("fee_bps", 5.0)),
            name=data.get("name", "paper"),
        )
        book.fills = [Fill(**f) for f in data.get("fills") or []]
        book.equity_curve = list(data.get("equity_curve") or [])
        return book

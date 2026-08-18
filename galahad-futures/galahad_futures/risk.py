"""Execution risk layer — sole place targets become order intents.

Hard caps: per-order notional, position notional, daily loss, leverage.
Session drawdown invalidation forces flat (no new risk).
Kill switch blocks live paths; paper is always allowed when mode=paper.

State machine (owned here, enriched by ``decision.SessionRisk``):

    ACTIVE ── daily loss breach ──▶ LOSS_HALTED ── recovery past
      ▲                                  │            floor + hysteresis
      └──────────────────────────────────┘
    ACTIVE ── drawdown trip ──▶ INVALIDATED (terminal force-flat)
    (live) ── kill_switch / !enable_live ──▶ LIVE_BLOCKED (per decision)
    (any)  ── execution-reported liquidation ──▶ LIQUIDATED (terminal)

Daily-loss semantics: hitting the floor force-flattens the book (target 0
is the only allowed action — never a frozen position) and new risk stays
blocked until equity recovers past floor + ``daily_loss_hysteresis``.
The hysteresis band prevents order flapping when equity hovers at the
floor, which matters for automated execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskConfig:
    max_order_notional: float = 5_000.0
    max_position_notional: float = 15_000.0
    max_daily_loss: float = 500.0
    max_leverage: float = 5.0
    # Pre-specified session invalidation: peak-to-trough equity drawdown fraction
    max_drawdown_pct: float = 0.15  # 15% from session peak → force flat
    # Recovery band above the daily-loss floor before LOSS_HALTED clears.
    # 0.0 = clear as soon as equity returns to the floor (legacy behavior).
    daily_loss_hysteresis: float = 0.0
    kill_switch: bool = True  # True = refuse live
    enable_live: bool = False
    mode: str = "paper"  # paper | live


@dataclass
class RiskDecision:
    allowed: bool
    target_signed_leverage: float
    reason: str = ""
    clipped: bool = False


@dataclass
class RiskGate:
    config: RiskConfig
    day_start_equity: float
    peak_equity: float = 0.0
    max_drawdown_seen: float = 0.0  # session peak-to-trough max (not final-only)
    invalidated: bool = False
    invalidation_reason: str = ""
    invalidation_events: list[dict[str, Any]] = field(default_factory=list)
    loss_halted: bool = False
    loss_halt_events: list[dict[str, Any]] = field(default_factory=list)
    rejects: list[dict[str, Any]] = field(default_factory=list)
    session_realized_loss: float = 0.0

    def __post_init__(self) -> None:
        if self.peak_equity <= 0:
            self.peak_equity = float(self.day_start_equity)

    # --- state helpers ---------------------------------------------------

    def daily_loss_floor(self) -> float:
        return float(self.day_start_equity - self.config.max_daily_loss)

    def dd_headroom(self) -> float:
        """Distance to the invalidation trip line, in drawdown units.

        Positive = below threshold (safe); <= 0 = trip already reached.
        """
        max_dd = float(self.config.max_drawdown_pct or 0.0)
        return max_dd - self.max_drawdown_seen

    def loss_headroom(self, equity: float) -> float:
        """Distance above the daily-loss floor (equity - floor)."""
        return float(equity) - self.daily_loss_floor()

    def live_blocked(self) -> bool:
        if self.config.mode != "live":
            return False
        if self.config.kill_switch or not self.config.enable_live:
            return True
        return False

    # --- state transitions ----------------------------------------------

    def update_equity(self, equity: float, *, ts: str = "") -> bool:
        """Update peak, running max drawdown, and trip invalidation if needed.

        Also drives the LOSS_HALTED state: entering when equity drops
        below the daily-loss floor, clearing only after recovery past
        floor + hysteresis.

        Returns True if invalidation just fired or was already active.
        """
        eq = float(equity)
        if eq > self.peak_equity:
            self.peak_equity = eq
        dd = 0.0
        if self.peak_equity > 0:
            dd = max(0.0, (self.peak_equity - eq) / self.peak_equity)
            if dd > self.max_drawdown_seen:
                self.max_drawdown_seen = dd
        if not self.invalidated:
            max_dd = float(self.config.max_drawdown_pct or 0.0)
            if max_dd > 0 and self.peak_equity > 0 and dd + 1e-12 >= max_dd:
                self.invalidated = True
                self.invalidation_reason = (
                    f"max_drawdown_pct={max_dd:.4f} peak={self.peak_equity:.6f} "
                    f"equity={eq:.6f} dd={dd:.4f}"
                )
                self.invalidation_events.append(
                    {
                        "ts": ts,
                        "peak_equity": self.peak_equity,
                        "equity": eq,
                        "drawdown": dd,
                        "max_drawdown_pct": max_dd,
                        "reason": self.invalidation_reason,
                    }
                )

        # Daily-loss halt with hysteresis recovery band.
        floor = self.daily_loss_floor()
        if not self.loss_halted and eq < floor:
            self.loss_halted = True
            self.loss_halt_events.append(
                {
                    "ts": ts,
                    "equity": eq,
                    "floor": floor,
                    "hysteresis": float(self.config.daily_loss_hysteresis),
                    "event": "halted",
                }
            )
        elif self.loss_halted and eq >= floor + float(self.config.daily_loss_hysteresis):
            self.loss_halted = False
            self.loss_halt_events.append(
                {
                    "ts": ts,
                    "equity": eq,
                    "floor": floor,
                    "hysteresis": float(self.config.daily_loss_hysteresis),
                    "event": "cleared",
                }
            )
        return self.invalidated

    def current_drawdown(self, equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - float(equity)) / self.peak_equity)

    # --- per-decision evaluation -----------------------------------------

    def filter_target(
        self,
        *,
        symbol: str,
        target_signed_leverage: float,
        mark: float,
        equity: float,
        current_qty: float,
        leverage: float,
        ts: str = "",
    ) -> RiskDecision:
        """Clip or reject a strategy target before any fill.

        Ordering: live gate → invalid inputs → terminal force-flats
        (invalidation, daily-loss halt) → sizing caps.
        """
        if self.live_blocked():
            d = RiskDecision(False, 0.0, reason="live_blocked_kill_switch_or_disabled")
            self._log_reject(symbol, target_signed_leverage, d.reason, ts)
            return d

        if mark <= 0 or equity <= 0:
            d = RiskDecision(False, 0.0, reason="invalid_mark_or_equity")
            self._log_reject(symbol, target_signed_leverage, d.reason, ts)
            return d

        # Session invalidation: only allowed action is flat (target 0).
        if self.invalidated:
            d = RiskDecision(True, 0.0, reason="invalidation_force_flat", clipped=True)
            if abs(target_signed_leverage) > 1e-12:
                self._log_reject(symbol, target_signed_leverage, "invalidation_block_new_risk", ts)
            return d

        # Daily-loss halt: force flat, no new risk. Never freeze a losing
        # position — reducing/flattening is always allowed.
        if self.loss_halted:
            d = RiskDecision(True, 0.0, reason="daily_loss_force_flat", clipped=True)
            if abs(target_signed_leverage) > 1e-12:
                self._log_reject(symbol, target_signed_leverage, "daily_loss_block_new_risk", ts)
            return d

        t = float(target_signed_leverage)
        clipped = False

        # Leverage cap on |target|
        max_lev = min(self.config.max_leverage, leverage if leverage > 0 else self.config.max_leverage)
        if abs(t) > max_lev + 1e-12:
            t = max_lev if t > 0 else -max_lev
            clipped = True

        desired_notional = abs(t) * equity
        if desired_notional > self.config.max_position_notional + 1e-9:
            t = (self.config.max_position_notional / equity) * (1.0 if t >= 0 else -1.0)
            desired_notional = abs(t) * equity
            clipped = True

        desired_qty = (t * equity) / mark
        delta_qty = desired_qty - current_qty
        order_notional = abs(delta_qty) * mark
        if order_notional > self.config.max_order_notional + 1e-9:
            max_delta = self.config.max_order_notional / mark
            if delta_qty > 0:
                delta_qty = max_delta
            else:
                delta_qty = -max_delta
            new_qty = current_qty + delta_qty
            t = (new_qty * mark) / equity
            if abs(t) * equity > self.config.max_position_notional:
                t = (self.config.max_position_notional / equity) * (1.0 if t >= 0 else -1.0)
            clipped = True
            order_notional = abs(delta_qty) * mark

        if abs(t) < 1e-12 and abs(target_signed_leverage) > 1e-12 and order_notional < 1e-9:
            d = RiskDecision(False, 0.0, reason="order_dust_after_caps", clipped=clipped)
            self._log_reject(symbol, target_signed_leverage, d.reason, ts)
            return d

        reason = "ok_clipped" if clipped else "ok"
        return RiskDecision(True, t, reason=reason, clipped=clipped)

    def _log_reject(self, symbol: str, target: float, reason: str, ts: str) -> None:
        self.rejects.append(
            {"ts": ts, "symbol": symbol, "target": target, "reason": reason}
        )

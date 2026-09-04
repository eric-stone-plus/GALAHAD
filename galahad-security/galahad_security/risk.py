"""Execution risk layer — sole place targets become order intents.

Ported from galahad-futures risk.py, adapted to cash equities: targets
are long-only weights (fraction of equity), not signed leverage; there
are no leverage, funding, margin, or liquidation concepts anywhere.

Hard caps: per-symbol weight, per-order notional, daily loss.
Session drawdown invalidation forces flat (no new risk).
Venue gate: venue order placement (the Alpaca paper endpoint) requires
``enable_alpaca_paper`` AND kill_switch off; mode=paper is always allowed.
There is no live-money mode to block — the only venue wired is paper.

State machine (owned here, enriched by ``decision.SessionRisk``):

    ACTIVE ── daily loss breach ──▶ LOSS_HALTED ── recovery past
      ▲                                  │            floor + hysteresis
      └──────────────────────────────────┘
    ACTIVE ── drawdown trip ──▶ INVALIDATED (terminal force-flat)
    (venue) ── kill_switch / !enable_alpaca_paper ──▶ VENUE_BLOCKED (per decision)

Daily-loss semantics: hitting the floor force-flattens the book (target 0
is the only allowed action — never a frozen position) and new risk stays
blocked until equity recovers past floor + ``daily_loss_hysteresis``.
The hysteresis band prevents order flapping when equity hovers at the
floor, which matters for automated execution.

Graduated de-risking ladder (optional, default OFF): ``derisk_ladder``
tiers scale targets down as session drawdown deepens — strictly earlier
and softer than the force-flat gates, never a weakening of them.
Malformed ladders are a hard config error at gate construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskConfig:
    max_weight: float = 0.25  # per-symbol target weight cap (fraction of equity)
    max_order_notional: float = 5_000.0
    max_daily_loss: float = 500.0
    # Pre-specified session invalidation: peak-to-trough equity drawdown fraction
    max_drawdown_pct: float = 0.15  # 15% from session peak → force flat
    # Recovery band above the daily-loss floor before LOSS_HALTED clears.
    # 0.0 = clear as soon as equity returns to the floor (legacy behavior).
    daily_loss_hysteresis: float = 0.0
    kill_switch: bool = True  # True = refuse venue order placement
    enable_alpaca_paper: bool = False  # venue passes only when True + kill_switch off
    mode: str = "paper"  # paper | venue  (venue = Alpaca paper endpoint; no live mode exists)
    # Graduated de-risking: [{drawdown: frac, leverage_multiplier: 0..1}, ...]
    # ascending in drawdown. Empty = OFF. Normalized by RiskGate to a tuple
    # of (drawdown, multiplier) pairs; malformed config raises ValueError.
    derisk_ladder: Any = None


def validate_derisk_ladder(ladder: Any) -> tuple[tuple[float, float], ...]:
    """Fail-closed ladder config validation; returns normalized tier pairs.

    Hard ``ValueError`` on: wrong container type, missing/non-numeric keys,
    drawdowns outside (0, 1] or non-ascending, multipliers outside [0, 1].
    """
    if ladder is None or ladder == () or ladder == []:
        return ()
    if not isinstance(ladder, (list, tuple)):
        raise ValueError(
            f"risk.derisk_ladder must be a list of tiers, got {type(ladder).__name__}"
        )
    out: list[tuple[float, float]] = []
    prev_dd = 0.0
    for i, tier in enumerate(ladder):
        if isinstance(tier, dict):
            raw_dd, raw_mult = tier.get("drawdown"), tier.get("leverage_multiplier")
        elif isinstance(tier, (list, tuple)) and len(tier) == 2:
            raw_dd, raw_mult = tier  # normalized pair (idempotent re-validation)
        else:
            raise ValueError(
                f"risk.derisk_ladder tier {i} malformed (need "
                f"{{drawdown, leverage_multiplier}}): {tier!r}"
            )
        try:
            dd, mult = float(raw_dd), float(raw_mult)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"risk.derisk_ladder tier {i} has non-numeric drawdown/"
                f"leverage_multiplier: {tier!r}"
            ) from exc
        if not (0.0 < dd <= 1.0):
            raise ValueError(
                f"risk.derisk_ladder tier {i}: drawdown {dd} outside (0, 1]"
            )
        if not (0.0 <= mult <= 1.0):
            raise ValueError(
                f"risk.derisk_ladder tier {i}: leverage_multiplier {mult} outside [0, 1]"
            )
        if dd <= prev_dd + 1e-12:
            raise ValueError(
                f"risk.derisk_ladder must be strictly ascending in drawdown "
                f"(tier {i}: {dd} after {prev_dd})"
            )
        prev_dd = dd
        out.append((dd, mult))
    return tuple(out)


@dataclass
class RiskDecision:
    allowed: bool
    target_weight: float
    reason: str = ""
    clipped: bool = False
    # Active de-risk ladder multiplier applied to this decision (1.0 = none).
    derisk_multiplier: float = 1.0


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

    def __post_init__(self) -> None:
        if self.peak_equity <= 0:
            self.peak_equity = float(self.day_start_equity)
        # Fail closed at construction: malformed ladder config raises here,
        # identically for every engine (all build gates via SessionRisk).
        self.config.derisk_ladder = validate_derisk_ladder(self.config.derisk_ladder)

    # --- state helpers ---------------------------------------------------

    def daily_loss_floor(self) -> float:
        return float(self.day_start_equity - self.config.max_daily_loss)

    def dd_headroom(self) -> float:
        """Distance to the invalidation trip line, in drawdown units."""
        max_dd = float(self.config.max_drawdown_pct or 0.0)
        return max_dd - self.max_drawdown_seen

    def loss_headroom(self, equity: float) -> float:
        """Distance above the daily-loss floor (equity - floor)."""
        return float(equity) - self.daily_loss_floor()

    def venue_blocked(self) -> bool:
        """Venue-order gate. mode=venue passes only with the kill switch
        off AND ``enable_alpaca_paper`` on. mode=paper is never blocked.
        There is no live-money mode in this package."""
        if self.config.mode != "venue":
            return False
        return bool(self.config.kill_switch or not self.config.enable_alpaca_paper)

    # --- state transitions ----------------------------------------------

    def update_equity(self, equity: float, *, ts: str = "") -> bool:
        """Update peak, running max drawdown, trip invalidation, and drive
        the LOSS_HALTED state (enter below the floor, clear only after
        recovery past floor + hysteresis).

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

    # --- graduated de-risking ladder --------------------------------------

    def derisk_multiplier(self, equity: float) -> float:
        """Active ladder multiplier at this equity (1.0 = ladder off/inactive).

        The deepest breached tier wins; a tier is breached at exactly its
        threshold. The ladder only ever scales targets down — it never
        weakens the force-flat gates, which fire at their own thresholds.
        """
        ladder = self.config.derisk_ladder
        if not ladder:
            return 1.0
        dd = self.current_drawdown(equity)
        mult = 1.0
        for tier_dd, tier_mult in ladder:
            if dd + 1e-12 >= tier_dd:
                mult = tier_mult
            else:
                break
        return mult

    def derisk_summary(self) -> dict[str, Any]:
        """Session-level ladder evidence for the summary ``derisk`` block."""
        ladder = self.config.derisk_ladder
        if not ladder:
            return {"ladder_enabled": False, "tiers_triggered": 0, "min_multiplier": 1.0}
        triggered = [
            mult for dd, mult in ladder if self.max_drawdown_seen + 1e-12 >= dd
        ]
        return {
            "ladder_enabled": True,
            "tiers_triggered": len(triggered),
            "min_multiplier": min(triggered) if triggered else 1.0,
        }

    # --- per-decision evaluation -----------------------------------------

    def filter_weight(
        self,
        *,
        symbol: str,
        target_weight: float,
        mark: float,
        equity: float,
        current_qty: float,
        ts: str = "",
    ) -> RiskDecision:
        """Clip or reject a strategy target weight before any fill.

        Ordering: venue gate → invalid inputs → terminal force-flats
        (invalidation, daily-loss halt) → long-only/weight/notional caps
        → de-risk ladder (strictly softer: only scales down).
        """
        if self.venue_blocked():
            d = RiskDecision(False, 0.0, reason="venue_blocked_kill_switch_or_disabled")
            self._log_reject(symbol, target_weight, d.reason, ts)
            return d

        if mark <= 0 or equity <= 0:
            d = RiskDecision(False, 0.0, reason="invalid_mark_or_equity")
            self._log_reject(symbol, target_weight, d.reason, ts)
            return d

        # Session invalidation: only allowed action is flat (target 0).
        if self.invalidated:
            d = RiskDecision(True, 0.0, reason="invalidation_force_flat", clipped=True)
            if abs(target_weight) > 1e-12:
                self._log_reject(symbol, target_weight, "invalidation_block_new_risk", ts)
            return d

        # Daily-loss halt: force flat, no new risk. Never freeze a losing
        # position — reducing/flattening is always allowed.
        if self.loss_halted:
            d = RiskDecision(True, 0.0, reason="daily_loss_force_flat", clipped=True)
            if abs(target_weight) > 1e-12:
                self._log_reject(symbol, target_weight, "daily_loss_block_new_risk", ts)
            return d

        # Long-only account: a negative target weight is a strategy bug.
        if target_weight < -1e-12:
            d = RiskDecision(False, 0.0, reason="negative_weight_long_only")
            self._log_reject(symbol, target_weight, d.reason, ts)
            return d

        w = float(target_weight)
        clipped = False

        # Per-symbol weight cap (also caps at 1.0: a cash account cannot
        # hold more than its equity in one symbol).
        cap = min(self.config.max_weight, 1.0)
        if w > cap + 1e-12:
            w = cap
            clipped = True

        # Per-order notional cap on the rebalance delta.
        desired_qty = (w * equity) / mark
        delta_qty = desired_qty - float(current_qty)
        order_notional = abs(delta_qty) * mark
        if order_notional > self.config.max_order_notional + 1e-9:
            max_delta = self.config.max_order_notional / mark
            delta_qty = max_delta if delta_qty > 0 else -max_delta
            w = max(0.0, (float(current_qty) + delta_qty) * mark / equity)
            clipped = True
            order_notional = abs(delta_qty) * mark

        # Graduated de-risking ladder: scale the capped target by the
        # deepest breached tier. Multiplier 0 is the ladder's own flatten
        # rung (distinct from invalidation, which fired above).
        mult = self.derisk_multiplier(equity)
        if mult <= 0.0:
            d = RiskDecision(
                True, 0.0, reason="derisk_force_flat", clipped=True, derisk_multiplier=0.0
            )
            if abs(target_weight) > 1e-12:
                self._log_reject(symbol, target_weight, "derisk_block_new_risk", ts)
            return d
        if mult < 1.0:
            w *= mult
            clipped = True

        if w < 1e-12 and abs(target_weight) > 1e-12 and order_notional < 1e-9:
            d = RiskDecision(False, 0.0, reason="order_dust_after_caps", clipped=clipped)
            self._log_reject(symbol, target_weight, d.reason, ts)
            return d

        reason = "ok_clipped" if clipped else "ok"
        return RiskDecision(True, w, reason=reason, clipped=clipped, derisk_multiplier=mult)

    def _log_reject(self, symbol: str, target: float, reason: str, ts: str) -> None:
        self.rejects.append(
            {"ts": ts, "symbol": symbol, "target": target, "reason": reason}
        )

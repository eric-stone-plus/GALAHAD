"""Decision layer — engine-agnostic session risk evaluation.

Ported from galahad-futures decision.py, adapted to cash equities:
targets are long-only weights; there is no liquidation input (a cash
account cannot be liquidated) and no live-money phase (the only venue
wired is the Alpaca paper endpoint).

The per-bar decision loop is shared verbatim by every execution backend
(offline reference cash book, Alpaca paper venue): pre-trade equity
snapshot, state transition, gate evaluation, execution, settlement.
Backends differ only in execution mechanics; the decision layer is the
single authority for *what* a position should be.

Design contract (see docs/architecture.md):

- **Pure and side-effect-free.** The decision layer never touches I/O,
  never places orders. Executors translate decisions into orders.
- **Deterministic.** Same (config, bar stream, executor-reported
  equity/position) in ⇒ same decision stream out. This is the audit
  spine for automated trading.
- **Terminal force-flats first.** Invalidation (drawdown trip) and the
  daily-loss halt both force target 0 — the only allowed action — and
  block all new risk. Reducing/flattening is never blocked.

Session phases (derived from the risk gate):

    ACTIVE         trading allowed
    LOSS_HALTED    daily-loss floor breached; force flat until equity
                   recovers past floor + hysteresis
    INVALIDATED    drawdown trip (terminal for the session)
    VENUE_BLOCKED  venue mode with kill switch / enable_alpaca_paper off

Transition table (illegal transitions fail closed with ValueError):

    ACTIVE → LOSS_HALTED → ACTIVE            (halt, recover)
    ACTIVE → INVALIDATED                     (terminal)
    LOSS_HALTED → INVALIDATED                (halt then trip)

Every decision record carries: seq (monotonic), ts, symbol,
phase_before/phase_after, dd_headroom (distance to the invalidation trip
line), loss_headroom (distance above the daily-loss floor), and the
active derisk_multiplier — the boundary-sensitivity instrumentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from galahad_security.risk import RiskConfig, RiskDecision, RiskGate

_PHASE_ORDER = ("ACTIVE", "LOSS_HALTED", "INVALIDATED", "VENUE_BLOCKED")
_TERMINAL = ("INVALIDATED",)


def _highest_phase(*phases: str) -> str:
    return max(phases, key=_PHASE_ORDER.index)


@dataclass
class SessionRisk:
    """RiskGate + per-bar decision bookkeeping (one instance per session).

    Executors feed their own equity estimate — each backend's equity is
    the quantity under test in venue reconciliation runs.
    """

    gate: RiskGate
    risk_decisions: list[dict[str, Any]] = field(default_factory=list)
    _seq: int = 0
    _last_phase: str | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], start_equity: float) -> "SessionRisk":
        risk_cfg_raw = dict(cfg.get("risk") or {})
        mode = str(cfg.get("mode", "paper")).lower()
        risk_cfg = RiskConfig(
            max_weight=float(risk_cfg_raw.get("max_weight", 0.25)),
            max_order_notional=float(risk_cfg_raw.get("max_order_notional", 5000)),
            max_daily_loss=float(risk_cfg_raw.get("max_daily_loss", 500)),
            max_drawdown_pct=float(risk_cfg_raw.get("max_drawdown_pct", 0.15)),
            daily_loss_hysteresis=float(risk_cfg_raw.get("daily_loss_hysteresis", 0.0)),
            kill_switch=bool(risk_cfg_raw.get("kill_switch", True)),
            enable_alpaca_paper=bool(risk_cfg_raw.get("enable_alpaca_paper", False)),
            derisk_ladder=risk_cfg_raw.get("derisk_ladder"),
            mode=mode,
        )
        return cls(gate=RiskGate(config=risk_cfg, day_start_equity=start_equity))

    # --- phase derivation ------------------------------------------------

    def phase(self) -> str:
        g = self.gate
        if g.invalidated:
            return "INVALIDATED"
        if g.loss_halted:
            return "LOSS_HALTED"
        if g.venue_blocked():
            return "VENUE_BLOCKED"
        return "ACTIVE"

    # --- per-bar evaluation ----------------------------------------------

    def update_equity(self, equity: float, *, ts: str = "") -> None:
        self.gate.update_equity(equity, ts=ts)

    def evaluate_weight(
        self,
        *,
        symbol: str,
        raw_weight: float,
        mark: float,
        pre_trade_equity: float,
        current_qty: float,
        ts: str,
    ) -> RiskDecision:
        """Evaluate one strategy target weight through the gate; record it.

        Fail-closed guard: a phase change that violates the transition
        table raises ValueError.
        """
        phase = self.phase()
        if self._last_phase is not None:
            self._assert_transition(self._last_phase, phase, ts=ts)
        phase_before = self._last_phase if self._last_phase is not None else phase
        self._last_phase = phase
        decision = self.gate.filter_weight(
            symbol=symbol,
            target_weight=raw_weight,
            mark=mark,
            equity=max(pre_trade_equity, 1e-9),
            current_qty=current_qty,
            ts=ts,
        )
        self._seq += 1
        self.risk_decisions.append(
            {
                "seq": self._seq,
                "ts": ts,
                "symbol": symbol,
                "phase_before": phase_before,
                "phase_after": phase,
                "raw_weight": raw_weight,
                "allowed": decision.allowed,
                "final_weight": decision.target_weight,
                "reason": decision.reason,
                "clipped": decision.clipped,
                "derisk_multiplier": decision.derisk_multiplier,
                "invalidated": self.gate.invalidated,
                "loss_halted": self.gate.loss_halted,
                "pre_trade_equity": pre_trade_equity,
                "pre_trade_drawdown": self.gate.current_drawdown(pre_trade_equity),
                "dd_headroom": self.gate.dd_headroom(),
                "loss_headroom": self.gate.loss_headroom(pre_trade_equity),
            }
        )
        return decision

    def _assert_transition(self, before: str, after: str, *, ts: str) -> None:
        """Terminal phases never revert; phase jumps must be legal."""
        if before == after:
            return
        if before in _TERMINAL:
            raise ValueError(
                f"illegal decision-layer transition {before} -> {after} at {ts}: "
                f"{before} is terminal"
            )
        if after in _TERMINAL:
            return  # terminal entry is always legal (drawdown trip)
        # Non-terminal transitions: ACTIVE <-> LOSS_HALTED only.
        if {before, after} != {"ACTIVE", "LOSS_HALTED"}:
            raise ValueError(
                f"illegal decision-layer transition {before} -> {after} at {ts}"
            )

"""Decision layer — engine-agnostic session risk evaluation.

The per-bar decision loop is shared verbatim by every execution backend
(paper reference book, NautilusTrader backtest engine, and future live
executors): pre-trade equity snapshot, state transition, gate
evaluation, execution, settlement. Backends differ only in execution
mechanics; the decision layer is the single authority for *what* a
position should be.

Design contract (see docs/architecture.md):

- **Pure and side-effect-free.** The decision layer never touches I/O,
  never places orders. Executors translate decisions into orders.
- **Deterministic.** Same (config, bar stream, executor-reported
  equity/position) in ⇒ same decision stream out. This is the audit
  spine for automated trading.
- **Terminal force-flats first.** Invalidation (drawdown trip) and the
  daily-loss halt both force target 0 — the only allowed action — and
  block all new risk. Reducing/flattening is never blocked.

Session phases (derived from the risk gate + executor reports):

    ACTIVE        trading allowed
    LOSS_HALTED   daily-loss floor breached; force flat until equity
                  recovers past floor + hysteresis
    INVALIDATED   drawdown trip (terminal for the session)
    LIVE_BLOCKED  live mode with kill switch / enable_live off
    LIQUIDATED    executor-reported liquidation (terminal)

Transition table (illegal transitions fail closed with ValueError):

    ACTIVE → LOSS_HALTED → ACTIVE            (halt, recover)
    ACTIVE → INVALIDATED                     (terminal)
    LOSS_HALTED → INVALIDATED                (halt then trip)
    any → LIQUIDATED                         (terminal, executor input)

Every decision record carries: seq (monotonic), ts, phase_before
(the phase entering the bar — the previous decision's phase),
phase_after (the phase at decision time), dd_headroom (distance to the
invalidation trip line) and loss_headroom (distance above the
daily-loss floor) — the headroom fields are the boundary-sensitivity
instrumentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from galahad_futures.risk import RiskConfig, RiskDecision, RiskGate

_PHASE_ORDER = ("ACTIVE", "LOSS_HALTED", "INVALIDATED", "LIVE_BLOCKED", "LIQUIDATED")
_TERMINAL = ("INVALIDATED", "LIQUIDATED")


def _highest_phase(*phases: str) -> str:
    return max(phases, key=_PHASE_ORDER.index)


@dataclass
class SessionRisk:
    """RiskGate + per-bar decision bookkeeping (one instance per session).

    Executors feed their own equity estimate — each backend's equity is
    the quantity under test in parity runs — and may report liquidation
    events via :meth:`note_liquidation`.
    """

    gate: RiskGate
    risk_decisions: list[dict[str, Any]] = field(default_factory=list)
    liquidated: bool = False
    liquidation_ts: str | None = None
    _seq: int = 0
    _last_phase: str | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], start_equity: float) -> "SessionRisk":
        risk_cfg_raw = dict(cfg.get("risk") or {})
        mode = str(cfg.get("mode", "paper")).lower()
        risk_cfg = RiskConfig(
            max_order_notional=float(risk_cfg_raw.get("max_order_notional", 5000)),
            max_position_notional=float(risk_cfg_raw.get("max_position_notional", 15000)),
            max_daily_loss=float(risk_cfg_raw.get("max_daily_loss", 500)),
            max_leverage=float(cfg.get("max_leverage", 5.0)),
            max_drawdown_pct=float(risk_cfg_raw.get("max_drawdown_pct", 0.15)),
            daily_loss_hysteresis=float(risk_cfg_raw.get("daily_loss_hysteresis", 0.0)),
            kill_switch=bool(risk_cfg_raw.get("kill_switch", True)),
            enable_live=bool(risk_cfg_raw.get("enable_live", False)),
            mode=mode,
        )
        return cls(gate=RiskGate(config=risk_cfg, day_start_equity=start_equity))

    # --- phase derivation ------------------------------------------------

    def phase(self) -> str:
        g = self.gate
        if self.liquidated:
            return "LIQUIDATED"
        if g.invalidated:
            return "INVALIDATED"
        if g.loss_halted:
            return "LOSS_HALTED"
        if g.live_blocked():
            return "LIVE_BLOCKED"
        return "ACTIVE"

    def note_liquidation(self, *, ts: str) -> None:
        """Executor reports a liquidation; the session becomes terminal."""
        if not self.liquidated:
            self.liquidated = True
            self.liquidation_ts = ts

    # --- per-bar evaluation ----------------------------------------------

    def update_equity(self, equity: float, *, ts: str = "") -> None:
        self.gate.update_equity(equity, ts=ts)

    def evaluate_target(
        self,
        *,
        symbol: str,
        raw_target: float,
        mark: float,
        pre_trade_equity: float,
        current_qty: float,
        leverage: float,
        ts: str,
    ) -> RiskDecision:
        """Evaluate one strategy target through the gate; record the decision.

        Fail-closed guards: a decision attempt after executor-reported
        liquidation, or a phase change that violates the transition
        table, raises ValueError.
        """
        if self.liquidated:
            raise ValueError(
                f"decision after liquidation at {self.liquidation_ts} (ts={ts})"
            )
        phase = self.phase()
        if self._last_phase is not None:
            self._assert_transition(self._last_phase, phase, ts=ts)
        phase_before = self._last_phase if self._last_phase is not None else phase
        self._last_phase = phase
        decision = self.gate.filter_target(
            symbol=symbol,
            target_signed_leverage=raw_target,
            mark=mark,
            equity=max(pre_trade_equity, 1e-9),
            current_qty=current_qty,
            leverage=leverage,
            ts=ts,
        )
        self._seq += 1
        self.risk_decisions.append(
            {
                "seq": self._seq,
                "ts": ts,
                "phase_before": phase_before,
                "phase_after": phase,
                "raw_target": raw_target,
                "allowed": decision.allowed,
                "final_target": decision.target_signed_leverage,
                "reason": decision.reason,
                "clipped": decision.clipped,
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
            return  # terminal entry is always legal (trip or liquidation)
        # Non-terminal transitions: ACTIVE <-> LOSS_HALTED only.
        if {before, after} != {"ACTIVE", "LOSS_HALTED"}:
            raise ValueError(
                f"illegal decision-layer transition {before} -> {after} at {ts}"
            )

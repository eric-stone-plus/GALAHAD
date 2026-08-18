"""Strategy go/no-go gate evaluation — the R3 gate-threshold table, codified.

Thresholds consolidated from the research notes behind this project:
  - the R3 gate-threshold table (v2)
  - the round-3 wiring decisions (reduced-frequency main path: turnover ≤30×,
    AUM ≤5e6, two-sided all-in friction in [0.002, 0.005] aligned with
    ``COST_TIERS``)

Every gate is fail-closed: a missing metric fails the gate and names what is
missing. Pure stdlib dataclasses; no new deps.

Gates (serial pipeline, any No-go kills the strategy):

  0. main_path       reduced-frequency main path: turnover / AUM / full-caliber friction / cost_tier
  1. contamination   counterfactual PC red line + post-cutoff data isolation
  2. statistics      effective-rank DSR / PBO / Zero-shot R² / e-BH FDR
  3. capacity        N≥20 seeds worst-path net Alpha / POV
  4. crowding        composite crowding Z_comp / index-futures discount quantile / conformal calibration window
  5. paper           forward paper-tracking duration / MinTRL / tracking deviation / win rate
  6. live            small-capital live observation / Sharpe decay / compliance order count
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

# Keep in sync with quantkit.backtest.COST_TIERS (duplicated to avoid importing
# numpy/pandas into this pure-stdlib module).
COST_TIERS: dict[str, float] = {"low": 0.002, "mid": 0.004, "high": 0.005}

__all__ = [
    "GateResult",
    "GATES_CONFIG",
    "COST_TIERS",
    "MAIN_PATH",
    "check_gate",
    "evaluate_gates",
    "report_to_dict",
]


# quiz2 round3 hard constants for the personal A-share reduced-frequency path
MAIN_PATH: dict[str, Any] = {
    "turnover_annual_max": 30.0,   # Annualized one-way turnover upper limit
    "aum_scale_max": 5_000_000.0,  # Capital scale upper limit (yuan)
    "friction_cost_min": 0.002,    # Full-caliber bilateral friction lower limit (= COST_TIERS["low"])
    "friction_cost_max": 0.005,    # Full-caliber bilateral friction upper limit (= COST_TIERS["high"])
}


GATES_CONFIG: dict[str, dict[str, Any]] = {
    "gate0_main_path": {
        "turnover_annual_max": MAIN_PATH["turnover_annual_max"],
        "aum_scale_max": MAIN_PATH["aum_scale_max"],
        "friction_cost_min": MAIN_PATH["friction_cost_min"],
        "friction_cost_max": MAIN_PATH["friction_cost_max"],
        "cost_tiers": COST_TIERS,
    },
    "gate1_contamination": {
        "pc_redline": 0.50,        # Counterfactual prediction consistency must be below 50% (substantial decline)
        "isolation_days": 365,     # post-cutoff data isolation
    },
    "gate2_statistics": {
        "dsr": 0.95,               # DSR confidence probability after effective rank correction
        "pbo": 0.05,               # PBO upper limit (R3-02 / quiz2 r3 tightened version)
        "zero_shot_r2": 0.02,      # Zero-shot R² passing line
        "ebh_fdr": 0.05,           # e-BH screening FDR
    },
    "gate3_capacity": {
        "n_seeds": 20,             # Monte Carlo seed count lower limit
        "worst_path_alpha": 0.0,   # 5th percentile worst-path net Alpha approval line
        "pov": 0.10,               # Single-order split POV upper limit (R3-02 version; R3-01 was 5%)
    },
    "gate4_crowding": {
        "z_comp": 2.0,             # Composite crowding trigger (deviation from mean 2σ)
        "basis_pct": 0.10,         # Index futures discount 252-day quantile warning
        "calib_window_min": 60,    # Conformal calibration window lower limit (days)
        "calib_window_max": 120,   # Conformal calibration window upper limit (days)
    },
    "gate5_paper": {
        "months": 6,               # Paper tracking ≥ 6 complete calendar months
        "mintrl_trades": 60,       # Independent virtual trade count lower limit
        "deviation": 0.15,         # Annualized tracking error upper limit
        "win_rate": 0.54,          # Weekly win rate additional line
    },
    "gate6_live": {
        "months": 3,               # Live trading observation period
        "sharpe_decay": 0.20,      # Sharpe ratio decay tolerance (20%)
        "daily_orders_warn": 18000,  # Daily order count warning line (red line 20000)
    },
}


@dataclass
class GateResult:
    """Single-gate outcome. ``passed=False`` is always accompanied by reasons."""

    gate: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - display helper
        status = "GO" if self.passed else "NO-GO"
        parts = [f"[{status}] {self.gate}"]
        if self.missing:
            parts.append(f"missing={self.missing}")
        if self.failures:
            parts.append(f"failures={self.failures}")
        return " ".join(parts)


def _need(metrics: Mapping[str, float], keys: list[str]) -> list[str]:
    return [k for k in keys if k not in metrics]


def _check(gate: str, metrics: Mapping[str, float], rules: list[tuple[str, str, float, str]]) -> GateResult:
    """Shared evaluator: rules are (metric_key, op, threshold, label)."""
    keys = [r[0] for r in rules]
    missing = _need(metrics, keys)
    failures: list[str] = []
    seen: dict[str, float] = {}
    for key, op, thr, label in rules:
        if key in missing:
            continue
        v = float(metrics[key])
        seen[key] = v
        ok = v >= thr if op == ">=" else v <= thr
        if not ok:
            failures.append(f"{label}: {v:g} fails {op} {thr:g}")
    return GateResult(gate=gate, passed=not missing and not failures,
                      failures=failures, missing=missing, metrics=seen)


def _check_main_path(metrics: Mapping[str, Any]) -> GateResult:
    """quiz2 round3 main-path envelope: turnover, AUM, friction, optional cost_tier."""
    c = GATES_CONFIG["gate0_main_path"]
    required = ["turnover_annual", "aum_scale", "friction_cost"]
    missing = _need(metrics, required)
    failures: list[str] = []
    seen: dict[str, float] = {}

    if "turnover_annual" not in missing:
        v = float(metrics["turnover_annual"])
        seen["turnover_annual"] = v
        if v > c["turnover_annual_max"]:
            failures.append(
                f"annualized single-side turnover: {v:g} fails <= {c['turnover_annual_max']:g}"
            )
    if "aum_scale" not in missing:
        v = float(metrics["aum_scale"])
        seen["aum_scale"] = v
        if v > c["aum_scale_max"]:
            failures.append(
                f"AUM scale: {v:g} fails <= {c['aum_scale_max']:g}"
            )
    if "friction_cost" not in missing:
        v = float(metrics["friction_cost"])
        seen["friction_cost"] = v
        lo, hi = c["friction_cost_min"], c["friction_cost_max"]
        if not (lo <= v <= hi):
            failures.append(f"full-caliber two-sided friction: {v:g} not in [{lo:g}, {hi:g}]")

    # Optional named cost_tier must resolve to COST_TIERS and match friction_cost
    # when both are present (fail-closed on unknown tier or mismatch).
    if "cost_tier" in metrics:
        tier = metrics["cost_tier"]
        if not isinstance(tier, str) or tier not in c["cost_tiers"]:
            failures.append(
                f"cost_tier: {tier!r} not in {sorted(c['cost_tiers'])}"
            )
        else:
            expected = float(c["cost_tiers"][tier])
            seen["cost_tier_rate"] = expected
            if "friction_cost" not in missing:
                actual = float(metrics["friction_cost"])
                if abs(actual - expected) > 1e-12:
                    failures.append(
                        f"cost_tier={tier} implies friction {expected:g} but friction_cost is {actual:g} (mismatch)"
                    )

    return GateResult(
        gate="gate0_main_path",
        passed=not missing and not failures,
        failures=failures,
        missing=missing,
        metrics=seen,
    )


def check_gate(gate: str, metrics: Mapping[str, Any]) -> GateResult:
    """Evaluate one gate. ``metrics`` is a flat mapping of measured values.

    Required metric keys per gate (see GATES_CONFIG for thresholds):
      gate0_main_path:     turnover_annual, aum_scale, friction_cost
                           (+ optional cost_tier in COST_TIERS)
      gate1_contamination: pc, isolation_days
      gate2_statistics:    dsr, pbo, zero_shot_r2, ebh_fdr
      gate3_capacity:      n_seeds, worst_path_alpha, pov
      gate4_crowding:      z_comp, basis_pct, calib_window
      gate5_paper:         paper_months, paper_trades, deviation, win_rate
      gate6_live:          live_months, sharpe_decay, daily_orders
    """
    c = GATES_CONFIG[gate]
    if gate == "gate0_main_path":
        return _check_main_path(metrics)
    if gate == "gate1_contamination":
        return _check(gate, metrics, [
            ("pc", "<=", c["pc_redline"], "counterfactual PC red line"),
            ("isolation_days", ">=", c["isolation_days"], "post-cutoff isolation days"),
        ])
    if gate == "gate2_statistics":
        return _check(gate, metrics, [
            ("dsr", ">=", c["dsr"], "effective-rank DSR"),
            ("pbo", "<=", c["pbo"], "PBO"),
            ("zero_shot_r2", ">=", c["zero_shot_r2"], "Zero-shot R²"),
            ("ebh_fdr", "<=", c["ebh_fdr"], "e-BH FDR"),
        ])
    if gate == "gate3_capacity":
        return _check(gate, metrics, [
            ("n_seeds", ">=", c["n_seeds"], "seed count"),
            ("worst_path_alpha", ">=", c["worst_path_alpha"], "worst-path net Alpha"),
            ("pov", "<=", c["pov"], "single-order POV"),
        ])
    if gate == "gate4_crowding":
        r = _check(gate, metrics, [
            ("z_comp", "<=", c["z_comp"], "composite crowding Z_comp"),
            ("basis_pct", ">=", c["basis_pct"], "discount quantile (below warning line)"),
        ])
        # Calibration window is interval determination: fails if below 60 or above 120
        if "calib_window" not in metrics:
            r.missing.append("calib_window")
            r.passed = False
        else:
            w = float(metrics["calib_window"])
            r.metrics["calib_window"] = w
            if not (c["calib_window_min"] <= w <= c["calib_window_max"]):
                r.failures.append(f"conformal calibration window: {w:g} not in [{c['calib_window_min']}, {c['calib_window_max']}]")
                r.passed = False
        return r
    if gate == "gate5_paper":
        return _check(gate, metrics, [
            ("paper_months", ">=", c["months"], "paper-tracking months"),
            ("paper_trades", ">=", c["mintrl_trades"], "MinTRL trade count"),
            ("deviation", "<=", c["deviation"], "annualized tracking error"),
            ("win_rate", ">=", c["win_rate"], "weekly win rate"),
        ])
    if gate == "gate6_live":
        return _check(gate, metrics, [
            ("live_months", ">=", c["months"], "live-observation months"),
            ("sharpe_decay", "<=", c["sharpe_decay"], "Sharpe decay"),
            ("daily_orders", "<=", c["daily_orders_warn"], "daily-order-count warning line"),
        ])
    raise KeyError(f"unknown gate: {gate}")


def evaluate_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Run all gates over one metrics mapping.

    Returns ``{"verdict": "GO"|"NO-GO", "gates": [GateResult, ...]}``.
    A strategy is GO only when every gate passes; evaluation always runs all
    gates so the caller sees every failure, not just the first.
    """
    order = [
        "gate0_main_path",
        "gate1_contamination", "gate2_statistics", "gate3_capacity",
        "gate4_crowding", "gate5_paper", "gate6_live",
    ]
    results = [check_gate(g, metrics) for g in order]
    verdict = "GO" if all(r.passed for r in results) else "NO-GO"
    return {"verdict": verdict, "gates": results}


def report_to_dict(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten an ``evaluate_gates`` report into a JSON-serializable dict.

    Pipeline exits call this to persist a ``gate_report.json`` next to their
    other outputs::

        report = evaluate_gates(metrics)
        Path("output/gate_report.json").write_text(
            json.dumps(report_to_dict(report), indent=2, ensure_ascii=False))

    Shape: ``{"verdict": ..., "gates": [{gate, passed, missing, failures,
    metrics}, ...], "missing": [all missing metric keys across gates]}``.
    """
    gates = [
        {
            "gate": r.gate,
            "passed": r.passed,
            "missing": list(r.missing),
            "failures": list(r.failures),
            "metrics": dict(r.metrics),
        }
        for r in report["gates"]
    ]
    missing = sorted({k for r in report["gates"] for k in r.missing})
    return {"verdict": report["verdict"], "gates": gates, "missing": missing}

"""Strategy go/no-go gate evaluation — R3 门控阈值表代码化.

Thresholds from:
  - ``finance/gemini/quiz1/round3/R3-01…/门控阈值表_手工汇编.md`` (v2)
  - quiz2 round3 wiring report (降频主路径: turnover ≤30×, AUM ≤5e6,
    two-sided all-in friction in [0.002, 0.005] aligned with ``COST_TIERS``)

Every gate is fail-closed: a missing metric fails the gate and names what is
missing. Pure stdlib dataclasses; no new deps.

Gates (serial pipeline, any No-go kills the strategy):

  0. main_path       降频主路径: 换手 / 资金 / 全口径摩擦 / cost_tier
  1. contamination   反事实 PC 红线 + post-cutoff 数据隔离
  2. statistics      有效秩 DSR / PBO / Zero-shot R² / e-BH FDR
  3. capacity        N≥20 种子最差路径净 Alpha / POV
  4. crowding        复合拥挤度 Z_comp / 期指贴水分位 / 保形校准窗口
  5. paper           前向纸面追踪时长 / MinTRL / 追踪偏离 / 胜率
  6. live            小资金实盘观察 / 夏普衰减 / 合规笔数
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
    "turnover_annual_max": 30.0,   # 年化单边换手上限
    "aum_scale_max": 5_000_000.0,  # 资金标尺上限（元）
    "friction_cost_min": 0.002,    # 全口径双边摩擦下限（= COST_TIERS["low"]）
    "friction_cost_max": 0.005,    # 全口径双边摩擦上限（= COST_TIERS["high"]）
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
        "pc_redline": 0.50,        # 反事实预测一致性须低于 50%（实质性下降）
        "isolation_days": 365,     # post-cutoff 数据隔离
    },
    "gate2_statistics": {
        "dsr": 0.95,               # 有效秩校正后 DSR 置信概率
        "pbo": 0.05,               # PBO 上限（R3-02 / quiz2 r3 收紧版）
        "zero_shot_r2": 0.02,      # Zero-shot R² 及格线
        "ebh_fdr": 0.05,           # e-BH 海选 FDR
    },
    "gate3_capacity": {
        "n_seeds": 20,             # 蒙特卡洛种子数下限
        "worst_path_alpha": 0.0,   # 第 5 百分位最差路径净 Alpha 放行线
        "pov": 0.10,               # 单笔拆分 POV 上限（R3-02 版；R3-01 曾为 5%）
    },
    "gate4_crowding": {
        "z_comp": 2.0,             # 复合拥挤度触发（偏离均值 2σ）
        "basis_pct": 0.10,         # 期指贴水 252 日分位预警
        "calib_window_min": 60,    # 保形校准窗口下限（日）
        "calib_window_max": 120,   # 保形校准窗口上限（日）
    },
    "gate5_paper": {
        "months": 6,               # 纸面追踪 ≥ 6 个完整自然月
        "mintrl_trades": 60,       # 独立虚拟交易笔数下限
        "deviation": 0.15,         # 年化追踪误差上限
        "win_rate": 0.54,          # 周度胜率附加线
    },
    "gate6_live": {
        "months": 3,               # 实盘观察期
        "sharpe_decay": 0.20,      # 夏普衰减容忍（两成）
        "daily_orders_warn": 18000,  # 日笔数预警线（红线 20000）
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
            failures.append(f"{label}: {v:g} 未过 {op} {thr:g}")
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
                f"年化单边换手: {v:g} 未过 <= {c['turnover_annual_max']:g}"
            )
    if "aum_scale" not in missing:
        v = float(metrics["aum_scale"])
        seen["aum_scale"] = v
        if v > c["aum_scale_max"]:
            failures.append(
                f"资金标尺: {v:g} 未过 <= {c['aum_scale_max']:g}"
            )
    if "friction_cost" not in missing:
        v = float(metrics["friction_cost"])
        seen["friction_cost"] = v
        lo, hi = c["friction_cost_min"], c["friction_cost_max"]
        if not (lo <= v <= hi):
            failures.append(f"全口径双边摩擦: {v:g} 不在 [{lo:g}, {hi:g}]")

    # Optional named cost_tier must resolve to COST_TIERS and match friction_cost
    # when both are present (fail-closed on unknown tier or mismatch).
    if "cost_tier" in metrics:
        tier = metrics["cost_tier"]
        if not isinstance(tier, str) or tier not in c["cost_tiers"]:
            failures.append(
                f"cost_tier: {tier!r} 不在 {sorted(c['cost_tiers'])}"
            )
        else:
            expected = float(c["cost_tiers"][tier])
            seen["cost_tier_rate"] = expected
            if "friction_cost" not in missing:
                actual = float(metrics["friction_cost"])
                if abs(actual - expected) > 1e-12:
                    failures.append(
                        f"cost_tier={tier} 期望摩擦 {expected:g} 与 friction_cost {actual:g} 不一致"
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
            ("pc", "<=", c["pc_redline"], "反事实 PC 红线"),
            ("isolation_days", ">=", c["isolation_days"], "post-cutoff 隔离天数"),
        ])
    if gate == "gate2_statistics":
        return _check(gate, metrics, [
            ("dsr", ">=", c["dsr"], "有效秩 DSR"),
            ("pbo", "<=", c["pbo"], "PBO"),
            ("zero_shot_r2", ">=", c["zero_shot_r2"], "Zero-shot R²"),
            ("ebh_fdr", "<=", c["ebh_fdr"], "e-BH FDR"),
        ])
    if gate == "gate3_capacity":
        return _check(gate, metrics, [
            ("n_seeds", ">=", c["n_seeds"], "种子数"),
            ("worst_path_alpha", ">=", c["worst_path_alpha"], "最差路径净 Alpha"),
            ("pov", "<=", c["pov"], "单笔 POV"),
        ])
    if gate == "gate4_crowding":
        r = _check(gate, metrics, [
            ("z_comp", "<=", c["z_comp"], "复合拥挤度 Z_comp"),
            ("basis_pct", ">=", c["basis_pct"], "贴水分位(预警线下)"),
        ])
        # 校准窗口是区间判定：低于 60 或高于 120 都失败
        if "calib_window" not in metrics:
            r.missing.append("calib_window")
            r.passed = False
        else:
            w = float(metrics["calib_window"])
            r.metrics["calib_window"] = w
            if not (c["calib_window_min"] <= w <= c["calib_window_max"]):
                r.failures.append(f"保形校准窗口: {w:g} 不在 [{c['calib_window_min']}, {c['calib_window_max']}]")
                r.passed = False
        return r
    if gate == "gate5_paper":
        return _check(gate, metrics, [
            ("paper_months", ">=", c["months"], "纸面追踪月数"),
            ("paper_trades", ">=", c["mintrl_trades"], "MinTRL 笔数"),
            ("deviation", "<=", c["deviation"], "年化追踪误差"),
            ("win_rate", ">=", c["win_rate"], "周度胜率"),
        ])
    if gate == "gate6_live":
        return _check(gate, metrics, [
            ("live_months", ">=", c["months"], "实盘观察月数"),
            ("sharpe_decay", "<=", c["sharpe_decay"], "夏普衰减"),
            ("daily_orders", "<=", c["daily_orders_warn"], "日笔数预警线"),
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

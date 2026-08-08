"""门控接线：quantkit.gates.evaluate_gates → output/gate_report.json。

fail-closed 语义（gates.py）：本流水线只能自测 gate0 的换手/资金标尺/摩擦
三项；pbo/dsr/拥挤度/纸面/实盘等证据不在本地产出，缺测即 NO-GO 并逐门点名。
外部证据（walk-forward 统计、拥挤度周评、纸面追踪……）以 JSON 合并进来，
实测键优先于证据文件同名键（stale 证据不能软化新测量）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantkit.gates import COST_TIERS, evaluate_gates, report_to_dict

from _common import OUT, resolve_fee_bps

EXTRA_METRICS_NAME = "gate_metrics.json"
REPORT_NAME = "gate_report.json"
MIN_TURNOVER_SPAN_DAYS = 28  # 跨度太短时年化换手噪声过大，宁可缺测 fail-closed


def _annual_turnover(out_dir: Path) -> float | None:
    """年化单边换手 = Σ|qty×price| / 平均权益 / 年数；fills 或 equity 缺失则缺测。"""
    fills_p = out_dir / "fills.csv"
    eq_p = next(
        (out_dir / n for n in ("lifecycle_equity.csv", "equity_marks.csv") if (out_dir / n).exists()),
        None,
    )
    if not fills_p.exists() or eq_p is None:
        return None
    fills = pd.read_csv(fills_p)
    eq = pd.read_csv(eq_p)
    date_col = "date" if "date" in eq.columns else "ts" if "ts" in eq.columns else None
    if fills.empty or eq.empty or "equity" not in eq.columns or date_col is None:
        return None
    dates = pd.to_datetime(eq[date_col], utc=True, format="mixed")
    span_days = (dates.iloc[-1] - dates.iloc[0]).days
    mean_equity = float(eq["equity"].astype(float).mean())
    if span_days < MIN_TURNOVER_SPAN_DAYS or mean_equity <= 0:
        return None
    notional = float((fills["qty"].abs() * fills["price"]).sum())
    return notional / mean_equity / (span_days / 365.25)


def collect_metrics(cfg: dict[str, Any], out_dir: Path = OUT) -> dict[str, Any]:
    """本流水线可实测的门控指标（gate0 三腿 + 摩擦口径）。"""
    metrics: dict[str, Any] = {}
    if cfg.get("initial_cash"):
        metrics["aum_scale"] = float(cfg["initial_cash"])
    fee_bps, tier = resolve_fee_bps(cfg)
    if tier is not None:
        metrics["cost_tier"] = tier
        metrics["friction_cost"] = COST_TIERS[tier]
    else:  # legacy fee_bps 是单边费，换算成全口径双边
        metrics["friction_cost"] = 2.0 * fee_bps / 10_000.0
    turnover = _annual_turnover(Path(out_dir))
    if turnover is not None:
        metrics["turnover_annual"] = turnover
    return metrics


def load_extra_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"extra gate metrics must be a JSON object: {path}")
    return data


def run_gate_report(
    cfg: dict[str, Any],
    out_dir: Path = OUT,
    extra_path: Path | str | None = None,
) -> dict[str, Any]:
    """跑六门 + gate0 并写 ``gate_report.json``；返回报告 dict。

    ``extra_path`` 显式给出但文件不存在时抛 FileNotFoundError（防止把
    打错的路径当成"无证据"静默 fail-closed）；缺省路径不存在则视为无证据。
    """
    out_dir = Path(out_dir)
    if extra_path is not None:
        extra_file = Path(extra_path)
        if not extra_file.exists():
            raise FileNotFoundError(f"extra metrics not found: {extra_file}")
    else:
        extra_file = out_dir / EXTRA_METRICS_NAME
    extra = load_extra_metrics(extra_file)
    measured = collect_metrics(cfg, out_dir)
    metrics = {**extra, **measured}  # 实测键优先

    report = report_to_dict(evaluate_gates(metrics))
    payload: dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "extra_metrics": str(extra_file) if extra else None,
            "measured_keys": sorted(measured),
            "extra_keys": sorted(extra),
        },
        "metrics": metrics,
        **report,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_NAME
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return payload

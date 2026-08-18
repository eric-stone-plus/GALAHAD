"""Gate wiring: quantkit.gates.evaluate_gates → output/gate_report.json.

fail-closed semantics (gates.py): this pipeline can only measure gate0's three
legs itself — turnover / AUM scale / friction. Evidence such as pbo/dsr/crowding/
paper/live is not produced locally; unmeasured gates are NO-GO and named one by
one. External evidence (walk-forward statistics, weekly crowding reviews, paper
tracking …) is merged in as JSON; measured keys take priority over same-named
keys from evidence files (stale evidence must not soften fresh measurements).
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
MIN_TURNOVER_SPAN_DAYS = 28  # When the span is too short, the annualized turnover noise is too high; better to fail-closed by missing measurements


def _annual_turnover(out_dir: Path) -> float | None:
    """Annualized single-side turnover = Σ|qty×price| / mean equity / years; unmeasured when fills or equity is missing."""
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
    """Gate metrics this pipeline can measure itself (gate0's three legs + friction basis)."""
    metrics: dict[str, Any] = {}
    if cfg.get("initial_cash"):
        metrics["aum_scale"] = float(cfg["initial_cash"])
    fee_bps, tier = resolve_fee_bps(cfg)
    if tier is not None:
        metrics["cost_tier"] = tier
        metrics["friction_cost"] = COST_TIERS[tier]
    else:  # legacy fee_bps is a unilateral fee; convert to a bilateral full-basis
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
    """Run the six gates + gate0 and write ``gate_report.json``; return the report dict.

    Raises FileNotFoundError when ``extra_path`` is given explicitly but the
    file does not exist (prevents a mistyped path from being silently treated
    as "no evidence" under fail-closed); a missing default path simply means
    no evidence.
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
    metrics = {**extra, **measured}  # Measured keys take priority

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

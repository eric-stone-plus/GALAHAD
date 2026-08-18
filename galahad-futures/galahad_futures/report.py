"""Summary + journal assembly shared by every execution backend.

The summary/journal shapes are the stable contract consumed by the
delivery platform (STAMMTISCH GALAHAD adapter) and by downstream
research tooling. Backends produce a ``result`` dict; this module turns
it into the on-disk artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def build_summary(
    result: dict[str, Any],
    *,
    cfg: dict[str, Any],
    symbol: str,
    interval: str,
    source_used: str,
    sample_kind: str,
    data_note: str | None,
    out_dir: Path,
    engine: str,
    engine_version: str,
    strategy_name: str,
) -> dict[str, Any]:
    """Assemble the paper-session summary dict (same shape for all engines)."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    funding_rate = float(cfg.get("funding_rate_per_bar", 0.0))
    summary: dict[str, Any] = {
        "run_id": run_id,
        "mode": str(cfg.get("mode", "paper")).lower(),
        "engine": engine,
        "engine_version": engine_version,
        "strategy": strategy_name,
        "strategy_kwargs": result.get("strategy_kwargs"),
        "symbol": symbol,
        "interval": interval,
        "bars": result.get("bars", 0),
        "source_used": source_used,
        "sample_kind": sample_kind,
        "data_note": data_note,
        "n_fills": result["n_fills"],
        "n_risk_rejects": result["n_risk_rejects"],
        "liquidated": result["liquidated"],
        "invalidated": result.get("invalidated", False),
        "invalidation_reason": result.get("invalidation_reason"),
        "peak_equity": result.get("peak_equity"),
        "max_drawdown": result.get("max_drawdown"),
        "initial_equity": result["initial_equity"],
        "final_equity": result["final_equity"],
        "equity_curve_len": result["equity_curve_len"],
        "funding_rate_per_bar": funding_rate,
        "total_funding": result["total_funding"],
        "n_funding_events": result["n_funding_events"],
        "status": "ok",
    }
    if summary["n_fills"] == 0 and not result["liquidated"]:
        summary["status"] = "no-trade but risk-idle OK"
        summary["idle_reason"] = "strategy flat or risk blocked all targets"
    if result.get("invalidated"):
        summary["status"] = "ok_invalidated" if summary["n_fills"] else summary["status"]
    return summary


def write_journal(
    summary: dict[str, Any],
    result: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Write journal/summary/equity artifacts; return summary with paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = summary["run_id"]

    journal = {
        "summary": summary,
        "engine": summary.get("engine"),
        "engine_version": summary.get("engine_version"),
        "fills": result["fills"],
        "equity_curve": result["equity_curve"],
        "risk_rejects": result["risk_rejects"],
        "risk_decisions_tail": result["risk_decisions_tail"],
        "liquidation_events": result["liquidation_events"],
        "invalidation_events": result.get("invalidation_events") or [],
        "funding_events": result["funding_events"],
        "positions": result["positions"],
    }

    journal_path = out_dir / f"paper_journal_{run_id}.json"
    summary_path = out_dir / "paper_last_summary.json"
    with journal_path.open("w", encoding="utf-8") as f:
        json.dump(journal, f, indent=2, ensure_ascii=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if result["equity_curve"]:
        eq_df = pd.DataFrame(result["equity_curve"])
        eq_path = out_dir / f"equity_curve_{run_id}.csv"
        eq_df.to_csv(eq_path, index=False)
        summary["equity_curve_path"] = str(eq_path)

    summary["journal_path"] = str(journal_path)
    summary["summary_path"] = str(summary_path)
    return summary

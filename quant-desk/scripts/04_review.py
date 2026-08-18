#!/usr/bin/env python3
"""Generate the review HTML report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, ensure_layout, load_cfg  # noqa: E402

from quantkit.review import (
    equity_performance,
    plot_drawdown_png,
    plot_equity_png,
    write_review_html,
)


def main() -> int:
    ensure_layout()
    cfg = load_cfg()

    # equity
    eq = None
    for name in ("lifecycle_equity.csv", "equity_marks.csv"):
        path = OUT / name
        if path.exists():
            df = pd.read_csv(path)
            if "equity" in df.columns:
                if "date" in df.columns:
                    idx = pd.to_datetime(df["date"])
                elif "ts" in df.columns:
                    idx = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
                else:
                    continue
                eq = pd.Series(df["equity"].values, index=idx, name="equity").astype(float)
                break
    if eq is None or eq.empty:
        print("ERROR: no equity curve", file=sys.stderr)
        return 1

    perf = equity_performance(eq)
    charts = OUT / "charts"
    eq_png = plot_equity_png(eq, charts / "equity.png", title="Portfolio equity")
    dd_png = plot_drawdown_png(eq, charts / "drawdown.png")

    holdings = pd.read_csv(OUT / "holdings_latest.csv") if (OUT / "holdings_latest.csv").exists() else None
    fills = pd.read_csv(OUT / "fills.csv") if (OUT / "fills.csv").exists() else None
    selection = None
    if (OUT / "selection_scores.csv").exists():
        selection = pd.read_csv(OUT / "selection_scores.csv")
        if len(selection) > 30:
            selection = selection.head(30)

    contrib = None
    if (OUT / "lifecycle_contribution.csv").exists():
        c = pd.read_csv(OUT / "lifecycle_contribution.csv", index_col=0).iloc[:, 0]
        contrib = c

    notes = [
        f"Desk: {cfg.get('name', 'quant_desk')}, initial cash {cfg.get('initial_cash')}",
        f"Selection mode: {(cfg.get('selection') or {}).get('mode')}, Top-N={(cfg.get('selection') or {}).get('top_n')}",
        "This report reviews the research/paper pipeline; for live trading connect vnpy_cloud_bridge or a broker interface.",
    ]
    if (OUT / "selection_meta.json").exists():
        meta = json.loads((OUT / "selection_meta.json").read_text(encoding="utf-8"))
        notes.insert(0, f"Latest selection: {meta.get('selected')} (asof {meta.get('asof')})")

    extra = {}
    if (OUT / "performance.json").exists():
        extra = {k: v for k, v in json.loads((OUT / "performance.json").read_text()).items()
                 if k.startswith("ret_")}

    html = write_review_html(
        title=f"Research review · {cfg.get('name', 'quant_desk')}",
        out_path=OUT / "review_latest.html",
        performance=perf,
        holdings=holdings,
        fills=fills,
        selection=selection,
        contribution=contrib,
        equity_png=eq_png,
        dd_png=dd_png,
        notes=notes,
        extra_metrics={k: f"{float(v)*100:+.2f}%" for k, v in extra.items()},
    )
    # also copy name for lifecycle if present
    print(f"Review report → {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

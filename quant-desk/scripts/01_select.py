#!/usr/bin/env python3
"""Cross-sectional selection: universe → score → filter → Top-N target weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, OUT, ensure_layout, load_cfg  # noqa: E402

from quantkit.selection import run_selection


def main() -> int:
    cfg = load_cfg()
    sel = cfg.get("selection") or {}
    p = argparse.ArgumentParser(description="quant_desk stock selection")
    p.add_argument("--top-n", type=int, default=int(sel.get("top_n", 5)))
    p.add_argument("--mode", default=sel.get("mode", "composite"))
    p.add_argument("--asof", default=None, help="YYYY-MM-DD, defaults to latest")
    args = p.parse_args()
    ensure_layout()

    universe = list(cfg.get("universe") or [])
    if not universe:
        print("ERROR: config.universe is empty", file=sys.stderr)
        return 1

    result = run_selection(
        universe,
        start=sel.get("history_start", "2020-01-01"),
        mode=args.mode,  # type: ignore[arg-type]
        top_n=args.top_n,
        market=(cfg.get("trading") or {}).get("market", "us"),
        provider=(cfg.get("trading") or {}).get("provider", "yahoo"),
        data_dir=DATA,
        asof=args.asof,
        weight_scheme=sel.get("weight_scheme", "equal"),  # type: ignore[arg-type]
        max_weight=float(sel.get("max_weight", 0.3)),
        min_price=sel.get("min_price"),
        max_vol_ann=sel.get("max_vol_ann"),
        require_uptrend=bool(sel.get("require_uptrend", False)),
    )

    scores_path = OUT / "selection_scores.csv"
    result.scores.to_csv(scores_path)
    result.rejected.to_csv(OUT / "selection_rejected.csv")
    weights = result.target_weights
    weights.to_csv(OUT / "target_weights.csv", header=["target_weight"])
    meta = {
        "asof": str(result.asof),
        "selected": result.selected,
        "weights": weights.to_dict(),
        **result.meta,
    }
    (OUT / "selection_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(f"\nSelected: {result.selected}")
    print(f"→ {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

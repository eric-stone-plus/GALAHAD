#!/usr/bin/env python3
"""01b_optimize.py — optimizer weights: read selection output and compute optimal weights with the index-enhancement optimizer.

Wiring: 01_select.py → 01b_optimize.py → 02_trade.py

Usage:
    quant-python scripts/01b_optimize.py [--te-max 0.05] [--turnover-cap 0.3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, OUT, ensure_layout, load_cfg  # noqa: E402

from quantkit.data import fetch_ohlcv
from quantkit.optimizer import index_enhanced_weights, lw_shrinkage_cov


def main() -> int:
    cfg = load_cfg()
    opt = cfg.get("optimizer") or {}
    p = argparse.ArgumentParser(description="quant_desk optimizer weights")
    p.add_argument("--te-max", type=float, default=float(opt.get("te_max", 0.05)))
    p.add_argument("--turnover-cap", type=float, default=float(opt.get("turnover_cap", 0.30)))
    p.add_argument("--long-only", action="store_true", default=True)
    args = p.parse_args()
    ensure_layout()

    # Load selection output
    weights_path = OUT / "target_weights.csv"
    meta_path = OUT / "selection_meta.json"
    if not weights_path.exists():
        print("ERROR: target_weights.csv not found; run 01_select.py first", file=sys.stderr)
        return 1

    tw = pd.read_csv(weights_path, index_col=0).iloc[:, 0]
    symbols = list(tw.index)
    benchmark_w = tw.values

    if not meta_path.exists():
        print("ERROR: selection_meta.json not found", file=sys.stderr)
        return 1

    meta = json.loads(meta_path.read_text())
    print(f"Selection: {len(symbols)} names, equal-weight benchmark")

    # Fetch historical returns for covariance estimation
    trading = cfg.get("trading") or {}
    market = trading.get("market", "us")
    provider = trading.get("provider", "yahoo")

    returns = {}
    for sym in symbols:
        df = fetch_ohlcv(
            sym,
            market=market,
            provider=provider,
            start="2022-01-01",
            data_dir=DATA,
        )
        if not df.empty:
            returns[sym] = df["close"].pct_change().dropna()

    if len(returns) < 2:
        print("ERROR: fewer than 2 usable names; cannot optimize", file=sys.stderr)
        return 1

    ret_df = pd.DataFrame(returns).dropna()
    print(f"Covariance matrix: {ret_df.shape[0]} days × {ret_df.shape[1]} names")

    # Estimate covariance
    cov = lw_shrinkage_cov(ret_df.values)

    # Score: use trailing momentum as alpha signal
    scores = ret_df.iloc[-60:].mean().values  # 60d avg return as score

    # Optimize
    try:
        opt_w = index_enhanced_weights(
            scores,
            benchmark_w,
            cov,
            te_max=args.te_max,
            turnover_cap=args.turnover_cap,
            long_only=args.long_only,
        )
    except Exception as e:
        print(f"Optimization failed, falling back to equal weight: {e}", file=sys.stderr)
        opt_w = benchmark_w

    # Output optimized weights
    opt_series = pd.Series(opt_w, index=symbols, name="optimized_weight")
    opt_series.to_csv(OUT / "optimized_weights.csv")

    # Also overwrite target_weights.csv for downstream 02_trade.py
    opt_series.to_csv(OUT / "target_weights.csv", header=["target_weight"])

    # Summary
    te = float(np.sqrt(np.sum((opt_w - benchmark_w) ** 2)))
    turnover = float(0.5 * np.sum(np.abs(opt_w - benchmark_w)))
    print(f"Optimization done: TE={te:.4f}, turnover={turnover:.4f}")
    print(f"Weight range: [{opt_w.min():.3f}, {opt_w.max():.3f}]")

    summary = {
        "n_symbols": len(symbols),
        "te_max": args.te_max,
        "turnover_cap": args.turnover_cap,
        "realized_te": te,
        "realized_turnover": turnover,
        "weights": opt_series.to_dict(),
    }
    (OUT / "optimizer_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"Overwrote target_weights.csv → {weights_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

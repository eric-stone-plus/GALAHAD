#!/usr/bin/env python3
"""Honest comparison: short TSMOM / long TSMOM / dual_ma on venue or cache bars."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
QUANT = ROOT.parent / "quant"
for p in (str(ROOT), str(QUANT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from galahad_futures.data import load_bars, sample_kind_for_source
from galahad_futures.engine import load_config, run_paper_on_bars
from galahad_futures.walkforward import bar_walk_forward_splits, oos_bar_slice

from quantkit.validation import deflated_sharpe_ratio, prob_backtest_overfitting


def _sharpe(rets) -> float:
    r = np.asarray(list(rets), dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return 0.0
    return float(r.mean() / (r.std(ddof=1) + 1e-12))


# Pre-specified families only — no open-ended grid
PRESETS = [
    {"id": "tsmom_48", "name": "tsmom", "kwargs": {"lookback": 48, "max_target_leverage": 1.5}},
    {"id": "tsmom_168", "name": "tsmom_long", "kwargs": {"lookback": 168, "max_target_leverage": 1.0}},
    {"id": "dual_ma_8_21", "name": "dual_ma", "kwargs": {"fast": 8, "slow": 21, "max_target_leverage": 2.0}},
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cache", choices=("auto", "cache", "rest", "fixture", "venue"))
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    symbol = args.symbol or str(cfg.get("symbol", "BTCUSDT"))
    interval = str(cfg.get("interval", "1h"))
    data_cfg = dict(cfg.get("data") or {})
    bars, src, note = load_bars(
        source=args.source,
        fixture_path=ROOT / data_cfg.get("fixture_path", "data/fixtures/btcusdt_1h.csv"),
        project_root=ROOT,
        symbol=symbol,
        interval=interval,
        limit=int(data_cfg.get("fetch_limit", 500)),
        rest_timeout=float(data_cfg.get("rest_timeout_sec", 12)),
        rest_url_template=data_cfg.get("rest_url_template"),
    )
    sample_kind = sample_kind_for_source(src)

    rows = []
    ret_cols = {}
    for preset in PRESETS:
        full = run_paper_on_bars(
            bars,
            cfg,
            symbol=symbol,
            strategy_name=preset["name"],
            strategy_kwargs=preset["kwargs"],
        )
        # Simple OOS: last 25% bars with warmup
        n = len(bars)
        oos_start = max(int(n * 0.75), 50)
        warmup = int(preset["kwargs"].get("lookback", preset["kwargs"].get("slow", 20)))
        w0 = max(0, oos_start - warmup)
        slice_bars = bars.iloc[w0:].reset_index(drop=True)
        oos = run_paper_on_bars(
            slice_bars,
            cfg,
            symbol=symbol,
            strategy_name=preset["name"],
            strategy_kwargs=preset["kwargs"],
            evaluate_from=oos_start - w0,
        )
        rets = oos.get("returns_oos") or []
        ret_cols[preset["id"]] = rets
        dsr = None
        if len(rets) >= 8:
            dsr = float(
                deflated_sharpe_ratio(np.asarray(rets, dtype=float), n_trials=1, sr_std=0.0)
            )
        rows.append(
            {
                "id": preset["id"],
                "strategy": preset["name"],
                "kwargs": preset["kwargs"],
                "full_final_equity": full["final_equity"],
                "full_n_fills": full["n_fills"],
                "full_total_funding": full["total_funding"],
                "full_invalidated": full.get("invalidated"),
                "full_max_drawdown": full.get("max_drawdown"),
                "oos_final_equity": oos["final_equity"],
                "oos_n_fills": oos.get("n_fills_oos", oos["n_fills"]),
                "oos_sharpe": _sharpe(rets),
                "oos_returns_len": len(rets),
                "deflated_sharpe_ratio": dsr,
            }
        )

    # Comparative PBO across the three pre-specified series (honest multi-trial)
    pbo = None
    pbo_err = None
    try:
        import pandas as pd

        min_len = min(len(v) for v in ret_cols.values() if len(v) > 0)
        if min_len >= 16:
            mat = pd.DataFrame({k: v[-min_len:] for k, v in ret_cols.items() if len(v) >= min_len})
            pbo = float(prob_backtest_overfitting(mat, n_blocks=4, metric="sharpe"))
    except Exception as e:  # noqa: BLE001
        pbo_err = f"{type(e).__name__}: {e}"

    flags = []
    if sample_kind == "synthetic_fixture" or src == "fixture":
        flags.append("SYNTHETIC_FIXTURE_NOT_EDGE_CLAIM")
    if sample_kind == "venue":
        flags.append("VENUE_OR_CACHE_SAMPLE")
    if pbo is not None and pbo > 0.5:
        flags.append("PBO_HIGH_RED")
    # None of the presets claim edge by default
    flags.append("COMPARISON_ONLY_NO_EDGE_CLAIM")

    report = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "symbol": symbol,
        "source_used": src,
        "sample_kind": sample_kind,
        "data_note": note,
        "bars": len(bars),
        "strategies": rows,
        "pbo_across_presets": pbo,
        "pbo_error": pbo_err,
        "policy_flags": flags,
        "doctrine": (
            "Pre-specified short TSMOM / long TSMOM / dual_ma comparison on one sample. "
            "Does not promote any row when PBO/DSR fail."
        ),
    }

    out = ROOT / "output" / "strategy_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(out)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("GALAHAD strategy comparison")
        print(f"  source: {src} ({sample_kind})  symbol={symbol}  bars={len(bars)}")
        print(f"  PBO(presets): {pbo}  flags={flags}")
        for r in rows:
            print(
                f"  {r['id']}: full_eq={r['full_final_equity']:.2f} "
                f"oos_sharpe={r['oos_sharpe']:.4f} dsr={r['deflated_sharpe_ratio']} "
                f"dd={r['full_max_drawdown']:.4f} inv={r['full_invalidated']}"
            )
        print(f"  report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

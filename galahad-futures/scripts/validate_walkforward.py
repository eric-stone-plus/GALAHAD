#!/usr/bin/env python3
"""Walk-forward validation for a pre-specified strategy on venue/cache bars."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QUANT = ROOT.parent / "quant"
for p in (str(ROOT), str(QUANT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from galahad_futures.data import load_bars, sample_kind_for_source
from galahad_futures.engine import load_config, run_paper_on_bars
from galahad_futures.strategy import strategy_kwargs_from_config
from galahad_futures.walkforward import bar_walk_forward_splits, oos_bar_slice

from quantkit.validation import deflated_sharpe_ratio, prob_backtest_overfitting


def _sharpe(rets: list[float] | np.ndarray) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return 0.0
    return float(r.mean() / (r.std(ddof=1) + 1e-12))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward validate TSMOM/dual_ma")
    ap.add_argument(
        "--source",
        choices=("auto", "cache", "rest", "fixture", "venue"),
        default="auto",
    )
    ap.add_argument("--strategy", default=None, help="override strategy.name")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=150)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    symbol = str(cfg.get("symbol", "BTCUSDT"))
    interval = str(cfg.get("interval", "1h"))
    data_cfg = dict(cfg.get("data") or {})
    fetch_limit = int(data_cfg.get("fetch_limit", 500))
    # Use full cache for WF (not bar_limit trim)
    bars, src, note = load_bars(
        source=args.source,
        fixture_path=ROOT / data_cfg.get("fixture_path", "data/fixtures/btcusdt_1h.csv"),
        project_root=ROOT,
        symbol=symbol,
        interval=interval,
        limit=fetch_limit,
        rest_timeout=float(data_cfg.get("rest_timeout_sec", 12)),
        rest_url_template=data_cfg.get("rest_url_template"),
    )
    sample_kind = sample_kind_for_source(src)
    strat_cfg = dict(cfg.get("strategy") or {})
    strat_name = args.strategy or str(strat_cfg.get("name", "tsmom"))
    strat_kw = strategy_kwargs_from_config(strat_cfg)
    warmup = int(strat_kw.get("lookback", strat_kw.get("slow", 48)))

    folds_meta = []
    oos_returns_all: list[float] = []
    try:
        splits = list(
            bar_walk_forward_splits(
                len(bars),
                n_folds=args.n_folds,
                min_train=min(args.min_train, max(80, len(bars) // 3)),
                purge=max(1, warmup // 10),
            )
        )
    except ValueError as e:
        report = {
            "status": "insufficient_bars",
            "error": str(e),
            "source_used": src,
            "sample_kind": sample_kind,
            "bars": len(bars),
        }
        out = ROOT / "output" / "walkforward_report.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(report)
        return 0

    for fi, (tr, te) in enumerate(splits):
        slice_bars, oos_start = oos_bar_slice(bars, tr, te, warmup=warmup)
        res = run_paper_on_bars(
            slice_bars,
            cfg,
            symbol=symbol,
            strategy_name=strat_name,
            strategy_kwargs=strat_kw,
            evaluate_from=oos_start,
        )
        rets = res.get("returns_oos") or []
        oos_returns_all.extend(rets)
        folds_meta.append(
            {
                "fold": fi,
                "train_n": int(len(tr)),
                "test_n": int(len(te)),
                "test_start": str(bars.iloc[int(te[0])]["ts"]),
                "test_end": str(bars.iloc[int(te[-1])]["ts"]),
                "final_equity": res["final_equity"],
                "n_fills": res["n_fills"],
                "n_fills_oos": res["n_fills_oos"],
                "total_funding": res["total_funding"],
                "n_funding_events": res["n_funding_events"],
                "sharpe_oos": _sharpe(rets),
                "liquidated": res["liquidated"],
                "n_oos_returns": len(rets),
            }
        )

    # DSR on concatenated OOS returns; n_trials=1 for pre-specified rule
    dsr = None
    pbo = None
    pbo_err = None
    if len(oos_returns_all) >= 8:
        r = np.asarray(oos_returns_all, dtype=float)
        dsr = float(deflated_sharpe_ratio(r, n_trials=1, sr_std=0.0))
        # Optional: compare to dual_ma as second series for toy PBO matrix
        try:
            dual = run_paper_on_bars(
                bars,
                cfg,
                symbol=symbol,
                strategy_name="dual_ma",
                strategy_kwargs={"fast": 8, "slow": 21, "max_target_leverage": 2.0},
            )
            dual_r = dual.get("returns_oos") or []
            # align lengths
            m = min(len(oos_returns_all), len(dual_r))
            if m >= 16:
                mat = pd.DataFrame(
                    {
                        strat_name: oos_returns_all[-m:],
                        "dual_ma_8_21": dual_r[-m:],
                    }
                )
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
    if dsr is not None and dsr <= 0.05:
        flags.append("DSR_WEAK")
    if dsr is not None and dsr > 0.05 and (pbo is None or pbo <= 0.5):
        flags.append("GATES_WEAK_PASS_ON_THIS_SAMPLE")
    if not folds_meta:
        flags.append("NO_FOLDS")

    report = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok" if folds_meta else "no_folds",
        "strategy": strat_name,
        "strategy_kwargs": strat_kw,
        "source_used": src,
        "sample_kind": sample_kind,
        "data_note": note,
        "bars": len(bars),
        "symbol": symbol,
        "n_folds": len(folds_meta),
        "folds": folds_meta,
        "oos_returns_len": len(oos_returns_all),
        "oos_sharpe": _sharpe(oos_returns_all),
        "deflated_sharpe_ratio": dsr,
        "pbo": pbo,
        "pbo_error": pbo_err,
        "policy_flags": flags,
        "funding_rate_per_bar": float(cfg.get("funding_rate_per_bar", 0.0)),
        "doctrine": (
            "Walk-forward OOS for a pre-specified rule; not a claim of edge. "
            "PBO vs dual_ma is comparative only."
        ),
    }

    out = ROOT / "output" / "walkforward_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(out)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("GALAHAD walk-forward validation")
        print(f"  strategy:   {strat_name}")
        print(f"  source:     {src} ({sample_kind})")
        print(f"  bars:       {len(bars)}")
        print(f"  folds:      {len(folds_meta)}")
        print(f"  oos_sharpe: {report['oos_sharpe']:.4f}")
        print(f"  DSR:        {dsr}")
        print(f"  PBO:        {pbo}")
        print(f"  flags:      {flags}")
        for f in folds_meta:
            print(
                f"    fold{f['fold']}: eq={f['final_equity']:.2f} "
                f"fills_oos={f['n_fills_oos']} sharpe={f['sharpe_oos']:.3f} "
                f"funding={f['total_funding']:.4f}"
            )
        print(f"  report:     {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

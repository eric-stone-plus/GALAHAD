#!/usr/bin/env python3
"""DSR / PBO statistical evaluation for futures paper strategies.

Runs a walk-forward split over the bars (train/test splits via
``galahad_futures.walkforward.bar_walk_forward_splits``) with the paper
reference engine, collects the out-of-sample return series per window, and
prints a statistics JSON::

    dsr         deflated Sharpe ratio (Bailey & LdP 2014), per-bar units
    pbo         probability of backtest overfitting (CSCV, Bailey et al. 2017)
    oos_sharpe  annualized Sharpe of the concatenated OOS returns
    n_windows   number of usable walk-forward windows
    dsr_pass    acceptance flag: DSR > 0
    pbo_flag    acceptance flag: PBO > 0.5

Advisory flags only (roadmap P1): the numbers are evidence for review, not
promotion gates.

Usage:
    python scripts/run_statistics.py [--source fixture] [--strategy tsmom]
                                     [--folds 3] [--min-train 60] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galahad_futures.data import load_bars, sample_kind_for_source  # noqa: E402
from galahad_futures.engine import load_config, run_paper_on_bars  # noqa: E402
from galahad_futures.statistics import evaluate_oos_statistics  # noqa: E402
from galahad_futures.strategy import strategy_kwargs_from_config  # noqa: E402
from galahad_futures.walkforward import bar_walk_forward_splits, oos_bar_slice  # noqa: E402

# 1h bars -> 8760 periods/year (same convention as scripts/walkforward_runner.py)
DEFAULT_PERIODS_PER_YEAR = 8760.0


def _periods_per_year(interval: str) -> float:
    """Annualization factor for a bar interval string (1h -> 8760)."""
    ivals = {"1m": 525600.0, "5m": 105120.0, "15m": 35040.0, "1h": 8760.0,
             "4h": 2190.0, "8h": 1095.0, "12h": 730.0, "1d": 365.0}
    key = interval.strip().lower()
    if key in ivals:
        return ivals[key]
    print(f"  warning: unknown interval {interval!r}, assuming 1h (8760/yr)",
          file=sys.stderr)
    return DEFAULT_PERIODS_PER_YEAR


def build_statistics_report(
    cfg: dict,
    bars,
    *,
    strategy_name: str | None = None,
    strategy_kwargs: dict | None = None,
    n_folds: int = 3,
    min_train: int = 60,
    test_size: int | None = None,
    purge: int = 5,
    warmup: int = 40,
    n_blocks: int = 16,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    source_used: str | None = None,
) -> dict:
    """Walk-forward OOS sweep with the paper engine -> statistics report.

    Returns the report dict (JSON-serializable); raises ValueError when no
    usable OOS window survives (fail-closed, same policy as
    ``evaluate_oos_statistics``).
    """
    symbol = str(cfg.get("symbol", "BTCUSDT"))
    strat_cfg = dict(cfg.get("strategy") or {})
    name = strategy_name or str(strat_cfg.get("name", "dual_ma"))
    kw = strategy_kwargs if strategy_kwargs is not None else strategy_kwargs_from_config(strat_cfg)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    n = len(bars)
    window_rows: list[dict] = []
    window_rets: list[list[float]] = []
    window_notes: list[str] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        bar_walk_forward_splits(
            n, n_folds=n_folds, min_train=min_train, test_size=test_size, purge=purge
        )
    ):
        slice_bars, oos_start = oos_bar_slice(bars, train_idx, test_idx, warmup=warmup)
        try:
            result = run_paper_on_bars(
                slice_bars,
                cfg,
                symbol=symbol,
                strategy_name=name,
                strategy_kwargs=dict(kw),
                evaluate_from=oos_start,
            )
        except Exception as exc:  # pragma: no cover - defensive; engine is deterministic
            window_notes.append(f"fold {fold_idx}: engine error: {exc}")
            continue
        rets = [float(v) for v in result.get("returns_oos") or []]
        if len(rets) < 3:
            window_notes.append(
                f"fold {fold_idx}: {len(rets)} OOS returns (< 3), skipped"
            )
            continue
        window_rets.append(rets)
        window_rows.append(
            {
                "window": fold_idx,
                "train_bars": int(len(train_idx)),
                "test_bars": int(len(test_idx)),
                "n_fills": int(result["n_fills"]),
                "n_risk_rejects": int(result["n_risk_rejects"]),
                "liquidated": bool(result["liquidated"]),
                "invalidated": bool(result["invalidated"]),
            }
        )

    if not window_rets:
        raise ValueError(
            "run_statistics: no usable OOS windows "
            f"({len(window_notes)} notes: {'; '.join(window_notes) or 'none'})"
        )

    stats = evaluate_oos_statistics(
        window_rets, n_blocks=n_blocks, periods_per_year=periods_per_year
    )
    report = {
        "schema": "galahad.statistics.v1",
        "run_id": run_id,
        "inputs": {
            "symbol": symbol,
            "strategy": name,
            "strategy_kwargs": dict(kw),
            "bars": n,
            "n_folds": n_folds,
            "min_train": min_train,
            "test_size": test_size,
            "purge": purge,
            "warmup": warmup,
            "n_blocks": n_blocks,
            "periods_per_year": periods_per_year,
            "source_used": source_used,
        },
        "statistics": {
            "dsr": stats["dsr"],
            "pbo": stats["pbo"],
            "oos_sharpe": stats["oos_sharpe"],
            "n_windows": stats["n_windows"],
            "oos_obs": stats["oos_obs"],
            "dsr_pass": stats["dsr_pass"],
            "pbo_flag": stats["pbo_flag"],
        },
        "windows": window_rows,
        "notes": window_notes,
        "warnings": stats["warnings"],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GALAHAD futures DSR/PBO statistics")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--source", default="fixture",
                    help="data source (default fixture for determinism)")
    ap.add_argument("--strategy", default=None, help="override strategy.name")
    ap.add_argument("--folds", type=int, default=3, help="walk-forward folds")
    ap.add_argument("--min-train", type=int, default=60, help="min train bars")
    ap.add_argument("--test-size", type=int, default=None,
                    help="OOS test bars per fold (default: remainder // folds)")
    ap.add_argument("--purge", type=int, default=5, help="purge bars before OOS")
    ap.add_argument("--warmup", type=int, default=40, help="indicator warmup bars")
    ap.add_argument("--n-blocks", type=int, default=16, help="CSCV blocks")
    ap.add_argument("--output-dir", default=None, help="override report output dir")
    ap.add_argument("--json", action="store_true", help="print report JSON only")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    cfg = load_config(args.config)
    data_cfg = dict(cfg.get("data") or {})
    symbol = str(cfg.get("symbol", "BTCUSDT"))
    interval = str(cfg.get("interval", "1h"))
    bars, source_used, _note = load_bars(
        source=args.source,
        fixture_path=data_cfg.get("fixture_path", "data/fixtures/btcusdt_1h.csv"),
        rest_url=None,
        rest_timeout=float(data_cfg.get("rest_timeout_sec", 12)),
        project_root=root,
        symbol=symbol,
        interval=interval,
        limit=int(data_cfg.get("fetch_limit", 500)),
        rest_url_template=data_cfg.get("rest_url_template"),
    )

    try:
        report = build_statistics_report(
            cfg,
            bars,
            strategy_name=args.strategy,
            n_folds=args.folds,
            min_train=args.min_train,
            test_size=args.test_size,
            purge=args.purge,
            warmup=args.warmup,
            n_blocks=args.n_blocks,
            periods_per_year=_periods_per_year(interval),
            source_used=sample_kind_for_source(source_used),
        )
    except ValueError as exc:
        print(f"run_statistics: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir) if args.output_dir else root / str(cfg.get("output_dir", "output"))
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"statistics_{report['run_id']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    report["report_path"] = str(path)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        s = report["statistics"]
        print("GALAHAD futures DSR/PBO statistics")
        print(f"  run_id:          {report['run_id']}")
        print(f"  strategy:        {report['inputs']['strategy']} "
              f"({report['inputs']['bars']} bars, {s['n_windows']} OOS windows)")
        print(f"  oos_sharpe:      {s['oos_sharpe']:.3f} (annualized, "
              f"{report['inputs']['periods_per_year']:.0f} periods/yr)")
        print(f"  dsr:             {s['dsr']:.4f}  -> dsr_pass={s['dsr_pass']} (DSR > 0)")
        print(f"  pbo:             {s['pbo']:.4f}  -> pbo_flag={s['pbo_flag']} (PBO > 0.5)")
        print(f"  report:          {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

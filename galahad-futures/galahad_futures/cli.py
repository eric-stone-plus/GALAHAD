"""CLI entry for futures paper sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    # Ensure project root on path when invoked as script
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from galahad_futures.engine import run_paper_session

    ap = argparse.ArgumentParser(
        description="GALAHAD Futures paper session (default: paper-only, fixture-capable)"
    )
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument(
        "--source",
        choices=("auto", "fixture", "rest", "cache", "venue"),
        default=None,
        help="override data.source (auto=rest→cache→parquet→fixture)",
    )
    ap.add_argument("--output-dir", default=None, help="override output directory")
    ap.add_argument(
        "--strategy",
        default=None,
        help="override strategy.name (tsmom | tsmom_long | dual_ma)",
    )
    ap.add_argument("--symbol", default=None, help="override symbol (e.g. ETHUSDT)")
    ap.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="override TSMOM lookback (pre-specified; e.g. 48 or 168)",
    )
    ap.add_argument(
        "--engine",
        choices=("paper", "nautilus"),
        default=None,
        help="execution backend (default: paper reference book; "
        "nautilus requires the optional nautilus_trader dependency)",
    )
    ap.add_argument("--json", action="store_true", help="print summary JSON only")
    args = ap.parse_args(argv)

    summary = run_paper_session(
        config_path=args.config,
        force_source=args.source,
        output_dir=args.output_dir,
        force_strategy=args.strategy,
        force_symbol=args.symbol,
        force_lookback=args.lookback,
        engine=args.engine,
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("GALAHAD Futures paper session")
        print(f"  status:         {summary['status']}")
        print(f"  engine:         {summary['engine']} ({summary['engine_version']})")
        print(f"  strategy:       {summary.get('strategy')}")
        print(f"  symbol:         {summary['symbol']}")
        print(f"  bars:           {summary['bars']}")
        print(f"  source:         {summary['source_used']}")
        if summary.get("data_note"):
            print(f"  data_note:      {summary['data_note']}")
        print(f"  fills:          {summary['n_fills']}")
        print(f"  risk_rejects:   {summary['n_risk_rejects']}")
        print(f"  equity_curve:   {summary['equity_curve_len']}")
        print(f"  initial_equity: {summary['initial_equity']:.4f}")
        print(f"  final_equity:   {summary['final_equity']:.4f}")
        print(f"  sample_kind:    {summary.get('sample_kind')}")
        print(f"  total_funding:  {summary.get('total_funding', 0):.6f}")
        print(f"  funding_events: {summary.get('n_funding_events', 0)}")
        print(f"  invalidated:    {summary.get('invalidated')}")
        print(f"  max_drawdown:   {summary.get('max_drawdown')}")
        print(f"  liquidated:     {summary['liquidated']}")
        print(f"  journal:        {summary.get('journal_path')}")
    # Exit 0 on successful paper plumbing; non-zero only on exception (raised)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

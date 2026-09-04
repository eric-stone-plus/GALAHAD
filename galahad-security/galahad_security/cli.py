"""CLI entry for equities paper sessions."""

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

    from galahad_security.engine import run_paper_session

    ap = argparse.ArgumentParser(
        description="GALAHAD Security paper session (default: offline paper book, fixture-capable)"
    )
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument(
        "--source",
        choices=("auto", "fixture", "cache"),
        default=None,
        help="override data.source (auto=cache→fixture; venue data only in venue mode)",
    )
    ap.add_argument("--output-dir", default=None, help="override output directory")
    ap.add_argument("--strategy", default=None, help="override strategy.name (dual_ma)")
    ap.add_argument(
        "--symbols",
        default=None,
        help="comma-separated symbol override (e.g. AAPL,MSFT)",
    )
    ap.add_argument(
        "--engine",
        choices=("paper", "alpaca_paper"),
        default=None,
        help="execution backend (default: paper reference cash book; "
        "alpaca_paper executes today's decision once on the Alpaca PAPER "
        "endpoint and requires ALPACA_PAPER_API_KEY/ALPACA_PAPER_API_SECRET "
        "plus risk.enable_alpaca_paper: true with the kill switch off)",
    )
    ap.add_argument("--json", action="store_true", help="print summary JSON only")
    args = ap.parse_args(argv)

    force_symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else None
    )
    try:
        summary = run_paper_session(
            config_path=args.config,
            force_source=args.source,
            output_dir=args.output_dir,
            force_strategy=args.strategy,
            force_symbols=force_symbols,
            engine=args.engine,
        )
    except RuntimeError as exc:
        # Fail closed with a clean operator-facing error (no traceback dump):
        # missing venue credentials, closed venue gate, venue transport errors.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("GALAHAD Security paper session")
        print(f"  status:         {summary['status']}")
        print(f"  mode:           {summary.get('mode')}  [paper money only — no live path exists]")
        print(f"  engine:         {summary['engine']} ({summary['engine_version']})")
        if summary.get("venue"):
            print(f"  venue:          {summary['venue']} (paper endpoint)")
        recon = summary.get("reconciliation")
        if recon:
            print(
                "  reconciliation: "
                f"submitted={recon['orders_submitted']} "
                f"filled={recon['orders_filled']} "
                f"position_mismatch={recon['position_mismatch']}"
            )
        print(f"  strategy:       {summary.get('strategy')}")
        print(f"  symbols:        {','.join(summary['symbols'])}")
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
        tca = summary.get("tca")
        if tca:
            print(
                f"  tca:            IS={tca['implementation_shortfall_usdt']:.4f} USD "
                f"({tca['implementation_shortfall_bps']:.2f} bps), "
                f"fees={tca['fee_cost_usdt']:.4f}"
            )
        derisk = summary.get("derisk")
        if derisk:
            print(
                f"  derisk:         enabled={derisk['ladder_enabled']} "
                f"tiers_triggered={derisk['tiers_triggered']} "
                f"min_multiplier={derisk['min_multiplier']}"
            )
        print(f"  invalidated:    {summary.get('invalidated')}")
        print(f"  max_drawdown:   {summary.get('max_drawdown')}")
        print(f"  journal:        {summary.get('journal_path')}")
    # Exit 0 on successful paper plumbing; non-zero only on error (above)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

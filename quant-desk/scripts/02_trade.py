#!/usr/bin/env python3
"""Paper trading: read target weights, rebalance at latest prices, and book the fills."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, OUT, STATE, ensure_layout, load_cfg, resolve_fee_bps  # noqa: E402

from quantkit.book import PaperBook
from quantkit.data import fetch_ohlcv


def latest_prices(symbols: list[str], cfg: dict) -> dict[str, float]:
    trading = cfg.get("trading") or {}
    out: dict[str, float] = {}
    for sym in symbols:
        df = fetch_ohlcv(
            sym,
            market=trading.get("market", "us"),
            provider=trading.get("provider", "yahoo"),
            start="2024-01-01",
            data_dir=DATA,
        )
        if not df.empty:
            out[sym] = float(df["close"].iloc[-1])
    return out


def main() -> int:
    cfg = load_cfg()
    p = argparse.ArgumentParser(description="quant_desk paper rebalance")
    p.add_argument("--reset", action="store_true", help="reset the ledger to initial cash")
    args = p.parse_args()
    ensure_layout()

    book_path = STATE / "paper_book.json"
    if args.reset or not book_path.exists():
        fee_bps, _ = resolve_fee_bps(cfg)
        book = PaperBook(
            cash=float(cfg.get("initial_cash", 1_000_000)),
            fee_bps=fee_bps,
            name=cfg.get("name", "paper"),
        )
    else:
        book = PaperBook.load(book_path)

    tw_path = OUT / "target_weights.csv"
    if not tw_path.exists():
        print("ERROR: run 01_select.py first to generate target_weights.csv", file=sys.stderr)
        return 1
    tw = pd.read_csv(tw_path, index_col=0).iloc[:, 0]
    tw = tw[tw > 0]

    # prices for target + current holdings
    symbols = sorted(set(tw.index) | set(book.positions.keys()))
    prices = latest_prices(symbols, cfg)
    if not prices:
        print("ERROR: unable to fetch prices", file=sys.stderr)
        return 1

    band = float((cfg.get("trading") or {}).get("rebalance_band", 0.02))
    ts = datetime.now(timezone.utc).isoformat()
    fills = book.rebalance_to_weights(tw, prices, ts=ts, band=band)
    mark = book.mark(prices, ts=ts)

    book.save(book_path)
    book.holdings_table(prices).to_csv(OUT / "holdings_latest.csv", index=False)
    book.fills_frame().to_csv(OUT / "fills.csv", index=False)
    book.equity_frame().to_csv(OUT / "equity_marks.csv", index=False)

    summary = {
        "ts": ts,
        "equity": mark["equity"],
        "cash": book.cash,
        "n_fills": len(fills),
        "positions": book.positions,
        "weights": mark["weights"],
    }
    (OUT / "trade_summary.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"Ledger → {book_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

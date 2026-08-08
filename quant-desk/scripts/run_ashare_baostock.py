#!/usr/bin/env python3
"""A-share lifecycle backtest using baostock data source."""

import sys, json
from pathlib import Path
import pandas as pd
import baostock as bs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _common import OUT, STATE, ensure_layout, load_cfg
from quantkit.book import PaperBook
from quantkit.portfolio import _rebalance_mask
from quantkit.review import (
    contribution_from_weights, equity_performance,
    plot_drawdown_png, plot_equity_png, write_review_html,
)
from quantkit.selection import apply_filters, score_universe, select_top_n
from quantkit.gates import COST_TIERS

def fetch_baostock_close(symbols, start, end):
    """Fetch close prices from baostock, return DataFrame with DatetimeIndex."""
    lg = bs.login()
    frames = {}
    for sym in symbols:
        # baostock uses sh.XXXXXX / sz.XXXXXX format
        prefix = "sh" if sym.startswith("6") else "sz"
        code = f"{prefix}.{sym}"
        rs = bs.query_history_k_data_plus(
            code, "date,close", start_date=start, end_date=end,
            frequency="d", adjustflag="2"  # 前复权
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if rows:
            df = pd.DataFrame(rows, columns=["date", "close"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").dropna()
            if not df.empty:
                frames[sym] = df["close"]
                print(f"  {sym}: {len(df)} bars")
    bs.logout()
    if not frames:
        return pd.DataFrame()
    panel = pd.DataFrame(frames)
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def main():
    cfg = load_cfg()
    sel_cfg = cfg.get("selection") or {}
    bf = cfg.get("backfill") or {}
    
    universe = list(cfg.get("universe") or [])
    start = bf.get("start", "2023-01-01")
    rebalance = bf.get("rebalance", "M")
    top_n = int(sel_cfg.get("top_n", 5))
    mode = sel_cfg.get("mode", "composite")
    ensure_layout()

    print(f"A-share lifecycle: {len(universe)} symbols, start={start}, rebalance={rebalance}")
    prices = fetch_baostock_close(universe, sel_cfg.get("history_start", "2020-01-01"), "2026-08-08")
    if prices.empty:
        print("ERROR: no data", file=sys.stderr)
        return 1

    cal = prices.loc[start:].index
    if len(cal) < 30:
        print("ERROR: too few trading days", file=sys.stderr)
        return 1

    rebal = _rebalance_mask(cal, rebalance)
    tier = cfg.get("cost_tier", "mid")
    fee_bps = COST_TIERS.get(tier, 0.004) * 10_000.0 / 2.0
    book = PaperBook(
        cash=float(cfg.get("initial_cash", 1_000_000)),
        fee_bps=fee_bps, name=cfg.get("name", "ashare_lifecycle"),
    )
    band = float((cfg.get("trading") or {}).get("rebalance_band", 0.02))
    weight_rows = []
    last_selected = []

    for i, dt in enumerate(cal):
        px_row = prices.loc[dt].dropna()
        price_map = {s: float(v) for s, v in px_row.items()}

        if bool(rebal.iloc[i]) or i == 0:
            hist = prices.loc[:dt]
            scores = score_universe(hist, mode=mode, asof=dt)
            passed, rejected = apply_filters(
                scores,
                min_price=sel_cfg.get("min_price"),
                max_vol_ann=sel_cfg.get("max_vol_ann"),
                require_uptrend=bool(sel_cfg.get("require_uptrend", False)),
            )
            selected, tw = select_top_n(
                passed, n=top_n,
                weight_scheme=sel_cfg.get("weight_scheme", "equal"),
                max_weight=float(sel_cfg.get("max_weight", 0.3)),
            )
            last_selected = selected
            for s in selected:
                if s not in price_map and s in prices.columns:
                    val = prices.loc[:dt, s].dropna()
                    if not val.empty:
                        price_map[s] = float(val.iloc[-1])
            book.rebalance_to_weights(tw, price_map, ts=str(dt.date()), band=band)

        for s in list(book.positions.keys()):
            if s not in price_map and s in prices.columns:
                val = prices.loc[:dt, s].dropna()
                if not val.empty:
                    price_map[s] = float(val.iloc[-1])
        mark = book.mark(price_map, ts=str(dt.date()))
        w = mark["weights"]
        row = {"date": dt, "equity": mark["equity"], "cash": mark["cash"]}
        for s, wv in w.items():
            row[f"w_{s}"] = wv
        weight_rows.append(row)

        if i % 60 == 0 or i == len(cal) - 1:
            print(f"  {dt.date()} equity={mark['equity']:.0f} pos={mark['n_positions']}")

    # persist
    eq_df = book.equity_frame()
    if "ts" in eq_df.columns:
        eq_df = eq_df.rename(columns={"ts": "date"})
    eq_df.to_csv(OUT / "ashare_lifecycle_equity.csv", index=False)
    book.fills_frame().to_csv(OUT / "ashare_fills.csv", index=False)
    book.save(STATE / "ashare_paper_book.json")

    last_px = {s: float(prices[s].dropna().iloc[-1]) for s in prices.columns if not prices[s].dropna().empty}
    holdings = book.holdings_table(last_px)
    holdings.to_csv(OUT / "ashare_holdings_latest.csv", index=False)

    eq = pd.Series(eq_df["equity"].values, index=pd.to_datetime(eq_df["date"]), name="equity")
    perf = equity_performance(eq)
    perf_path = OUT / "ashare_performance.json"
    perf_path.write_text(json.dumps(perf.as_dict(), indent=2, default=str), encoding="utf-8")

    charts = OUT / "charts"
    eq_png = plot_equity_png(eq, charts / "ashare_equity.png", title="A-share lifecycle equity")
    dd_png = plot_drawdown_png(eq, charts / "ashare_drawdown.png")

    notes = [
        f"A-share lifecycle: {start} → {rebalance}, {mode} Top-{top_n}",
        f"Universe {len(universe)} symbols, cost_tier={tier}",
        f"Selected: {last_selected}",
        f"CAGR {perf.cagr*100:.1f}% · Sharpe {perf.sharpe:.2f} · MaxDD {perf.max_drawdown*100:.1f}%",
        "Data: baostock (前复权)",
    ]
    wdf = pd.DataFrame(weight_rows).set_index("date")
    wcols = [c for c in wdf.columns if c.startswith("w_")]
    weights = wdf[wcols].rename(columns=lambda c: c[2:])
    asset_rets = prices.pct_change().reindex(weights.index).fillna(0.0)
    contrib = contribution_from_weights(weights, asset_rets)

    html = write_review_html(
        title=f"A-share lifecycle · {cfg.get('name', 'quant_desk')}",
        out_path=OUT / "ashare_lifecycle_report.html",
        performance=perf, holdings=holdings,
        fills=book.fills_frame(), selection=None,
        contribution=contrib,
        equity_png=eq_png, dd_png=dd_png, notes=notes,
    )
    print(f"\nCAGR={perf.cagr*100:.1f}% Sharpe={perf.sharpe:.2f} MaxDD={perf.max_drawdown*100:.1f}%")
    print(f"Report → {html}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

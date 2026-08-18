#!/usr/bin/env python3
"""Grid search for A-share lifecycle backtest using baostock.

Runs multiple configurations (mode × top_n × rebalance × universe) on real
baostock data and saves results to output/ashare_grid_results.json.
"""

import sys
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import baostock as bs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from quantkit.book import PaperBook
from quantkit.portfolio import _rebalance_mask
from quantkit.review import equity_performance
from quantkit.selection import apply_filters, score_universe, select_top_n
from quantkit.gates import COST_TIERS

OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ── Universes ──────────────────────────────────────────────────────────────
UNIVERSE_BLUE = [
    "600519",  # Kweichow Moutai
    "000858",  # Wuliangye
    "601318",  # Ping An
    "600036",  # China Merchants Bank
    "000333",  # Midea Group
    "600900",  # Yangtze Power
    "002415",  # Hikvision
    "600276",  # Hengrui Medicine
    "000001",  # Ping An Bank
    "601888",  # China Duty Free
    "300750",  # CATL
    "002594",  # BYD
]

UNIVERSE_EXPANDED = UNIVERSE_BLUE + [
    "300059",  # Eastmoney
    "002475",  # Luxshare Precision
    "603259",  # WuXi AppTec
    "600584",  # JCET
    "002371",  # NAURA Technology
]

# ── Fetch helpers ──────────────────────────────────────────────────────────

def fetch_baostock_close(symbols, start, end):
    """Fetch close prices from baostock, return DataFrame with DatetimeIndex."""
    lg = bs.login()
    frames = {}
    for sym in symbols:
        prefix = "sh" if sym.startswith("6") else "sz"
        code = f"{prefix}.{sym}"
        rs = bs.query_history_k_data_plus(
            code, "date,close", start_date=start, end_date=end,
            frequency="d", adjustflag="2"  # forward-adjusted
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


def run_backtest(prices, start, rebalance, mode, top_n, fee_bps, band=0.02,
                 sel_cfg=None):
    """Run a single lifecycle backtest, return performance dict."""
    if sel_cfg is None:
        sel_cfg = {}

    cal = prices.loc[start:].index
    if len(cal) < 30:
        return {"error": True, "note": "too few trading days"}

    rebal = _rebalance_mask(cal, rebalance)
    book = PaperBook(cash=1_000_000.0, fee_bps=fee_bps, name=f"grid_{mode}")
    weight_rows = []
    last_selected = []

    for i, dt in enumerate(cal):
        px_row = prices.loc[dt].dropna()
        price_map = {s: float(v) for s, v in px_row.items()}

        if bool(rebal.iloc[i]) or i == 0:
            hist = prices.loc[:dt]
            scores = score_universe(hist, mode=mode, asof=dt)
            passed, _ = apply_filters(
                scores,
                min_price=sel_cfg.get("min_price", 3.0),
                max_vol_ann=sel_cfg.get("max_vol_ann", 1.5),
                require_uptrend=bool(sel_cfg.get("require_uptrend", False)),
            )
            selected, tw = select_top_n(
                passed, n=top_n,
                weight_scheme="equal",
                max_weight=0.40,
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
        weight_rows.append({
            "date": dt,
            "equity": mark["equity"],
            "cash": mark["cash"],
        })

    eq_df = book.equity_frame()
    if "ts" in eq_df.columns:
        eq_df = eq_df.rename(columns={"ts": "date"})
    eq = pd.Series(eq_df["equity"].values,
                    index=pd.to_datetime(eq_df["date"]), name="equity")
    perf = equity_performance(eq)
    return {
        "error": False,
        "cagr": perf.cagr,
        "total_return": perf.total_return,
        "sharpe": perf.sharpe,
        "sortino": getattr(perf, "sortino", 0),
        "max_drawdown": perf.max_drawdown,
        "vol_ann": perf.vol_ann,
        "win_rate": perf.win_rate,
        "n_days": perf.n_days,
        "final_equity": perf.final_equity,
        "start": str(perf.start),
        "end": str(perf.end),
        "last_selected": last_selected,
    }


# ── Grid definition ────────────────────────────────────────────────────────
GRID = [
    # (label, mode, top_n, rebalance, universe_key, universe, require_uptrend)
    ("momentum_3_W_blue",        "momentum",        3, "W", "blue",      UNIVERSE_BLUE,      False),
    ("momentum_3_M_blue",        "momentum",        3, "M", "blue",      UNIVERSE_BLUE,      False),
    ("momentum_5_W_blue",        "momentum",        5, "W", "blue",      UNIVERSE_BLUE,      False),
    ("momentum_5_M_blue",        "momentum",        5, "M", "blue",      UNIVERSE_BLUE,      False),
    ("composite_3_W_blue",       "composite",       3, "W", "blue",      UNIVERSE_BLUE,      False),
    ("composite_5_W_blue",       "composite",       5, "W", "blue",      UNIVERSE_BLUE,      False),
    ("mean_reversion_3_W_blue",  "mean_reversion",  3, "W", "blue",      UNIVERSE_BLUE,      False),
    ("mean_reversion_5_W_blue",  "mean_reversion",  5, "W", "blue",      UNIVERSE_BLUE,      False),
    ("quality_trend_3_W_blue",   "quality_trend",   3, "W", "blue",      UNIVERSE_BLUE,      False),
    ("quality_trend_5_W_blue",   "quality_trend",   5, "W", "blue",      UNIVERSE_BLUE,      False),
    # Expanded universe
    ("momentum_3_W_expanded",    "momentum",        3, "W", "expanded",  UNIVERSE_EXPANDED,  False),
    ("momentum_5_W_expanded",    "momentum",        5, "W", "expanded",  UNIVERSE_EXPANDED,  False),
    ("mean_reversion_3_W_expanded", "mean_reversion", 3, "W", "expanded", UNIVERSE_EXPANDED, False),
    ("mean_reversion_5_W_expanded", "mean_reversion", 5, "W", "expanded", UNIVERSE_EXPANDED, False),
    ("quality_trend_3_W_expanded",  "quality_trend",  3, "W", "expanded", UNIVERSE_EXPANDED, False),
    ("quality_trend_5_W_expanded",  "quality_trend",  5, "W", "expanded", UNIVERSE_EXPANDED, False),
    ("composite_3_W_expanded",   "composite",        3, "W", "expanded",  UNIVERSE_EXPANDED,  False),
    ("composite_5_W_expanded",   "composite",        5, "W", "expanded",  UNIVERSE_EXPANDED,  False),
    # Mean reversion monthly on expanded (bearish market hypothesis)
    ("mean_reversion_5_M_expanded", "mean_reversion", 5, "M", "expanded", UNIVERSE_EXPANDED, False),
    ("mean_reversion_3_M_expanded", "mean_reversion", 3, "M", "expanded", UNIVERSE_EXPANDED, False),
    # Quality trend monthly on expanded
    ("quality_trend_5_M_expanded",  "quality_trend",  5, "M", "expanded", UNIVERSE_EXPANDED, False),
    ("quality_trend_3_M_expanded",  "quality_trend",  3, "M", "expanded", UNIVERSE_EXPANDED, False),
    # With require_uptrend = True (defensive filter)
    ("momentum_5_W_blue_uptrend",      "momentum",       5, "W", "blue", UNIVERSE_BLUE,      True),
    ("mean_reversion_5_W_blue_uptrend","mean_reversion", 5, "W", "blue", UNIVERSE_BLUE,      True),
    ("quality_trend_5_W_blue_uptrend", "quality_trend",  5, "W", "blue", UNIVERSE_BLUE,      True),
]


def main():
    start = "2023-01-01"
    history_start = "2020-01-01"
    tier = "mid"
    fee_bps = COST_TIERS[tier] * 10_000.0 / 2.0

    # Fetch data for the largest universe (expanded = 17 symbols)
    print("=" * 60)
    print("Fetching baostock data for expanded universe (17 symbols)...")
    print("=" * 60)
    prices = fetch_baostock_close(UNIVERSE_EXPANDED, history_start, "2026-08-08")
    if prices.empty:
        print("ERROR: no data fetched", file=sys.stderr)
        return 1
    print(f"\nPrice matrix: {prices.shape[0]} days × {prices.shape[1]} symbols")
    print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    results = []
    best_sharpe = -999
    best_label = ""

    for label, mode, top_n, rebal, ukey, universe, uptrend in GRID:
        print(f"\n{'─' * 50}")
        print(f"Running: {label} ({mode}, Top-{top_n}, {rebal})")
        sub_prices = prices[[c for c in universe if c in prices.columns]]
        if sub_prices.shape[1] < top_n:
            print(f"  SKIP: only {sub_prices.shape[1]} symbols, need {top_n}")
            results.append({
                "label": label, "mode": mode, "top_n": top_n,
                "rebalance": rebal, "universe": ukey,
                "error": True, "note": "insufficient symbols",
            })
            continue

        sel_cfg = {"min_price": 3.0, "max_vol_ann": 1.5, "require_uptrend": uptrend}
        perf = run_backtest(
            sub_prices, start, rebal, mode, top_n, fee_bps, sel_cfg=sel_cfg
        )
        perf["label"] = label
        perf["mode"] = mode
        perf["top_n"] = top_n
        perf["rebalance"] = rebal
        perf["universe"] = ukey
        perf["require_uptrend"] = uptrend
        perf["n_symbols"] = int(sub_prices.shape[1])

        if not perf.get("error"):
            cagr_pct = perf["cagr"] * 100
            sharpe = perf["sharpe"]
            mdd_pct = perf["max_drawdown"] * 100
            print(f"  CAGR={cagr_pct:+.1f}%  Sharpe={sharpe:.2f}  "
                  f"MaxDD={mdd_pct:.1f}%  Win={perf['win_rate']*100:.0f}%")
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_label = label
        else:
            print(f"  ERROR: {perf.get('note', 'unknown')}")

        results.append(perf)

    # Save results
    out_path = OUT / "ashare_grid_results.json"
    # Convert numpy types for JSON serialization
    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    clean = json.loads(json.dumps(results, default=sanitize))
    out_path.write_text(json.dumps(clean, indent=2), encoding="utf-8")

    # Print summary
    print(f"\n{'=' * 60}")
    print("GRID SEARCH SUMMARY (sorted by Sharpe)")
    print(f"{'=' * 60}")
    valid = [r for r in results if not r.get("error")]
    valid.sort(key=lambda r: r.get("sharpe", -999), reverse=True)
    print(f"{'Label':<38} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'Win%':>5}")
    print("─" * 70)
    for r in valid:
        print(f"{r['label']:<38} {r['cagr']*100:>+6.1f}% {r['sharpe']:>7.2f} "
              f"{r['max_drawdown']*100:>6.1f}% {r['win_rate']*100:>4.0f}%")

    if best_label:
        print(f"\n★ Best by Sharpe: {best_label} (Sharpe={best_sharpe:.2f})")

    print(f"\nResults saved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

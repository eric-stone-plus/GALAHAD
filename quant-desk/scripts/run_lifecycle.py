#!/usr/bin/env python3
"""历史生命周期：按再平衡频率 选股→调仓→盯市，最后评价+复盘 HTML。

  quant-python scripts/run_lifecycle.py
  quant-python scripts/run_lifecycle.py --start 2023-01-01 --rebalance M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, OUT, STATE, ensure_layout, load_cfg, resolve_fee_bps  # noqa: E402
from _gates import run_gate_report  # noqa: E402

from quantkit.book import PaperBook
from quantkit.data.panel import fetch_close_panel
from quantkit.portfolio import _rebalance_mask
from quantkit.review import (
    contribution_from_weights,
    equity_performance,
    plot_drawdown_png,
    plot_equity_png,
    write_review_html,
)
from quantkit.selection import apply_filters, score_universe, select_top_n


def main() -> int:
    cfg = load_cfg()
    sel_cfg = cfg.get("selection") or {}
    bf = cfg.get("backfill") or {}
    p = argparse.ArgumentParser(description="quant_desk 历史生命周期")
    p.add_argument("--start", default=bf.get("start", "2023-01-01"))
    p.add_argument("--rebalance", default=bf.get("rebalance", "M"), choices=list("DWMQ"))
    p.add_argument("--top-n", type=int, default=int(sel_cfg.get("top_n", 5)))
    p.add_argument("--mode", default=sel_cfg.get("mode", "composite"))
    p.add_argument("--conformal", action="store_true", help="apply DtACI conformal weight scaling")
    args = p.parse_args()
    ensure_layout()

    universe = list(cfg.get("universe") or [])
    print(f"下载股票池 {len(universe)} 只 …")
    # need history before start for factors
    prices = fetch_close_panel(
        universe,
        market=(cfg.get("trading") or {}).get("market", "us"),
        provider=(cfg.get("trading") or {}).get("provider", "yahoo"),
        start=sel_cfg.get("history_start", "2020-01-01"),
        data_dir=DATA,
    )
    if prices.empty:
        print("ERROR: 价格面板为空", file=sys.stderr)
        return 1

    # trading calendar from start
    cal = prices.loc[args.start:].index
    if len(cal) < 30:
        print("ERROR: 回测区间太短", file=sys.stderr)
        return 1

    rebal = _rebalance_mask(cal, args.rebalance)  # type: ignore[arg-type]
    fee_bps, cost_tier = resolve_fee_bps(cfg)
    book = PaperBook(
        cash=float(cfg.get("initial_cash", 1_000_000)),
        fee_bps=fee_bps,
        name=cfg.get("name", "lifecycle"),
    )
    band = float((cfg.get("trading") or {}).get("rebalance_band", 0.02))

    weight_rows: list[dict] = []
    last_selection = None
    last_selected: list[str] = []
    daily_returns: list[dict] = []  # for conformal
    prev_prices: dict[str, float] = {}  # for computing returns

    for i, dt in enumerate(cal):
        px_row = prices.loc[dt].dropna()
        price_map = {s: float(v) for s, v in px_row.items()}

        if bool(rebal.iloc[i]) or i == 0:
            # score using data up to dt
            hist = prices.loc[:dt]
            scores = score_universe(hist, mode=args.mode, asof=dt)  # type: ignore[arg-type]
            passed, rejected = apply_filters(
                scores,
                min_price=sel_cfg.get("min_price"),
                max_vol_ann=sel_cfg.get("max_vol_ann"),
                require_uptrend=bool(sel_cfg.get("require_uptrend", False)),
            )
            selected, tw = select_top_n(
                passed,
                n=args.top_n,
                weight_scheme=sel_cfg.get("weight_scheme", "equal"),  # type: ignore[arg-type]
                max_weight=float(sel_cfg.get("max_weight", 0.3)),
            )
            last_selection = scores
            last_selected = selected
            # ensure prices for selected
            for s in selected:
                if s not in price_map and s in prices.columns:
                    val = prices.loc[:dt, s].dropna()
                    if not val.empty:
                        price_map[s] = float(val.iloc[-1])
            # Conformal weight scaling: vol-based uncertainty adjustment
            if args.conformal and len(daily_returns) > 30:
                ret_df = pd.DataFrame(daily_returns).set_index("date")
                # Compute rolling vol for each selected symbol
                vol = ret_df.ewm(halflife=20, min_periods=10).std().iloc[-1]
                vol_median = vol.median()
                if vol_median > 0:
                    adj_tw = {}
                    for s, w in tw.items():
                        if s in vol.index and not pd.isna(vol[s]):
                            # Scale down high-vol assets, scale up low-vol
                            vol_ratio = vol[s] / vol_median
                            scale = 1.0 / max(vol_ratio, 0.5)  # floor at 0.5
                            scale = min(scale, 1.5)  # cap at 1.5
                            adj_tw[s] = w * scale
                        else:
                            adj_tw[s] = w
                    # Renormalize
                    total = sum(adj_tw.values())
                    if total > 0:
                        tw = {s: v / total for s, v in adj_tw.items()}

            book.rebalance_to_weights(tw, price_map, ts=str(dt.date()), band=band)

        # daily mark
        # fill missing held prices
        for s in list(book.positions.keys()):
            if s not in price_map and s in prices.columns:
                val = prices.loc[:dt, s].dropna()
                if not val.empty:
                    price_map[s] = float(val.iloc[-1])
        mark = book.mark(price_map, ts=str(dt.date()))

        # Track daily returns for conformal
        ret_row = {"date": dt}
        for s in prices.columns:
            if s in price_map and s in prev_prices and prev_prices[s] > 0:
                ret_row[s] = price_map[s] / prev_prices[s] - 1.0
        daily_returns.append(ret_row)
        prev_prices = dict(price_map)

        w = mark["weights"]
        row = {"date": dt, "equity": mark["equity"], "cash": mark["cash"]}
        for s, wv in w.items():
            row[f"w_{s}"] = wv
        weight_rows.append(row)

        if i % 40 == 0 or i == len(cal) - 1:
            print(f"  {dt.date()} equity={mark['equity']:.0f} pos={mark['n_positions']} selected={last_selected}")

    # persist
    eq_df = book.equity_frame()
    # normalize ts column name to date for evaluate
    if "ts" in eq_df.columns:
        eq_df = eq_df.rename(columns={"ts": "date"})
    eq_df.to_csv(OUT / "lifecycle_equity.csv", index=False)
    book.fills_frame().to_csv(OUT / "fills.csv", index=False)
    book.save(STATE / "paper_book.json")

    # last holdings
    last_px = {s: float(prices[s].dropna().iloc[-1]) for s in prices.columns if not prices[s].dropna().empty}
    holdings = book.holdings_table(last_px)
    holdings.to_csv(OUT / "holdings_latest.csv", index=False)

    # weights panel + contribution
    wdf = pd.DataFrame(weight_rows).set_index("date")
    wcols = [c for c in wdf.columns if c.startswith("w_")]
    weights = wdf[wcols].rename(columns=lambda c: c[2:])
    weights.to_csv(OUT / "lifecycle_weights.csv")
    asset_rets = prices.pct_change().reindex(weights.index).fillna(0.0)
    contrib = contribution_from_weights(weights, asset_rets)
    contrib.to_csv(OUT / "lifecycle_contribution.csv", header=["contribution"])

    if last_selection is not None:
        last_selection.to_csv(OUT / "selection_scores.csv")
        meta = {
            "asof": str(last_selection["asof"].iloc[0]) if "asof" in last_selection else None,
            "selected": last_selected,
            "mode": args.mode,
            "top_n": args.top_n,
            "rebalance": args.rebalance,
            "start": args.start,
        }
        (OUT / "selection_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        tw = pd.Series(
            {s: float(weights.iloc[-1].get(s, 0.0)) for s in last_selected},
            name="target_weight",
        )
        tw.to_csv(OUT / "target_weights.csv", header=["target_weight"])

    # performance
    eq = pd.Series(eq_df["equity"].values, index=pd.to_datetime(eq_df["date"]), name="equity")
    perf = equity_performance(eq)
    (OUT / "performance.json").write_text(
        json.dumps(perf.as_dict(), indent=2, default=str), encoding="utf-8"
    )

    # gates: go/no-go over the metrics this run can measure (fail-closed)
    gate_report = run_gate_report(cfg, OUT)
    print(f"门控 verdict: {gate_report['verdict']} → {OUT / 'gate_report.json'}")

    # charts + HTML
    charts = OUT / "charts"
    eq_png = plot_equity_png(eq, charts / "equity.png", title="Lifecycle portfolio equity")
    dd_png = plot_drawdown_png(eq, charts / "drawdown.png")

    notes = [
        f"回测起点 {args.start}，再平衡 {args.rebalance}，选股 {args.mode} Top-{args.top_n}",
        f"股票池 {len(universe)} 只；成本假设 cost_tier={cost_tier or 'legacy'}（单边 {fee_bps:g} bps）",
        f"期末持仓：{last_selected}",
        f"总收益 {_pct(perf.total_return)} · CAGR {_pct(perf.cagr)} · Sharpe {perf.sharpe:.2f} · MaxDD {_pct(perf.max_drawdown)}",
        f"门控 verdict：{gate_report['verdict']}（缺证据 fail-closed，详见 output/gate_report.json）",
        "纸面撮合：按收盘价调仓，忽略涨跌停/冲击成本以外的简单费率。",
    ]
    html = write_review_html(
        title=f"生命周期复盘 · {cfg.get('name', 'quant_desk')}",
        out_path=OUT / "lifecycle_report.html",
        performance=perf,
        holdings=holdings,
        fills=book.fills_frame(),
        selection=last_selection.reset_index() if last_selection is not None else None,
        contribution=contrib,
        equity_png=eq_png,
        dd_png=dd_png,
        notes=notes,
    )
    # mirror as review_latest
    (OUT / "review_latest.html").write_text(html.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps(perf.as_dict(), indent=2, default=str))
    print(f"\n生命周期报告 → {html}")
    return 0


def _pct(x: float) -> str:
    return f"{x*100:+.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())

"""Performance evaluation and period review (投研复盘)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantkit.backtest import _cagr, _max_drawdown, _sharpe
from quantkit.paths import ensure_dirs
from quantkit.report import metrics_table


@dataclass
class PerformanceSummary:
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    vol_ann: float
    win_rate: float
    n_days: int
    start: str
    end: str
    final_equity: float
    start_equity: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "vol_ann": self.vol_ann,
            "win_rate": self.win_rate,
            "n_days": self.n_days,
            "start": self.start,
            "end": self.end,
            "final_equity": self.final_equity,
            "start_equity": self.start_equity,
        }


def equity_performance(equity: pd.Series, periods_per_year: float = 252.0) -> PerformanceSummary:
    eq = equity.dropna().astype(float)
    if eq.empty:
        return PerformanceSummary(0, 0, 0, 0, 0, 0, 0, "", "", 0, 0)
    rets = eq.pct_change().dropna()
    vol = float(rets.std() * np.sqrt(periods_per_year)) if len(rets) else 0.0
    return PerformanceSummary(
        total_return=float(eq.iloc[-1] / eq.iloc[0] - 1),
        cagr=_cagr(eq, periods_per_year),
        sharpe=_sharpe(rets, periods_per_year),
        max_drawdown=_max_drawdown(eq),
        vol_ann=vol,
        win_rate=float((rets > 0).mean()) if len(rets) else 0.0,
        n_days=len(eq),
        start=str(eq.index[0].date()) if hasattr(eq.index[0], "date") else str(eq.index[0]),
        end=str(eq.index[-1].date()) if hasattr(eq.index[-1], "date") else str(eq.index[-1]),
        final_equity=float(eq.iloc[-1]),
        start_equity=float(eq.iloc[0]),
    )


def contribution_from_weights(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """Approximate period PnL contribution by asset: sum(w_{t-1} * r_t)."""
    w = weights.shift(1).fillna(0.0)
    r = asset_returns.reindex_like(w).fillna(0.0)
    contrib = (w * r).sum(axis=0).sort_values(ascending=False)
    return contrib.rename("contribution")


def trade_stats(fills: pd.DataFrame) -> dict[str, Any]:
    if fills is None or fills.empty:
        return {"n_fills": 0, "buy_notional": 0.0, "sell_notional": 0.0, "fees": 0.0}
    notional = fills["qty"].astype(float) * fills["price"].astype(float)
    buys = fills["side"].str.upper().eq("BUY")
    return {
        "n_fills": int(len(fills)),
        "buy_notional": float(notional[buys].sum()) if buys.any() else 0.0,
        "sell_notional": float(notional[~buys].sum()) if (~buys).any() else 0.0,
        "fees": float(fills["fee"].astype(float).sum()) if "fee" in fills else 0.0,
        "n_symbols": int(fills["symbol"].nunique()),
    }


def plot_equity_png(equity: pd.Series, path: Path, title: str = "Equity") -> Path:
    ensure_dirs(path.parent)
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)
    ax.plot(equity.index, equity.values, color="#2563eb", lw=1.4)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_drawdown_png(equity: pd.Series, path: Path) -> Path:
    ensure_dirs(path.parent)
    dd = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=140)
    ax.fill_between(dd.index, dd.values, 0, color="#ef4444", alpha=0.45)
    ax.set_title("Drawdown")
    ax.set_ylabel("DD")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x) * 100:+.2f}%"


def _num(x: float | None, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):,.{nd}f}"


def write_review_html(
    *,
    title: str,
    out_path: Path,
    performance: PerformanceSummary,
    holdings: pd.DataFrame | None = None,
    fills: pd.DataFrame | None = None,
    selection: pd.DataFrame | None = None,
    contribution: pd.Series | None = None,
    equity_png: Path | None = None,
    dd_png: Path | None = None,
    notes: list[str] | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> Path:
    """Chinese-friendly HTML review report."""
    ensure_dirs(out_path.parent)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    tstats = trade_stats(fills if fills is not None else pd.DataFrame())
    perf = performance.as_dict()

    def df_html(df: pd.DataFrame | None) -> str:
        if df is None or df.empty:
            return "<p class='muted'>暂无数据</p>"
        return df.to_html(index=False, border=0, classes="tbl", escape=True, na_rep="—")

    contrib_html = "<p class='muted'>暂无</p>"
    if contribution is not None and not contribution.empty:
        cdf = contribution.rename("贡献").reset_index()
        cdf.columns = ["标的", "贡献"]
        cdf["贡献"] = cdf["贡献"].map(_pct)
        contrib_html = df_html(cdf)

    eq_img = ""
    if equity_png and equity_png.exists():
        eq_img = f'<img src="data:image/png;base64,{_img_b64(equity_png)}" alt="equity"/>'
    dd_img = ""
    if dd_png and dd_png.exists():
        dd_img = f'<img src="data:image/png;base64,{_img_b64(dd_png)}" alt="drawdown"/>'

    notes = notes or []
    notes_li = "\n".join(f"<li>{n}</li>" for n in notes) or "<li class='muted'>无</li>"

    extra = extra_metrics or {}
    extra_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in extra.items()
    )

    # format fills/holdings lightly
    hold_show = holdings
    if holdings is not None and not holdings.empty:
        hold_show = holdings.copy()
        for c in ("weight",):
            if c in hold_show:
                hold_show[c] = hold_show[c].map(lambda x: _pct(x) if pd.notna(x) else "—")
        for c in ("avg_cost", "price", "market_value", "unrealized_pnl", "qty"):
            if c in hold_show:
                hold_show[c] = hold_show[c].map(lambda x: _num(x, 4 if c == "qty" else 2) if pd.notna(x) else "—")

    fill_show = fills
    if fills is not None and not fills.empty:
        fill_show = fills.tail(50).copy()

    html = f"""<!DOCTYPE html>
<html lang="zh-Hans">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
:root {{
  --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#9aa8b8;
  --border:#2a3648; --good:#22c55e; --bad:#ef4444; --accent:#3b82f6;
}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
background:var(--bg);color:var(--text);line-height:1.55}}
.wrap{{max-width:1100px;margin:0 auto;padding:2rem 1.2rem 4rem}}
h1{{font-size:1.6rem;margin:0 0 .4rem}}
h2{{font-size:1.12rem;margin:1.8rem 0 .7rem;border-bottom:1px solid var(--border);padding-bottom:.35rem}}
.meta{{color:var(--muted);font-size:.92rem}}
.disclaimer{{background:#2a2210;border:1px solid #5c4a1f;color:#f5d98a;padding:.75rem 1rem;border-radius:8px;margin:1rem 0}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.7rem;margin:1rem 0}}
.kpi .item{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.7rem .85rem}}
.kpi .label{{color:var(--muted);font-size:.72rem;text-transform:uppercase}}
.kpi .val{{font-size:1.12rem;font-weight:700;margin-top:.15rem}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem;margin:.6rem 0 1rem;overflow-x:auto}}
table.tbl{{width:100%;border-collapse:collapse;font-size:.9rem}}
table.tbl th,table.tbl td{{padding:.42rem .55rem;border-bottom:1px solid var(--border);text-align:left}}
table.tbl th{{color:var(--muted);font-size:.78rem}}
img{{width:100%;border-radius:10px;border:1px solid var(--border);background:#fff}}
.muted{{color:var(--muted)}}
footer{{margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border);color:var(--muted);font-size:.85rem}}
</style>
</head>
<body>
<div class="wrap">
  <h1>{title}</h1>
  <div class="meta">生成时间：{now} · 区间：{perf['start']} → {perf['end']} · 交易日数：{perf['n_days']}</div>
  <div class="disclaimer">研究 / 纸面交易复盘报告。数据可能延迟。不构成投资建议。</div>

  <div class="kpi">
    <div class="item"><div class="label">总收益率</div><div class="val">{_pct(perf['total_return'])}</div></div>
    <div class="item"><div class="label">CAGR</div><div class="val">{_pct(perf['cagr'])}</div></div>
    <div class="item"><div class="label">Sharpe</div><div class="val">{_num(perf['sharpe'])}</div></div>
    <div class="item"><div class="label">最大回撤</div><div class="val">{_pct(perf['max_drawdown'])}</div></div>
    <div class="item"><div class="label">年化波动</div><div class="val">{_pct(perf['vol_ann'])}</div></div>
    <div class="item"><div class="label">日胜率</div><div class="val">{_pct(perf['win_rate'])}</div></div>
    <div class="item"><div class="label">期末权益</div><div class="val">{_num(perf['final_equity'], 0)}</div></div>
    <div class="item"><div class="label">成交笔数</div><div class="val">{tstats['n_fills']}</div></div>
  </div>

  <h2>1. 收益与风险</h2>
  <div class="card">
    <table class="tbl">
      <tr><th>指标</th><th>数值</th></tr>
      <tr><td>期初权益</td><td>{_num(perf['start_equity'], 2)}</td></tr>
      <tr><td>期末权益</td><td>{_num(perf['final_equity'], 2)}</td></tr>
      <tr><td>总收益</td><td>{_pct(perf['total_return'])}</td></tr>
      <tr><td>CAGR</td><td>{_pct(perf['cagr'])}</td></tr>
      <tr><td>Sharpe</td><td>{_num(perf['sharpe'])}</td></tr>
      <tr><td>最大回撤</td><td>{_pct(perf['max_drawdown'])}</td></tr>
      <tr><td>年化波动</td><td>{_pct(perf['vol_ann'])}</td></tr>
      {extra_rows}
    </table>
  </div>
  <div class="card">{eq_img or "<p class='muted'>无净值图</p>"}</div>
  <div class="card">{dd_img or "<p class='muted'>无回撤图</p>"}</div>

  <h2>2. 持仓（最新）</h2>
  <div class="card">{df_html(hold_show)}</div>

  <h2>3. 成交明细（最近 50 笔）</h2>
  <div class="card">
    <p class="muted">买入名义 {_num(tstats['buy_notional'], 0)} · 卖出名义 {_num(tstats['sell_notional'], 0)} · 费用 {_num(tstats['fees'], 2)}</p>
    {df_html(fill_show)}
  </div>

  <h2>4. 选股截面</h2>
  <div class="card">{df_html(selection)}</div>

  <h2>5. 收益贡献（近似）</h2>
  <div class="card">{contrib_html}</div>

  <h2>6. 复盘要点</h2>
  <div class="card"><ul>{notes_li}</ul></div>

  <footer>quant-analysis · quant_desk · quantkit.review</footer>
</div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def qs_metrics(returns: pd.Series, benchmark: pd.Series | None = None) -> pd.Series:
    """Optional quantstats metrics table."""
    return metrics_table(returns, benchmark)

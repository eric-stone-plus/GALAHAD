"""Performance reporting via quantstats (+ plain matplotlib fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quantkit.paths import ensure_dirs


def equity_to_returns(equity: pd.Series) -> pd.Series:
    r = equity.pct_change().dropna()
    r.name = "strategy"
    return r


def metrics_table(returns: pd.Series, benchmark: pd.Series | None = None) -> pd.Series:
    """Key metrics; uses quantstats when available."""
    try:
        import quantstats as qs

        qs.extend_pandas()
        rows: dict[str, Any] = {
            "CAGR": qs.stats.cagr(returns),
            "Sharpe": qs.stats.sharpe(returns),
            "Max Drawdown": qs.stats.max_drawdown(returns),
            "Volatility": qs.stats.volatility(returns),
            "Sortino": qs.stats.sortino(returns),
            "Calmar": qs.stats.calmar(returns),
            "Win Rate": qs.stats.win_rate(returns),
        }
        if benchmark is not None:
            b = benchmark.reindex(returns.index).fillna(0.0)
            rows["Beta"] = qs.stats.beta(returns, b)
            rows["Alpha"] = qs.stats.greeks(returns, b).get("alpha", float("nan")) if hasattr(qs.stats, "greeks") else float("nan")
        return pd.Series(rows, name="metric")
    except Exception:
        # fallback
        ann = 252
        mu = returns.mean() * ann
        vol = returns.std() * (ann ** 0.5)
        equity = (1 + returns).cumprod()
        dd = (equity / equity.cummax() - 1).min()
        sharpe = mu / vol if vol else 0.0
        return pd.Series(
            {
                "CAGR": mu,
                "Sharpe": sharpe,
                "Max Drawdown": dd,
                "Volatility": vol,
                "Win Rate": float((returns > 0).mean()),
            },
            name="metric",
        )


def save_tearsheet(
    returns: pd.Series,
    output_html: Path | str,
    benchmark: pd.Series | None = None,
    title: str = "Strategy Tearsheet",
) -> Path:
    """Write quantstats HTML tearsheet. Falls back to a minimal HTML table."""
    path = Path(output_html)
    ensure_dirs(path.parent)
    try:
        import quantstats as qs

        qs.reports.html(
            returns,
            benchmark=benchmark,
            output=str(path),
            title=title,
            download_filename=path.name,
        )
        return path
    except Exception as exc:
        # minimal fallback
        metrics = metrics_table(returns, benchmark)
        body = metrics.to_frame().to_html()
        path.write_text(
            f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
            f"<body><h1>{title}</h1><p>quantstats full report unavailable: {exc}</p>{body}</body></html>",
            encoding="utf-8",
        )
        return path


def save_metrics_csv(returns: pd.Series, path: Path | str, benchmark: pd.Series | None = None) -> Path:
    p = Path(path)
    ensure_dirs(p.parent)
    metrics_table(returns, benchmark).to_csv(p, header=["value"])
    return p

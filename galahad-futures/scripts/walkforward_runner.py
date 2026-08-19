"""Walk-forward backtest runner: multi-symbol × multi-strategy.

Runs expanding-window walk-forward OOS evaluation on galahad_futures strategies.
Outputs JSON results + comprehensive dark-themed HTML report.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# galahad_futures imports
from galahad_futures.data import load_bars, write_synthetic_fixture
from galahad_futures.engine import load_config, run_paper_on_bars
from galahad_futures.strategy import build_strategy
from galahad_futures.walkforward import bar_walk_forward_splits, oos_bar_slice

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["dual_ma", "tsmom", "tsmom_long", "rsi", "bollinger"]
STRATEGY_LABELS = {
    "dual_ma": "Dual MA (8/21)",
    "tsmom": "TSMOM (48h)",
    "tsmom_long": "TSMOM Long (168h)",
    "rsi": "RSI (14, 70/30)",
    "bollinger": "Bollinger (20, 2σ)",
}
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
N_FOLDS = 4
MIN_TRAIN = 150
WARMUP = 50  # bars of indicator warmup before OOS starts counting


def synthetic_fixture_seed(symbol: str) -> int:
    """Process-stable seed for the synthetic fallback of ``symbol``.

    ``hash()`` is randomized per interpreter (PYTHONHASHSEED), so it must
    not seed anything meant to be reproducible across runs.
    """
    return zlib.crc32(symbol.upper().encode("utf-8")) % 10000


def load_symbol_data(symbol: str) -> tuple[pd.DataFrame, str]:
    """Load bars for a symbol: cache → REST → fixture fallback."""
    cfg = load_config()
    data_cfg = dict(cfg.get("data") or {})
    interval = str(cfg.get("interval", "1h"))
    fetch_limit = int(data_cfg.get("fetch_limit", 500))

    try:
        bars, source_used, note = load_bars(
            source="auto",
            fixture_path=data_cfg.get("fixture_path", "data/fixtures/btcusdt_1h.csv"),
            rest_url=None,
            rest_timeout=float(data_cfg.get("rest_timeout_sec", 12)),
            project_root=PROJECT_ROOT,
            symbol=symbol,
            interval=interval,
            limit=fetch_limit,
            rest_url_template=data_cfg.get("rest_url_template"),
        )
        return bars, source_used
    except Exception as e:
        print(f"  ⚠ Failed to load {symbol}: {e}")
        # Deterministic synthetic fixture as last resort. Written under
        # output/ (runtime scratch): data/fixtures holds committed canonical
        # fixtures and must never be overwritten by ad-hoc synthetic data.
        fix_path = OUTPUT_DIR / f"synthetic_{symbol.lower()}_1h.csv"
        price_map = {"BTCUSDT": 60000, "ETHUSDT": 3500, "SOLUSDT": 150, "BNBUSDT": 600}
        start = price_map.get(symbol, 1000)
        seed = synthetic_fixture_seed(symbol)
        write_synthetic_fixture(fix_path, n=500, start_price=start, seed=seed)
        bars = load_bars(source="fixture", fixture_path=str(fix_path), project_root=PROJECT_ROOT)
        return bars[0], "fixture_synthetic"


def _equity_values(equity_curve: list, initial_equity: float) -> list[float]:
    """Extract float equity values from dict entries or plain floats."""
    values: list[float] = []
    for e in equity_curve:
        if isinstance(e, dict):
            values.append(float(e.get("equity", e.get("eq", initial_equity))))
        else:
            values.append(float(e))
    return values


def chain_oos_equity_curves(
    oos_equity_curves: list[list], initial_equity: float
) -> list[float]:
    """Chain per-fold OOS equity curves into one compounded curve.

    Every fold restarts from ``initial_equity`` after its own warmup, so
    the curves cannot be aggregated by splicing levels: that would inject
    a fake jump at each fold boundary (fold k's final equity vs the next
    fold's reset) and drop earlier folds' compounded P&L from the totals.
    Chaining on per-bar returns keeps each fold's percentage path and
    compounds across folds.
    """
    chained = [float(initial_equity)]
    for curve in oos_equity_curves:
        eq = _equity_values(curve, initial_equity)
        for prev, curr in zip(eq, eq[1:]):
            if prev > 1e-9:
                chained.append(chained[-1] * curr / prev)
            else:
                chained.append(curr)
    return chained


def compute_metrics(equity_curve: list[dict], initial_equity: float) -> dict:
    """Compute performance metrics from equity curve entries."""
    eq_raw = _equity_values(equity_curve, initial_equity)
    if not equity_curve or len(eq_raw) < 2:
        return {
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "n_bars": 0,
            "final_equity": initial_equity,
        }

    eq = np.array(eq_raw, dtype=float)
    n = len(eq)
    final = float(eq[-1])
    total_ret = (final / initial_equity - 1.0) * 100.0

    # CAGR: assume 1h bars → hours per year = 8760
    hours = n
    if hours > 0 and final > 0:
        years = hours / 8760.0
        cagr = ((final / initial_equity) ** (1.0 / max(years, 1e-6)) - 1.0) * 100.0
    else:
        cagr = 0.0

    # Per-bar returns
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
    if len(rets) > 1 and np.std(rets) > 1e-12:
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(8760))
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-9)
    max_dd = float(np.max(dd)) * 100.0

    # Win rate (positive bar returns)
    win_rate = float(np.sum(rets > 0) / max(len(rets), 1) * 100.0)

    return {
        "total_return_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "n_bars": n,
        "final_equity": round(final, 2),
    }


def run_walkforward_single(
    bars: pd.DataFrame,
    symbol: str,
    strategy_name: str,
    n_folds: int = N_FOLDS,
    min_train: int = MIN_TRAIN,
) -> dict:
    """Run walk-forward backtest for one symbol × strategy."""
    cfg = load_config()
    n = len(bars)
    if n < min_train + 20:
        return {"error": f"insufficient bars ({n} < {min_train + 20})"}

    folds_results = []
    oos_equity_curves = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        bar_walk_forward_splits(n, n_folds=n_folds, min_train=min_train, purge=5)
    ):
        # Get bars slice with warmup for indicators
        slice_bars, oos_start = oos_bar_slice(bars, train_idx, test_idx, warmup=WARMUP)

        try:
            result = run_paper_on_bars(
                slice_bars,
                cfg,
                symbol=symbol,
                strategy_name=strategy_name,
                strategy_kwargs={},
                evaluate_from=oos_start,
            )
        except Exception as e:
            folds_results.append({"fold": fold_idx, "error": str(e)})
            continue

        # OOS equity curve
        oos_curve = result["equity_curve"]
        if oos_start < len(oos_curve):
            oos_eq = oos_curve[oos_start:]
        else:
            oos_eq = oos_curve

        metrics = compute_metrics(oos_eq, cfg.get("initial_equity", 10000))
        metrics["fold"] = fold_idx
        metrics["train_bars"] = len(train_idx)
        metrics["test_bars"] = len(test_idx)
        metrics["n_fills"] = result["n_fills"]
        metrics["n_risk_rejects"] = result["n_risk_rejects"]
        metrics["liquidated"] = result["liquidated"]
        metrics["invalidated"] = result.get("invalidated", False)

        folds_results.append(metrics)
        oos_equity_curves.append(oos_eq)

    # Aggregate OOS metrics across folds. Each fold restarts at
    # initial_equity after its own warmup, so chain the per-fold curves on
    # returns — never splice levels (fake boundary jumps, lost compounding).
    all_oos_eq = chain_oos_equity_curves(
        oos_equity_curves, cfg.get("initial_equity", 10000)
    )

    agg = compute_metrics(all_oos_eq, cfg.get("initial_equity", 10000))

    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "strategy_label": STRATEGY_LABELS.get(strategy_name, strategy_name),
        "total_bars": n,
        "n_folds": len(folds_results),
        "folds": folds_results,
        "aggregate": agg,
        "oos_equity_all": all_oos_eq,
    }


def run_all() -> dict:
    """Run full walk-forward matrix: symbols × strategies."""
    results = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "n_folds": N_FOLDS,
        "min_train": MIN_TRAIN,
        "warmup": WARMUP,
        "combinations": [],
    }

    # Load data for all symbols
    data_cache: dict[str, pd.DataFrame] = {}
    data_sources: dict[str, str] = {}
    for sym in SYMBOLS:
        print(f"Loading {sym}...")
        bars, source = load_symbol_data(sym)
        data_cache[sym] = bars
        data_sources[sym] = source
        print(f"  → {len(bars)} bars ({source})")

    results["data_sources"] = data_sources

    # Run matrix
    for sym in SYMBOLS:
        bars = data_cache[sym]
        for strat in STRATEGIES:
            combo_key = f"{strat}×{sym}"
            print(f"\n{'='*60}")
            print(f"Running {combo_key} ({len(bars)} bars)...")
            try:
                res = run_walkforward_single(bars, sym, strat)
                res["combo_key"] = combo_key
                results["combinations"].append(res)
                agg = res.get("aggregate", {})
                status = "✅" if not res.get("error") else "❌"
                print(f"  {status} Return={agg.get('total_return_pct', 'N/A')}% "
                      f"Sharpe={agg.get('sharpe', 'N/A')} "
                      f"MaxDD={agg.get('max_drawdown_pct', 'N/A')}%")
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                traceback.print_exc()
                results["combinations"].append({
                    "combo_key": combo_key,
                    "symbol": sym,
                    "strategy": strat,
                    "error": str(e),
                })

    return results


# ── HTML Report Generator ─────────────────────────────────────────────────

def generate_html_report(results: dict) -> str:
    """Generate a comprehensive dark-themed HTML report."""
    combos = results["combinations"]
    run_id = results["run_id"]

    # Build comparison table rows
    table_rows = ""
    for c in combos:
        agg = c.get("aggregate", {})
        error = c.get("error", "")
        if error:
            table_rows += f"""
            <tr class="error-row">
                <td>{c.get('strategy_label', c.get('strategy', '?'))}</td>
                <td>{c.get('symbol', '?')}</td>
                <td colspan="6" class="error-cell">Error: {error}</td>
            </tr>"""
            continue

        ret = agg.get("total_return_pct", 0)
        cagr = agg.get("cagr_pct", 0)
        sharpe = agg.get("sharpe", 0)
        mdd = agg.get("max_drawdown_pct", 0)
        wr = agg.get("win_rate_pct", 0)
        final = agg.get("final_equity", 0)
        n_folds = c.get("n_folds", 0)

        ret_class = "positive" if ret > 0 else "negative" if ret < 0 else ""
        sharpe_class = "positive" if sharpe > 0.5 else "negative" if sharpe < -0.5 else ""
        mdd_class = "negative" if mdd > 10 else "warning" if mdd > 5 else "positive"

        table_rows += f"""
            <tr>
                <td>{c.get('strategy_label', c.get('strategy', '?'))}</td>
                <td><span class="symbol-badge">{c.get('symbol', '?')}</span></td>
                <td class="{ret_class}">{ret:+.2f}%</td>
                <td class="{ret_class}">{cagr:+.2f}%</td>
                <td class="{sharpe_class}">{sharpe:.3f}</td>
                <td class="{mdd_class}">{mdd:.2f}%</td>
                <td>{wr:.1f}%</td>
                <td>${final:,.0f}</td>
            </tr>"""

    # Build cards for each combination
    cards_html = ""
    for c in combos:
        if c.get("error"):
            cards_html += f"""
            <div class="card error-card">
                <div class="card-header">
                    <h3>{c.get('strategy_label', c.get('strategy', '?'))} — {c.get('symbol', '?')}</h3>
                    <span class="badge badge-error">ERROR</span>
                </div>
                <p class="error-msg">{c['error']}</p>
            </div>"""
            continue

        agg = c.get("aggregate", {})
        folds = c.get("folds", [])
        strategy = c.get("strategy", "?")
        symbol = c.get("symbol", "?")
        label = c.get("strategy_label", strategy)
        data_src = results.get("data_sources", {}).get(symbol, "?")

        # Fold detail table
        fold_rows = ""
        for f in folds:
            if f.get("error"):
                fold_rows += f'<tr><td>Fold {f.get("fold","?")}</td><td colspan="5" class="error-cell">{f["error"]}</td></tr>'
                continue
            fr = f.get("total_return_pct", 0)
            fs = f.get("sharpe", 0)
            fmdd = f.get("max_drawdown_pct", 0)
            fwr = f.get("win_rate_pct", 0)
            fclass = "positive" if fr > 0 else "negative" if fr < 0 else ""
            fold_rows += f"""
                <tr>
                    <td>Fold {f.get('fold', '?')}</td>
                    <td>{f.get('train_bars', 0)}/{f.get('test_bars', 0)}</td>
                    <td class="{fclass}">{fr:+.2f}%</td>
                    <td>{fs:.3f}</td>
                    <td>{fmdd:.2f}%</td>
                    <td>{fwr:.1f}%</td>
                    <td>{'💀' if f.get('liquidated') else '⚠️' if f.get('invalidated') else '✅'}</td>
                </tr>"""

        ret = agg.get("total_return_pct", 0)
        ret_class = "positive" if ret > 0 else "negative"
        sharpe = agg.get("sharpe", 0)
        sharpe_class = "positive" if sharpe > 0.5 else "negative" if sharpe < -0.5 else ""
        mdd = agg.get("max_drawdown_pct", 0)
        mdd_class = "negative" if mdd > 10 else "warning" if mdd > 5 else "positive"

        # Mini equity sparkline data (every Nth point to keep HTML size reasonable)
        oos_eq_raw = c.get("oos_equity_all", [])
        # Extract float values — could be dicts with 'equity' key or plain floats
        oos_eq = []
        for v in oos_eq_raw:
            if isinstance(v, dict):
                oos_eq.append(float(v.get("equity", v.get("eq", 0))))
            else:
                oos_eq.append(float(v))
        if len(oos_eq) > 200:
            step = max(1, len(oos_eq) // 200)
            spark = [oos_eq[i] for i in range(0, len(oos_eq), step)]
        else:
            spark = oos_eq
        spark_json = json.dumps([round(v, 2) for v in spark])

        cards_html += f"""
        <div class="card">
            <div class="card-header">
                <h3>{label} — {symbol}</h3>
                <span class="badge {'badge-green' if ret > 0 else 'badge-red'}">{ret:+.2f}%</span>
            </div>
            <div class="card-meta">Data: {data_src} | Bars: {c.get('total_bars', 0)} | Folds: {c.get('n_folds', 0)}</div>

            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-label">Total Return</span>
                    <span class="metric-value {ret_class}">{ret:+.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">CAGR</span>
                    <span class="metric-value {ret_class}">{agg.get('cagr_pct', 0):+.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Sharpe</span>
                    <span class="metric-value {sharpe_class}">{sharpe:.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Drawdown</span>
                    <span class="metric-value {mdd_class}">{mdd:.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Win Rate</span>
                    <span class="metric-value">{agg.get('win_rate_pct', 0):.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Final Equity</span>
                    <span class="metric-value">${agg.get('final_equity', 0):,.0f}</span>
                </div>
            </div>

            <div class="sparkline-container">
                <h4>OOS Equity Curve (aggregated across folds)</h4>
                <canvas class="sparkline" data-values='{spark_json}' width="700" height="120"></canvas>
            </div>

            <details>
                <summary>Fold Details</summary>
                <table class="fold-table">
                    <thead><tr><th>Fold</th><th>Train/Test</th><th>Return</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th><th>Status</th></tr></thead>
                    <tbody>{fold_rows}</tbody>
                </table>
            </details>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GALAHAD Walk-Forward Backtest Report — {run_id}</title>
<style>
:root {{
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #1c2128;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --blue: #58a6ff;
    --purple: #bc8cff;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    max-width: 1200px;
    margin: 0 auto;
    line-height: 1.5;
}}
h1 {{ color: var(--text); font-size: 1.8em; margin-bottom: 4px; }}
h2 {{ color: var(--text); font-size: 1.3em; margin: 32px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
h3 {{ color: var(--text); font-size: 1.1em; }}
h4 {{ color: var(--text-dim); font-size: 0.9em; margin: 12px 0 6px; }}
.subtitle {{ color: var(--text-dim); font-size: 0.9em; margin-bottom: 24px; }}

/* Comparison table */
.comparison-table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
.comparison-table th, .comparison-table td {{
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    font-size: 0.9em;
}}
.comparison-table th {{ color: var(--text-dim); font-weight: 600; background: var(--surface2); position: sticky; top: 0; }}
.comparison-table tr:hover {{ background: var(--surface2); }}

/* Cards */
.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
}}
.card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
.card-meta {{ color: var(--text-dim); font-size: 0.85em; margin-bottom: 16px; }}
.error-card {{ border-color: var(--red); }}
.error-msg {{ color: var(--red); font-size: 0.9em; }}

/* Metrics grid */
.metrics-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 16px 0; }}
.metric {{ background: var(--surface2); padding: 12px; border-radius: 6px; text-align: center; }}
.metric-label {{ display: block; color: var(--text-dim); font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; }}
.metric-value {{ display: block; font-size: 1.15em; font-weight: 700; margin-top: 4px; }}

/* Sparkline */
.sparkline-container {{ margin: 16px 0; }}
canvas.sparkline {{ background: var(--surface2); border-radius: 4px; width: 100%; height: 120px; }}

/* Badges */
.badge {{ padding: 4px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }}
.badge-green {{ background: rgba(63, 185, 80, 0.15); color: var(--green); }}
.badge-red {{ background: rgba(248, 81, 73, 0.15); color: var(--red); }}
.badge-error {{ background: rgba(248, 81, 73, 0.2); color: var(--red); }}
.symbol-badge {{ background: rgba(88, 166, 255, 0.12); color: var(--blue); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85em; }}

/* Fold table */
.fold-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
.fold-table th, .fold-table td {{ padding: 8px 12px; font-size: 0.85em; border-bottom: 1px solid var(--border); }}
.fold-table th {{ color: var(--text-dim); }}

/* Color classes */
.positive {{ color: var(--green); }}
.negative {{ color: var(--red); }}
.warning {{ color: var(--yellow); }}
.error-cell {{ color: var(--red); font-style: italic; }}
.error-row {{ opacity: 0.7; }}

details {{ margin: 12px 0; }}
summary {{ cursor: pointer; color: var(--blue); font-size: 0.9em; }}
summary:hover {{ text-decoration: underline; }}

.footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border); color: var(--text-dim); font-size: 0.8em; text-align: center; }}

@media (max-width: 768px) {{
    .metrics-grid {{ grid-template-columns: repeat(3, 1fr); }}
    body {{ padding: 12px; }}
}}
</style>
</head>
<body>
<h1>⚔️ GALAHAD Walk-Forward Backtest Report</h1>
<p class="subtitle">Run ID: {run_id} | {len(combos)} combinations | {N_FOLDS}-fold expanding window | Min train: {MIN_TRAIN} bars | Purge: 5 bars</p>

<h2>📊 Comparison Matrix</h2>
<table class="comparison-table">
    <thead>
        <tr>
            <th>Strategy</th><th>Symbol</th><th>Return</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th><th>Final Equity</th>
        </tr>
    </thead>
    <tbody>
        {table_rows}
    </tbody>
</table>

<h2>📋 Detailed Results</h2>
{cards_html}

<div class="footer">
    GALAHAD Futures Walk-Forward Engine | Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br>
    Data: Binance Vision REST → local cache → synthetic fixture | Strategies: pre-specified (no re-fit)
</div>

<script>
// Draw sparklines
document.querySelectorAll('canvas.sparkline').forEach(canvas => {{
    const values = JSON.parse(canvas.dataset.values);
    if (!values || values.length < 2) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const pad = 8;

    // Grid line at initial equity
    const initY = h - pad - ((values[0] - min) / range) * (h - 2 * pad);
    ctx.strokeStyle = 'rgba(139, 148, 158, 0.3)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad, initY);
    ctx.lineTo(w - pad, initY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Equity line
    ctx.strokeStyle = values[values.length-1] >= values[0] ? '#3fb950' : '#f85149';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {{
        const x = pad + (i / (values.length - 1)) * (w - 2 * pad);
        const y = h - pad - ((values[i] - min) / range) * (h - 2 * pad);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }}
    ctx.stroke();

    // Fill area
    const lastX = pad + ((values.length - 1) / (values.length - 1)) * (w - 2 * pad);
    ctx.lineTo(lastX, h - pad);
    ctx.lineTo(pad, h - pad);
    ctx.closePath();
    ctx.fillStyle = values[values.length-1] >= values[0] ? 'rgba(63,185,80,0.08)' : 'rgba(248,81,73,0.08)';
    ctx.fill();
}});
</script>
</body>
</html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("⚔️  GALAHAD Walk-Forward Backtest Runner")
    print("=" * 60)

    results = run_all()

    # Save raw JSON
    json_path = OUTPUT_DIR / "walkforward_results.json"
    # Strip large equity data from JSON to keep it manageable
    results_slim = json.loads(json.dumps(results, default=str))
    for c in results_slim.get("combinations", []):
        c.pop("oos_equity_all", None)
    with json_path.open("w") as f:
        json.dump(results_slim, f, indent=2, ensure_ascii=False)
    print(f"\n✅ JSON results: {json_path}")

    # Generate HTML
    html = generate_html_report(results)
    html_path = OUTPUT_DIR / "walkforward_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"✅ HTML report: {html_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<22} {'Symbol':<10} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'WinRate':>10}")
    print("-" * 80)
    for c in results["combinations"]:
        agg = c.get("aggregate", {})
        if c.get("error"):
            print(f"{c.get('strategy','?'):<22} {c.get('symbol','?'):<10} {'ERROR':>10}")
            continue
        print(f"{c.get('strategy_label','?'):<22} {c.get('symbol','?'):<10} "
              f"{agg.get('total_return_pct',0):>+9.2f}% {agg.get('sharpe',0):>10.3f} "
              f"{agg.get('max_drawdown_pct',0):>9.2f}% {agg.get('win_rate_pct',0):>9.1f}%")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Academic validation report for dual-MA paper variants.

Builds a small (fast,slow) grid of paper equity paths on the same bars,
computes per-variant Sharpe, DSR (honest n_trials), and CSCV PBO via quantkit.

This is the statistical gate — not a claim of edge until PBO/DSR pass policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QUANT = ROOT.parent / "quant"
for p in (str(ROOT), str(QUANT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from galahad_futures.book import FuturesPaperBook
from galahad_futures.data import load_bars, sample_kind_for_source
from galahad_futures.risk import RiskConfig, RiskGate
from galahad_futures.strategy import DualMAConfig, DualMAStrategy

from quantkit.validation import deflated_sharpe_ratio, prob_backtest_overfitting


def _bar_returns_from_equity(eq: list[float]) -> np.ndarray:
    a = np.asarray(eq, dtype=float)
    if len(a) < 3:
        return np.array([])
    r = np.diff(a) / np.maximum(a[:-1], 1e-9)
    return r[np.isfinite(r)]


def run_variant(
    bars: pd.DataFrame,
    *,
    symbol: str,
    fast: int,
    slow: int,
    max_lev: float,
    initial: float,
    fee_bps: float,
    default_leverage: float,
    risk: RiskConfig,
    funding_rate_per_bar: float = 0.0,
) -> dict:
    strat = DualMAStrategy(
        DualMAConfig(fast=fast, slow=slow, max_target_leverage=max_lev)
    )
    targets = strat.targets(bars)
    book = FuturesPaperBook(
        wallet=initial,
        fee_bps=fee_bps,
        default_leverage=default_leverage,
        max_leverage=max(default_leverage, max_lev + 1),
        funding_rate_per_bar=funding_rate_per_bar,
    )
    book.set_leverage(symbol, default_leverage)
    gate = RiskGate(config=risk, day_start_equity=initial)
    for i, row in bars.iterrows():
        ts = str(row["ts"])
        mark = float(row["close"])
        marks = {symbol: mark}
        raw = float(targets.iloc[i])
        pos = book.position(symbol)
        d = gate.filter_target(
            symbol=symbol,
            target_signed_leverage=raw,
            mark=mark,
            equity=max(book.equity(marks), 1e-9),
            current_qty=pos.qty,
            leverage=pos.leverage,
            ts=ts,
        )
        if d.allowed:
            book.apply_target(symbol, d.target_signed_leverage, mark, ts=ts)
        book.mark_to_market(marks, ts=ts)
        if book.liquidated:
            break
    eq = [float(s["equity"]) for s in book.equity_curve]
    rets = _bar_returns_from_equity(eq)
    sharpe = float(rets.mean() / (rets.std(ddof=1) + 1e-12) * np.sqrt(24 * 365)) if len(rets) > 2 else 0.0
    # bar is 1h → ~24*365 periods/year for annualization of simple sharpe
    return {
        "fast": fast,
        "slow": slow,
        "n_fills": len(book.fills),
        "final_equity": eq[-1] if eq else initial,
        "n_bars": len(eq),
        "sharpe_ann_approx": sharpe,
        "returns": rets,
        "liquidated": book.liquidated,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate dual-MA paper grid (DSR/PBO)")
    ap.add_argument(
        "--source",
        choices=("fixture", "auto", "rest", "cache", "venue"),
        default="auto",
        help="prefer auto (rest→cache→fixture); use cache after a successful venue pull",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    symbol = str(cfg.get("symbol", "BTCUSDT"))
    interval = str(cfg.get("interval", "1h"))
    data_cfg = dict(cfg.get("data") or {})
    limit = int(cfg.get("bar_limit", 120))
    fetch_limit = int(data_cfg.get("fetch_limit", max(limit, 500)))
    bars, src, note = load_bars(
        source=args.source,
        fixture_path=ROOT / data_cfg.get("fixture_path", "data/fixtures/btcusdt_1h.csv"),
        rest_url=None,
        rest_timeout=float(data_cfg.get("rest_timeout_sec", 12)),
        project_root=ROOT,
        symbol=symbol,
        interval=interval,
        limit=fetch_limit,
        rest_url_template=data_cfg.get("rest_url_template"),
    )
    if len(bars) > limit:
        bars = bars.iloc[-limit:].reset_index(drop=True)
    sample_kind = sample_kind_for_source(src)
    funding_rate = float(cfg.get("funding_rate_per_bar", 0.0))

    risk_raw = dict(cfg.get("risk") or {})
    risk = RiskConfig(
        max_order_notional=float(risk_raw.get("max_order_notional", 5000)),
        max_position_notional=float(risk_raw.get("max_position_notional", 15000)),
        max_daily_loss=float(risk_raw.get("max_daily_loss", 500)),
        max_leverage=float(cfg.get("max_leverage", 5)),
        mode="paper",
        kill_switch=True,
        enable_live=False,
    )
    initial = float(cfg.get("initial_equity", 10_000))
    fee = float(cfg.get("fee_bps", 4))
    lev = float(cfg.get("default_leverage", 3))
    max_t = float((cfg.get("strategy") or {}).get("max_target_leverage", 2))

    # Small pre-specified grid (honest trial count for DSR)
    grid = [(5, 20), (8, 21), (10, 30), (12, 48), (20, 50)]
    variants = []
    for f, s in grid:
        if f >= s:
            continue
        variants.append(
            run_variant(
                bars,
                symbol=symbol,
                fast=f,
                slow=s,
                max_lev=max_t,
                initial=initial,
                fee_bps=fee,
                default_leverage=lev,
                risk=risk,
                funding_rate_per_bar=funding_rate,
            )
        )

    # Align returns to common length (min)
    min_len = min(len(v["returns"]) for v in variants) if variants else 0
    if min_len < 8:
        report = {
            "status": "insufficient_returns",
            "source_used": src,
            "sample_kind": sample_kind,
            "data_note": note,
            "n_variants": len(variants),
            "min_len": min_len,
        }
    else:
        cols = {}
        for v in variants:
            name = f"ma_{v['fast']}_{v['slow']}"
            cols[name] = v["returns"][-min_len:]
        mat = pd.DataFrame(cols)
        # Pick default config (8,21) as "selected" for DSR; n_trials = grid size
        selected = next((v for v in variants if v["fast"] == 8 and v["slow"] == 21), variants[0])
        rets_sel = selected["returns"][-min_len:]
        # Per-bar Sharpe for each trial → sr_std required by quantkit DSR API
        sr_trials = []
        for v in variants:
            rr = v["returns"][-min_len:]
            if len(rr) < 3:
                continue
            sr_trials.append(float(np.mean(rr) / (np.std(rr, ddof=1) + 1e-12)))
        sr_std = float(np.std(sr_trials, ddof=1)) if len(sr_trials) > 1 else 0.0
        sr_per = float(np.mean(rets_sel) / (np.std(rets_sel, ddof=1) + 1e-12))
        dsr = float(
            deflated_sharpe_ratio(
                rets_sel,
                n_trials=len(variants),
                sr_std=sr_std,
                periods_per_year=1.0,
            )
        )
        # PBO: use fewer blocks if series short
        n_blocks = 8 if min_len >= 32 else 4
        try:
            pbo = float(prob_backtest_overfitting(mat, n_blocks=n_blocks, metric="sharpe"))
        except Exception as e:  # noqa: BLE001
            pbo = None
            pbo_err = f"{type(e).__name__}: {e}"
        else:
            pbo_err = None

        # Policy flags (research — not live)
        flags = []
        if sample_kind == "synthetic_fixture" or src == "fixture":
            flags.append("SYNTHETIC_FIXTURE_NOT_EDGE_CLAIM")
        if sample_kind == "venue":
            flags.append("VENUE_OR_CACHE_SAMPLE")
        if pbo is not None and pbo > 0.5:
            flags.append("PBO_HIGH_RED")
        if dsr <= 0:
            flags.append("DSR_NONPOSITIVE")
        if "PBO_HIGH_RED" not in flags and "DSR_NONPOSITIVE" not in flags:
            flags.append("GATES_WEAK_PASS_ON_THIS_SAMPLE")

        report = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "ok",
            "source_used": src,
            "sample_kind": sample_kind,
            "data_note": note,
            "bars": len(bars),
            "symbol": symbol,
            "funding_rate_per_bar": funding_rate,
            "n_variants": len(variants),
            "grid": [{"fast": v["fast"], "slow": v["slow"], "final_equity": v["final_equity"],
                       "n_fills": v["n_fills"], "sharpe_ann_approx": v["sharpe_ann_approx"],
                       "liquidated": v["liquidated"]} for v in variants],
            "selected": {"fast": selected["fast"], "slow": selected["slow"],
                         "final_equity": selected["final_equity"],
                         "sharpe_ann_approx": selected["sharpe_ann_approx"],
                         "sr_per_bar": sr_per},
            "n_trials_for_dsr": len(variants),
            "deflated_sharpe_ratio": dsr,
            "pbo": pbo,
            "pbo_error": pbo_err,
            "pbo_n_blocks": n_blocks,
            "returns_len": min_len,
            "policy_flags": flags,
            "doctrine": (
                "Dual-MA paper equity is plumbing until DSR>0 and PBO<=0.5 under honest grids "
                "on venue history; synthetic fixture alone is never an edge claim. "
                "see docs/strategy_research.md and public strategy_foundations.md"
            ),
        }

    out = ROOT / "output" / "validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        # strip heavy arrays if any
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("GALAHAD strategy validation")
        print(f"  status:     {report.get('status')}")
        print(f"  source:     {report.get('source_used')}")
        print(f"  variants:   {report.get('n_variants')}")
        if report.get("status") == "ok":
            print(f"  selected:   MA({report['selected']['fast']},{report['selected']['slow']})")
            print(f"  final_eq:   {report['selected']['final_equity']:.4f}")
            print(f"  Sharpe~ann: {report['selected']['sharpe_ann_approx']:.4f}")
            print(f"  DSR:        {report['deflated_sharpe_ratio']:.4f}")
            print(f"  PBO:        {report.get('pbo')}")
            print(f"  flags:      {report.get('policy_flags')}")
        print(f"  report:     {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

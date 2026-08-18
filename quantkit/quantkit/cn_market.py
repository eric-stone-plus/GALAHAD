"""A-share (CN) market microstructure helpers for research backtests.

Covers common mid/low-frequency constraints:
  - price-limit bands (main board 10%, ST 5%, STAR/ChiNext 20% simplified)
  - suspension detection (zero volume / flat price heuristics)
  - tradeable mask (cannot open new buys at limit-up; cannot sell at limit-down optional)
  - adjust-mode notes when using akshare qfq/hfq

These are **research approximations**, not exchange rule engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Board = Literal["main", "st", "chinext_star", "auto"]


def infer_board(symbol: str, name: str | None = None) -> Board:
    """Best-effort board inference from code / name."""
    code = symbol.replace(".SZ", "").replace(".SH", "").replace(".ss", "").replace(".sz", "")
    n = (name or "").upper()
    if "ST" in n or "st" in (name or ""):
        return "st"
    # ChiNext 300xxx, STAR 688xxx
    if code.startswith("300") or code.startswith("301") or code.startswith("688"):
        return "chinext_star"
    return "main"


def limit_pct(board: Board) -> float:
    if board == "st":
        return 0.05
    if board == "chinext_star":
        return 0.20
    return 0.10


def limit_band(
    prev_close: pd.Series,
    board: Board = "main",
    *,
    pct: float | None = None,
) -> pd.DataFrame:
    """Theoretical limit-up / limit-down from previous close."""
    p = pct if pct is not None else limit_pct(board)
    # A-share tick rounding is more complex; use 2-decimal approx for research
    up = (prev_close * (1 + p)).round(2)
    down = (prev_close * (1 - p)).round(2)
    return pd.DataFrame({"limit_up": up, "limit_down": down})


def detect_limit_flags(
    ohlcv: pd.DataFrame,
    board: Board = "main",
    *,
    tol: float = 0.002,
) -> pd.DataFrame:
    """Flag limit-up / limit-down bars using close vs prev_close bands.

    ``tol`` allows for rounding / noise (fraction of price).
    """
    df = ohlcv.copy()
    prev = df["close"].shift(1)
    band = limit_band(prev, board=board)
    # also treat high==limit_up & close near high as limit-up day
    near_up = (df["close"] >= band["limit_up"] * (1 - tol)) | (
        (df["high"] >= band["limit_up"] * (1 - tol))
        & (df["close"] >= band["limit_up"] * (1 - 2 * tol))
    )
    near_down = (df["close"] <= band["limit_down"] * (1 + tol)) | (
        (df["low"] <= band["limit_down"] * (1 + tol))
        & (df["close"] <= band["limit_down"] * (1 + 2 * tol))
    )
    out = pd.DataFrame(
        {
            "prev_close": prev,
            "limit_up": band["limit_up"],
            "limit_down": band["limit_down"],
            "is_limit_up": near_up.fillna(False),
            "is_limit_down": near_down.fillna(False),
        },
        index=df.index,
    )
    return out


def detect_suspension(
    ohlcv: pd.DataFrame,
    *,
    min_volume: float = 1.0,
    flat_price: bool = True,
) -> pd.Series:
    """Heuristic suspension: volume ~ 0, optionally open=high=low=close."""
    vol = ohlcv["volume"].fillna(0.0) if "volume" in ohlcv.columns else pd.Series(0.0, index=ohlcv.index)
    suspended = vol < min_volume
    if flat_price and {"open", "high", "low", "close"}.issubset(ohlcv.columns):
        flat = (
            (ohlcv["open"] == ohlcv["close"])
            & (ohlcv["high"] == ohlcv["low"])
            & (ohlcv["open"] == ohlcv["high"])
        )
        # only count flat+zero vol as suspend; pure flat with volume may be halt-like
        suspended = suspended | (flat & (vol < min_volume))
    return suspended.fillna(False).rename("is_suspended")


@dataclass
class CNTradeRules:
    """How to turn raw bars into a tradeable mask for backtests."""

    board: Board = "main"
    block_buy_limit_up: bool = True
    block_sell_limit_down: bool = True
    block_suspended: bool = True


def tradeable_mask(
    ohlcv: pd.DataFrame,
    rules: CNTradeRules | None = None,
    *,
    side: Literal["buy", "sell", "both"] = "both",
) -> pd.Series:
    """True when a mid-freq strategy may assume a fill is possible.

    For long-only research:
      - cannot *open* on limit-up (often unfillable)
      - cannot *exit* on limit-down (optional)
      - cannot trade when suspended
    """
    rules = rules or CNTradeRules()
    flags = detect_limit_flags(ohlcv, board=rules.board)
    susp = detect_suspension(ohlcv) if rules.block_suspended else pd.Series(False, index=ohlcv.index)

    ok = ~susp
    if side in ("buy", "both") and rules.block_buy_limit_up:
        ok = ok & ~flags["is_limit_up"]
    if side in ("sell", "both") and rules.block_sell_limit_down:
        ok = ok & ~flags["is_limit_down"]
    return ok.fillna(False).rename("tradeable")


def apply_cn_constraints_to_signal(
    signal: pd.Series,
    ohlcv: pd.DataFrame,
    rules: CNTradeRules | None = None,
) -> pd.Series:
    """Zero-out entries that would require unfillable buys; freeze holds on suspend.

    Logic for long-only signal in {0,1}:
      - if want to go 0→1 but not buy-tradeable → stay 0
      - if want to go 1→0 but not sell-tradeable → stay 1
      - if suspended → keep previous position
    """
    rules = rules or CNTradeRules()
    sig = signal.reindex(ohlcv.index).fillna(0.0).astype(float).clip(0.0, 1.0)
    buy_ok = tradeable_mask(ohlcv, rules, side="buy")
    sell_ok = tradeable_mask(ohlcv, rules, side="sell")
    susp = detect_suspension(ohlcv) if rules.block_suspended else pd.Series(False, index=ohlcv.index)

    pos = []
    prev = 0.0
    for dt in ohlcv.index:
        target = float(sig.loc[dt])
        if bool(susp.loc[dt]):
            pos.append(prev)
            continue
        if target > prev + 1e-12:  # increase / open
            pos.append(target if buy_ok.loc[dt] else prev)
        elif target < prev - 1e-12:  # reduce / close
            pos.append(target if sell_ok.loc[dt] else prev)
        else:
            pos.append(target)
        prev = pos[-1]
    return pd.Series(pos, index=ohlcv.index, name="signal_cn")


def annotate_ohlcv(
    ohlcv: pd.DataFrame,
    symbol: str = "",
    board: Board = "auto",
    name: str | None = None,
) -> pd.DataFrame:
    """Return OHLCV plus limit/suspend/tradeable columns."""
    b: Board = infer_board(symbol, name) if board == "auto" else board
    out = ohlcv.copy()
    flags = detect_limit_flags(out, board=b)
    out = out.join(flags)
    out["is_suspended"] = detect_suspension(out)
    out["tradeable_buy"] = tradeable_mask(out, CNTradeRules(board=b), side="buy")
    out["tradeable_sell"] = tradeable_mask(out, CNTradeRules(board=b), side="sell")
    out["board"] = b
    out["limit_pct"] = limit_pct(b)
    return out


def validate_adjust(
    raw_close: pd.Series,
    adj_close: pd.Series,
    *,
    max_jump: float = 0.25,
) -> pd.DataFrame:
    """Sanity-check adjusted vs raw series for corporate-action spikes.

    Large single-day gaps in adj that are absent in raw often mean bad adjust
    joins; large gaps in both are ordinary moves or limit events.
    """
    r = raw_close.pct_change()
    a = adj_close.pct_change()
    flag = (a.abs() > max_jump) & (r.abs() < max_jump * 0.5)
    return pd.DataFrame(
        {
            "raw_ret": r,
            "adj_ret": a,
            "adj_jump_suspicious": flag.fillna(False),
        }
    )


def cn_fee_bps(
    *,
    commission_bps: float = 2.5,
    stamp_tax_bps_sell: float = 5.0,
    side: Literal["buy", "sell", "roundtrip"] = "roundtrip",
) -> float:
    """Rough A-share cost in bps (commission + sell-side stamp tax)."""
    if side == "buy":
        return commission_bps
    if side == "sell":
        return commission_bps + stamp_tax_bps_sell
    # round-trip: buy commission + sell commission + stamp
    return 2 * commission_bps + stamp_tax_bps_sell

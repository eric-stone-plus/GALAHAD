"""Data layer — per-symbol daily OHLCV bars, offline-first.

Tiers (``source``):

- ``fixture`` — data/fixtures/{symbol}_1d.csv, deterministic synthetic
  (``write_synthetic_fixture`` / scripts/gen_fixture.py); always
  available, always labeled ``synthetic_fixture``, never presented as
  venue data. A missing fixture file is regenerated deterministically.
- ``cache`` — data/cache/{symbol}_1d.csv, written by venue fetches
  (``sample_kind: venue``). Missing → FileNotFoundError (fail closed).
- ``auto`` (default) — cache → fixture.
- ``venue`` — raises: venue data is fetched only by the venue engine
  (``venue_alpaca``), which writes the cache tier. This module stays
  pure-offline.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

OHLCV_COLS = ["ts", "open", "high", "low", "close", "volume"]
INTERVAL = "1d"


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    lower = {c.lower(): c for c in out.columns}
    for want in ("open", "high", "low", "close", "volume"):
        if want not in out.columns and want in lower:
            rename[lower[want]] = want
    if rename:
        out = out.rename(columns=rename)
    if "ts" not in out.columns:
        for cand in ("timestamp", "datetime", "date", "time"):
            if cand in out.columns:
                out = out.rename(columns={cand: "ts"})
                break
            if cand in lower:
                out = out.rename(columns={lower[cand]: "ts"})
                break
    missing = [c for c in OHLCV_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    out = out[OHLCV_COLS].copy()
    out["ts"] = out["ts"].astype(str)
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = out[c].astype(float)
    return out.reset_index(drop=True)


def fixture_path_for(project_root: Path, symbol: str) -> Path:
    return Path(project_root) / "data" / "fixtures" / f"{symbol}_1d.csv"


def cache_path_for(project_root: Path, symbol: str) -> Path:
    return Path(project_root) / "data" / "cache" / f"{symbol}_1d.csv"


def load_fixture(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"fixture not found: {path}")
    return _normalize_bars(pd.read_csv(path))


def write_synthetic_fixture(
    path: str | Path,
    *,
    n: int = 250,
    start_price: float = 100.0,
    seed: int = 42,
    drift_up: float = 0.35,
    drift_down: float = -0.30,
) -> Path:
    """Deterministic synthetic daily OHLCV with a mid-series trend flip.

    Deterministic by construction (integer hash noise, no RNG state), so
    committed fixtures regenerate byte-identically. The trend flip exists
    so the dual-MA null produces both long and flat regimes.
    """
    import math

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    px = float(start_price)
    t0 = pd.Timestamp("2024-01-02", tz="UTC")  # a trading Tuesday
    day = 0
    i = 0
    while i < n:
        ts = t0 + pd.Timedelta(days=day)
        day += 1
        if ts.weekday() >= 5:  # skip weekends — daily equity bars
            continue
        if i < n // 2:
            drift = drift_up + 0.08 * math.sin(i / 3.0)
        else:
            drift = drift_down + 0.06 * math.sin(i / 4.0)
        noise = ((seed * 1103515245 + i * 12345) % 1000) / 1000.0 - 0.5
        o = px
        c = max(1.0, px * (1.0 + (drift + noise * 0.6) / 100.0))
        h = max(o, c) * (1.0 + abs(noise) * 0.004)
        l = min(o, c) * (1.0 - abs(noise) * 0.004)
        vol = 1_000_000.0 + (i % 17) * 30_000.0
        rows.append(
            {
                "ts": ts.strftime("%Y-%m-%d"),
                "open": round(o, 4),
                "high": round(h, 4),
                "low": round(l, 4),
                "close": round(c, 4),
                "volume": vol,
            }
        )
        px = c
        i += 1
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OHLCV_COLS)
        w.writeheader()
        w.writerows(rows)
    return path


def save_cache(path: str | Path, bars: pd.DataFrame) -> Path:
    """Write venue-fetched bars to the CSV cache tier."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_bars(bars).to_csv(path, index=False)
    return path


def load_bars(
    *,
    source: str = "auto",
    project_root: str | Path,
    symbols: list[str] | tuple[str, ...],
    limit: int = 250,
    fixture_seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], str, str | None]:
    """Load per-symbol daily bars. Returns (bars_by_symbol, source_used, note).

    Symbols sharing a tier share its source label; a per-symbol fixture
    fallback under ``auto`` is reported in the note (never silently).
    """
    root = Path(project_root)
    source = str(source or "auto").lower()
    if source == "venue":
        raise RuntimeError(
            "data source 'venue' is fetch-on-venue-engine only: use "
            "--engine alpaca_paper (the venue path fetches latest bars from "
            "the Alpaca data API and write-caches them); this module is "
            "pure-offline by design"
        )
    if source not in ("auto", "fixture", "cache"):
        raise ValueError(f"unknown data source: {source!r} (expected auto | fixture | cache)")

    out: dict[str, pd.DataFrame] = {}
    used: list[str] = []
    notes: list[str] = []
    for idx, symbol in enumerate(symbols):
        cache_path = cache_path_for(root, symbol)
        fix_path = fixture_path_for(root, symbol)
        df: pd.DataFrame | None = None
        if source in ("auto", "cache") and cache_path.is_file():
            df = _normalize_bars(pd.read_csv(cache_path))
            used.append("cache")
        elif source == "cache":
            raise FileNotFoundError(f"cache missing for {symbol}: {cache_path}")
        if df is None:
            if not fix_path.is_file():
                write_synthetic_fixture(
                    fix_path,
                    n=max(limit, 250),
                    start_price=100.0 + 25.0 * idx,
                    seed=fixture_seed + idx,
                )
                notes.append(f"generated deterministic fixture for {symbol}")
            df = load_fixture(fix_path)
            used.append("fixture")
        if len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        out[symbol] = df
    source_used = "fixture" if all(u == "fixture" for u in used) else "cache"
    if len(set(used)) > 1:
        source_used = "mixed"
        notes.append("symbols resolved across tiers: " + ",".join(f"{s}:{u}" for s, u in zip(symbols, used)))
    return out, source_used, ("; ".join(notes) if notes else None)


def sample_kind_for_source(source_used: str) -> str:
    return "synthetic_fixture" if source_used == "fixture" else "venue"

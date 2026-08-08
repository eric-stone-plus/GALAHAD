"""Simple parquet cache under a project data/ directory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantkit.paths import ensure_dirs


def cache_path(data_dir: Path | str, key: str) -> Path:
    """Stable cache file path for a logical key (symbol + timeframe etc.)."""
    safe = (
        key.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("^", "")
    )
    return Path(data_dir) / "cache" / f"{safe}.parquet"


def load_cache(path: Path | str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def save_cache(df: pd.DataFrame, path: Path | str) -> Path:
    p = Path(path)
    ensure_dirs(p.parent)
    df.to_parquet(p)
    return p

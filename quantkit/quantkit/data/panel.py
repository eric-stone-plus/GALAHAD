"""Multi-symbol panel download helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from quantkit.data.ohlcv import fetch_ohlcv


def fetch_close_panel(
    symbols: Iterable[str],
    *,
    market: str = "us",
    provider: str = "auto",
    start: str | None = "2020-01-01",
    end: str | None = None,
    data_dir: Path | str | None = None,
    force_refresh: bool = False,
    field: str = "close",
) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for sym in symbols:
        df = fetch_ohlcv(
            sym,
            market=market,  # type: ignore[arg-type]
            provider=provider,  # type: ignore[arg-type]
            start=start,
            end=end,
            data_dir=data_dir,
            force_refresh=force_refresh,
        )
        if df is not None and not df.empty and field in df.columns:
            cols[sym] = df[field].rename(sym)
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols).sort_index().ffill()
    return panel

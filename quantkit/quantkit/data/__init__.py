"""Multi-market market-data loaders with optional parquet cache."""

from __future__ import annotations

from quantkit.data.cache import cache_path, load_cache, save_cache
from quantkit.data.consensus import (
    ConsensusTolerance,
    OHLCVConsensusError,
    OHLCVConsensusResult,
    OHLCVIdentity,
    OHLCVSecurityIdentity,
    OHLCVSource,
    build_ohlcv_consensus,
)
from quantkit.data.ohlcv import fetch_ohlcv, normalize_ohlcv
from quantkit.data.panel import fetch_close_panel

__all__ = [
    "fetch_ohlcv",
    "normalize_ohlcv",
    "build_ohlcv_consensus",
    "ConsensusTolerance",
    "OHLCVConsensusError",
    "OHLCVConsensusResult",
    "OHLCVIdentity",
    "OHLCVSecurityIdentity",
    "OHLCVSource",
    "fetch_close_panel",
    "cache_path",
    "load_cache",
    "save_cache",
]

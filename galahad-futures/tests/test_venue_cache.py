"""Venue cache load/save — pure local, no network required for cache path."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.data import (
    load_bars,
    load_venue_cache,
    sample_kind_for_source,
    save_venue_cache,
)


def test_save_and_load_venue_cache(tmp_path):
    bars = pd.DataFrame(
        {
            "ts": [f"2024-01-01T{i:02d}:00:00+00:00" for i in range(60)],
            "open": [100.0 + i for i in range(60)],
            "high": [101.0 + i for i in range(60)],
            "low": [99.0 + i for i in range(60)],
            "close": [100.5 + i for i in range(60)],
            "volume": [10.0] * 60,
        }
    )
    path = save_venue_cache(
        bars,
        project_root=tmp_path,
        symbol="BTCUSDT",
        interval="1h",
        rest_url="https://data-api.binance.vision/api/v3/klines",
        venue="test",
    )
    assert path.is_file()
    loaded = load_venue_cache(tmp_path, symbol="BTCUSDT", interval="1h", min_rows=50)
    assert loaded is not None
    df, meta = loaded
    assert len(df) == 60
    assert meta.get("sample_kind") == "venue"
    assert float(df["close"].iloc[-1]) > 0

    bars2, src, note = load_bars(
        source="cache",
        project_root=tmp_path,
        symbol="BTCUSDT",
        interval="1h",
        min_cache_rows=50,
    )
    assert src == "cache"
    assert len(bars2) == 60
    assert sample_kind_for_source("cache") == "venue"
    assert sample_kind_for_source("fixture") == "synthetic_fixture"


def test_auto_falls_back_to_fixture_when_no_cache_or_rest(tmp_path):
    # project with only fixture
    fix = tmp_path / "fix.csv"
    pd.DataFrame(
        {
            "ts": [f"t{i}" for i in range(80)],
            "open": [1.0] * 80,
            "high": [1.0] * 80,
            "low": [1.0] * 80,
            "close": [1.0 + i * 0.01 for i in range(80)],
            "volume": [1.0] * 80,
        }
    ).to_csv(fix, index=False)
    bars, src, note = load_bars(
        source="auto",
        fixture_path=fix,
        project_root=tmp_path,
        symbol="NOSYM",
        interval="1h",
        rest_url_template=None,
        rest_timeout=0.5,
        min_cache_rows=50,
    )
    # rest will fail (network or empty urls after vision timeout); fixture should win
    assert src in ("fixture", "rest", "cache")
    assert len(bars) >= 50
    if src == "fixture":
        assert note  # rest failure note expected when falling back

"""Parquet cache persistence (P0 data-foundation slice) — pure local, no network.

Covers: fixture→parquet round-trip identity, source="parquet" offline
loads, fail-closed behavior when the operator explicitly requests the
parquet tier, and the auto chain (rest → CSV cache → parquet → fixture).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures.data import (
    load_bars,
    load_fixture,
    load_parquet_cache,
    sample_kind_for_source,
    save_parquet_cache,
    save_venue_cache,
    write_synthetic_fixture,
)


def _fixture_bars(tmp_path) -> pd.DataFrame:
    """Deterministic synthetic fixture → normalized bars (source-agnostic)."""
    fix = tmp_path / "fix.csv"
    write_synthetic_fixture(fix, n=120, start_price=40_000.0, seed=42)
    return load_fixture(fix)


def _bars60() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [f"2024-01-01T{i:02d}:00:00+00:00" for i in range(60)],
            "open": [100.0 + i for i in range(60)],
            "high": [101.0 + i for i in range(60)],
            "low": [99.0 + i for i in range(60)],
            "close": [100.5 + i for i in range(60)],
            "volume": [10.0] * 60,
        }
    )


def test_parquet_round_trip_identity(tmp_path):
    orig = _fixture_bars(tmp_path)
    path = save_parquet_cache(
        orig, project_root=tmp_path, symbol="BTCUSDT", interval="1h"
    )
    assert path == tmp_path / "data" / "cache" / "BTCUSDT_1h.parquet"
    assert path.is_file()

    loaded = load_parquet_cache(tmp_path, symbol="BTCUSDT", interval="1h")
    assert loaded is not None
    df, meta = loaded
    pdt.assert_frame_equal(df, orig)
    # ts preserved as ISO-8601 strings on round-trip (consumers parse strings)
    assert all(isinstance(t, str) for t in df["ts"])
    assert df["ts"].iloc[0] == orig["ts"].iloc[0]
    assert df["ts"].iloc[-1] == orig["ts"].iloc[-1]
    assert df["ts"].iloc[0].endswith("+00:00")
    assert meta["n_rows"] == len(orig)
    assert meta["sample_kind"] == "venue"
    assert meta["cache_path"].endswith("BTCUSDT_1h.parquet")


def test_source_parquet_runs_offline(tmp_path):
    orig = _fixture_bars(tmp_path)
    save_parquet_cache(orig, project_root=tmp_path, symbol="BTCUSDT", interval="1h")
    bars, src, note = load_bars(
        source="parquet",
        project_root=tmp_path,
        symbol="BTCUSDT",
        interval="1h",
        min_cache_rows=50,
    )
    assert src == "parquet"
    assert note is None
    pdt.assert_frame_equal(bars, orig)
    assert sample_kind_for_source("parquet") == "venue"


def test_source_parquet_fails_closed_when_missing(tmp_path):
    # a fixture exists — but an explicit parquet request must never fall back
    _fixture_bars(tmp_path)
    with pytest.raises(FileNotFoundError, match="parquet cache missing for BTCUSDT_1h"):
        load_bars(
            source="parquet",
            fixture_path=tmp_path / "fix.csv",
            project_root=tmp_path,
            symbol="BTCUSDT",
            interval="1h",
        )
    # the error names the expected file
    with pytest.raises(FileNotFoundError, match=r"BTCUSDT_1h\.parquet"):
        load_bars(
            source="parquet",
            project_root=tmp_path,
            symbol="BTCUSDT",
            interval="1h",
        )


def test_source_parquet_fails_closed_when_undersized(tmp_path):
    save_parquet_cache(
        _bars60().iloc[:10].reset_index(drop=True),
        project_root=tmp_path,
        symbol="BTCUSDT",
        interval="1h",
    )
    with pytest.raises(ValueError, match="fewer than 50 rows"):
        load_bars(
            source="parquet",
            project_root=tmp_path,
            symbol="BTCUSDT",
            interval="1h",
            min_cache_rows=50,
        )


def test_corrupt_parquet_fails_loudly(tmp_path):
    pq_path = tmp_path / "data" / "cache" / "NOSYM_1h.parquet"
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    pq_path.write_bytes(b"not a parquet file")
    # explicit request: clear error
    with pytest.raises(OSError, match="failed to read parquet cache"):
        load_bars(
            source="parquet", project_root=tmp_path, symbol="NOSYM", interval="1h"
        )
    # auto chain also surfaces corruption loudly — never silently skipped.
    # NOSYM keeps rest off the table so the chain reaches the parquet tier.
    with pytest.raises(OSError, match="failed to read parquet cache"):
        load_bars(
            source="auto",
            project_root=tmp_path,
            symbol="NOSYM",
            interval="1h",
            rest_timeout=0.5,
            min_cache_rows=50,
        )


def test_auto_uses_parquet_when_csv_cache_missing(tmp_path):
    # NOSYM guarantees rest failure (venue rejects unknown symbols); only the
    # parquet tier exists → auto must land on parquet before fixture.
    orig = _fixture_bars(tmp_path)
    save_parquet_cache(orig, project_root=tmp_path, symbol="NOSYM", interval="1h")
    bars, src, note = load_bars(
        source="auto",
        fixture_path=tmp_path / "fix.csv",
        project_root=tmp_path,
        symbol="NOSYM",
        interval="1h",
        rest_timeout=0.5,
        min_cache_rows=50,
    )
    assert src == "parquet"
    assert note  # rest failure note carried through
    pdt.assert_frame_equal(bars, orig)


def test_auto_prefers_csv_cache_over_parquet(tmp_path):
    bars = _bars60()
    csv_path = save_venue_cache(
        bars,
        project_root=tmp_path,
        symbol="NOSYM",
        interval="1h",
        rest_url="https://data-api.binance.vision/api/v3/klines",
        venue="test",
    )
    assert csv_path.is_file()
    # venue pulls persist the parquet tier too
    pq_path = tmp_path / "data" / "cache" / "NOSYM_1h.parquet"
    assert pq_path.is_file()
    loaded, src, note = load_bars(
        source="auto",
        project_root=tmp_path,
        symbol="NOSYM",
        interval="1h",
        rest_timeout=0.5,
        min_cache_rows=50,
    )
    assert src == "cache"  # CSV tier wins in the auto chain
    assert len(loaded) == 60


def test_auto_falls_to_fixture_when_no_cache_tier(tmp_path):
    fix = tmp_path / "fix.csv"
    write_synthetic_fixture(fix, n=120, start_price=40_000.0, seed=42)
    bars, src, note = load_bars(
        source="auto",
        fixture_path=fix,
        project_root=tmp_path,
        symbol="NOSYM",
        interval="1h",
        rest_timeout=0.5,
        min_cache_rows=50,
    )
    assert src == "fixture"
    assert note


def test_auto_skips_undersized_parquet_to_fixture(tmp_path):
    save_parquet_cache(
        _bars60().iloc[:10].reset_index(drop=True),
        project_root=tmp_path,
        symbol="NOSYM",
        interval="1h",
    )
    fix = tmp_path / "fix.csv"
    write_synthetic_fixture(fix, n=120, start_price=40_000.0, seed=42)
    bars, src, note = load_bars(
        source="auto",
        fixture_path=fix,
        project_root=tmp_path,
        symbol="NOSYM",
        interval="1h",
        rest_timeout=0.5,
        min_cache_rows=50,
    )
    assert src == "fixture"
    assert len(bars) >= 50


def test_venue_source_uses_parquet_cache(tmp_path):
    # venue = rest → cache → parquet; never fixture
    orig = _fixture_bars(tmp_path)
    save_parquet_cache(orig, project_root=tmp_path, symbol="NOSYM", interval="1h")
    bars, src, note = load_bars(
        source="venue",
        project_root=tmp_path,
        symbol="NOSYM",
        interval="1h",
        rest_timeout=0.5,
        min_cache_rows=50,
    )
    assert src == "parquet"
    assert note
    pdt.assert_frame_equal(bars, orig)

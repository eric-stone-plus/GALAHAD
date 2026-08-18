"""Bar data adapters: venue REST + durable cache (CSV/parquet) + synthetic fixture.

Workspace data policy:
  - Prefer ``data-api.binance.vision`` REST (spot klines) — often direct-reachable
  - Optional fapi USDT-M template when provided
  - Successful venue pulls cache under ``data/cache/`` for offline re-runs:
    CSV tier ({symbol}_{interval}.csv + meta sidecar) and the P0 parquet
    tier ({symbol}_{interval}.parquet, self-describing single file)
  - ``source="parquet"`` runs fully offline from the parquet cache
  - Fixture is always available but labeled synthetic (never silent as venue)
"""

from __future__ import annotations

import csv
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

OHLCV_COLS = ["ts", "open", "high", "low", "close", "volume"]

# Parquet cache metadata is embedded in the file's schema key-value metadata
# (single-file, self-describing tier — no sidecar needed).
PARQUET_META_KEY = "galahad_futures_meta"

# Workspace-allowed public vision endpoint (spot klines; no API key)
VISION_KLINES = (
    "https://data-api.binance.vision/api/v3/klines"
    "?symbol={symbol}&interval={interval}&limit={limit}"
)
FAPI_KLINES = (
    "https://fapi.binance.com/fapi/v1/klines"
    "?symbol={symbol}&interval={interval}&limit={limit}"
)


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
        for cand in ("timestamp", "datetime", "date", "time", "open_time"):
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
    out = out.reset_index(drop=True)
    return out


def load_fixture(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"fixture not found: {path}")
    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return _normalize_bars(df)


def fetch_rest_klines(*, url: str, timeout: float = 12.0) -> pd.DataFrame:
    """Fetch Binance-style klines JSON array and normalize to OHLCV."""
    req = urllib.request.Request(url, headers={"User-Agent": "galahad-futures/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.load(resp)
    if not isinstance(raw, list) or not raw:
        raise ValueError("empty or invalid klines payload")
    rows = []
    for k in raw:
        open_time_ms = int(k[0])
        ts = pd.Timestamp(open_time_ms, unit="ms", tz="UTC").isoformat()
        rows.append(
            {
                "ts": ts,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )
    return _normalize_bars(pd.DataFrame(rows))


def cache_paths(
    project_root: Path,
    *,
    symbol: str,
    interval: str,
) -> tuple[Path, Path, Path]:
    """(csv, meta sidecar, parquet) cache paths for a symbol/interval."""
    cache_dir = project_root / "data" / "cache"
    stem = f"{symbol}_{interval}"
    return (
        cache_dir / f"{stem}.csv",
        cache_dir / f"{stem}.meta.json",
        cache_dir / f"{stem}.parquet",
    )


def _import_pyarrow() -> tuple[Any, Any]:
    """Lazy optional pyarrow import for the parquet tier.

    pyarrow is a project dependency (added with the P0 parquet slice) but
    stays a lazy import so the rest of the data layer loads without it.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - declared dependency
        raise ImportError(
            "parquet persistence requires pyarrow — install with "
            "`uv pip install --python .venv/bin/python pyarrow`"
        ) from e
    return pa, pq


def save_parquet_cache(
    bars: pd.DataFrame,
    *,
    project_root: Path,
    symbol: str,
    interval: str,
    venue: str = "venue",
    sample_kind: str = "venue",
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Persist normalized bars as a self-describing parquet cache file.

    Any normalized OHLCV frame (rest, cache, or fixture source) can be
    persisted this way; metadata is embedded in the parquet schema so the
    file round-trips alone at ``data/cache/{symbol}_{interval}.parquet``.
    The ``ts`` column is preserved as ISO-8601 strings.
    """
    pa, pq = _import_pyarrow()
    _, _, pq_path = cache_paths(project_root, symbol=symbol, interval=interval)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    bars = _normalize_bars(bars)
    meta: dict[str, Any] = {
        "ts_saved": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": symbol,
        "interval": interval,
        "n_rows": len(bars),
        "venue": venue,
        "first_ts": str(bars["ts"].iloc[0]) if len(bars) else None,
        "last_ts": str(bars["ts"].iloc[-1]) if len(bars) else None,
        "last_close": float(bars["close"].iloc[-1]) if len(bars) else None,
        "sample_kind": sample_kind,
    }
    if extra_meta:
        meta.update(extra_meta)
    table = pa.Table.from_pandas(bars)
    md = dict(table.schema.metadata or {})
    md[PARQUET_META_KEY.encode()] = json.dumps(meta).encode("utf-8")
    table = table.replace_schema_metadata(md)
    pq.write_table(table, pq_path)
    return pq_path


def load_parquet_cache(
    project_root: Path,
    *,
    symbol: str,
    interval: str,
    min_rows: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Load the parquet cache tier; None when the file is absent.

    Fail-closed: an existing-but-unreadable parquet file raises (a corrupt
    cache is surfaced loudly, never silently skipped). Rows below
    ``min_rows`` are treated as no-cache, matching the CSV tier.
    """
    _, _, pq_path = cache_paths(project_root, symbol=symbol, interval=interval)
    if not pq_path.is_file():
        return None
    _pa, pq = _import_pyarrow()
    try:
        schema = pq.read_schema(pq_path)
        bars = _normalize_bars(pq.read_table(pq_path).to_pandas())
    except Exception as e:  # noqa: BLE001
        raise OSError(
            f"failed to read parquet cache {pq_path}: {type(e).__name__}: {e}"
        ) from e
    if len(bars) < min_rows:
        return None
    meta: dict[str, Any] = {}
    raw = (schema.metadata or {}).get(PARQUET_META_KEY.encode())
    if raw:
        try:
            meta = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            meta = {}
    meta.setdefault("sample_kind", "venue")
    meta["cache_path"] = str(pq_path)
    return bars, meta


def save_venue_cache(
    bars: pd.DataFrame,
    *,
    project_root: Path,
    symbol: str,
    interval: str,
    rest_url: str,
    venue: str = "binance_vision",
) -> Path:
    csv_path, meta_path, _pq_path = cache_paths(
        project_root, symbol=symbol, interval=interval
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(csv_path, index=False)
    # P0 parquet tier: every venue pull also lands a self-describing
    # parquet file so offline re-runs can use source="parquet".
    save_parquet_cache(
        bars,
        project_root=project_root,
        symbol=symbol,
        interval=interval,
        venue=venue,
    )
    meta = {
        "ts_saved": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": symbol,
        "interval": interval,
        "n_rows": len(bars),
        "venue": venue,
        "rest_url": rest_url.split("?")[0],
        "first_ts": str(bars["ts"].iloc[0]) if len(bars) else None,
        "last_ts": str(bars["ts"].iloc[-1]) if len(bars) else None,
        "last_close": float(bars["close"].iloc[-1]) if len(bars) else None,
        "sample_kind": "venue",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path


def load_venue_cache(
    project_root: Path,
    *,
    symbol: str,
    interval: str,
    min_rows: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    csv_path, meta_path, _pq_path = cache_paths(
        project_root, symbol=symbol, interval=interval
    )
    if not csv_path.is_file():
        return None
    bars = load_fixture(csv_path)
    if len(bars) < min_rows:
        return None
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    meta.setdefault("sample_kind", "venue")
    meta["cache_path"] = str(csv_path)
    return bars, meta


def build_rest_urls(
    *,
    symbol: str,
    interval: str,
    limit: int,
    rest_url_template: str | None = None,
    prefer_vision: bool = True,
) -> list[tuple[str, str]]:
    """Ordered (venue_label, url) candidates."""
    out: list[tuple[str, str]] = []
    if prefer_vision:
        out.append(
            (
                "binance_vision",
                VISION_KLINES.format(symbol=symbol, interval=interval, limit=limit),
            )
        )
    if rest_url_template:
        out.append(
            (
                "rest_template",
                rest_url_template.format(symbol=symbol, interval=interval, limit=limit),
            )
        )
    # Always offer fapi as last resort for USDT-M semantics
    fapi = FAPI_KLINES.format(symbol=symbol, interval=interval, limit=limit)
    if not any(u[1] == fapi for u in out):
        out.append(("binance_fapi", fapi))
    return out


def fetch_and_cache_venue_bars(
    *,
    project_root: Path,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    rest_url_template: str | None = None,
    rest_timeout: float = 12.0,
    prefer_vision: bool = True,
) -> tuple[pd.DataFrame, str, str | None, Path | None]:
    """Try venue REST URLs in order; on success write cache.

    Returns (bars, source_used, error_note, cache_path_or_None).
    source_used is ``rest`` on success.
    """
    errors: list[str] = []
    for venue, url in build_rest_urls(
        symbol=symbol,
        interval=interval,
        limit=limit,
        rest_url_template=rest_url_template,
        prefer_vision=prefer_vision,
    ):
        try:
            bars = fetch_rest_klines(url=url, timeout=rest_timeout)
            if len(bars) < 10:
                errors.append(f"{venue}:too_few_rows={len(bars)}")
                continue
            path = save_venue_cache(
                bars,
                project_root=project_root,
                symbol=symbol,
                interval=interval,
                rest_url=url,
                venue=venue,
            )
            return bars, "rest", None, path
        except Exception as e:  # noqa: BLE001
            errors.append(f"{venue}:{type(e).__name__}:{e}")
    note = "; ".join(errors) if errors else "no_urls"
    return pd.DataFrame(), "none", note, None


def load_bars(
    *,
    source: str = "auto",
    fixture_path: str | Path | None = None,
    rest_url: str | None = None,
    rest_timeout: float = 12.0,
    project_root: Path | None = None,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    rest_url_template: str | None = None,
    min_cache_rows: int = 50,
) -> tuple[pd.DataFrame, str, str | None]:
    """Return (bars, source_used, error_note).

    source:
      fixture  — synthetic/offline fixture only
      cache    — venue cache only (CSV tier)
      parquet  — parquet cache only; fully offline, fails closed if missing
      rest     — live REST only (may raise if all fail and no fallback requested)
      auto     — rest → cache (CSV) → parquet → fixture
      venue    — rest → cache → parquet (never synthetic fixture; raises if all fail)

    source_used values: rest | cache | parquet | fixture
    """
    note: str | None = None
    root = project_root or Path.cwd()
    fix = Path(fixture_path) if fixture_path else None
    if fix and not fix.is_absolute():
        fix = root / fix

    if source == "fixture":
        if fix is None:
            raise ValueError("fixture_path required for source=fixture")
        return load_fixture(fix), "fixture", None

    if source == "cache":
        cached = load_venue_cache(
            root, symbol=symbol, interval=interval, min_rows=min_cache_rows
        )
        if cached is None:
            raise FileNotFoundError(f"venue cache missing for {symbol}_{interval}")
        bars, meta = cached
        return bars, "cache", None

    if source == "parquet":
        cached = load_parquet_cache(
            root, symbol=symbol, interval=interval, min_rows=min_cache_rows
        )
        if cached is None:
            pq_path = root / "data" / "cache" / f"{symbol}_{interval}.parquet"
            if pq_path.is_file():
                raise ValueError(
                    f"parquet cache for {symbol}_{interval} has fewer than "
                    f"{min_cache_rows} rows (expected ≥ {min_cache_rows})"
                )
            raise FileNotFoundError(
                f"parquet cache missing for {symbol}_{interval} — expected {pq_path}"
            )
        bars, meta = cached
        return bars, "parquet", None

    # Build template list; honor explicit rest_url as highest priority override
    tmpl = rest_url_template
    if rest_url and not tmpl:
        # single URL override — wrap as only template path via special handling
        try:
            bars = fetch_rest_klines(url=rest_url, timeout=rest_timeout)
            path = save_venue_cache(
                bars,
                project_root=root,
                symbol=symbol,
                interval=interval,
                rest_url=rest_url,
                venue="rest_url",
            )
            return bars, "rest", None
        except Exception as e:  # noqa: BLE001
            note = f"rest_failed: {type(e).__name__}: {e}"
            if source == "rest":
                # try cache before raise
                cached = load_venue_cache(
                    root, symbol=symbol, interval=interval, min_rows=min_cache_rows
                )
                if cached:
                    return cached[0], "cache", note
                pq_cached = load_parquet_cache(
                    root, symbol=symbol, interval=interval, min_rows=min_cache_rows
                )
                if pq_cached is not None:
                    return pq_cached[0], "parquet", note
                raise
    else:
        bars, src, err, _path = fetch_and_cache_venue_bars(
            project_root=root,
            symbol=symbol,
            interval=interval,
            limit=limit,
            rest_url_template=tmpl,
            rest_timeout=rest_timeout,
            prefer_vision=True,
        )
        if src == "rest" and len(bars) > 0:
            return bars, "rest", None
        note = err

    # Cache fallback (CSV tier)
    cached = load_venue_cache(
        root, symbol=symbol, interval=interval, min_rows=min_cache_rows
    )
    if cached is not None:
        return cached[0], "cache", note

    # Parquet cache fallback (P0 offline tier)
    pq_cached = load_parquet_cache(
        root, symbol=symbol, interval=interval, min_rows=min_cache_rows
    )
    if pq_cached is not None:
        return pq_cached[0], "parquet", note

    if source == "venue":
        raise FileNotFoundError(
            f"venue bars unavailable (rest+cache failed): {note}"
        )

    # Fixture fallback (auto only)
    if fix is None or not fix.is_file():
        raise FileNotFoundError(
            f"no bars available (source={source}, note={note}, fixture={fix})"
        )
    return load_fixture(fix), "fixture", note


def sample_kind_for_source(source_used: str) -> str:
    if source_used in ("rest", "cache", "parquet"):
        return "venue"
    if source_used == "fixture":
        return "synthetic_fixture"
    return "unknown"


def write_synthetic_fixture(
    path: str | Path,
    *,
    n: int = 120,
    start_price: float = 40_000.0,
    seed: int = 42,
) -> Path:
    """Deterministic synthetic OHLCV with a mid-series trend flip for dual-MA fills."""
    import math

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    px = float(start_price)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(n):
        if i < n // 2:
            drift = 80.0 + 15.0 * math.sin(i / 3.0)
        else:
            drift = -90.0 + 10.0 * math.sin(i / 4.0)
        noise = ((seed * 1103515245 + i * 12345) % 1000) / 1000.0 - 0.5
        o = px
        c = max(1.0, px + drift + noise * 40.0)
        h = max(o, c) + abs(noise) * 20.0
        l = min(o, c) - abs(noise) * 20.0
        vol = 100.0 + (i % 17) * 3.0
        ts = (t0 + pd.Timedelta(hours=i)).isoformat()
        rows.append(
            {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": vol}
        )
        px = c
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OHLCV_COLS)
        w.writeheader()
        w.writerows(rows)
    return path

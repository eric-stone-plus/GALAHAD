"""Market perception — prices + optional attention pointers. No orders.

Layers (decoupled from execution):
  L0 attention: fin-daily report path, firecrawl health (pointers only)
  L3 venue: multi-symbol last prices (numeric bars/targets feed only)

Network failure is recorded as fetch_error; process still returns a snapshot.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Public spot vision ticker (workspace data caliber: vision REST, no keys)
VISION_PRICE = "https://data-api.binance.vision/api/v3/ticker/price?symbol={symbol}"
VISION_MULTI = (
    "https://data-api.binance.vision/api/v3/ticker/price"
    "?symbols=%5B%22BTCUSDT%22%2C%22ETHUSDT%22%2C%22SOLUSDT%22%5D"
)
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class PerceptionSnapshot:
    ts: str
    symbols: dict[str, float] = field(default_factory=dict)
    source: str = "unknown"
    fetch_error: str | None = None
    firecrawl: dict[str, Any] = field(default_factory=dict)
    attention: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def fetch_prices_rest(
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    *,
    timeout: float = 10.0,
) -> tuple[dict[str, float], str | None]:
    """Return ({symbol: price}, error_or_None). Partial success keeps what we got."""
    prices: dict[str, float] = {}
    errors: list[str] = []
    # Try multi endpoint first
    try:
        req = urllib.request.Request(
            VISION_MULTI, headers={"User-Agent": "galahad-perception/0.1"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        if isinstance(data, list):
            for row in data:
                sym = str(row.get("symbol", ""))
                px = float(row["price"])
                if sym and px > 0:
                    prices[sym] = px
            if prices:
                return prices, None
    except Exception as e:  # noqa: BLE001
        errors.append(f"multi:{type(e).__name__}:{e}")

    for sym in symbols:
        if sym in prices:
            continue
        url = VISION_PRICE.format(symbol=sym)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "galahad-perception/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                row = json.load(resp)
            px = float(row["price"])
            if px > 0:
                prices[sym] = px
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sym}:{type(e).__name__}:{e}")

    err = "; ".join(errors) if errors else None
    if not prices and err is None:
        err = "empty_payload"
    return prices, err


def fetch_prices_from_fixture(fixture_path: Path) -> dict[str, float]:
    """Last close from OHLCV fixture as offline price proxy."""
    import pandas as pd

    df = pd.read_csv(fixture_path)
    col = "close" if "close" in df.columns else df.columns[-2]
    last = float(df[col].iloc[-1])
    return {"BTCUSDT": last}


def probe_firecrawl(base: str = "http://127.0.0.1:3002", timeout: float = 3.0) -> dict[str, Any]:
    out: dict[str, Any] = {"url": base, "ok": False}
    try:
        req = urllib.request.Request(base + "/", headers={"User-Agent": "galahad-perception/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(200).decode("utf-8", errors="replace")
            out["ok"] = resp.status == 200 or "Firecrawl" in body or body.startswith("{")
            out["status"] = getattr(resp, "status", 200)
            out["snippet"] = body[:120]
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def find_latest_fin_daily(reports_root: Path) -> dict[str, Any]:
    """Pointer to newest fin-daily HTML if present (attention layer only)."""
    info: dict[str, Any] = {"reports_root": str(reports_root), "path": None}
    if not reports_root.is_dir():
        info["note"] = "reports_root_missing"
        return info
    candidates = sorted(reports_root.glob("**/fin-daily-*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        p = candidates[0]
        info["path"] = str(p)
        info["mtime_utc"] = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    else:
        info["note"] = "no_fin_daily_html"
    return info


def build_snapshot(
    *,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    rest_timeout: float = 10.0,
    fixture_path: Path | None = None,
    force_offline: bool = False,
    firecrawl_url: str = "http://127.0.0.1:3002",
    reports_root: Path | None = None,
) -> PerceptionSnapshot:
    ts = utc_now()
    prices: dict[str, float] = {}
    source = "rest"
    fetch_error: str | None = None

    if not force_offline:
        prices, fetch_error = fetch_prices_rest(symbols, timeout=rest_timeout)
        if prices:
            source = "rest"
        elif fixture_path and fixture_path.is_file():
            prices = fetch_prices_from_fixture(fixture_path)
            source = "fixture"
            fetch_error = fetch_error or "rest_empty_fallback_fixture"
        else:
            source = "none"
            fetch_error = fetch_error or "no_prices"
    else:
        if fixture_path and fixture_path.is_file():
            prices = fetch_prices_from_fixture(fixture_path)
            source = "fixture"
        else:
            source = "none"
            fetch_error = "force_offline_no_fixture"

    fc = probe_firecrawl(firecrawl_url)
    att: dict[str, Any] = {}
    if reports_root is not None:
        att["fin_daily"] = find_latest_fin_daily(reports_root)

    status = "ok" if prices else "degraded"
    if fetch_error and not prices:
        status = "fetch_failed"

    return PerceptionSnapshot(
        ts=ts,
        symbols=prices,
        source=source,
        fetch_error=fetch_error,
        firecrawl=fc,
        attention=att,
        status=status,
    )


def write_snapshot(
    snap: PerceptionSnapshot,
    paths: list[Path],
) -> list[str]:
    """Write JSON snapshot to one or more destinations. Returns written paths."""
    payload = json.dumps(snap.to_dict(), indent=2, ensure_ascii=False)
    written: list[str] = []
    for p in paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload, encoding="utf-8")
        written.append(str(p))
    return written


def run_perception(
    *,
    project_root: Path | None = None,
    ops_state: Path | None = None,
    force_offline: bool = False,
    rest_timeout: float = 10.0,
) -> dict[str, Any]:
    """One-shot perception entry. Always returns summary dict; exit-friendly."""
    root = project_root or Path(__file__).resolve().parent.parent
    fixture = root / "data" / "fixtures" / "btcusdt_1h.csv"
    # Optional attention-reports root: env override, else repo-local reports/
    reports = Path(os.environ.get("GALAHAD_REPORTS_ROOT", root / "reports"))
    snap = build_snapshot(
        fixture_path=fixture if fixture.is_file() else None,
        force_offline=force_offline,
        rest_timeout=rest_timeout,
        reports_root=reports if reports.is_dir() else None,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_paths: list[Path] = [
        root / "output" / "perception_last.json",
        root / "output" / f"perception_{stamp}.json",
    ]
    if ops_state is not None:
        ops_state = Path(ops_state)
        out_paths.append(ops_state / "snapshots" / f"perception_{stamp}.json")
        out_paths.append(ops_state / "last_perception.json")

    written = write_snapshot(snap, out_paths)
    summary = {
        "ts": snap.ts,
        "status": snap.status,
        "source": snap.source,
        "n_symbols": len(snap.symbols),
        "symbols": snap.symbols,
        "fetch_error": snap.fetch_error,
        "firecrawl_ok": bool(snap.firecrawl.get("ok")),
        "written": written,
    }
    return summary

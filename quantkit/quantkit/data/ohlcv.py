"""Unified OHLCV fetch for US / CN / HK / crypto.

Providers
---------
- yahoo   : yfinance  (US equities/indexes, many HK via ``XXXX.HK``)
- akshare : A-share daily / HK via akshare (network dependent)
- ccxt    : crypto OHLCV (default exchange: binance)

All loaders return a normalized DataFrame::

    columns = open, high, low, close, volume
    index   = DatetimeIndex (UTC-naive, sorted, unique)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from quantkit.data.cache import cache_path, load_cache, save_cache

Market = Literal["us", "cn", "hk", "crypto", "auto"]
Provider = Literal["yahoo", "akshare", "ccxt", "auto"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and index for OHLCV bars."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    out = df.copy()
    # flatten MultiIndex columns if present (yfinance multi-ticker edge)
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "timestamp": "date",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    if not isinstance(out.index, pd.DatetimeIndex):
        for col in ("date", "datetime", "Date", "Datetime"):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col])
                out = out.set_index(col)
                break
        else:
            out.index = pd.to_datetime(out.index)

    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()

    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]
    out = out[cols].astype(float, errors="ignore")
    out = out.dropna(subset=["close"], how="any")
    return out


def _resolve_provider(symbol: str, market: Market, provider: Provider) -> Provider:
    if provider != "auto":
        return provider
    if market == "crypto" or "/" in symbol or symbol.upper().endswith("USDT"):
        return "ccxt"
    if market in ("cn", "hk"):
        return "akshare"
    # US + HK yahoo tickers like 0700.HK
    return "yahoo"


def _fetch_yahoo(symbol: str, start: str | None, end: str | None, interval: str) -> pd.DataFrame:
    import yfinance as yf

    kwargs: dict = {"interval": interval, "auto_adjust": True, "progress": False}
    if start:
        kwargs["start"] = start
    if end:
        kwargs["end"] = end
    if not start and not end:
        kwargs["period"] = "2y"
    raw = yf.download(symbol, **kwargs)
    if isinstance(raw.columns, pd.MultiIndex):
        # single-ticker download sometimes returns MultiIndex level0 = field
        try:
            raw.columns = raw.columns.get_level_values(0)
        except Exception:
            pass
    return normalize_ohlcv(raw)


def _fetch_akshare_cn(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    import akshare as ak

    code = symbol.replace(".SZ", "").replace(".SH", "").replace(".ss", "").replace(".sz", "")
    start_s = (start or "20180101").replace("-", "")
    end_s = (end or datetime.now().strftime("%Y-%m-%d")).replace("-", "")
    # Eastmoney daily
    raw = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_s,
        end_date=end_s,
        adjust="qfq",
    )
    return normalize_ohlcv(raw)


def _fetch_akshare_hk(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    import akshare as ak

    # Accept 0700 / 00700 / 0700.HK
    code = symbol.upper().replace(".HK", "").zfill(5)
    raw = ak.stock_hk_hist(
        symbol=code,
        period="daily",
        start_date=(start or "20180101").replace("-", ""),
        end_date=(end or datetime.now().strftime("%Y-%m-%d")).replace("-", ""),
        adjust="qfq",
    )
    return normalize_ohlcv(raw)


def _fetch_ccxt(
    symbol: str,
    start: str | None,
    end: str | None,
    timeframe: str = "1d",
    exchange_id: str = "binance",
) -> pd.DataFrame:
    import ccxt

    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    # Normalize BTCUSDT → BTC/USDT
    pair = symbol.upper()
    if "/" not in pair:
        if pair.endswith("USDT"):
            pair = pair[:-4] + "/USDT"
        elif pair.endswith("USD"):
            pair = pair[:-3] + "/USD"
    since_ms = None
    if start:
        since_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000) if end else None

    rows: list[list] = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        if end_ms and cursor >= end_ms:
            break
        if cursor is None:
            break

    if not rows:
        return normalize_ohlcv(pd.DataFrame())

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df = df.set_index("date")
    if end_ms:
        df = df[df.index <= pd.Timestamp(end_ms, unit="ms")]
    return normalize_ohlcv(df)


def fetch_ohlcv(
    symbol: str,
    *,
    market: Market = "auto",
    provider: Provider = "auto",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    data_dir: Path | str | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    exchange: str = "binance",
) -> pd.DataFrame:
    """Fetch OHLCV with optional on-disk parquet cache.

    Parameters
    ----------
    symbol :
        e.g. ``AAPL``, ``^GSPC``, ``600519``, ``0700.HK``, ``BTC/USDT``
    market :
        ``us`` | ``cn`` | ``hk`` | ``crypto`` | ``auto``
    provider :
        ``yahoo`` | ``akshare`` | ``ccxt`` | ``auto``
    data_dir :
        Project ``data/`` folder; when set, bars are cached under ``data/cache/``.
    """
    prov = _resolve_provider(symbol, market, provider)
    cache_key = f"{prov}_{market}_{symbol}_{interval}_{start or 'na'}_{end or 'na'}"
    cpath = cache_path(data_dir, cache_key) if data_dir else None

    if use_cache and cpath and not force_refresh:
        cached = load_cache(cpath)
        if cached is not None and not cached.empty:
            return normalize_ohlcv(cached)

    if prov == "yahoo":
        df = _fetch_yahoo(symbol, start, end, interval)
    elif prov == "akshare":
        m = market
        if m == "auto":
            m = "hk" if ".HK" in symbol.upper() or (symbol.isdigit() and len(symbol) in (4, 5)) else "cn"
        if m == "hk":
            df = _fetch_akshare_hk(symbol, start, end)
        else:
            df = _fetch_akshare_cn(symbol, start, end)
    elif prov == "ccxt":
        # map common interval names
        tf = {"1d": "1d", "1h": "1h", "4h": "4h", "1m": "1m"}.get(interval, interval)
        df = _fetch_ccxt(symbol, start, end, timeframe=tf, exchange_id=exchange)
    else:
        raise ValueError(f"Unknown provider: {prov}")

    if use_cache and cpath is not None and not df.empty:
        save_cache(df, cpath)
    return df

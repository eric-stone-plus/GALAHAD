"""Technical indicators — thin wrappers around pandas-ta + pure pandas fallbacks."""

from __future__ import annotations

import pandas as pd


def add_returns(df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    import numpy as np

    out = df.copy()
    out["ret_1"] = out[col].pct_change()
    ratio = out[col] / out[col].shift(1)
    out["log_ret_1"] = np.log(ratio.where(ratio > 0))
    return out


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    ef = ema(series, fast)
    es = ema(series, slow)
    line = ef - es
    sig = ema(line, signal)
    hist = line - sig
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": hist})


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window, min_periods=window).std()
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": mid + n_std * std,
            "bb_lower": mid - n_std * std,
        }
    )


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a standard indicator pack used by demos and factor pipelines."""
    out = df.copy()
    c = out["close"]
    out["sma_20"] = sma(c, 20)
    out["sma_50"] = sma(c, 50)
    out["sma_200"] = sma(c, 200)
    out["ema_12"] = ema(c, 12)
    out["ema_26"] = ema(c, 26)
    out["rsi_14"] = rsi(c, 14)
    out = out.join(macd(c))
    out = out.join(bollinger(c))
    out["atr_14"] = atr(out)
    out["ret_1"] = c.pct_change()
    out["ret_5"] = c.pct_change(5)
    out["ret_20"] = c.pct_change(20)
    out["vol_20"] = out["ret_1"].rolling(20).std()
    # optional pandas-ta extras when available
    try:
        import pandas_ta as ta  # noqa: F401

        bb = ta.bbands(c, length=20)
        if bb is not None and not bb.empty:
            # keep pure columns already present; skip overwrite
            pass
    except Exception:
        pass
    return out

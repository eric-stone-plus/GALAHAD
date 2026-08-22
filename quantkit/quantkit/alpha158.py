"""QLib Alpha158 factor set — exact pandas/numpy port.

Factor definitions are ported from microsoft/qlib (MIT license),
``qlib/contrib/data/loader.py`` ``Alpha158DL.get_feature_config`` with
operator semantics taken from ``qlib/data/ops.py`` and
``qlib/data/_libs/rolling.pyx``:

  - rolling operators use ``min_periods=1`` (partial windows at the head);
  - ``Std`` is the pandas sample std (ddof=1);
  - ``Slope``/``Rsquare``/``Resi`` are rolling OLS of the series on integer
    bar positions (newest bar = largest x), NaN points skipped, undefined
    with fewer than 2 valid points; ``Rsquare`` and ``Corr`` are masked to
    NaN on (near-)flat windows (rolling std within ``atol=2e-05`` of 0);
  - ``Rank`` is the trailing percentile of the latest value within the
    window, normalized to (0, 1] (pandas ``rolling.rank(pct=True)``);
  - ``IdxMax``/``IdxMin`` are the 1-based position of the window argmax /
    argmin; ``Greater``/``Less`` are elementwise maximum / minimum;
  - division guards (``+1e-12``) appear exactly where QLib's expressions
    place them.

Attribution: factor definitions (c) Microsoft Corporation, MIT license,
https://github.com/microsoft/qlib (``qlib/contrib/data/loader.py``,
class ``Alpha158DL``).

Documented deviation: VWAP is approximated as ``(high + low + close) / 3``
because the OHLCV bars used by quantkit do not carry a traded ``amount``
(QLib defines ``$vwap = $amount / $volume``).

No lookahead: every factor value at row ``t`` is a function of rows
``<= t`` only (trailing windows and backward shifts exclusively).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

__all__ = ["alpha158", "FEATURE_NAMES_DEFAULT"]

_EPS = 1e-12  # QLib division guard, placed exactly as in the QLib expressions
_FLAT_ATOL = 2e-05  # QLib flat-window mask for Rsquare / Corr (ops.py)

_KBAR_NAMES = [
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
]

# Rolling operators in QLib's definition order; "LOW" produces MIN* columns.
_ROLLING_OPS = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "LOW", "QTLU", "QTLD",
    "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN",
    "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN",
    "VSUMD",
]

_ROLLING_WINDOWS_DEFAULT = [5, 10, 20, 30, 60]

# QLib Alpha158DL default: kbar + price(OPEN/HIGH/LOW/VWAP @ 0) + all rolling
# operators on [5, 10, 20, 30, 60]. Note: no "volume" block by default.
_DEFAULT_CONFIG = {
    "kbar": {},
    "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
    "rolling": {},
}

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _factor_names(config: dict) -> list[str]:
    """Ordered factor names for a config, mirroring Alpha158DL name emission."""
    names: list[str] = []
    if "kbar" in config:
        names += _KBAR_NAMES
    if "price" in config:
        windows = config["price"].get("windows", range(5))
        features = config["price"].get(
            "feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]
        )
        names += [f.upper() + str(d) for f in features for d in windows]
    if "volume" in config:
        names += ["VOLUME" + str(d) for d in config["volume"].get("windows", range(5))]
    if "rolling" in config:
        windows = config["rolling"].get("windows", _ROLLING_WINDOWS_DEFAULT)
        include = config["rolling"].get("include")
        exclude = config["rolling"].get("exclude", [])
        for op in _ROLLING_OPS:
            if op in exclude or (include is not None and op not in include):
                continue
            stem = "MIN" if op == "LOW" else op
            names += [stem + str(d) for d in windows]
    return names


FEATURE_NAMES_DEFAULT = _factor_names(_DEFAULT_CONFIG)


def _rolling_regression(s: pd.Series, n: int) -> pd.DataFrame:
    """Rolling OLS of ``s`` on integer bar positions (QLib Slope/Rsquare/Resi).

    Matches ``qlib.data._libs.rolling``: x is the 1-based position of each
    point inside the trailing window (oldest = 1, newest = window length),
    NaN points are skipped, and results are NaN with fewer than 2 valid
    points or a degenerate design. Returns columns ``slope``, ``rsquare``
    (r-value squared) and ``resi`` (residual of the newest point). Slope,
    r-squared and residuals are invariant to a constant x-shift, so partial
    head windows match QLib's absolute-position convention exactly.
    """
    vals = s.to_numpy(dtype=float)
    n_obs = len(vals)
    slope = np.full(n_obs, np.nan)
    rsquare = np.full(n_obs, np.nan)
    resi = np.full(n_obs, np.nan)
    for t in range(n_obs):
        w = vals[max(0, t - n + 1) : t + 1]
        valid = ~np.isnan(w)
        m = int(valid.sum())
        if m < 2 or np.isnan(w[-1]):
            continue
        x = np.arange(1, len(w) + 1, dtype=float)[valid]
        y = w[valid]
        sx, sy = x.sum(), y.sum()
        sxx, syy, sxy = (x * x).sum(), (y * y).sum(), (x * y).sum()
        den_x = m * sxx - sx * sx
        den_y = m * syy - sy * sy
        num = m * sxy - sx * sy
        if den_x <= 0.0:
            continue
        b = num / den_x
        a = (sy - b * sx) / m
        slope[t] = b
        if den_y > 0.0:
            rsquare[t] = num * num / (den_x * den_y)
        resi[t] = w[-1] - (a + b * len(w))
    return pd.DataFrame(
        {"slope": slope, "rsquare": rsquare, "resi": resi}, index=s.index
    )


def _rolling_corr(left: pd.Series, right: pd.Series, n: int) -> pd.Series:
    """Rolling Pearson correlation with QLib Corr's flat-window mask."""
    res = left.rolling(n, min_periods=1).corr(right)
    flat_left = np.isclose(left.rolling(n, min_periods=1).std(), 0.0, atol=_FLAT_ATOL)
    flat_right = np.isclose(right.rolling(n, min_periods=1).std(), 0.0, atol=_FLAT_ATOL)
    return res.mask(flat_left | flat_right)


def _rolling_idx(s: pd.Series, n: int, fn) -> pd.Series:
    """1-based position of the window argextremum (QLib IdxMax/IdxMin)."""
    return s.rolling(n, min_periods=1).apply(lambda w: float(fn(w) + 1), raw=True)


def alpha158(ohlcv: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Compute the QLib Alpha158 factor set from a single asset's OHLCV bars.

    Parameters
    ----------
    ohlcv :
        Columns ``open``, ``high``, ``low``, ``close``, ``volume`` with a
        DatetimeIndex (same convention as
        :func:`quantkit.factors.build_feature_frame`).
    config :
        QLib-style block config; a block is computed iff its key is present:

        - ``"kbar": {}`` — 9 hard-coded candle factors;
        - ``"price": {"windows": [...], "feature": [...]}`` — raw prices at
          ``d`` days ago, divided by the latest close;
        - ``"volume": {"windows": [...]}`` — raw volume at ``d`` days ago,
          divided by ``(latest volume + 1e-12)``;
        - ``"rolling": {"windows": [...], "include": [...], "exclude": [...]}``
          — rolling-operator factors; ``include=None`` selects the default
          operator set.

        ``None`` (default) reproduces QLib's Alpha158DL defaults: kbar +
        OPEN/HIGH/LOW/VWAP at window 0 + all 29 rolling operators on
        [5, 10, 20, 30, 60] — 158 columns.

    Returns
    -------
    DataFrame of factor columns in QLib's emission order; under the default
    config the columns are exactly ``FEATURE_NAMES_DEFAULT``.
    """
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise TypeError("ohlcv must have a DatetimeIndex (quantkit convention)")
    missing = [c for c in _REQUIRED_COLUMNS if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"ohlcv is missing required columns: {missing}")

    cfg = copy.deepcopy(_DEFAULT_CONFIG) if config is None else config

    o = ohlcv["open"].astype(float)
    h = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    c = ohlcv["close"].astype(float)
    v = ohlcv["volume"].astype(float)
    # QLib $vwap = $amount/$volume; approximated here (see module docstring).
    vwap = (h + low + c) / 3.0
    base = {"OPEN": o, "HIGH": h, "LOW": low, "CLOSE": c, "VWAP": vwap}

    cols: dict[str, pd.Series] = {}

    if "kbar" in cfg:
        hl = h - low + _EPS
        cols["KMID"] = (c - o) / o
        cols["KLEN"] = (h - low) / o
        cols["KMID2"] = (c - o) / hl
        cols["KUP"] = (h - np.maximum(o, c)) / o
        cols["KUP2"] = (h - np.maximum(o, c)) / hl
        cols["KLOW"] = (np.minimum(o, c) - low) / o
        cols["KLOW2"] = (np.minimum(o, c) - low) / hl
        cols["KSFT"] = (2 * c - h - low) / o
        cols["KSFT2"] = (2 * c - h - low) / hl

    if "price" in cfg:
        windows = cfg["price"].get("windows", range(5))
        features = cfg["price"].get(
            "feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]
        )
        for field in features:
            s = base[field.upper()]
            for d in windows:
                d = int(d)
                cols[field.upper() + str(d)] = s / c if d == 0 else s.shift(d) / c

    if "volume" in cfg:
        for d in cfg["volume"].get("windows", range(5)):
            d = int(d)
            shifted = v if d == 0 else v.shift(d)
            cols["VOLUME" + str(d)] = shifted / (v + _EPS)

    if "rolling" in cfg:
        windows = [int(d) for d in cfg["rolling"].get("windows", _ROLLING_WINDOWS_DEFAULT)]
        include = cfg["rolling"].get("include")
        exclude = cfg["rolling"].get("exclude", [])

        def use(op: str) -> bool:
            return op not in exclude and (include is None or op in include)

        delta_c = c - c.shift(1)
        delta_v = v - v.shift(1)
        gain_c = delta_c.clip(lower=0)  # QLib Greater(x, 0); NaN propagates
        loss_c = (-delta_c).clip(lower=0)
        gain_v = delta_v.clip(lower=0)
        loss_v = (-delta_v).clip(lower=0)
        up_day = c > c.shift(1)  # NaN comparison -> False, as np.greater
        down_day = c < c.shift(1)
        wvma_x = (c / c.shift(1) - 1).abs() * v
        reg_cache: dict[int, pd.DataFrame] = {}
        imax_cache: dict[int, pd.Series] = {}
        imin_cache: dict[int, pd.Series] = {}

        for op in _ROLLING_OPS:
            if not use(op):
                continue
            for d in windows:
                if op == "ROC":
                    cols[f"ROC{d}"] = c.shift(d) / c
                elif op == "MA":
                    cols[f"MA{d}"] = c.rolling(d, min_periods=1).mean() / c
                elif op == "STD":
                    cols[f"STD{d}"] = c.rolling(d, min_periods=1).std() / c
                elif op in ("BETA", "RSQR", "RESI"):
                    if d not in reg_cache:
                        reg_cache[d] = _rolling_regression(c, d)
                    reg = reg_cache[d]
                    if op == "BETA":
                        cols[f"BETA{d}"] = reg["slope"] / c
                    elif op == "RSQR":
                        flat = np.isclose(
                            c.rolling(d, min_periods=1).std(), 0.0, atol=_FLAT_ATOL
                        )
                        cols[f"RSQR{d}"] = reg["rsquare"].mask(flat)
                    else:
                        cols[f"RESI{d}"] = reg["resi"] / c
                elif op == "MAX":
                    cols[f"MAX{d}"] = h.rolling(d, min_periods=1).max() / c
                elif op == "LOW":
                    cols[f"MIN{d}"] = low.rolling(d, min_periods=1).min() / c
                elif op == "QTLU":
                    cols[f"QTLU{d}"] = c.rolling(d, min_periods=1).quantile(0.8) / c
                elif op == "QTLD":
                    cols[f"QTLD{d}"] = c.rolling(d, min_periods=1).quantile(0.2) / c
                elif op == "RANK":
                    cols[f"RANK{d}"] = c.rolling(d, min_periods=1).rank(pct=True)
                elif op == "RSV":
                    lo_d = low.rolling(d, min_periods=1).min()
                    hi_d = h.rolling(d, min_periods=1).max()
                    cols[f"RSV{d}"] = (c - lo_d) / (hi_d - lo_d + _EPS)
                elif op in ("IMAX", "IMIN", "IMXD"):
                    if d not in imax_cache:
                        imax_cache[d] = _rolling_idx(h, d, np.argmax)
                        imin_cache[d] = _rolling_idx(low, d, np.argmin)
                    if op == "IMAX":
                        cols[f"IMAX{d}"] = imax_cache[d] / d
                    elif op == "IMIN":
                        cols[f"IMIN{d}"] = imin_cache[d] / d
                    else:
                        cols[f"IMXD{d}"] = (imax_cache[d] - imin_cache[d]) / d
                elif op == "CORR":
                    cols[f"CORR{d}"] = _rolling_corr(c, np.log(v + 1), d)
                elif op == "CORD":
                    cols[f"CORD{d}"] = _rolling_corr(
                        c / c.shift(1), np.log(v / v.shift(1) + 1), d
                    )
                elif op in ("CNTP", "CNTN", "CNTD"):
                    cntp = up_day.rolling(d, min_periods=1).mean()
                    cntn = down_day.rolling(d, min_periods=1).mean()
                    if op == "CNTP":
                        cols[f"CNTP{d}"] = cntp
                    elif op == "CNTN":
                        cols[f"CNTN{d}"] = cntn
                    else:
                        cols[f"CNTD{d}"] = cntp - cntn
                elif op in ("SUMP", "SUMN", "SUMD"):
                    abs_sum = delta_c.abs().rolling(d, min_periods=1).sum() + _EPS
                    gain_sum = gain_c.rolling(d, min_periods=1).sum()
                    loss_sum = loss_c.rolling(d, min_periods=1).sum()
                    if op == "SUMP":
                        cols[f"SUMP{d}"] = gain_sum / abs_sum
                    elif op == "SUMN":
                        cols[f"SUMN{d}"] = loss_sum / abs_sum
                    else:
                        cols[f"SUMD{d}"] = (gain_sum - loss_sum) / abs_sum
                elif op == "VMA":
                    cols[f"VMA{d}"] = v.rolling(d, min_periods=1).mean() / (v + _EPS)
                elif op == "VSTD":
                    cols[f"VSTD{d}"] = v.rolling(d, min_periods=1).std() / (v + _EPS)
                elif op == "WVMA":
                    num = wvma_x.rolling(d, min_periods=1).std()
                    den = wvma_x.rolling(d, min_periods=1).mean() + _EPS
                    cols[f"WVMA{d}"] = num / den
                elif op in ("VSUMP", "VSUMN", "VSUMD"):
                    abs_sum = delta_v.abs().rolling(d, min_periods=1).sum() + _EPS
                    gain_sum = gain_v.rolling(d, min_periods=1).sum()
                    loss_sum = loss_v.rolling(d, min_periods=1).sum()
                    if op == "VSUMP":
                        cols[f"VSUMP{d}"] = gain_sum / abs_sum
                    elif op == "VSUMN":
                        cols[f"VSUMN{d}"] = loss_sum / abs_sum
                    else:
                        cols[f"VSUMD{d}"] = (gain_sum - loss_sum) / abs_sum

    out = pd.DataFrame(cols, index=ohlcv.index)
    return out[_factor_names(cfg)]

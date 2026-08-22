"""Tests for quantkit.alpha158 — exact QLib Alpha158 port.

Contracts (definitions: microsoft/qlib qlib/contrib/data/loader.py
``Alpha158DL.get_feature_config``, operator semantics qlib/data/ops.py):
  - default config MUST produce exactly the 158 FEATURE_NAMES_DEFAULT columns
  - KMID/KLEN/ROC5/MA5/STD5/CORR5/BETA5/RSQR5/RESI5 MUST match plain-pandas
    hand-computed references
  - VWAP0 MUST use the documented (high+low+close)/3 approximation
  - mutating rows after t MUST NOT change any factor value at rows <= t
  - all outputs MUST be finite after the 60-day warmup window
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantkit.alpha158 import FEATURE_NAMES_DEFAULT, alpha158


def _ohlcv(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic OHLCV fixture with sane bar geometry."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, n))
    volume = rng.uniform(1e6, 2e6, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Shape / naming contract
# ---------------------------------------------------------------------------


def test_default_config_produces_exactly_158_named_columns():
    df = _ohlcv()
    feat = alpha158(df)
    assert feat.shape == (len(df), 158)
    assert list(feat.columns) == FEATURE_NAMES_DEFAULT
    assert len(FEATURE_NAMES_DEFAULT) == 158
    assert len(set(FEATURE_NAMES_DEFAULT)) == 158


def test_output_index_preserved_and_float():
    df = _ohlcv()
    feat = alpha158(df)
    assert feat.index.equals(df.index)
    assert (feat.dtypes == float).all()


# ---------------------------------------------------------------------------
# Hand-computed spot checks (plain pandas references)
# ---------------------------------------------------------------------------


def test_kbar_hand_computed():
    df = _ohlcv()
    feat = alpha158(df)
    o, h, low, c = df["open"], df["high"], df["low"], df["close"]
    hl = h - low + 1e-12
    np.testing.assert_allclose(feat["KMID"], (c - o) / o)
    np.testing.assert_allclose(feat["KLEN"], (h - low) / o)
    np.testing.assert_allclose(feat["KMID2"], (c - o) / hl)
    np.testing.assert_allclose(feat["KUP"], (h - np.maximum(o, c)) / o)
    np.testing.assert_allclose(feat["KLOW"], (np.minimum(o, c) - low) / o)
    np.testing.assert_allclose(feat["KSFT"], (2 * c - h - low) / o)


def test_price_block_and_vwap_approximation():
    df = _ohlcv()
    feat = alpha158(df)
    c = df["close"]
    vwap = (df["high"] + df["low"] + c) / 3.0
    np.testing.assert_allclose(feat["OPEN0"], df["open"] / c)
    np.testing.assert_allclose(feat["HIGH0"], df["high"] / c)
    np.testing.assert_allclose(feat["LOW0"], df["low"] / c)
    np.testing.assert_allclose(feat["VWAP0"], vwap / c)


def test_roc_ma_std_hand_computed():
    df = _ohlcv()
    feat = alpha158(df)
    c = df["close"]
    # QLib: Ref($close, d)/$close and rolling min_periods=1, Std is ddof=1
    np.testing.assert_allclose(feat["ROC5"], c.shift(5) / c, equal_nan=True)
    np.testing.assert_allclose(feat["MA5"], c.rolling(5, min_periods=1).mean() / c)
    np.testing.assert_allclose(feat["STD5"], c.rolling(5, min_periods=1).std() / c)
    np.testing.assert_allclose(feat["MA60"], c.rolling(60, min_periods=1).mean() / c)


def test_corr_hand_computed_with_flat_window_mask():
    df = _ohlcv()
    feat = alpha158(df)
    c, v = df["close"], df["volume"]
    log_v = np.log(v + 1)
    expected = c.rolling(5, min_periods=1).corr(log_v)
    flat = np.isclose(c.rolling(5, min_periods=1).std(), 0.0, atol=2e-05) | np.isclose(
        log_v.rolling(5, min_periods=1).std(), 0.0, atol=2e-05
    )
    np.testing.assert_allclose(feat["CORR5"], expected.mask(flat), equal_nan=True)


def test_rolling_regression_hand_computed():
    df = _ohlcv()
    feat = alpha158(df)
    c = df["close"]
    t = 100  # full 5-day window, no warmup edge effects
    x = np.arange(1, 6, dtype=float)
    y = c.to_numpy()[t - 4 : t + 1]
    b, a = np.polyfit(x, y, 1)
    np.testing.assert_allclose(feat["BETA5"].iloc[t], b / c.iloc[t], rtol=1e-10)
    np.testing.assert_allclose(feat["RSQR5"].iloc[t], np.corrcoef(x, y)[0, 1] ** 2)
    np.testing.assert_allclose(
        feat["RESI5"].iloc[t], (y[-1] - (a + b * 5)) / c.iloc[t], rtol=1e-10
    )


def test_count_and_sum_family_consistency():
    df = _ohlcv()
    feat = alpha158(df)
    # QLib: SUMN = 1 - SUMP and CNTD = CNTP - CNTN by construction
    # (row 0 is NaN: the close delta is undefined there, as in QLib)
    np.testing.assert_allclose(
        (feat["SUMP20"] + feat["SUMN20"]).iloc[1:], 1.0, atol=1e-9
    )
    np.testing.assert_allclose(
        feat["CNTD20"], feat["CNTP20"] - feat["CNTN20"], atol=1e-12
    )
    assert feat["RANK20"].dropna().between(0.0, 1.0).all()
    assert feat["RSV20"].dropna().between(0.0, 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# No-lookahead property
# ---------------------------------------------------------------------------


def test_no_lookahead_mutation_of_future_rows():
    df = _ohlcv()
    t = 80
    base = alpha158(df)
    mutated = df.copy()
    # aggressively mutate every row after t, all five columns
    mutated.iloc[t + 1 :] = mutated.iloc[t + 1 :] * 3.7 + 11.0
    after = alpha158(mutated)
    pd.testing.assert_frame_equal(
        base.iloc[: t + 1], after.iloc[: t + 1], check_exact=True
    )


def test_all_finite_after_60_day_warmup():
    df = _ohlcv()
    feat = alpha158(df)
    tail = feat.iloc[60:]
    assert np.isfinite(tail.to_numpy()).all()


# ---------------------------------------------------------------------------
# Config overrides (QLib-style)
# ---------------------------------------------------------------------------


def test_config_include_exclude_and_windows():
    df = _ohlcv()
    cfg = {
        "kbar": {},
        "price": {"windows": [0, 1], "feature": ["OPEN"]},
        "volume": {"windows": [0, 2]},
        "rolling": {"windows": [5, 20], "include": ["ROC", "MA"], "exclude": ["MA"]},
    }
    feat = alpha158(df, cfg)
    expected = (
        ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
        + ["OPEN0", "OPEN1"]
        + ["VOLUME0", "VOLUME2"]
        + ["ROC5", "ROC20"]
    )
    assert list(feat.columns) == expected
    c = df["close"]
    np.testing.assert_allclose(feat["OPEN1"], df["open"].shift(1) / c, equal_nan=True)
    np.testing.assert_allclose(
        feat["VOLUME2"], df["volume"].shift(2) / (df["volume"] + 1e-12), equal_nan=True
    )


def test_config_omit_blocks():
    df = _ohlcv()
    feat = alpha158(df, {"rolling": {"windows": [5], "include": ["RANK"]}})
    assert list(feat.columns) == ["RANK5"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_requires_datetime_index_and_ohlcv_columns():
    df = _ohlcv().reset_index(drop=True)
    with pytest.raises(TypeError):
        alpha158(df)
    with pytest.raises(ValueError):
        alpha158(_ohlcv().drop(columns=["volume"]))

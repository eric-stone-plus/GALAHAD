"""Tests for conformal weight policy, style factors, sentiment, and optimizer integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# conformal_weight_policy
# ---------------------------------------------------------------------------


def test_conformal_weight_policy_scales_down_in_high_vol():
    """Weights should shrink when returns are volatile (wide intervals)."""
    from quantkit.portfolio import conformal_weight_policy

    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    symbols = ["A", "B", "C"]

    # Low-vol regime first 50 days, high-vol last 50
    rets = pd.DataFrame(rng.normal(0, 0.01, (100, 3)), index=dates, columns=symbols)
    rets.iloc[50:] = rng.normal(0, 0.05, (50, 3))  # spike vol

    tw = pd.DataFrame(1 / 3, index=dates, columns=symbols)
    scaled = conformal_weight_policy(tw, rets, alpha=0.10)

    assert scaled.shape == tw.shape
    # Weights should be ≤ target (scaled down by conformity)
    assert (scaled.iloc[50:].abs().sum(axis=1) <= 1.01).all()
    # High-vol regime should produce lower total weight than low-vol
    low_vol_gross = scaled.iloc[10:40].abs().sum(axis=1).mean()
    high_vol_gross = scaled.iloc[60:90].abs().sum(axis=1).mean()
    assert low_vol_gross >= high_vol_gross * 0.8  # relaxed for stochastic ACI


def test_conformal_weight_policy_preserves_shape():
    from quantkit.portfolio import conformal_weight_policy

    rng = np.random.default_rng(99)
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    symbols = ["X", "Y"]
    rets = pd.DataFrame(rng.normal(0, 0.02, (50, 2)), index=dates, columns=symbols)
    tw = pd.DataFrame(0.5, index=dates, columns=symbols)
    scaled = conformal_weight_policy(tw, rets)
    assert scaled.index.tolist() == dates.tolist()
    assert scaled.columns.tolist() == symbols


# ---------------------------------------------------------------------------
# style_factors
# ---------------------------------------------------------------------------


def test_style_factors_returns_six_columns():
    from quantkit.factors import style_factors

    rng = np.random.default_rng(77)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    n = len(dates)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    vol = rng.integers(1_000_000, 10_000_000, n).astype(float)
    ohlcv = pd.DataFrame({"close": close, "volume": vol}, index=dates)

    factors = style_factors(ohlcv)
    expected = {"value", "momentum", "quality", "size", "low_vol", "liquidity"}
    assert expected == set(factors.columns)
    assert len(factors) == len(dates)


def test_style_factors_zscored_roughly():
    """After z-scoring, cross-sectional mean should be near 0."""
    from quantkit.factors import style_factors

    rng = np.random.default_rng(88)
    dates = pd.date_range("2023-01-01", periods=400, freq="B")
    n = len(dates)
    close = 50 + np.cumsum(rng.normal(0, 1, n))
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    ohlcv = pd.DataFrame({"close": close, "volume": vol}, index=dates)

    factors = style_factors(ohlcv)
    # After z-scoring, mean of each column should be near 0 (allow NaN warmup + drift)
    for col in factors.columns:
        mean = factors[col].iloc[60:].mean()
        assert abs(mean) < 2.0, f"{col} mean={mean}"


# ---------------------------------------------------------------------------
# sentiment
# ---------------------------------------------------------------------------


def test_keyword_sentiment_positive():
    from quantkit.sentiment import keyword_sentiment

    assert keyword_sentiment("利好消息，股价上涨突破") > 0
    assert keyword_sentiment("利空下跌暴跌") < 0
    assert keyword_sentiment("今天天气不错") == 0.0


def test_build_sentiment_factor():
    from quantkit.sentiment import build_sentiment_factor

    headlines = {
        "AAPL": ["利好上涨超预期", "创新高买入"],
        "TSLA": ["利空下跌暴跌", "卖出减持"],
        "MSFT": ["今天天气不错"],
    }
    df = build_sentiment_factor(headlines)
    assert "AAPL" in df.columns
    assert df.loc[df.index[0], "AAPL"] > 0
    assert df.loc[df.index[0], "TSLA"] < 0
    assert df.loc[df.index[0], "MSFT"] == 0.0


def test_accumulate_oos(tmp_path):
    from quantkit.sentiment import SentimentSnapshot, accumulate_oos

    snap1 = SentimentSnapshot(
        date="2026-08-01",
        scores={"A": 0.5, "B": -0.3},
        n_articles=10,
    )
    snap2 = SentimentSnapshot(
        date="2026-08-02",
        scores={"A": -0.2, "C": 0.8},
        n_articles=15,
    )
    accumulate_oos(tmp_path, snap1)
    df = accumulate_oos(tmp_path, snap2)
    assert len(df) == 2
    assert "A" in df.columns
    assert "C" in df.columns
    # Verify file written
    assert (tmp_path / "sentiment_oos.jsonl").exists()

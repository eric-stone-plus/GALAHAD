"""Tests for quantkit.factor_eval — per-factor IC / RankIC table.

Contracts:
  - a factor identical to the forward return MUST score IC = RankIC = 1.0
  - independent noise MUST score |IC| near 0
  - alignment MUST be pairwise on the index intersection (NaN-safe)
  - the table MUST be sorted by the headline statistic (RankIC when rank=True)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantkit.factor_eval import factor_ic_table


def _frame(n: int = 600, seed: int = 11) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    fwd = pd.Series(rng.normal(0.0, 0.02, n), index=idx, name="fwd_ret")
    factors = pd.DataFrame(
        {
            "perfect": fwd.to_numpy(),
            "anti": -fwd.to_numpy(),
            "noise": rng.normal(0.0, 1.0, n),
        },
        index=idx,
    )
    return factors, fwd


def test_known_answers_perfect_anti_noise():
    factors, fwd = _frame()
    table = factor_ic_table(factors, fwd, n_blocks=10)
    assert table.loc["perfect", "ic_mean"] == pytest.approx(1.0)
    assert table.loc["perfect", "rank_ic_mean"] == pytest.approx(1.0)
    assert table.loc["perfect", "ic_std"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc["anti", "ic_mean"] == pytest.approx(-1.0)
    assert abs(table.loc["noise", "ic_mean"]) < 0.1
    assert abs(table.loc["noise", "rank_ic_mean"]) < 0.1
    # identical per-block ICs -> zero std -> ICIR undefined (NaN), not inf
    assert np.isnan(table.loc["perfect", "icir"])


def test_rank_flag_controls_sorting():
    factors, fwd = _frame()
    table = factor_ic_table(factors, fwd, rank=True)
    key = table["rank_ic_mean"].abs()
    assert (key.to_numpy()[:-1] >= key.to_numpy()[1:] - 1e-12).all()
    assert table.index[0] in ("perfect", "anti")


def test_pairwise_nan_alignment_and_count():
    factors, fwd = _frame(n=200)
    f = factors["perfect"].copy()
    f.iloc[:25] = np.nan  # 25 NaN in the factor only
    y = fwd.copy()
    y.iloc[100:110] = np.nan  # 10 NaN in the label only
    table = factor_ic_table(pd.DataFrame({"f": f}), y, n_blocks=5)
    assert table.loc["f", "n_obs"] == 200 - 25 - 10
    # the remaining sample is still perfectly correlated
    assert table.loc["f", "ic_mean"] == pytest.approx(1.0)


def test_index_intersection_not_positional():
    factors, fwd = _frame(n=120)
    # label covers only the tail; alignment must use the index
    table = factor_ic_table(pd.DataFrame({"f": factors["perfect"]}), fwd.iloc[40:])
    assert table.loc["f", "n_obs"] == 80
    assert table.loc["f", "ic_mean"] == pytest.approx(1.0)


def test_short_or_constant_factor_yields_nan_stats():
    factors, fwd = _frame(n=50)
    const = pd.Series(1.0, index=factors.index)
    short = factors["perfect"].iloc[:4]  # below min_block_obs
    table = factor_ic_table(
        pd.DataFrame({"const": const, "short": short}), fwd, n_blocks=5
    )
    assert np.isnan(table.loc["const", "ic_mean"])
    assert table.loc["short", "n_valid_blocks"] == 0
    assert np.isnan(table.loc["short", "rank_ic_mean"])


def test_invalid_n_blocks_rejected():
    factors, fwd = _frame(n=30)
    with pytest.raises(ValueError):
        factor_ic_table(factors, fwd, n_blocks=0)

"""Bar-based walk-forward splits for hourly venue samples.

Month-based quantkit.walk_forward_splits needs multi-month calendars; 500×1h
bars span ~3 weeks, so we use expanding train + contiguous OOS test blocks
with an optional purge gap. Strategy rules are pre-specified (no re-fit).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def bar_walk_forward_splits(
    n: int,
    *,
    n_folds: int = 4,
    min_train: int = 120,
    test_size: int | None = None,
    purge: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) position arrays.

    Expanding window: train grows; test blocks tile the tail after min_train.
    """
    if n < min_train + 10:
        raise ValueError(f"need n >= min_train+10, got n={n}, min_train={min_train}")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    remainder = n - min_train
    if test_size is None:
        test_size = max(20, remainder // n_folds)
    purge = max(0, int(purge))
    start = min_train
    fold = 0
    while fold < n_folds and start < n:
        test_end = min(n, start + test_size)
        if test_end - start < 5:
            break
        train_end = start - purge if purge > 0 else start
        if train_end < 20:
            start = test_end
            fold += 1
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(start, test_end)
        yield train_idx, test_idx
        start = test_end
        fold += 1


def oos_bar_slice(
    bars: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    warmup: int,
) -> pd.DataFrame:
    """Bars for paper on OOS: include ``warmup`` history before test for indicators.

    Returns a frame whose last len(test_idx) rows are the OOS evaluation region
    when the engine is run with evaluate_from set — simpler approach: return
    bars.iloc[test_start - warmup : test_end] and let caller only score the tail.
    """
    t0 = int(test_idx[0])
    t1 = int(test_idx[-1]) + 1
    w0 = max(0, t0 - warmup)
    return bars.iloc[w0:t1].reset_index(drop=True), t0 - w0  # slice, oos_start_in_slice

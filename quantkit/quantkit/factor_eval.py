"""Per-factor IC / RankIC evaluation against forward returns.

Analysis-paradigm companion to :mod:`quantkit.alpha158`: ranks every factor
of a factor frame by its information coefficient against a forward-return
label. This is deliberately separate from the walk-forward fold machinery in
:mod:`quantkit.factors` / :mod:`quantkit.validation` (model-level OOS
evaluation); here each factor is scored on its own, on the full sample,
split into contiguous time blocks so that IC mean/std/ICIR are block
statistics rather than a single whole-sample correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["factor_ic_table"]


def _block_stats(f: np.ndarray, y: np.ndarray, blocks: list[np.ndarray]) -> tuple[list[float], list[float]]:
    """Per-block Pearson (IC) and Spearman (RankIC) correlations."""
    ics: list[float] = []
    rics: list[float] = []
    for b in blocks:
        fb, yb = f[b], y[b]
        if fb.std() == 0.0 or yb.std() == 0.0:
            continue  # constant block carries no correlation information
        ics.append(float(np.corrcoef(fb, yb)[0, 1]))
        rics.append(float(pd.Series(fb).corr(pd.Series(yb), method="spearman")))
    return ics, rics


def _mean_std_ir(stats: list[float]) -> tuple[float, float, float]:
    if not stats:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(stats))
    std = float(np.std(stats, ddof=1)) if len(stats) > 1 else float("nan")
    ir = mean / std if std and std > 0.0 else float("nan")
    return mean, std, ir


def factor_ic_table(
    factors: pd.DataFrame,
    fwd_ret: pd.Series,
    rank: bool = True,
    n_blocks: int = 10,
    min_block_obs: int = 5,
) -> pd.DataFrame:
    """Per-factor IC / RankIC mean, std, ICIR and observation count.

    For each column of ``factors``, the factor and ``fwd_ret`` are aligned
    on their index intersection and NaNs dropped pairwise. The aligned
    sample is split into ``n_blocks`` contiguous blocks (in index order) and
    the Pearson (IC) and Spearman (RankIC) correlation with the forward
    return is computed per block; blocks shorter than ``min_block_obs`` or
    with a constant side are skipped.

    Parameters
    ----------
    factors : T×K factor frame (e.g. the output of ``quantkit.alpha158``).
    fwd_ret : T-vector of forward returns aligned with ``factors``.
    rank : if True (default) the table is sorted by ``|rank_ic_mean|``
        descending, otherwise by ``|ic_mean|`` — the desk typically screens
        on RankIC, which is robust to factor outliers.
    n_blocks : number of contiguous evaluation blocks.
    min_block_obs : minimum valid observations for a block to contribute.

    Returns
    -------
    DataFrame indexed by factor name with columns ``ic_mean``, ``ic_std``,
    ``icir``, ``rank_ic_mean``, ``rank_ic_std``, ``rank_icir``, ``n_obs``,
    ``n_valid_blocks`` (ICIR = mean/std, NaN when fewer than 2 valid blocks
    or zero std).
    """
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    y = fwd_ret.astype(float)

    rows: dict[str, dict] = {}
    for name in factors.columns:
        aligned = pd.concat(
            [factors[name].astype(float), y], axis=1, keys=["f", "y"]
        ).dropna()
        n_obs = len(aligned)
        blocks = [
            b
            for b in np.array_split(np.arange(n_obs), min(n_blocks, max(n_obs, 1)))
            if len(b) >= min_block_obs
        ]
        ics, rics = _block_stats(
            aligned["f"].to_numpy(), aligned["y"].to_numpy(), blocks
        )
        ic_mean, ic_std, icir = _mean_std_ir(ics)
        ric_mean, ric_std, ricir = _mean_std_ir(rics)
        rows[name] = dict(
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            rank_ic_mean=ric_mean,
            rank_ic_std=ric_std,
            rank_icir=ricir,
            n_obs=n_obs,
            n_valid_blocks=len(ics),
        )

    table = pd.DataFrame.from_dict(rows, orient="index")
    key = "rank_ic_mean" if rank else "ic_mean"
    return table.reindex(table[key].abs().sort_values(ascending=False).index)

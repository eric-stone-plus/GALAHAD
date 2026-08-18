"""News sentiment cross-sectional factor.

Fetches news headlines from RSS feeds, scores sentiment via LLM or keyword
heuristic, and builds a cross-sectional factor for portfolio construction.

Designed for manual/semi-automated OOS accumulation — no cron, no auto-push.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "keyword_sentiment",
    "build_sentiment_factor",
    "SentimentSnapshot",
]


# Keyword heuristic — fast, deterministic, no LLM dependency

_POSITIVE = {
    "利好", "上涨", "突破", "创新高", "超预期", "增长", "盈利", "买入",
    "推荐", "看多", "反弹", "放量", "涨停", "利好消息", "利多",
    "bullish", "upgrade", "beat", "outperform", "buy", "strong",
}
_NEGATIVE = {
    "利空", "下跌", "跌破", "创新低", "不及预期", "下降", "亏损", "卖出",
    "减持", "看空", "暴跌", "缩量", "跌停", "利空消息", "利空",
    "bearish", "downgrade", "miss", "underperform", "sell", "weak",
}


def keyword_sentiment(text: str) -> float:
    """Score sentiment in [-1, 1] via keyword matching.

    Returns 0.0 if no keywords found.  Deterministic, no LLM needed.
    """
    text_lower = text.lower()
    pos = sum(1 for w in _POSITIVE if w in text_lower)
    neg = sum(1 for w in _NEGATIVE if w in text_lower)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


@dataclass
class SentimentSnapshot:
    """One date's cross-sectional sentiment scores."""
    date: str
    scores: dict[str, float]  # symbol → sentiment ∈ [-1, 1]
    n_articles: int
    source: str = "keyword"

    def to_frame(self) -> pd.DataFrame:
        return pd.Series(self.scores, name=self.date).to_frame().T


def build_sentiment_factor(
    headlines: dict[str, list[str]],
    *,
    dates: list[str] | None = None,
) -> pd.DataFrame:
    """Build daily cross-sectional sentiment factor from headline lists.

    Parameters
    ----------
    headlines : {symbol: [headline_strings]} — one list per symbol.
        Each headline is free text (RSS title, news summary, etc.).
    dates : optional date list to align output. If None, use today.

    Returns
    -------
    DataFrame with dates as index, symbols as columns, sentiment ∈ [-1, 1].
    """
    symbols = sorted(headlines.keys())
    if dates is None:
        dates = [datetime.now().strftime("%Y-%m-%d")]

    rows = []
    for d in dates:
        row = {}
        for sym in symbols:
            arts = headlines.get(sym, [])
            if arts:
                scores = [keyword_sentiment(a) for a in arts]
                row[sym] = float(np.mean(scores)) if scores else 0.0
            else:
                row[sym] = 0.0
        rows.append(row)

    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))


def accumulate_oos(
    snapshot_dir: str | Path,
    new_snapshot: SentimentSnapshot,
) -> pd.DataFrame:
    """Append a new sentiment snapshot to the OOS accumulation file.

    Loads existing snapshots from `snapshot_dir/sentiment_oos.jsonl`,
    appends the new one, and returns the full accumulated DataFrame.
    """
    snap_dir = Path(snapshot_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    oos_path = snap_dir / "sentiment_oos.jsonl"

    # Load existing
    records: list[dict[str, Any]] = []
    if oos_path.exists():
        for line in oos_path.read_text().strip().split("\n"):
            if line.strip():
                records.append(json.loads(line))

    # Append new
    records.append({
        "date": new_snapshot.date,
        "scores": new_snapshot.scores,
        "n_articles": new_snapshot.n_articles,
        "source": new_snapshot.source,
    })

    # Write back
    with oos_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Build DataFrame
    dates = [r["date"] for r in records]
    all_syms: set[str] = set()
    for r in records:
        all_syms.update(r["scores"].keys())
    syms = sorted(all_syms)

    data = []
    for r in records:
        data.append({s: r["scores"].get(s, 0.0) for s in syms})

    return pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="date"))

"""Announcement event study — pre-event abnormal return (drift) analysis.

An offline numerical boundary in the spirit of :mod:`quantkit.semiconductor`:
it accepts already-admitted return series and an explicit event list, performs
no acquisition, symbol lookup, calendar inference, or recommendation.

Given per-symbol daily close series and (symbol, event_date) pairs, the module
fits a market model per event on an estimation window and reports cumulative
abnormal returns (CARs) around the event.  The headline diagnostic for
informed-trading screening is the *pre-event* CAR: systematic abnormal drift
in the days **before** a public disclosure is the classic footprint of
information leakage ahead of the announcement.  Pre-event drift is evidence
of association only, never of wrongdoing by any identified party.

Conventions:
- Returns are simple daily close-to-close returns.
- The market series is the equal-weight mean of all supplied series
  (self-contained; callers may pass a benchmark instead).
- Events with insufficient estimation or window data are skipped and
  reported, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Event:
    symbol: str
    date: pd.Timestamp
    label: str = ""


@dataclass
class EventResult:
    event: Event
    car_pre: float | None = None  # cumulative AR over (pre_start, -1]
    car_post: float | None = None  # cumulative AR over [0, post_end]
    beta: float | None = None
    alpha: float | None = None
    n_est: int = 0
    skipped_reason: str | None = None


@dataclass
class StudyReport:
    results: list[EventResult] = field(default_factory=list)

    @property
    def used(self) -> list[EventResult]:
        return [r for r in self.results if r.skipped_reason is None]

    def mean_car(self, window: str = "pre") -> float | None:
        values = [
            r.car_pre if window == "pre" else r.car_post
            for r in self.used
            if (r.car_pre if window == "pre" else r.car_post) is not None
        ]
        return float(np.mean(values)) if values else None

    def t_stat(self, window: str = "pre") -> float | None:
        values = [
            r.car_pre if window == "pre" else r.car_post
            for r in self.used
            if (r.car_pre if window == "pre" else r.car_post) is not None
        ]
        if len(values) < 2:
            return None
        arr = np.asarray(values, dtype=float)
        sd = arr.std(ddof=1)
        return float(arr.mean() / (sd / np.sqrt(len(arr)))) if sd > 0 else None


def _returns(closes: pd.Series) -> pd.Series:
    return closes.pct_change().dropna()


def event_study(
    closes: Mapping[str, pd.Series],
    events: Sequence[Event],
    *,
    estimation: int = 120,
    gap: int = 5,
    pre: int = 20,
    post: int = 5,
    market: pd.Series | None = None,
) -> StudyReport:
    """Run a market-model event study.

    Parameters
    ----------
    closes: symbol -> daily close series (DatetimeIndex, ascending).
    events: (symbol, date) pairs; ``date`` is the disclosure day (t=0).
    estimation: length of the market-model estimation window (trading days),
        ending ``gap`` days before t=0 so the window is not contaminated.
    pre/post: CAR windows in trading days relative to t=0.
    market: optional benchmark close series; default is the equal-weight
    mean of all supplied series.
    """
    returns = {s: _returns(c.astype(float)) for s, c in closes.items()}
    if market is not None:
        market_returns = _returns(market.astype(float))
    else:
        frame = pd.DataFrame({s: r for s, r in returns.items() if len(r)})
        market_returns = frame.mean(axis=1)

    report = StudyReport()
    for event in events:
        r = returns.get(event.symbol)
        if r is None or len(r) == 0:
            report.results.append(EventResult(event=event, skipped_reason="no price series"))
            continue
        loc = r.index.searchsorted(event.date)
        if loc >= len(r) or r.index[loc] != event.date:
            # align to the first trading day >= event date
            if loc >= len(r):
                report.results.append(EventResult(event=event, skipped_reason="event after last bar"))
                continue
        est_end = loc - gap
        est_start = est_end - estimation
        if est_start < 0:
            report.results.append(
                EventResult(event=event, skipped_reason=f"estimation window too short ({max(est_end,0)} bars)")
            )
            continue
        y = r.iloc[est_start:est_end].to_numpy()
        x = market_returns.reindex(r.index[est_start:est_end]).to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < max(30, estimation // 4):
            report.results.append(EventResult(event=event, skipped_reason="market model underdetermined"))
            continue
        beta, alpha = np.polyfit(x[mask], y[mask], 1)
        post_end = min(loc + post + 1, len(r))
        if post_end <= loc:
            report.results.append(EventResult(event=event, skipped_reason="no post window"))
            continue

        def car(a: int, b: int) -> float:
            window_index = r.index[a:b]
            r_win = r.iloc[a:b].to_numpy()
            m_win = market_returns.reindex(window_index).to_numpy()
            ar = r_win - (alpha + beta * m_win)
            return float(np.nansum(ar))

        result = EventResult(
            event=event,
            car_pre=car(max(loc - pre, 0), loc),
            car_post=car(loc + 1, post_end),
            beta=float(beta),
            alpha=float(alpha),
            n_est=int(mask.sum()),
        )
        report.results.append(result)
    return report

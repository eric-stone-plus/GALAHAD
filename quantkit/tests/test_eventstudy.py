"""Tests for quantkit.eventstudy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantkit.eventstudy import Event, StudyReport, event_study


def _series(seed: int, n: int = 400, drift: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    rets = rng.normal(0.0005 + drift, 0.02, n)
    return pd.Series(np.cumprod(1 + rets) * 100.0, index=idx)


def test_no_drift_gives_small_pre_car() -> None:
    closes = {f"s{i}": _series(i) for i in range(6)}
    idx = list(closes.values())[0].index
    events = [Event(f"s{i}", idx[300], label="plain") for i in range(6)]
    report = event_study(closes, events, estimation=120, pre=20, post=5)
    assert len(report.used) == 6
    assert abs(report.mean_car("pre")) < 0.10  # noise only
    assert report.t_stat("pre") is not None


def test_pre_event_drift_is_detected() -> None:
    # One symbol drifts up strongly over the 20 sessions before t=0.
    closes = {f"s{i}": _series(i) for i in range(6)}
    s = "leaky"
    base = _series(99)
    rets = base.pct_change().copy()
    idx = base.index
    t0 = 300
    rets.iloc[t0 - 20 : t0] = rets.iloc[t0 - 20 : t0] + 0.012  # +1.2%/day leak
    closes[s] = (rets.fillna(0) + 1).cumprod() * 100.0
    events = [Event(s, idx[t0], label="leak")]
    report = event_study(closes, events, estimation=120, pre=20, post=5)
    (result,) = report.used
    assert result.car_pre > 0.15  # the planted leak must show up
    assert result.n_est >= 30


def test_events_without_data_are_reported_not_dropped() -> None:
    closes = {"s0": _series(0)}
    idx = closes["s0"].index
    events = [
        Event("missing", idx[300]),
        Event("s0", idx[-1] + pd.Timedelta(days=10)),  # beyond the last bar
        Event("s0", idx[10]),  # not enough history for estimation
    ]
    report = event_study(closes, events, estimation=120)
    assert len(report.used) == 0
    reasons = {r.skipped_reason for r in report.results}
    assert any("no price series" in (x or "") for x in reasons)
    assert any("after last bar" in (x or "") for x in reasons)
    assert any("estimation" in (x or "") for x in reasons)


def test_report_accessors_handle_empty() -> None:
    empty = StudyReport()
    assert empty.mean_car("pre") is None
    assert empty.t_stat("pre") is None

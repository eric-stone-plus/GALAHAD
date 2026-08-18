"""Offline contract tests for the unified OHLCV providers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from quantkit.data import ohlcv


class _BaoResult:
    def __init__(self, rows, error_code="0"):
        self.error_code = error_code
        self.error_msg = "fixture"
        self._rows = iter(rows)
        self._current = None

    def next(self):
        try:
            self._current = next(self._rows)
        except StopIteration:
            return False
        return True

    def get_row_data(self):
        return self._current


def test_auto_provider_routing():
    assert ohlcv._resolve_provider("600519", "cn", "auto") == "baostock"
    assert ohlcv._resolve_provider("600519", "auto", "auto") == "baostock"
    assert ohlcv._resolve_provider("600519.SS", "auto", "auto") == "baostock"
    assert ohlcv._resolve_provider("0700.HK", "hk", "auto") == "yahoo"
    assert ohlcv._resolve_provider("0700.HK", "auto", "auto") == "yahoo"
    assert ohlcv._resolve_provider("00700", "auto", "auto") == "yahoo"
    assert ohlcv._resolve_provider("600519.SS", "us", "auto") == "yahoo"
    assert ohlcv._resolve_provider("430047.BJ", "cn", "auto") == "akshare"
    assert ohlcv._resolve_provider("830799", "auto", "auto") == "akshare"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("600519", "sh.600519"),
        ("600519.SH", "sh.600519"),
        ("600519.SS", "sh.600519"),
        ("sh.600519", "sh.600519"),
        ("000001.SZ", "sz.000001"),
    ],
)
def test_baostock_symbol_normalization(raw, expected):
    assert ohlcv._baostock_symbol(raw) == expected


@pytest.mark.parametrize("raw", ["430047", "830799.BJ", "bj.920002"])
def test_baostock_rejects_beijing_exchange(raw):
    with pytest.raises(ValueError, match="does not support Beijing Exchange"):
        ohlcv._baostock_symbol(raw)


def test_baostock_fetch_is_forward_adjusted_and_normalized(monkeypatch):
    calls = []
    rows = [
        ["2026-08-13", "1338", "1359.6", "1337", "1355.29", "3235348"],
        ["2026-08-14", "1355", "1359", "1338.14", "1341.99", "2985315"],
        ["2026-08-15", "", "", "", "", ""],
    ]
    fake = SimpleNamespace(
        login=lambda: SimpleNamespace(error_code="0", error_msg="success"),
        logout=lambda: calls.append(("logout",)),
        query_history_k_data_plus=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or _BaoResult(rows)
        ),
    )
    monkeypatch.setitem(sys.modules, "baostock", fake)

    frame = ohlcv._fetch_baostock_cn("600519.SH", "2026-08-13", "2026-08-14", "1d")

    query_args, query_kwargs = calls[0]
    assert query_args == ("sh.600519", "date,open,high,low,close,volume")
    assert query_kwargs == {
        "start_date": "2026-08-13",
        "end_date": "2026-08-14",
        "frequency": "d",
        "adjustflag": "2",
    }
    assert calls[-1] == ("logout",)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert list(frame.index) == [pd.Timestamp("2026-08-13"), pd.Timestamp("2026-08-14")]
    assert frame.loc[pd.Timestamp("2026-08-14"), "close"] == pytest.approx(1341.99)


def test_baostock_logout_happens_when_query_fails(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        login=lambda: SimpleNamespace(error_code="0", error_msg="success"),
        logout=lambda: calls.append("logout"),
        query_history_k_data_plus=lambda *args, **kwargs: _BaoResult([], error_code="1001"),
    )
    monkeypatch.setitem(sys.modules, "baostock", fake)

    with pytest.raises(RuntimeError, match="BaoStock query failed: 1001"):
        ohlcv._fetch_baostock_cn("600519", None, None, "1d")
    assert calls == ["logout"]


def test_baostock_rejects_intraday_before_login(monkeypatch):
    monkeypatch.setitem(sys.modules, "baostock", None)
    with pytest.raises(ValueError, match="interval='1d'"):
        ohlcv._fetch_baostock_cn("600519", None, None, "1h")


@pytest.mark.parametrize("raw", ["430047.BJ", "bj.830799"])
def test_akshare_fallback_strips_beijing_exchange_affixes(monkeypatch, raw):
    calls = []
    fake = SimpleNamespace(
        stock_zh_a_hist=lambda **kwargs: (
            calls.append(kwargs)
            or pd.DataFrame(
                [{"日期": "2026-08-14", "开盘": 1, "最高": 1, "最低": 1, "收盘": 1, "成交量": 1}]
            )
        )
    )
    monkeypatch.setitem(sys.modules, "akshare", fake)

    frame = ohlcv._fetch_akshare_cn(raw, "2026-08-14", "2026-08-14")

    assert calls[0]["symbol"] in {"430047", "830799"}
    assert frame.loc[pd.Timestamp("2026-08-14"), "close"] == 1


def test_baostock_normalizes_compact_dates_and_rejects_reverse_range(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        login=lambda: SimpleNamespace(error_code="0", error_msg="success"),
        logout=lambda: None,
        query_history_k_data_plus=lambda *args, **kwargs: (
            calls.append(kwargs) or _BaoResult([])
        ),
    )
    monkeypatch.setitem(sys.modules, "baostock", fake)

    ohlcv._fetch_baostock_cn("600519", "20260813", "20260814", "1d")
    assert calls[0]["start_date"] == "2026-08-13"
    assert calls[0]["end_date"] == "2026-08-14"

    with pytest.raises(ValueError, match="start date"):
        ohlcv._fetch_baostock_cn("600519", "2026-08-15", "2026-08-14", "1d")


def test_fetch_routes_auto_cn_to_baostock_and_uses_source_specific_cache(monkeypatch, tmp_path):
    seen = []
    expected = pd.DataFrame(
        {"close": [1341.99]},
        index=pd.DatetimeIndex(["2026-08-14"], name="date"),
    )

    def fake_fetch(symbol, start, end, interval):
        seen.append((symbol, start, end, interval))
        return expected

    monkeypatch.setattr(ohlcv, "_fetch_baostock_cn", fake_fetch)
    result = ohlcv.fetch_ohlcv(
        "600519.SS",
        start="2026-08-14",
        end="2026-08-14",
        data_dir=tmp_path,
    )

    assert seen == [("600519.SS", "2026-08-14", "2026-08-14", "1d")]
    pd.testing.assert_frame_equal(result, expected)
    assert (tmp_path / "cache" / "baostock_auto_600519.SS_1d_2026-08-14_2026-08-14.parquet").is_file()


def test_fetch_normalizes_bare_hk_code_for_yahoo(monkeypatch, tmp_path):
    seen = []
    expected = pd.DataFrame(
        {"close": [390.0]},
        index=pd.DatetimeIndex(["2026-08-14"], name="date"),
    )

    def fake_fetch(symbol, start, end, interval):
        seen.append(symbol)
        return expected

    monkeypatch.setattr(ohlcv, "_fetch_yahoo", fake_fetch)
    result = ohlcv.fetch_ohlcv(
        "0700",
        start="2026-08-14",
        end="2026-08-14",
        data_dir=tmp_path,
    )

    assert seen == ["0700.HK"]
    pd.testing.assert_frame_equal(result, expected)

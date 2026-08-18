from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quantkit.data.consensus import (
    ConsensusTolerance,
    OHLCVConsensusError,
    OHLCVIdentity,
    OHLCVSecurityIdentity,
    OHLCVSource,
    build_ohlcv_consensus,
)


IDENTITY = OHLCVIdentity(
    symbol="931743.CSI",
    market="XSHG",
    interval="1d",
    calendar="XSHG",
    currency="CNY",
)
SECURITY_IDENTITY = OHLCVSecurityIdentity(
    symbol=IDENTITY.symbol,
    market=IDENTITY.market,
    currency=IDENTITY.currency,
)
ARTIFACT_DIGEST = "1" * 64
DEFAULT_SESSIONS = ["2026-08-10", "2026-08-11"]


def assert_canonical_reason_codes(value: object) -> None:
    """Every emitted reason-code array is unique and lexically ordered."""

    if isinstance(value, dict):
        for key, child in value.items():
            if key == "reason_codes":
                assert isinstance(child, list)
                assert child == sorted(set(child))
            assert_canonical_reason_codes(child)
    elif isinstance(value, list):
        for child in value:
            assert_canonical_reason_codes(child)


def frame(
    closes: list[float],
    *,
    dates: list[str] | None = None,
    volume: float = 1_000.0,
) -> pd.DataFrame:
    dates = dates or [f"2026-08-{day:02d}" for day in range(10, 10 + len(closes))]
    return pd.DataFrame(
        {
            "open": [value - 1.0 for value in closes],
            "high": [value + 1.0 for value in closes],
            "low": [value - 2.0 for value in closes],
            "close": closes,
            "volume": [volume] * len(closes),
        },
        index=pd.to_datetime(dates),
    )


def source(
    provider: str,
    group: str,
    data: pd.DataFrame,
    *,
    source_id: str | None = None,
    price_basis: str = "raw",
    volume_unit: str = "shares",
) -> OHLCVSource:
    return OHLCVSource(
        provider=provider,
        independence_group=group,
        price_basis=price_basis,
        volume_unit=volume_unit,
        frame=data,
        identity=SECURITY_IDENTITY,
        input_artifact_sha256=ARTIFACT_DIGEST,
        source_id=source_id,
    )


def wide_tolerance() -> ConsensusTolerance:
    return ConsensusTolerance(
        price_pass=0.003,
        price_warning=0.02,
        price_quarantine=0.20,
        volume_pass=0.02,
        volume_warning=0.20,
        volume_quarantine=0.50,
    )


def test_clean_three_source_median_consensus_and_manifest_shape() -> None:
    result = build_ohlcv_consensus(
        IDENTITY,
        [
            source("alpha", "group-a", frame([100.0, 105.0])),
            source("beta", "group-b", frame([100.1, 105.1])),
            source("gamma", "group-c", frame([99.9, 104.9])),
        ],
        sessions=DEFAULT_SESSIONS,
        tolerance=wide_tolerance(),
    )

    assert result.status == "accepted"
    assert result.accepted_bars["close"].tolist() == pytest.approx([100.0, 105.0])
    manifest = result.to_manifest()
    assert manifest["schema"] == "quantkit.ohlcv-consensus.v1"
    assert manifest["identity"] == {
        "symbol": "931743.CSI",
        "market": "XSHG",
        "interval": "1d",
        "calendar": "XSHG",
        "currency": "CNY",
        "price_basis": "raw",
        "volume_unit": "shares",
    }
    assert manifest["bars"] == [
        {
            "date": "2026-08-10",
            "open": 99.0,
            "high": 101.0,
            "low": 98.0,
            "close": 100.0,
            "volume": 1000.0,
        },
        {
            "date": "2026-08-11",
            "open": 104.0,
            "high": 106.0,
            "low": 103.0,
            "close": 105.0,
            "volume": 1000.0,
        },
    ]
    assert len(manifest["bars_sha256"]) == 64
    assert len(manifest["output_sha256"]) == 64
    json.dumps(manifest, allow_nan=False)


def test_inputs_are_not_mutated() -> None:
    first = frame([100.0])
    second = frame([100.1])
    before_first = first.copy(deep=True)
    before_second = second.copy(deep=True)

    build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", first), source("beta", "b", second)],
        sessions=DEFAULT_SESSIONS,
        tolerance=wide_tolerance(),
    )

    assert_frame_equal(first, before_first, check_exact=True)
    assert_frame_equal(second, before_second, check_exact=True)


def test_union_alignment_never_forward_fills_missing_dates() -> None:
    result = build_ohlcv_consensus(
        IDENTITY,
        [
            source(
                "alpha",
                "a",
                frame([100.0, 101.0], dates=["2026-08-10", "2026-08-11"]),
            ),
            source("beta", "b", frame([100.1], dates=["2026-08-10"])),
        ],
        sessions=DEFAULT_SESSIONS,
        tolerance=wide_tolerance(),
    )

    assert result.accepted_bars.index.strftime("%Y-%m-%d").tolist() == ["2026-08-10"]
    second_session = result.to_manifest()["diagnostics"][1]
    assert second_session["date"] == "2026-08-11"
    assert second_session["status"] == "quarantined"
    beta = next(item for item in second_session["sources"] if item["provider"] == "beta")
    assert beta["status"] == "missing"


def test_two_independent_group_minimum_and_duplicate_group_does_not_count() -> None:
    one_session = ["2026-08-10"]
    with pytest.raises(OHLCVConsensusError, match="at least two independent"):
        build_ohlcv_consensus(
            IDENTITY,
            [
                source("alpha", "shared", frame([100.0])),
                source("beta", "shared", frame([100.0])),
            ],
            sessions=one_session,
        )

    result = build_ohlcv_consensus(
        IDENTITY,
        [
            source("alpha", "shared", frame([100.0])),
            source("alpha", "shared", frame([100.0]), source_id="alpha-backup"),
            source("beta", "independent", frame([100.1])),
        ],
        sessions=one_session,
        tolerance=wide_tolerance(),
    )
    assert result.status == "accepted"
    assert result.to_manifest()["diagnostics"][0]["supporting_groups"] == [
        "independent",
        "shared",
    ]


def test_two_source_agreement_accepts_but_unresolved_disagreement_quarantines() -> None:
    one_session = ["2026-08-10"]
    agreed = build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", frame([100.0])), source("beta", "b", frame([100.2]))],
        sessions=one_session,
        tolerance=wide_tolerance(),
    )
    disagreed = build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", frame([100.0])), source("beta", "b", frame([130.0]))],
        sessions=one_session,
        tolerance=wide_tolerance(),
    )

    assert agreed.status == "accepted"
    assert disagreed.status == "quarantined"
    assert disagreed.accepted_bars.empty
    assert "independent_group_disagreement" in disagreed.diagnostics[0]["reason_codes"]


def test_pass_warning_and_quarantine_bands_are_total() -> None:
    tolerance = ConsensusTolerance(
        price_pass=0.01,
        price_warning=0.03,
        price_quarantine=0.10,
        volume_pass=0.01,
        volume_warning=0.03,
        volume_quarantine=0.10,
    )
    passed = build_ohlcv_consensus(
        IDENTITY,
        [source("a", "a", frame([100.0])), source("b", "b", frame([100.5]))],
        sessions=DEFAULT_SESSIONS,
        tolerance=tolerance,
    )
    warned = build_ohlcv_consensus(
        IDENTITY,
        [source("a", "a", frame([100.0])), source("b", "b", frame([102.0]))],
        sessions=DEFAULT_SESSIONS,
        tolerance=tolerance,
    )
    quarantined = build_ohlcv_consensus(
        IDENTITY,
        [source("a", "a", frame([100.0])), source("b", "b", frame([106.0]))],
        sessions=DEFAULT_SESSIONS,
        tolerance=tolerance,
    )

    assert passed.diagnostics[0]["status"] == "accepted"
    assert warned.diagnostics[0]["status"] == "warning"
    assert quarantined.status == "quarantined"
    assert "quarantine_band" in quarantined.diagnostics[0]["reason_codes"]


def test_one_bad_outlier_is_excluded_from_three_groups() -> None:
    result = build_ohlcv_consensus(
        IDENTITY,
        [
            source("alpha", "a", frame([100.0])),
            source("beta", "b", frame([100.2])),
            source("gamma", "c", frame([180.0])),
        ],
        sessions=DEFAULT_SESSIONS,
        tolerance=wide_tolerance(),
    )

    assert result.status == "partial"
    assert result.accepted_bars.iloc[0]["close"] == pytest.approx(100.1)
    diagnostic = result.diagnostics[0]
    assert diagnostic["status"] == "warning"
    assert diagnostic["supporting_groups"] == ["a", "b"]
    gamma = next(item for item in diagnostic["sources"] if item["provider"] == "gamma")
    assert gamma["status"] == "excluded"
    assert gamma["reason_codes"] == ["beyond_quarantine_band", "group_outlier"]
    assert_canonical_reason_codes(result.to_manifest())


def test_ambiguous_two_of_three_bridge_cluster_is_quarantined() -> None:
    tolerance = ConsensusTolerance(
        price_pass=0.01,
        price_warning=0.055,
        price_quarantine=0.20,
        volume_pass=0.01,
        volume_warning=0.055,
        volume_quarantine=0.20,
    )
    result = build_ohlcv_consensus(
        IDENTITY,
        [
            source("alpha", "a", frame([100.0])),
            source("beta", "b", frame([105.0])),
            source("gamma", "c", frame([110.0])),
        ],
        sessions=DEFAULT_SESSIONS,
        tolerance=tolerance,
    )

    assert result.status == "quarantined"
    assert result.accepted_bars.empty
    assert "ambiguous_support_cluster" in result.diagnostics[0]["reason_codes"]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda data: data.assign(close=np.nan), "non_finite_ohlcv"),
        (lambda data: data.assign(close=np.inf), "non_finite_ohlcv"),
        (lambda data: data.assign(close=-1.0), "non_positive_price"),
        (lambda data: data.assign(volume=-1.0), "negative_volume"),
        (lambda data: data.assign(high=50.0), "invalid_ohlc_geometry"),
    ],
)
def test_invalid_rows_are_diagnostic_and_fail_closed(mutate, reason: str) -> None:
    bad = mutate(frame([100.0]))
    result = build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", bad), source("beta", "b", frame([100.0]))],
        sessions=DEFAULT_SESSIONS,
        tolerance=wide_tolerance(),
    )

    assert result.status == "quarantined"
    invalid = next(item for item in result.diagnostics[0]["sources"] if item["provider"] == "alpha")
    assert invalid["status"] == "invalid"
    assert reason in invalid["reason_codes"]


def test_basis_and_volume_unit_mismatches_are_contract_errors() -> None:
    with pytest.raises(OHLCVConsensusError, match="price bases"):
        build_ohlcv_consensus(
            IDENTITY,
            [
                source("alpha", "a", frame([100.0]), price_basis="raw"),
                source("beta", "b", frame([100.0]), price_basis="split_adjusted"),
            ],
            sessions=DEFAULT_SESSIONS,
        )
    with pytest.raises(OHLCVConsensusError, match="volume units"):
        build_ohlcv_consensus(
            IDENTITY,
            [
                source("alpha", "a", frame([100.0]), volume_unit="shares"),
                source("beta", "b", frame([100.0]), volume_unit="lots"),
            ],
            sessions=DEFAULT_SESSIONS,
        )


def test_hashes_are_deterministic_and_source_order_invariant() -> None:
    sources = [
        source("alpha", "a", frame([100.0, 101.0])),
        source("beta", "b", frame([100.1, 101.1])),
        source("gamma", "c", frame([99.9, 100.9])),
    ]
    first = build_ohlcv_consensus(
        IDENTITY, sources, sessions=DEFAULT_SESSIONS, tolerance=wide_tolerance()
    )
    second = build_ohlcv_consensus(
        IDENTITY,
        list(reversed(sources)),
        sessions=DEFAULT_SESSIONS,
        tolerance=wide_tolerance(),
    )

    assert first.to_manifest() == second.to_manifest()
    assert (
        first.to_manifest()["output_sha256"]
        == second.to_manifest()["output_sha256"]
    )
    assert (
        first.to_manifest()["bars_sha256"]
        == second.to_manifest()["bars_sha256"]
    )
    assert [item["frame_sha256"] for item in first.to_manifest()["sources"]] == [
        item["frame_sha256"] for item in second.to_manifest()["sources"]
    ]


def test_original_row_order_changes_frame_hash_but_not_accepted_output_hash() -> None:
    ordered = frame([100.0, 101.0])
    reversed_rows = ordered.iloc[::-1]
    first = build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", ordered), source("beta", "b", ordered.copy())],
        sessions=DEFAULT_SESSIONS,
    )
    second = build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", reversed_rows), source("beta", "b", ordered.copy())],
        sessions=DEFAULT_SESSIONS,
    )

    first_sources = {item["provider"]: item for item in first.to_manifest()["sources"]}
    second_sources = {item["provider"]: item for item in second.to_manifest()["sources"]}
    assert first_sources["alpha"]["frame_sha256"] != second_sources["alpha"]["frame_sha256"]
    assert first.to_manifest()["bars_sha256"] == second.to_manifest()["bars_sha256"]
    assert first.to_manifest()["output_sha256"] != second.to_manifest()["output_sha256"]


def test_source_identity_and_artifact_digest_are_mandatory_and_bound() -> None:
    wrong_identity = OHLCVSecurityIdentity(
        symbol="000001.SH", market="XSHG", currency="CNY"
    )
    with pytest.raises(OHLCVConsensusError, match="security identity"):
        build_ohlcv_consensus(
            IDENTITY,
            [
                OHLCVSource(
                    provider="alpha",
                    independence_group="a",
                    price_basis="raw",
                    volume_unit="shares",
                    frame=frame([100.0]),
                    identity=wrong_identity,
                    input_artifact_sha256=ARTIFACT_DIGEST,
                ),
                source("beta", "b", frame([100.0])),
            ],
            sessions=DEFAULT_SESSIONS,
        )
    with pytest.raises(OHLCVConsensusError, match="input_artifact_sha256"):
        build_ohlcv_consensus(
            IDENTITY,
            [
                OHLCVSource(
                    provider="alpha",
                    independence_group="a",
                    price_basis="raw",
                    volume_unit="shares",
                    frame=frame([100.0]),
                    identity=SECURITY_IDENTITY,
                ),
                source("beta", "b", frame([100.0])),
            ],
            sessions=DEFAULT_SESSIONS,
        )

    result = build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", frame([100.0])), source("beta", "b", frame([100.0]))],
        sessions=DEFAULT_SESSIONS,
    )
    assert {
        item["input_artifact_sha256"] for item in result.to_manifest()["sources"]
    } == {ARTIFACT_DIGEST}
    assert all(len(item["frame_sha256"]) == 64 for item in result.to_manifest()["sources"])


def test_actual_session_calendar_quarantines_matching_weekend_rows() -> None:
    result = build_ohlcv_consensus(
        IDENTITY,
        [
            source("alpha", "a", frame([100.0], dates=["2026-08-15"])),
            source("beta", "b", frame([100.0], dates=["2026-08-15"])),
        ],
        sessions=["2026-08-14", "2026-08-17"],
    )

    assert result.status == "quarantined"
    assert result.accepted_bars.empty
    weekend = next(item for item in result.diagnostics if item["date"] == "2026-08-15")
    assert weekend["reason_codes"] == ["outside_actual_session_calendar"]
    assert all(
        item["status"] == "excluded" and item["reason_codes"] == ["outside_actual_session_calendar"]
        for item in weekend["sources"]
    )


def test_calendar_must_be_explicit_sorted_and_daily_interval_only() -> None:
    sources = [source("alpha", "a", frame([100.0])), source("beta", "b", frame([100.0]))]
    with pytest.raises(OHLCVConsensusError, match="sessions must contain"):
        build_ohlcv_consensus(IDENTITY, sources, sessions=[])
    with pytest.raises(OHLCVConsensusError, match="strictly increasing"):
        build_ohlcv_consensus(
            IDENTITY, sources, sessions=["2026-08-11", "2026-08-10"]
        )
    intraday_identity = OHLCVIdentity(
        symbol=IDENTITY.symbol,
        market=IDENTITY.market,
        interval="1h",
        calendar=IDENTITY.calendar,
        currency=IDENTITY.currency,
    )
    with pytest.raises(OHLCVConsensusError, match="only the 1d interval"):
        build_ohlcv_consensus(
            intraday_identity, sources, sessions=DEFAULT_SESSIONS
        )


def test_calendar_only_missing_session_is_present_in_diagnostics() -> None:
    result = build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", frame([100.0])), source("beta", "b", frame([100.0]))],
        sessions=["2026-08-10", "2026-08-11"],
    )

    assert result.status == "partial"
    missing = result.diagnostics[1]
    assert missing["date"] == "2026-08-11"
    assert missing["status"] == "quarantined"
    assert "insufficient_independent_groups" in missing["reason_codes"]
    assert {item["status"] for item in missing["sources"]} == {"missing"}
    manifest = result.to_manifest()
    assert manifest["calendar"]["sessions"] == ["2026-08-10", "2026-08-11"]
    assert len(manifest["calendar"]["sessions_sha256"]) == 64


def test_manifest_and_diagnostics_are_detached_copies() -> None:
    result = build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", frame([100.0])), source("beta", "b", frame([100.1]))],
        sessions=DEFAULT_SESSIONS,
        tolerance=wide_tolerance(),
    )
    original = copy.deepcopy(result.to_manifest())
    changed = result.to_manifest()
    changed["bars"][0]["close"] = 0
    changed["diagnostics"][0]["sources"][0]["status"] = "invalid"
    detached_diagnostics = list(result.diagnostics)
    detached_diagnostics[0]["status"] = "quarantined"
    detached_bars = result.accepted_bars
    detached_bars.iloc[0, detached_bars.columns.get_loc("close")] = 0

    assert result.to_manifest() == original
    assert result.diagnostics[0]["status"] == "accepted"
    assert result.accepted_bars.iloc[0]["close"] == pytest.approx(100.05)


def test_empty_and_all_quarantined_results_never_look_successful() -> None:
    empty = frame([])
    empty_result = build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", empty), source("beta", "b", empty)],
        sessions=DEFAULT_SESSIONS,
    )
    rejected = build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", frame([100.0])), source("beta", "b", frame([150.0]))],
        sessions=DEFAULT_SESSIONS,
        tolerance=wide_tolerance(),
    )

    assert empty_result.status == "quarantined"
    assert empty_result.accepted_bars.empty
    assert empty_result.to_manifest()["bars"] == []
    assert rejected.status == "quarantined"
    assert rejected.accepted_bars.empty


def test_conflicting_duplicate_provider_captures_do_not_cast_a_vote() -> None:
    result = build_ohlcv_consensus(
        IDENTITY,
        [
            source("alpha", "a", frame([100.0]), source_id="alpha-primary"),
            source("alpha", "a", frame([160.0]), source_id="alpha-backup"),
            source("beta", "b", frame([100.0])),
        ],
        sessions=DEFAULT_SESSIONS,
        tolerance=wide_tolerance(),
    )

    assert result.status == "quarantined"
    assert "insufficient_independent_groups" in result.diagnostics[0]["reason_codes"]
    alpha_decisions = [
        item for item in result.diagnostics[0]["sources"] if item["provider"] == "alpha"
    ]
    assert {item["status"] for item in alpha_decisions} == {"excluded"}
    assert all(
        "provider_internal_disagreement" in item["reason_codes"]
        for item in alpha_decisions
    )


def test_duplicate_sessions_are_rejected_instead_of_silently_deduplicated() -> None:
    duplicated = frame(
        [100.0, 100.1], dates=["2026-08-10", "2026-08-10"]
    )
    result = build_ohlcv_consensus(
        IDENTITY,
        [source("alpha", "a", duplicated), source("beta", "b", frame([100.0]))],
        sessions=DEFAULT_SESSIONS,
        tolerance=wide_tolerance(),
    )

    assert result.status == "quarantined"
    alpha = next(item for item in result.diagnostics[0]["sources"] if item["provider"] == "alpha")
    assert "duplicate_session" in alpha["reason_codes"]

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from quantkit.data.consensus import (
    ConsensusTolerance,
    OHLCVIdentity,
    OHLCVSecurityIdentity,
    OHLCVSource,
    build_ohlcv_consensus,
)
from quantkit.semiconductor import (
    AuxiliaryDiagnostic,
    ControlSpec,
    InvalidationFlag,
    LeadLagCalculation,
    LeadLagConfig,
    LeadLagError,
    MethodRuntime,
    ValidatedReturnSeries,
    VenueSession,
    build_semiconductor_lead_lag,
    build_verified_session_map,
)


RUNTIME = MethodRuntime("a" * 64, "b" * 64)


def sessions(start: str, count: int) -> list[str]:
    day = dt.date.fromisoformat(start)
    result: list[str] = []
    while len(result) < count:
        if day.weekday() < 5:
            result.append(day.isoformat())
        day += dt.timedelta(days=1)
    return result


def manifest_bytes(
    *,
    symbol: str,
    market: str,
    calendar: str,
    currency: str,
    dates: list[str],
    closes: np.ndarray,
    gaps: np.ndarray | None = None,
) -> bytes:
    gaps = np.zeros(len(dates), dtype=float) if gaps is None else gaps
    opens = np.asarray(closes, dtype=float).copy()
    for position in range(1, len(dates)):
        opens[position] = closes[position - 1] * math.exp(float(gaps[position]))
    bars = []
    for date, opened, closed in zip(dates, opens, closes):
        high = max(float(opened), float(closed)) * 1.001
        low = min(float(opened), float(closed)) * 0.999
        bars.append(
            {
                "date": date,
                "open": float(opened),
                "high": high,
                "low": low,
                "close": float(closed),
                "volume": 1_000_000.0,
            }
        )
    frame = pd.DataFrame.from_records(bars).set_index(pd.to_datetime(dates))
    frame.index.name = None
    frame = frame[["open", "high", "low", "close", "volume"]]
    identity = OHLCVIdentity(symbol, market, "1d", calendar, currency)
    security = OHLCVSecurityIdentity(symbol, market, currency)
    sources = [
        OHLCVSource(
            provider=f"provider-{position}",
            independence_group=f"group-{position}",
            price_basis="raw",
            volume_unit="shares",
            frame=frame.copy(),
            identity=security,
            input_artifact_sha256=str(position + 1) * 64,
        )
        for position in range(2)
    ]
    result = build_ohlcv_consensus(
        identity,
        sources,
        sessions=dates,
        tolerance=ConsensusTolerance(),
    )
    assert result.status == "accepted"
    return json.dumps(
        result.to_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def schedule(
    dates: list[str], *, market: str, future: int = 0
) -> tuple[VenueSession, ...]:
    all_dates = list(dates)
    if future:
        all_dates.extend(sessions(all_dates[-1], future + 1)[1:])
    result = []
    for date in all_dates:
        if market == "US":
            opened = f"{date}T13:30:00Z"
            closed = f"{date}T20:00:00Z"
        else:
            opened = f"{date}T01:30:00Z"
            closed = f"{date}T07:00:00Z"
        result.append(VenueSession(date, opened, closed))
    return tuple(result)


def study_fixture(count: int = 270, *, seed: int = 7):
    rng = np.random.default_rng(seed)
    target_dates = sessions("2024-01-02", count)
    source_dates = sessions("2024-01-01", count)
    driver_returns = rng.normal(0.0, 0.012, count - 1)
    driver_closes = np.r_[100.0, 100.0 * np.exp(np.cumsum(driver_returns))]
    gap_returns = np.r_[0.0, 0.30 * driver_returns + rng.normal(0, 0.003, count - 1)]
    intraday_returns = np.r_[0.0, 0.15 * driver_returns + rng.normal(0, 0.004, count - 1)]
    target_closes = np.empty(count)
    target_closes[0] = 100.0
    for position in range(1, count):
        target_closes[position] = target_closes[position - 1] * math.exp(
            gap_returns[position] + intraday_returns[position]
        )
    control_returns = rng.normal(0, 0.006, count - 1)
    control_closes = np.r_[50.0, 50.0 * np.exp(np.cumsum(control_returns))]
    driver_bytes = manifest_bytes(
        symbol="SOXX", market="US", calendar="XNYS", currency="USD",
        dates=source_dates, closes=driver_closes,
    )
    target_bytes = manifest_bytes(
        symbol="931743.CSI", market="CN", calendar="XSHG", currency="CNY",
        dates=target_dates, closes=target_closes, gaps=gap_returns,
    )
    control_bytes = manifest_bytes(
        symbol="SPY", market="US", calendar="XNYS", currency="USD",
        dates=source_dates, closes=control_closes,
    )
    driver = ValidatedReturnSeries(
        "us_semiconductor_driver", "source_return", "source_close", driver_bytes
    )
    target_gap = ValidatedReturnSeries(
        "ashare_semiconductor_target", "target_gap", "target_session", target_bytes
    )
    target_intraday = ValidatedReturnSeries(
        "ashare_semiconductor_target", "target_intraday", "target_session", target_bytes
    )
    control = ValidatedReturnSeries(
        "us_market_control", "control_return", "source_close", control_bytes
    )
    target_schedule = schedule(target_dates, market="CN", future=7)
    mapping = build_verified_session_map(
        "US", "CN", schedule(source_dates, market="US"), target_schedule
    )
    config = LeadLagConfig(
        bootstrap_repetitions=199,
        required_controls=(
            ControlSpec("us_market_control", "broad_market", "source_close", "c" * 64),
        ),
    )
    return {
        "driver": driver,
        "target_gap": target_gap,
        "target_intraday": target_intraday,
        "controls": [control],
        "session_map": mapping,
        "estimation_session": target_dates[-1],
        "runtime": RUNTIME,
        "config": config,
    }


def build(**overrides):
    values = study_fixture()
    values.update(overrides)
    return build_semiconductor_lead_lag(**values)


def reseal(document: dict) -> bytes:
    def canonical(value: object) -> bytes:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        ).encode()

    unsealed = {key: value for key, value in document.items() if key != "output_sha256"}
    document["output_sha256"] = hashlib.sha256(
        b"quantkit.ohlcv-consensus-manifest.v1\0" + canonical(unsealed)
    ).hexdigest()
    return canonical(document)


def reseal_calculation(document: dict) -> dict:
    document.pop("output_sha256", None)
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    document["output_sha256"] = hashlib.sha256(
        b"quantkit.semiconductor-lead-lag.v1\0" + canonical
    ).hexdigest()
    return document


def component_digest(domain: str, value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(domain.encode() + b"\0" + canonical).hexdigest()


def auxiliary(
    family_id: str,
    status: str,
    as_of_session: str,
    *,
    validated_intraday_data: bool = False,
    intraday_session_count: int = 0,
    intraday_observation_count: int = 0,
) -> AuxiliaryDiagnostic:
    document = {
        "schema": "quantkit.semiconductor-method-diagnostic.v1",
        "family_id": family_id,
        "status": status,
        "as_of_session": as_of_session,
        "validated_intraday_data": validated_intraday_data,
        "intraday_session_count": intraday_session_count,
        "intraday_observation_count": intraday_observation_count,
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    semantic = hashlib.sha256(
        b"quantkit.semiconductor-method-diagnostic.v1\0" + payload
    ).hexdigest()
    document["output_sha256"] = semantic
    exact = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return AuxiliaryDiagnostic(
        family_id,
        status,
        as_of_session,
        semantic,
        exact,
        validated_intraday_data,
        intraday_session_count,
        intraday_observation_count,
    )


def test_real_consensus_manifests_derive_returns_and_seal_45_member_family() -> None:
    first = build().to_manifest()
    second = build().to_manifest()

    assert first == second
    assert first["schema"] == "quantkit.semiconductor-lead-lag.v1"
    assert first["family"]["test_count"] == 45
    assert first["family"]["windows"] == [63, 126, 252]
    assert first["family"]["horizons"] == [1, 2, 3, 4, 5]
    assert first["family"]["outcomes"] == ["gap", "intraday", "total"]
    assert first["inputs"]["driver"]["observation_count"] == 269
    assert first["inputs"]["target_gap"]["observation_count"] == 269
    assert first["inputs"]["target_intraday"]["observation_count"] == 270
    assert first["inputs"]["target_gap"]["manifest_bytes_sha256"] == first["inputs"]["target_intraday"]["manifest_bytes_sha256"]
    assert first["inputs"]["target_gap"]["returns_sha256"] != first["inputs"]["target_intraday"]["returns_sha256"]
    assert first["runtime_sha256"] != first["method_sha256"]
    json.dumps(first, allow_nan=False)


def test_manifest_integrity_tampering_is_rejected() -> None:
    values = study_fixture()
    target = json.loads(values["target_gap"].consensus_manifest_bytes)
    target["bars"][10]["volume"] += 1.0
    tampered = json.dumps(target, sort_keys=True, separators=(",", ":")).encode()
    bad_gap = ValidatedReturnSeries(
        "ashare_semiconductor_target", "target_gap", "target_session", tampered
    )
    bad_intraday = ValidatedReturnSeries(
        "ashare_semiconductor_target", "target_intraday", "target_session", tampered
    )
    values.update(target_gap=bad_gap, target_intraday=bad_intraday)
    with pytest.raises(LeadLagError, match="bars_sha256 does not match"):
        build_semiconductor_lead_lag(**values)

    target = json.loads(values["driver"].consensus_manifest_bytes)
    target["output_sha256"] = "0" * 64
    bad = ValidatedReturnSeries(
        "us_semiconductor_driver", "source_return", "source_close",
        json.dumps(target, sort_keys=True, separators=(",", ":")).encode(),
    )
    values = study_fixture()
    values["driver"] = bad
    with pytest.raises(LeadLagError, match="output_sha256 does not match"):
        build_semiconductor_lead_lag(**values)


def test_exact_manifest_bytes_hash_is_distinct_and_target_legs_must_match() -> None:
    values = study_fixture()
    semantic = json.loads(values["target_gap"].consensus_manifest_bytes)
    pretty = json.dumps(semantic, indent=2).encode()
    pretty_gap = ValidatedReturnSeries(
        "ashare_semiconductor_target", "target_gap", "target_session", pretty
    )
    pretty_intraday = ValidatedReturnSeries(
        "ashare_semiconductor_target", "target_intraday", "target_session", pretty
    )
    values.update(target_gap=pretty_gap, target_intraday=pretty_intraday)
    output = build_semiconductor_lead_lag(**values).to_manifest()
    assert output["inputs"]["target_gap"]["manifest_bytes_sha256"] == hashlib.sha256(pretty).hexdigest()
    assert output["inputs"]["target_gap"]["manifest_bytes_sha256"] != output["inputs"]["target_gap"]["consensus_output_sha256"]

    values = study_fixture()
    values["target_intraday"] = pretty_intraday
    with pytest.raises(LeadLagError, match="identical manifest bytes"):
        build_semiconductor_lead_lag(**values)


def test_rehashed_fake_upstream_provenance_stays_inside_explicit_boundary() -> None:
    values = study_fixture()
    driver_doc = json.loads(values["driver"].consensus_manifest_bytes)
    driver_doc["sources"][0]["input_artifact_sha256"] = "f" * 64
    forged = reseal(driver_doc)
    values["driver"] = ValidatedReturnSeries(
        "us_semiconductor_driver", "source_return", "source_close", forged
    )
    manifest = build_semiconductor_lead_lag(**values).to_manifest()
    assert manifest["provenance_boundary"] == "calculation_integrity_relative_to_upstream_accepted_manifests_and_schedules"
    assert manifest["publication_status"] == "ABSTAIN"


def test_verified_schedule_map_handles_collision_half_day_dst_and_unmatched_close() -> None:
    sources = (
        VenueSession("2026-07-02", "2026-07-02T13:30:00Z", "2026-07-02T20:00:00Z"),
        VenueSession("2026-07-03", "2026-07-03T13:30:00Z", "2026-07-03T17:00:00Z"),
        VenueSession("2026-07-06", "2026-07-06T13:30:00Z", "2026-07-06T20:00:00Z"),
        VenueSession("2026-11-27", "2026-11-27T14:30:00Z", "2026-11-27T18:00:00Z"),
    )
    targets = (
        VenueSession("2026-07-06", "2026-07-06T01:30:00Z", "2026-07-06T07:00:00Z"),
        VenueSession("2026-07-07", "2026-07-07T01:30:00Z", "2026-07-07T07:00:00Z"),
    )
    mapped = build_verified_session_map("US", "CN", sources, targets)
    assert mapped.links[0].source_session == "2026-07-03"
    assert mapped.excluded_collisions[0]["source_session"] == "2026-07-02"
    assert mapped.links[1].source_session == "2026-07-06"
    assert mapped.unmatched_source_closes[0]["source_session"] == "2026-11-27"
    forged = copy.copy(mapped)
    object.__setattr__(forged, "links", tuple(reversed(mapped.links)))
    values = study_fixture()
    values["session_map"] = forged
    with pytest.raises(LeadLagError, match="does not match"):
        build_semiconductor_lead_lag(**values)


def test_missing_observations_and_post_estimation_data_fail_closed() -> None:
    values = study_fixture()
    document = json.loads(values["target_gap"].consensus_manifest_bytes)
    document["calendar"]["sessions"].pop(100)
    # A producer can reseal a semantically self-consistent but non-contiguous slice.
    calendar = document["calendar"]["sessions"]
    canonical = json.dumps(calendar, sort_keys=True, separators=(",", ":")).encode()
    document["calendar"]["sessions_sha256"] = hashlib.sha256(
        b"quantkit.ohlcv-session-calendar.v1\0" + canonical
    ).hexdigest()
    document["bars"].pop(100)
    bars = document["bars"]
    canonical = json.dumps(bars, sort_keys=True, separators=(",", ":")).encode()
    document["bars_sha256"] = hashlib.sha256(
        b"quantkit.ohlcv-accepted-bars.v1\0" + canonical
    ).hexdigest()
    payload = reseal(document)
    values["target_gap"] = ValidatedReturnSeries(
        "ashare_semiconductor_target", "target_gap", "target_session", payload
    )
    values["target_intraday"] = ValidatedReturnSeries(
        "ashare_semiconductor_target", "target_intraday", "target_session", payload
    )
    with pytest.raises(LeadLagError, match="omits an actual scheduled session"):
        build_semiconductor_lead_lag(**values)

    values = study_fixture()
    values["estimation_session"] = values["session_map"].target_schedule[-10].session
    with pytest.raises(LeadLagError, match="observations after estimation"):
        build_semiconductor_lead_lag(**values)


def test_controls_are_preregistered_and_timing_domains_are_distinct() -> None:
    values = study_fixture()
    values["controls"] = []
    with pytest.raises(LeadLagError, match="preregistered control set"):
        build_semiconductor_lead_lag(**values)

    values = study_fixture()
    control = values["controls"][0]
    values["controls"] = [
        ValidatedReturnSeries(
            control.series_id, control.return_kind, "target_previous_close",
            control.consensus_manifest_bytes,
        )
    ]
    with pytest.raises(LeadLagError, match="timing differs"):
        build_semiconductor_lead_lag(**values)


def test_source_close_control_unused_tail_round_trips() -> None:
    values = study_fixture()
    control_document = json.loads(
        values["controls"][0].consensus_manifest_bytes
    )
    original_dates = control_document["calendar"]["sessions"]
    extended_dates = original_dates + sessions(original_dates[-1], 3)[1:]
    original_closes = np.asarray(
        [bar["close"] for bar in control_document["bars"]], dtype=float
    )
    extended_closes = np.concatenate(
        [original_closes, original_closes[-1] * np.asarray([1.001, 1.002])]
    )
    values["controls"] = [
        ValidatedReturnSeries(
            "us_market_control",
            "control_return",
            "source_close",
            manifest_bytes(
                symbol="SPY",
                market="US",
                calendar="XNYS",
                currency="USD",
                dates=extended_dates,
                closes=extended_closes,
            ),
        )
    ]
    values["session_map"] = build_verified_session_map(
        "US",
        "CN",
        schedule(extended_dates, market="US"),
        values["session_map"].target_schedule,
    )

    manifest = build_semiconductor_lead_lag(**values).to_manifest()
    assert (
        manifest["inputs"]["controls"][0]["last_observation_session"]
        > manifest["estimation_session"]
    )
    assert LeadLagCalculation(manifest).to_manifest() == manifest


def test_effect_decay_expiry_uses_future_target_schedule_and_boundary_is_inclusive() -> None:
    current = build().to_manifest()
    identified = [item for item in current["results"] if item["expiry"]["expiry_session"]]
    assert identified
    expiry = identified[0]["expiry"]["expiry_session"]
    on_boundary = build(evaluation_session=expiry).to_manifest()
    matching = next(
        item for item in on_boundary["results"]
        if item["window_sessions"] == identified[0]["window_sessions"]
        and item["outcome"] == identified[0]["outcome"]
        and item["horizon_sessions"] == identified[0]["horizon_sessions"]
    )
    assert matching["expiry"]["expired"] is False
    mapping = study_fixture()["session_map"]
    expiry_position = [s.session for s in mapping.target_schedule].index(expiry)
    later = mapping.target_schedule[expiry_position + 1].session
    expired = build(evaluation_session=later).to_manifest()
    matching = next(
        item for item in expired["results"]
        if item["window_sessions"] == identified[0]["window_sessions"]
        and item["outcome"] == identified[0]["outcome"]
        and item["horizon_sessions"] == identified[0]["horizon_sessions"]
    )
    assert matching["status"] == "ABSTAIN"
    assert matching["statistical_status"] == "expired"


def test_invalidation_is_edge_scoped_post_estimation_and_resolvable() -> None:
    values = study_fixture()
    mapping = values["session_map"]
    estimate_position = [s.session for s in mapping.target_schedule].index(values["estimation_session"])
    effective = mapping.target_schedule[estimate_position + 1].session
    resolved = mapping.target_schedule[estimate_position + 3].session
    flag = InvalidationFlag(
        "methodology_change", effective, "d" * 64,
        "us_semiconductor_driver", "ashare_semiconductor_target", resolved,
    )
    values.update(evaluation_session=effective, invalidations=[flag])
    invalidated = build_semiconductor_lead_lag(**values).to_manifest()
    assert all(item["publication_status"] == "ABSTAIN" for item in invalidated["results"])
    assert all(item["status"] == "ABSTAIN" for item in invalidated["results"])
    assert all(
        item["statistical_status"] == "invalidated"
        for item in invalidated["results"]
    )

    values["evaluation_session"] = resolved
    resolved_output = build_semiconductor_lead_lag(**values).to_manifest()
    assert all(
        item["statistical_status"] != "invalidated"
        for item in resolved_output["results"]
    )

    values = study_fixture()
    values["invalidations"] = [
        InvalidationFlag(
            "unrelated", effective, "e" * 64,
            "other_driver", "ashare_semiconductor_target",
        )
    ]
    with pytest.raises(LeadLagError, match="different association"):
        build_semiconductor_lead_lag(**values)

    values = study_fixture()
    historical = values["session_map"].target_schedule[-20].session
    values["invalidations"] = [
        InvalidationFlag(
            "unresolved_reclassification", historical, "f" * 64,
            "us_semiconductor_driver", "ashare_semiconductor_target",
        )
    ]
    unresolved = build_semiconductor_lead_lag(**values).to_manifest()
    assert all(item["status"] == "ABSTAIN" for item in unresolved["results"])
    assert all(
        item["statistical_status"] == "invalidated"
        for item in unresolved["results"]
    )


def test_auxiliary_schools_cannot_promote_and_falsification_can_only_demote() -> None:
    baseline = build().to_manifest()
    positive = auxiliary(
        "fundamental_supply_chain", "corroborates", baseline["evaluation_session"]
    )
    with_positive = build(auxiliary_diagnostics=[positive]).to_manifest()
    for before, after in zip(baseline["results"], with_positive["results"]):
        assert before["historical_primary_status"] == after["historical_primary_status"]
        if before["publication_status"] == "ABSTAIN":
            assert after["publication_status"] == "ABSTAIN"

    negative = auxiliary(
        "technical_regime", "falsifies", baseline["evaluation_session"]
    )
    with_negative = build(auxiliary_diagnostics=[negative]).to_manifest()
    assert all(item["publication_status"] == "ABSTAIN" for item in with_negative["results"])
    assert any("auxiliary_falsification" in item["reason_codes"] for item in with_negative["results"])


def test_microstructure_requires_validated_intraday_coverage() -> None:
    values = study_fixture()
    diagnostic = auxiliary(
        "microstructure", "neutral", values["estimation_session"],
        validated_intraday_data=True,
        intraday_session_count=59,
        intraday_observation_count=10_000,
    )
    values["auxiliary_diagnostics"] = [diagnostic]
    with pytest.raises(LeadLagError, match="lacks validated intraday coverage"):
        build_semiconductor_lead_lag(**values)


def test_auxiliary_status_or_counts_cannot_be_forged_outside_artifact() -> None:
    values = study_fixture()
    valid = auxiliary(
        "technical_regime", "neutral", values["estimation_session"]
    )
    forged = AuxiliaryDiagnostic(
        valid.family_id,
        "corroborates",
        valid.as_of_session,
        valid.artifact_sha256,
        valid.artifact_bytes,
    )
    values["auxiliary_diagnostics"] = [forged]
    with pytest.raises(LeadLagError, match="differ from artifact"):
        build_semiconductor_lead_lag(**values)


def test_calibration_never_claims_current_shock_application_or_eligibility() -> None:
    calculation = build()
    manifest = calculation.to_manifest()
    assert manifest["artifact_type"] == "historical_calibration"
    assert manifest["current_shock_application"] is None
    assert calculation.status == "ABSTAIN"
    assert calculation.statistical_status == manifest["statistical_status"]
    assert manifest["status"] == "ABSTAIN"
    assert manifest["publication_status"] == "ABSTAIN"
    assert all(item["status"] == "ABSTAIN" for item in manifest["results"])
    assert all(item["publication_status"] == "ABSTAIN" for item in manifest["results"])
    assert all("current_shock_not_applied" in item["reason_codes"] for item in manifest["results"])
    assert manifest["method"]["publication_requires_external_global_family_adjustment"] is True


def test_caller_hashes_and_self_consistent_schedules_never_open_publication() -> None:
    values = study_fixture()
    values["config"] = dataclasses.replace(
        values["config"],
        global_search_family_sha256="d" * 64,
        preregistration_sha256="e" * 64,
    )
    # These are bindings supplied by the caller, not proof that an external
    # preregistration or exchange-calendar audit occurred.
    with_hash_assertions = build_semiconductor_lead_lag(**values)
    accepted = [
        item
        for item in with_hash_assertions.manifest["results"]
        if item["statistical_status"] == "accepted"
    ]
    assert accepted  # exercise the dangerous historical-statistical pass branch
    assert with_hash_assertions.statistical_status == "accepted_associations"
    assert with_hash_assertions.status == "ABSTAIN"
    assert with_hash_assertions.manifest["current_shock_application"] is None
    assert with_hash_assertions.manifest["summary"]["publication_status"] == "ABSTAIN"
    assert all(
        item["status"] == item["publication_status"] == "ABSTAIN"
        for item in accepted
    )

    values = study_fixture()
    source = tuple(
        VenueSession(item.session, item.open_at, item.close_at)
        for item in values["session_map"].source_schedule
    )
    target = tuple(
        VenueSession(item.session, item.open_at, item.close_at)
        for item in values["session_map"].target_schedule
    )
    # A caller can construct internally consistent tuples.  Rebuilding proves
    # deterministic mapping only; it does not make those tuples an audited
    # exchange-calendar artifact, so publication remains closed.
    values["session_map"] = build_verified_session_map("US", "CN", source, target)
    with_caller_schedule = build_semiconductor_lead_lag(**values)
    assert with_caller_schedule.status == "ABSTAIN"
    assert with_caller_schedule.manifest["provenance_boundary"].endswith(
        "accepted_manifests_and_schedules"
    )


def test_result_wrapper_rejects_action_postures_and_detaches_caller_mapping() -> None:
    manifest = build().to_manifest()
    detached = LeadLagCalculation(manifest)
    manifest["status"] = "BUY"
    manifest["results"][0]["publication_status"] = "SELL"
    assert detached.status == "ABSTAIN"
    assert detached.manifest["results"][0]["publication_status"] == "ABSTAIN"

    forged = detached.to_manifest()
    forged["status"] = "BUY"
    with pytest.raises(LeadLagError, match="publication posture must remain ABSTAIN"):
        LeadLagCalculation(forged)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest.__setitem__("execute_order", "BUY"),
            "strict contract",
        ),
        (
            lambda manifest: manifest["results"][0].__setitem__(
                "recommended_action", "BUY"
            ),
            "strict contract",
        ),
        (
            lambda manifest: manifest["results"][0].__setitem__(
                "standardized_driver_coefficient", 0.0
            ),
            "standardized coefficient sign is inconsistent",
        ),
        (
            lambda manifest: manifest["results"][0].__setitem__(
                "statistical_status", "accepted"
            ),
            "statistical_status is inconsistent",
        ),
        (
            lambda manifest: manifest["results"][0].__setitem__(
                "historical_primary_status", "accepted"
            ),
            "historical_primary_status is inconsistent",
        ),
        (
            lambda manifest: manifest["summary"]["statistical_counts"].__setitem__(
                "accepted", 45
            ),
            "statistical_counts do not match results",
        ),
        (
            lambda manifest: manifest.__setitem__(
                "statistical_status", "accepted_associations"
            ),
            "overall statistical_status does not match results",
        ),
    ],
)
def test_result_wrapper_rejects_resealed_historical_tampering(
    mutate, message: str
) -> None:
    manifest = build().to_manifest()
    mutate(manifest)
    reseal_calculation(manifest)

    with pytest.raises(LeadLagError, match=message):
        LeadLagCalculation(manifest)


@pytest.mark.parametrize(
    "path",
    [
        ("method",),
        ("config",),
        ("runtime",),
        ("inputs",),
        ("inputs", "driver"),
        ("inputs", "session_map"),
        ("inputs", "invalidations"),
        ("family",),
        ("results", 0, "oos_comparison"),
        ("results", 0, "primary_gates"),
        ("summary",),
        ("summary", "statistical_counts"),
    ],
)
def test_result_wrapper_rejects_resealed_unknown_fields_at_every_layer(path) -> None:
    manifest = build().to_manifest()
    target = manifest
    for key in path:
        target = target[key]
    target["buy_signal"] = True
    reseal_calculation(manifest)

    with pytest.raises(LeadLagError, match="strict contract"):
        LeadLagCalculation(manifest)


def test_result_wrapper_rejects_resealed_numeric_and_family_drift() -> None:
    baseline = build().to_manifest()

    def rejects(mutate, message: str) -> None:
        manifest = copy.deepcopy(baseline)
        mutate(manifest)
        reseal_calculation(manifest)
        with pytest.raises(LeadLagError, match=message):
            LeadLagCalculation(manifest)

    rejects(
        lambda manifest: manifest["results"][0].__setitem__(
            "driver_coefficient",
            manifest["results"][0]["driver_coefficient"] + 1e-6,
        ),
        "same-design outcome coefficient",
    )
    rejects(
        lambda manifest: manifest["results"][0].__setitem__(
            "raw_probability",
            manifest["results"][0]["raw_probability"] + 1e-6,
        ),
        "bootstrap probability grid",
    )
    rejects(
        lambda manifest: manifest["results"][0].__setitem__(
            "fdr_q_value",
            manifest["results"][0]["fdr_q_value"] - 1e-9,
        ),
        "family-wide FDR values are inconsistent",
    )
    rejects(
        lambda manifest: manifest["results"][0]["primary_gates"].__setitem__(
            "sample_size",
            not manifest["results"][0]["primary_gates"]["sample_size"],
        ),
        "primary_gates are inconsistent",
    )
    rejects(
        lambda manifest: manifest["results"][0]["reason_codes"].append(
            "buy_signal_confirmed"
        ),
        "reason_codes are inconsistent",
    )
    rejects(
        lambda manifest: manifest["results"][0].__setitem__(
            "base_first_session", "1900-01-01"
        ),
        "observation sessions are inconsistent",
    )
    rejects(
        lambda manifest: manifest["results"][0]["structural_break"].__setitem__(
            "split_observation",
            manifest["results"][0]["structural_break"]["split_observation"] + 1,
        ),
        "structural-break split is inconsistent",
    )
    rejects(
        lambda manifest: manifest["results"][0]["oos_comparison"].__setitem__(
            "prediction_count", manifest["results"][0]["usable_observations"]
        ),
        "OOS prediction_count is inconsistent",
    )

    def drift_half_life(manifest: dict) -> None:
        for result in manifest["results"][:5]:
            current = result["effect_decay"]["estimated_half_life_sessions"]
            assert current is not None
            result["effect_decay"]["estimated_half_life_sessions"] = current + 0.01
            result["expiry"]["estimated_half_life_sessions"] = current + 0.01

    rejects(drift_half_life, "effect-decay values are inconsistent")

    def drift_expiry(manifest: dict) -> None:
        expiry = manifest["results"][0]["expiry"]
        for key in (
            "recalibration_session",
            "half_life_session",
            "maximum_validity_session",
            "expiry_session",
        ):
            expiry[key] = "2099-01-01"
        expiry["expired"] = False
        expiry["not_expired_at_historical_evaluation"] = True

    rejects(drift_expiry, "expiry does not match the target schedule")

    rejects(
        lambda manifest: manifest["results"].append(
            copy.deepcopy(manifest["results"][0])
        ),
        "results are invalid",
    )
    rejects(
        lambda manifest: manifest["results"].pop(),
        "results are invalid",
    )
    rejects(
        lambda manifest: manifest["results"].__setitem__(
            1, copy.deepcopy(manifest["results"][0])
        ),
        "do not exactly cover the fixed family",
    )


def test_result_wrapper_rejects_resealed_component_hash_drift() -> None:
    baseline = build().to_manifest()

    for path in (
        ("method_sha256",),
        ("runtime_sha256",),
        ("config_sha256",),
        ("inputs", "session_map", "session_map_sha256"),
        ("inputs", "invalidations", "sha256"),
        ("inputs", "auxiliary_diagnostics", "sha256"),
    ):
        manifest = copy.deepcopy(baseline)
        target = manifest
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = "0" * 64
        reseal_calculation(manifest)
        with pytest.raises(LeadLagError, match="does not match"):
            LeadLagCalculation(manifest)


def test_result_wrapper_rejects_resealed_same_economic_driver_and_target() -> None:
    manifest = build().to_manifest()
    driver = manifest["inputs"]["driver"]
    target = manifest["inputs"]["target_gap"]
    for key in ("asset_id", "market", "calendar"):
        driver[key] = target[key]
    session_map = manifest["inputs"]["session_map"]
    session_map["source_market"] = target["market"]
    session_map["session_map_sha256"] = component_digest(
        "quantkit.semiconductor-session-map.v1",
        {
            "source_market": session_map["source_market"],
            "target_market": session_map["target_market"],
            "source_schedule_sha256": session_map["source_schedule_sha256"],
            "target_schedule_sha256": session_map["target_schedule_sha256"],
            "rule": session_map["rule"],
            "links": session_map["links"],
            "excluded_collisions": session_map["excluded_collisions"],
            "unmatched_source_closes": session_map["unmatched_source_closes"],
        },
    )
    reseal_calculation(manifest)

    with pytest.raises(LeadLagError, match="different economic series"):
        LeadLagCalculation(manifest)


@pytest.mark.parametrize(
    "field_path",
    [
        ("driver_coefficient",),
        ("adjacent_window_drift_check", "driver_coefficient"),
        ("structural_break", "first_coefficient"),
        ("structural_break", "second_coefficient"),
    ],
)
def test_result_wrapper_rejects_resealed_outcome_coefficient_drift(
    field_path: tuple[str, ...],
) -> None:
    manifest = build().to_manifest()
    total_members = [
        item
        for item in manifest["results"]
        if item["window_sessions"] == 63 and item["outcome"] == "total"
    ]
    for item in total_members:
        target = item
        for key in field_path[:-1]:
            target = target[key]
        value = target[field_path[-1]]
        assert value is not None
        target[field_path[-1]] = value * 2.0
    if field_path == ("driver_coefficient",):
        # Scaling the entire horizon response preserves its half-life, so only
        # the same-design cross-outcome identity can reject this attack.
        for item in total_members:
            assert item["effect_decay"] == total_members[0]["effect_decay"]
    reseal_calculation(manifest)

    expected_message = (
        "same-design standardized coefficient scales"
        if field_path == ("driver_coefficient",)
        else "same-design outcome coefficient"
    )
    with pytest.raises(LeadLagError, match=expected_message):
        LeadLagCalculation(manifest)


def test_same_design_coefficient_check_allows_legitimate_cancellation_round_trip() -> None:
    values = study_fixture(seed=7)
    driver_document = json.loads(values["driver"].consensus_manifest_bytes)
    driver_closes = np.asarray(
        [bar["close"] for bar in driver_document["bars"]], dtype=float
    )
    driver_returns = np.concatenate(
        [[0.0], np.log(driver_closes[1:] / driver_closes[:-1])]
    )
    target_document = json.loads(
        values["target_gap"].consensus_manifest_bytes
    )
    target_dates = target_document["calendar"]["sessions"]
    rng = np.random.default_rng(771)
    total_returns = np.concatenate(
        [[0.0], rng.normal(0.0, 0.002, len(target_dates) - 1)]
    )
    gap_returns = 10_000.0 * driver_returns
    target_closes = np.empty(len(target_dates))
    target_closes[0] = 100.0
    for position in range(1, len(target_dates)):
        target_closes[position] = target_closes[position - 1] * math.exp(
            float(total_returns[position])
        )
    payload = manifest_bytes(
        symbol="931743.CSI",
        market="CN",
        calendar="XSHG",
        currency="CNY",
        dates=target_dates,
        closes=target_closes,
        gaps=gap_returns,
    )
    values["target_gap"] = ValidatedReturnSeries(
        "ashare_semiconductor_target", "target_gap", "target_session", payload
    )
    values["target_intraday"] = ValidatedReturnSeries(
        "ashare_semiconductor_target",
        "target_intraday",
        "target_session",
        payload,
    )

    manifest = build_semiconductor_lead_lag(**values).to_manifest()
    assert LeadLagCalculation(manifest).to_manifest() == manifest


def test_same_design_coefficient_check_allows_near_collinear_round_trip() -> None:
    values = study_fixture(seed=7)
    driver_document = json.loads(values["driver"].consensus_manifest_bytes)
    dates = driver_document["calendar"]["sessions"]
    driver_closes = np.asarray(
        [bar["close"] for bar in driver_document["bars"]], dtype=float
    )
    driver_returns = np.log(driver_closes[1:] / driver_closes[:-1])
    rng = np.random.default_rng(991)
    control_returns = driver_returns + 1e-11 * rng.normal(
        size=len(driver_returns)
    )
    control_closes = np.concatenate(
        [[50.0], 50.0 * np.exp(np.cumsum(control_returns))]
    )
    payload = manifest_bytes(
        symbol="SPY",
        market="US",
        calendar="XNYS",
        currency="USD",
        dates=dates,
        closes=control_closes,
    )
    values["controls"] = [
        ValidatedReturnSeries(
            "us_market_control", "control_return", "source_close", payload
        )
    ]

    manifest = build_semiconductor_lead_lag(**values).to_manifest()
    assert LeadLagCalculation(manifest).to_manifest() == manifest


def test_same_design_coefficient_check_handles_constant_outcome_null_shape() -> None:
    values = study_fixture()
    target_document = json.loads(
        values["target_gap"].consensus_manifest_bytes
    )
    target_dates = target_document["calendar"]["sessions"]
    target_closes = np.asarray(
        [bar["close"] for bar in target_document["bars"]], dtype=float
    )
    payload = manifest_bytes(
        symbol="931743.CSI",
        market="CN",
        calendar="XSHG",
        currency="CNY",
        dates=target_dates,
        closes=target_closes,
        gaps=np.zeros(len(target_dates)),
    )
    values["target_gap"] = ValidatedReturnSeries(
        "ashare_semiconductor_target", "target_gap", "target_session", payload
    )
    values["target_intraday"] = ValidatedReturnSeries(
        "ashare_semiconductor_target",
        "target_intraday",
        "target_session",
        payload,
    )
    manifest = build_semiconductor_lead_lag(**values).to_manifest()
    gap_members = [item for item in manifest["results"] if item["outcome"] == "gap"]
    assert all(item["driver_coefficient"] is None for item in gap_members)
    assert LeadLagCalculation(manifest).to_manifest() == manifest

    additive_paths = (
        ("driver_coefficient",),
        ("adjacent_window_drift_check", "driver_coefficient"),
        ("structural_break", "first_coefficient"),
        ("structural_break", "second_coefficient"),
    )
    for path in additive_paths:
        forged = copy.deepcopy(manifest)
        for item in forged["results"]:
            if item["outcome"] != "total":
                continue
            target = item
            for key in path[:-1]:
                target = target[key]
            value = target[path[-1]]
            assert value is not None
            target[path[-1]] = value * 2.0
        reseal_calculation(forged)
        expected_message = (
            "same-design standardized coefficient scales"
            if path == ("driver_coefficient",)
            else "same-design outcome coefficient"
        )
        with pytest.raises(LeadLagError, match=expected_message):
            LeadLagCalculation(forged)

    forged = copy.deepcopy(manifest)
    for item in forged["results"]:
        if item["outcome"] == "total":
            standardized = item["standardized_driver_coefficient"]
            assert standardized is not None
            item["standardized_driver_coefficient"] = standardized * 1_000_000.0
    reseal_calculation(forged)
    with pytest.raises(
        LeadLagError, match="same-design standardized coefficient scales"
    ):
        LeadLagCalculation(forged)

    forged = copy.deepcopy(manifest)
    total = next(
        item
        for item in forged["results"]
        if item["window_sessions"] == 63
        and item["horizon_sessions"] == 1
        and item["outcome"] == "total"
    )
    oos = total["oos_comparison"]
    assert oos["full_mse"] is not None and oos["null_mse"] is not None
    oos["full_mse"] *= 1_000_000.0
    oos["null_mse"] *= 1_000_000.0
    oos["relative_improvement"] = 1.0 - oos["full_mse"] / oos["null_mse"]
    reseal_calculation(forged)
    with pytest.raises(LeadLagError, match="same-design OOS residual norm"):
        LeadLagCalculation(forged)


def test_result_wrapper_rejects_resealed_adjacent_nullability_drift() -> None:
    manifest = build().to_manifest()
    result = manifest["results"][0]
    assert result["driver_coefficient"] is not None
    assert result["adjacent_window_drift_check"]["driver_coefficient"] is not None
    result["adjacent_window_drift_check"]["driver_coefficient"] = None
    result["adjacent_window_drift_check"]["same_direction"] = False
    result["primary_gates"]["adjacent_window_drift"] = False
    if "adjacent_window_drift_not_confirmed" not in result["reason_codes"]:
        result["reason_codes"].append("adjacent_window_drift_not_confirmed")
        result["reason_codes"].sort()
    reseal_calculation(manifest)

    with pytest.raises(LeadLagError, match="adjacent coefficient nullability"):
        LeadLagCalculation(manifest)


def test_result_wrapper_rejects_resealed_structural_nullability_drift() -> None:
    manifest = build().to_manifest()
    result = manifest["results"][0]
    structural = result["structural_break"]
    assert structural["probability"] is not None
    structural.update(
        probability=None,
        first_coefficient=None,
        second_coefficient=None,
        passed=False,
    )
    result["primary_gates"]["no_detected_structural_break"] = False
    if "structural_break_detected" not in result["reason_codes"]:
        result["reason_codes"].append("structural_break_detected")
        result["reason_codes"].sort()
    reseal_calculation(manifest)

    with pytest.raises(
        LeadLagError, match="same-design structural-break nullability"
    ):
        LeadLagCalculation(manifest)


def test_result_wrapper_rejects_resealed_same_design_oos_count_drift() -> None:
    manifest = build().to_manifest()
    result = manifest["results"][0]
    original = result["oos_comparison"]["prediction_count"]
    assert original > manifest["config"]["oos_min_predictions"]
    result["oos_comparison"]["prediction_count"] = original - 1
    reseal_calculation(manifest)

    with pytest.raises(LeadLagError, match="same-design OOS prediction counts"):
        LeadLagCalculation(manifest)


def test_result_wrapper_rejects_resealed_same_design_oos_mse_drift() -> None:
    manifest = build().to_manifest()
    result = next(
        item
        for item in manifest["results"]
        if item["window_sessions"] == 63
        and item["horizon_sessions"] == 1
        and item["outcome"] == "total"
    )
    oos = result["oos_comparison"]
    assert oos["full_mse"] is not None and oos["null_mse"] is not None
    oos["full_mse"] *= 1_000_000.0
    oos["null_mse"] *= 1_000_000.0
    oos["relative_improvement"] = 1.0 - oos["full_mse"] / oos["null_mse"]
    reseal_calculation(manifest)

    with pytest.raises(LeadLagError, match="same-design OOS residual norm"):
        LeadLagCalculation(manifest)


def test_result_wrapper_rejects_resealed_standardized_coefficient_sign() -> None:
    manifest = build().to_manifest()
    result = manifest["results"][0]
    standardized = result["standardized_driver_coefficient"]
    assert standardized is not None and standardized != 0.0
    result["standardized_driver_coefficient"] = -standardized
    reseal_calculation(manifest)

    with pytest.raises(LeadLagError, match="standardized coefficient sign"):
        LeadLagCalculation(manifest)


def test_result_wrapper_rejects_resealed_standardized_coefficient_scale() -> None:
    manifest = build().to_manifest()
    result = next(
        item
        for item in manifest["results"]
        if item["window_sessions"] == 63
        and item["horizon_sessions"] == 1
        and item["outcome"] == "total"
    )
    standardized = result["standardized_driver_coefficient"]
    assert standardized is not None and standardized != 0.0
    result["standardized_driver_coefficient"] = standardized * 1_000_000.0
    reseal_calculation(manifest)

    with pytest.raises(
        LeadLagError, match="same-design standardized coefficient scales"
    ):
        LeadLagCalculation(manifest)


def test_standardized_scale_check_allows_exact_zero_coefficient_round_trip() -> None:
    values = study_fixture()
    driver_document = json.loads(values["driver"].consensus_manifest_bytes)
    source_dates = driver_document["calendar"]["sessions"]
    target_document = json.loads(
        values["target_gap"].consensus_manifest_bytes
    )
    target_dates = target_document["calendar"]["sessions"]
    count = len(target_dates)
    step = math.log(2.0)
    driver_returns = np.zeros(count - 1)
    driver_returns[219] = step
    driver_returns[220] = -step
    driver_closes = np.concatenate(
        [[100.0], 100.0 * np.exp(np.cumsum(driver_returns))]
    )
    target_returns = np.zeros(count)
    target_returns[210] = step
    target_returns[211] = -step
    target_closes = np.empty(count)
    target_closes[0] = 100.0
    for position in range(1, count):
        target_closes[position] = target_closes[position - 1] * math.exp(
            float(target_returns[position])
        )
    driver_payload = manifest_bytes(
        symbol="SOXX",
        market="US",
        calendar="XNYS",
        currency="USD",
        dates=source_dates,
        closes=driver_closes,
    )
    target_payload = manifest_bytes(
        symbol="931743.CSI",
        market="CN",
        calendar="XSHG",
        currency="CNY",
        dates=target_dates,
        closes=target_closes,
        gaps=np.zeros(count),
    )
    values["driver"] = ValidatedReturnSeries(
        "us_semiconductor_driver", "source_return", "source_close", driver_payload
    )
    values["target_gap"] = ValidatedReturnSeries(
        "ashare_semiconductor_target",
        "target_gap",
        "target_session",
        target_payload,
    )
    values["target_intraday"] = ValidatedReturnSeries(
        "ashare_semiconductor_target",
        "target_intraday",
        "target_session",
        target_payload,
    )
    values["controls"] = []
    values["config"] = dataclasses.replace(
        values["config"], required_controls=()
    )

    manifest = build_semiconductor_lead_lag(**values).to_manifest()
    assert any(
        item["driver_coefficient"]
        == item["standardized_driver_coefficient"]
        == 0.0
        for item in manifest["results"]
    )
    assert LeadLagCalculation(manifest).to_manifest() == manifest


def test_structural_nullability_allows_same_design_split_rank_failure() -> None:
    values = study_fixture()
    control_document = json.loads(
        values["controls"][0].consensus_manifest_bytes
    )
    dates = control_document["calendar"]["sessions"]
    rng = np.random.default_rng(12345)
    control_returns = np.concatenate(
        [
            np.zeros(170),
            rng.normal(0.0, 0.006, len(dates) - 1 - 170),
        ]
    )
    control_closes = np.concatenate(
        [[50.0], 50.0 * np.exp(np.cumsum(control_returns))]
    )
    payload = manifest_bytes(
        symbol="SPY",
        market="US",
        calendar="XNYS",
        currency="USD",
        dates=dates,
        closes=control_closes,
    )
    values["controls"] = [
        ValidatedReturnSeries(
            "us_market_control", "control_return", "source_close", payload
        )
    ]

    manifest = build_semiconductor_lead_lag(**values).to_manifest()
    long_window = [
        item for item in manifest["results"] if item["window_sessions"] == 252
    ]
    assert all(item["driver_coefficient"] is not None for item in long_window)
    assert all(
        item["structural_break"]["probability"] is None
        for item in long_window
    )
    assert LeadLagCalculation(manifest).to_manifest() == manifest


def test_short_history_and_unidentified_model_are_valid_quarantines() -> None:
    short_values = study_fixture(count=50)
    short = build_semiconductor_lead_lag(**short_values).to_manifest()
    assert all(item["historical_primary_status"] == "quarantined" for item in short["results"])
    assert all(item["expected_observations"] == 0 for item in short["results"])
    assert LeadLagCalculation(short).to_manifest() == short

    boundary_values = study_fixture(count=63)
    boundary = build_semiconductor_lead_lag(**boundary_values).to_manifest()
    assert LeadLagCalculation(boundary).to_manifest() == boundary

    values = study_fixture()
    driver_document = json.loads(values["driver"].consensus_manifest_bytes)
    source_dates = driver_document["calendar"]["sessions"]
    constant_driver = manifest_bytes(
        symbol="SOXX",
        market="US",
        calendar="XNYS",
        currency="USD",
        dates=source_dates,
        closes=np.full(len(source_dates), 100.0),
    )
    values["driver"] = ValidatedReturnSeries(
        "us_semiconductor_driver",
        "source_return",
        "source_close",
        constant_driver,
    )
    unidentified = build_semiconductor_lead_lag(**values).to_manifest()
    assert all(item["historical_primary_status"] == "quarantined" for item in unidentified["results"])
    assert all("model_not_identified" in item["reason_codes"] for item in unidentified["results"])
    assert LeadLagCalculation(unidentified).to_manifest() == unidentified


def test_low_oos_prediction_branch_round_trips_as_descriptive_history() -> None:
    values = study_fixture()
    values["config"] = dataclasses.replace(
        values["config"], oos_min_predictions=10_000
    )
    manifest = build_semiconductor_lead_lag(**values).to_manifest()
    assert all(item["oos_comparison"]["passed"] is False for item in manifest["results"])
    assert all(item["oos_comparison"]["full_mse"] is None for item in manifest["results"])
    assert LeadLagCalculation(manifest).to_manifest() == manifest

    forged = copy.deepcopy(manifest)
    forged["results"][0]["oos_comparison"]["prediction_count"] = forged[
        "results"
    ][0]["usable_observations"]
    reseal_calculation(forged)
    with pytest.raises(LeadLagError, match="OOS prediction_count is inconsistent"):
        LeadLagCalculation(forged)


def test_config_rejects_unregistered_windows_horizons_and_weak_bootstrap() -> None:
    with pytest.raises(LeadLagError, match="windows must be exactly"):
        LeadLagConfig(windows=(20, 63, 126))
    with pytest.raises(LeadLagError, match="horizons must be exactly"):
        LeadLagConfig(horizons=(1, 2))
    with pytest.raises(LeadLagError, match="bootstrap_repetitions"):
        LeadLagConfig(bootstrap_repetitions=99)


def test_total_horizon_return_telescopes_gap_and_intraday_components() -> None:
    output = build().to_manifest()
    additive_paths = (
        ("driver_coefficient",),
        ("adjacent_window_drift_check", "driver_coefficient"),
        ("structural_break", "first_coefficient"),
        ("structural_break", "second_coefficient"),
    )
    for window in (63, 126, 252):
        for horizon in (1, 2, 3, 4, 5):
            items = {
                item["outcome"]: item for item in output["results"]
                if item["window_sessions"] == window
                and item["horizon_sessions"] == horizon
            }
            assert items["total"]["usable_observations"] == items["gap"]["usable_observations"]
            assert items["total"]["usable_observations"] == items["intraday"]["usable_observations"]
            for path in additive_paths:
                values = []
                for outcome in ("gap", "intraday", "total"):
                    value = items[outcome]
                    for key in path:
                        value = value[key]
                    values.append(value)
                if all(value is None for value in values):
                    continue
                assert sum(value is None for value in values) <= 1
                normalized = [
                    0.0 if value is None else float(value) for value in values
                ]
                scale = max(*(abs(value) for value in normalized), 1.0)
                assert abs(normalized[2] - sum(normalized[:2])) <= max(
                    1e-12, 1e-10 * scale
                )

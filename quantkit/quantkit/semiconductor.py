"""Deterministic semiconductor lead-lag association calculations.

This module is an offline numerical boundary.  It accepts hash-bound return
series that an upstream caller has already admitted, plus explicit actual-
session links.  It performs no acquisition, symbol lookup, calendar inference,
portfolio construction, recommendation, or execution.

The fixed daily family contains 63/126/252-session windows, horizons one
through five, and cumulative opening-gap, intraday, and total log returns.
Outputs deliberately use conditional-association language only.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "quantkit.semiconductor-lead-lag.v1"
METHOD_NAME = "semiconductor_conditional_lead_lag"
METHOD_VERSION = "1.0.0"
CALENDAR_HASH_DOMAIN = "quantkit.ohlcv-session-calendar.v1"
RETURNS_HASH_DOMAIN = "quantkit.validated-returns.v1"
SESSION_MAP_HASH_DOMAIN = "quantkit.semiconductor-session-map.v1"
CONFIG_HASH_DOMAIN = "quantkit.semiconductor-lead-lag-config.v1"
OUTPUT_HASH_DOMAIN = "quantkit.semiconductor-lead-lag.v1"
CONSENSUS_SCHEMA = "quantkit.ohlcv-consensus.v1"
BARS_HASH_DOMAIN = "quantkit.ohlcv-accepted-bars.v1"
CONSENSUS_OUTPUT_HASH_DOMAIN = "quantkit.ohlcv-consensus-manifest.v1"
SCHEDULE_HASH_DOMAIN = "quantkit.venue-session-schedule.v1"
DIAGNOSTIC_HASH_DOMAIN = "quantkit.semiconductor-method-diagnostic.v1"
INVALIDATION_HASH_DOMAIN = "quantkit.semiconductor-invalidations.v1"
METHOD_HASH_DOMAIN = "quantkit.semiconductor-method.v1"
RUNTIME_HASH_DOMAIN = "quantkit.semiconductor-runtime.v1"
MAX_MANIFEST_BYTES = 32 * 1024 * 1024

REQUIRED_WINDOWS = (63, 126, 252)
REQUIRED_HORIZONS = (1, 2, 3, 4, 5)
OUTCOMES = ("gap", "intraday", "total")

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_TEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_INVALIDATION_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)

_CONSENSUS_TOP_FIELDS = frozenset(
    {
        "schema",
        "status",
        "identity",
        "policy",
        "calendar",
        "bars",
        "bars_sha256",
        "sources",
        "diagnostics",
        "output_sha256",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "symbol",
        "market",
        "interval",
        "calendar",
        "currency",
        "price_basis",
        "volume_unit",
    }
)
_BAR_FIELDS = frozenset({"date", "open", "high", "low", "close", "volume"})
_METHOD_FAMILIES = {
    "statistical_econometric_lead_lag": "primary_estimator",
    "factor_risk_model_confounds": "controls_only",
    "event_driven_pit": "labels_and_invalidation_only",
    "fundamental_supply_chain": "corroboration_or_falsification_only",
    "technical_regime": "regime_diagnostic_or_falsification_only",
    "microstructure": "intraday_diagnostic_or_falsification_only",
}
_AUXILIARY_FAMILIES = frozenset(
    {"fundamental_supply_chain", "technical_regime", "microstructure"}
)
_AUXILIARY_STATUSES = frozenset({"corroborates", "neutral", "falsifies"})
_TIMING_DOMAINS = frozenset(
    {"source_close", "target_session", "target_previous_close"}
)

_CALCULATION_TOP_FIELDS = frozenset(
    {
        "schema",
        "status",
        "statistical_status",
        "artifact_type",
        "publication_status",
        "current_shock_application",
        "scope",
        "provenance_boundary",
        "method",
        "method_sha256",
        "runtime",
        "runtime_sha256",
        "config",
        "config_sha256",
        "estimation_session",
        "evaluation_session",
        "inputs",
        "family",
        "results",
        "summary",
        "output_sha256",
    }
)
_METHOD_FIELDS = frozenset(
    {
        "name",
        "version",
        "estimator",
        "inference",
        "multiplicity",
        "family_scope",
        "publication_requires_external_global_family_adjustment",
        "method_families",
    }
)
_METHOD_FAMILY_FIELDS = frozenset({"family_id", "role"})
_RUNTIME_FIELDS = frozenset({"code_sha256", "environment_sha256"})
_CONTROL_SPEC_FIELDS = frozenset(
    {"series_id", "role", "timing_domain", "definition_sha256"}
)
_INPUT_FIELDS = frozenset(
    {
        "driver",
        "target_gap",
        "target_intraday",
        "controls",
        "session_map",
        "invalidations",
        "auxiliary_diagnostics",
    }
)
_LINEAGE_FIELDS = frozenset(
    {
        "series_id",
        "asset_id",
        "market",
        "calendar",
        "currency",
        "price_basis",
        "return_kind",
        "timing_domain",
        "return_basis",
        "status",
        "first_observation_session",
        "last_observation_session",
        "observation_count",
        "consensus_output_sha256",
        "manifest_bytes_sha256",
        "bars_sha256",
        "calendar_sessions_sha256",
        "returns_sha256",
    }
)
_CONTROL_LINEAGE_FIELDS = _LINEAGE_FIELDS | frozenset(
    {"role", "definition_sha256"}
)
_SESSION_MAP_FIELDS = frozenset(
    {
        "source_market",
        "target_market",
        "source_schedule",
        "target_schedule",
        "source_schedule_sha256",
        "target_schedule_sha256",
        "rule",
        "links",
        "excluded_collisions",
        "unmatched_source_closes",
        "session_map_sha256",
    }
)
_VENUE_SESSION_FIELDS = frozenset({"session", "open_at", "close_at"})
_SESSION_LINK_FIELDS = frozenset(
    {"source_session", "source_close_at", "target_session", "target_open_at"}
)
_EXCLUDED_COLLISION_FIELDS = _SESSION_LINK_FIELDS | frozenset({"reason_codes"})
_UNMATCHED_CLOSE_FIELDS = frozenset(
    {"source_session", "source_close_at", "reason_codes"}
)
_HASHED_RECORDS_FIELDS = frozenset({"records", "sha256"})
_INVALIDATION_FIELDS = frozenset(
    {
        "code",
        "effective_session",
        "resolved_session",
        "driver_series_id",
        "target_series_id",
        "evidence_sha256",
    }
)
_AUXILIARY_RECORD_FIELDS = frozenset(
    {
        "family_id",
        "role",
        "status",
        "as_of_session",
        "artifact_sha256",
        "artifact_bytes_sha256",
        "validated_intraday_data",
        "intraday_session_count",
        "intraday_observation_count",
    }
)
_FAMILY_FIELDS = frozenset(
    {
        "study_family_id",
        "primary_driver_series_id",
        "primary_target_series_id",
        "windows",
        "horizons",
        "outcomes",
        "test_count",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "window_sessions",
        "outcome",
        "horizon_sessions",
        "base_first_session",
        "base_last_session",
        "expected_observations",
        "usable_observations",
        "required_observations",
        "parameter_count",
        "driver_coefficient",
        "standardized_driver_coefficient",
        "bootstrap_interval_95",
        "raw_probability",
        "fdr_q_value",
        "adjacent_window_drift_check",
        "oos_comparison",
        "structural_break",
        "effect_decay",
        "expiry",
        "primary_gates",
        "context_gates",
        "historical_primary_status",
        "statistical_status",
        "status",
        "publication_status",
        "reason_codes",
    }
)
_ADJACENT_FIELDS = frozenset(
    {"shift_sessions", "driver_coefficient", "same_direction"}
)
_OOS_FIELDS = frozenset(
    {
        "passed",
        "embargo_sessions",
        "prediction_count",
        "full_mse",
        "null_mse",
        "relative_improvement",
    }
)
_STRUCTURAL_BREAK_FIELDS = frozenset(
    {
        "passed",
        "method",
        "split_observation",
        "probability",
        "first_coefficient",
        "second_coefficient",
    }
)
_EFFECT_DECAY_FIELDS = frozenset(
    {"basis", "estimated_half_life_sessions"}
)
_EXPIRY_FIELDS = frozenset(
    {
        "not_expired_at_historical_evaluation",
        "estimated_half_life_sessions",
        "recalibration_session",
        "half_life_session",
        "maximum_validity_session",
        "expiry_session",
        "evaluation_session",
        "expired",
    }
)
_PRIMARY_GATE_FIELDS = frozenset(
    {
        "sample_size",
        "bootstrap_interval",
        "family_wide_fdr",
        "standardized_magnitude",
        "purged_embargoed_oos",
        "adjacent_window_drift",
        "no_detected_structural_break",
        "not_expired_at_historical_evaluation",
        "preregistered_controls",
        "global_search_family_registered",
        "preregistration_bound",
    }
)
_CONTEXT_GATE_FIELDS = frozenset(
    {"no_active_point_in_time_invalidation", "no_auxiliary_falsification"}
)
_SUMMARY_FIELDS = frozenset({"publication_status", "statistical_counts"})
_RESULT_STATUSES = (
    "accepted",
    "descriptive_only",
    "quarantined",
    "expired",
    "invalidated",
)
_PRIMARY_STATUSES = frozenset(
    {"accepted", "descriptive_only", "quarantined", "expired"}
)
_OVERALL_STATUSES = frozenset(
    {
        "accepted_associations",
        "descriptive_only",
        "quarantined",
        "expired",
        "invalidated",
    }
)


class LeadLagError(ValueError):
    """Raised when the calculation contract cannot be evaluated safely."""


@dataclass(frozen=True)
class ValidatedReturnSeries:
    """Return derivation request over exact accepted consensus-manifest bytes.

    ``manifest_bytes_sha256`` binds the exact upstream manifest bytes and is
    intentionally separate from the producer's semantic ``output_sha256``.  A
    caller cannot supply return values or lineage hashes: the kernel strictly
    validates the manifest and deterministically derives them from its bars.
    """

    series_id: str
    return_kind: str
    timing_domain: str
    consensus_manifest_bytes: bytes = field(repr=False, compare=False)


@dataclass(frozen=True)
class SessionLink:
    """One schedule-verified completed close mapped to the next actual open."""

    source_session: str
    source_close_at: str
    target_session: str
    target_open_at: str


@dataclass(frozen=True)
class VenueSession:
    """One explicit actual-session label with regular open and close times."""

    session: str
    open_at: str
    close_at: str


@dataclass(frozen=True)
class VerifiedSessionMap:
    """Detached schedule-derived one-to-one link set and collision audit."""

    source_market: str
    target_market: str
    source_schedule: tuple[VenueSession, ...]
    target_schedule: tuple[VenueSession, ...]
    links: tuple[SessionLink, ...]
    excluded_collisions: tuple[dict[str, Any], ...]
    unmatched_source_closes: tuple[dict[str, Any], ...]
    source_schedule_sha256: str
    target_schedule_sha256: str
    session_map_sha256: str


@dataclass(frozen=True)
class AuxiliaryDiagnostic:
    """Expected bindings for one replay-verified non-primary artifact."""

    family_id: str
    status: str
    as_of_session: str
    artifact_sha256: str
    artifact_bytes: bytes = field(repr=False, compare=False)
    validated_intraday_data: bool = False
    intraday_session_count: int = 0
    intraday_observation_count: int = 0


@dataclass(frozen=True)
class MethodRuntime:
    """Caller-supplied reproducibility bindings for code and environment."""

    code_sha256: str
    environment_sha256: str


@dataclass(frozen=True)
class ControlSpec:
    """Preregistered factor/risk control identity and timing contract."""

    series_id: str
    role: str
    timing_domain: str
    definition_sha256: str


@dataclass(frozen=True)
class InvalidationFlag:
    """Categorical invalidation evidence; it never enters the regression."""

    code: str
    effective_session: str
    evidence_sha256: str
    driver_series_id: str
    target_series_id: str
    resolved_session: str | None = None


@dataclass(frozen=True)
class LeadLagConfig:
    """Preregistered calculation and acceptance policy."""

    config_version: str = "semiconductor-daily-v1"
    windows: tuple[int, ...] = REQUIRED_WINDOWS
    horizons: tuple[int, ...] = REQUIRED_HORIZONS
    min_observations: int = 40
    observations_per_parameter: int = 10
    bootstrap_repetitions: int = 1999
    bootstrap_block_length: int = 5
    interval_alpha: float = 0.05
    fdr_alpha: float = 0.05
    min_standardized_coefficient: float = 0.10
    oos_min_train: int = 40
    oos_min_predictions: int = 10
    oos_min_relative_improvement: float = 0.01
    oos_embargo_sessions: int = 1
    adjacent_shift_sessions: int = 5
    break_alpha: float = 0.05
    recalibration_sessions: int = 5
    max_validity_sessions: int = 5
    random_seed: int = 1729
    method_family_version: str = "semiconductor-multilens-v1"
    study_family_id: str = "ashare-semiconductor-us-close-v1"
    global_search_family_sha256: str | None = None
    preregistration_sha256: str | None = None
    primary_driver_series_id: str = "us_semiconductor_driver"
    primary_target_series_id: str = "ashare_semiconductor_target"
    required_controls: tuple[ControlSpec, ...] = ()
    microstructure_min_sessions: int = 60
    microstructure_min_observations: int = 500

    def __post_init__(self) -> None:
        if tuple(self.windows) != REQUIRED_WINDOWS:
            raise LeadLagError("windows must be exactly 63, 126, and 252 sessions")
        if tuple(self.horizons) != REQUIRED_HORIZONS:
            raise LeadLagError("horizons must be exactly one through five sessions")
        _plain_text(self.config_version, "config_version")
        _plain_text(self.method_family_version, "method_family_version")
        _plain_text(self.study_family_id, "study_family_id")
        _plain_text(self.primary_driver_series_id, "primary_driver_series_id")
        _plain_text(self.primary_target_series_id, "primary_target_series_id")
        if self.global_search_family_sha256 is not None:
            _digest(
                self.global_search_family_sha256,
                "global_search_family_sha256",
            )
        if self.preregistration_sha256 is not None:
            _digest(self.preregistration_sha256, "preregistration_sha256")
        control_ids: set[str] = set()
        for position, spec in enumerate(self.required_controls):
            if not isinstance(spec, ControlSpec):
                raise LeadLagError(
                    f"required_controls[{position}] must be ControlSpec"
                )
            series_id = _plain_text(spec.series_id, "required control series_id")
            _plain_text(spec.role, "required control role")
            if spec.timing_domain not in {
                "source_close",
                "target_previous_close",
            }:
                raise LeadLagError("required control timing_domain is invalid")
            _digest(spec.definition_sha256, "required control definition_sha256")
            if series_id in control_ids:
                raise LeadLagError("required control series_id values must be unique")
            control_ids.add(series_id)
        for name, value, minimum in (
            ("min_observations", self.min_observations, 20),
            ("observations_per_parameter", self.observations_per_parameter, 5),
            ("bootstrap_repetitions", self.bootstrap_repetitions, 199),
            ("bootstrap_block_length", self.bootstrap_block_length, 2),
            ("oos_min_train", self.oos_min_train, 20),
            ("oos_min_predictions", self.oos_min_predictions, 5),
            ("oos_embargo_sessions", self.oos_embargo_sessions, 1),
            ("adjacent_shift_sessions", self.adjacent_shift_sessions, 1),
            ("recalibration_sessions", self.recalibration_sessions, 1),
            ("max_validity_sessions", self.max_validity_sessions, 1),
            ("microstructure_min_sessions", self.microstructure_min_sessions, 1),
            (
                "microstructure_min_observations",
                self.microstructure_min_observations,
                1,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise LeadLagError(f"{name} must be an integer >= {minimum}")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise LeadLagError("random_seed must be an integer")
        for name, value in (
            ("interval_alpha", self.interval_alpha),
            ("fdr_alpha", self.fdr_alpha),
            ("break_alpha", self.break_alpha),
        ):
            if not _finite_number(value) or not 0 < float(value) < 0.5:
                raise LeadLagError(f"{name} must be in (0, 0.5)")
        if (
            not _finite_number(self.min_standardized_coefficient)
            or float(self.min_standardized_coefficient) < 0
        ):
            raise LeadLagError("min_standardized_coefficient must be non-negative")
        if (
            not _finite_number(self.oos_min_relative_improvement)
            or not 0 <= float(self.oos_min_relative_improvement) < 1
        ):
            raise LeadLagError("oos_min_relative_improvement must be in [0, 1)")
        if self.recalibration_sessions > self.max_validity_sessions:
            raise LeadLagError(
                "recalibration_sessions cannot exceed max_validity_sessions"
            )


@dataclass(frozen=True, init=False)
class LeadLagCalculation:
    """Validated immutable historical-calibration output.

    The public type deliberately accepts only a complete, internally
    consistent manifest.  It stores canonical bytes instead of retaining the
    caller's mapping, so later mutation cannot turn the conventional status
    into an action-like value.
    """

    _document_bytes: bytes = field(repr=False, compare=False)

    def __init__(self, document: Mapping[str, Any]) -> None:
        normalized = _validate_calculation_document(document)
        object.__setattr__(self, "_document_bytes", _canonical_json(normalized))

    @property
    def status(self) -> str:
        """Return the fail-closed publication posture.

        ``status`` is intentionally *not* the historical estimator's outcome.
        Consumers that inspect only this conventional property must never turn
        a calibrated association into a current market view by accident.
        """

        return "ABSTAIN"

    @property
    def statistical_status(self) -> str:
        """Return the historical calibration outcome, never an action signal."""

        return str(self.to_manifest()["statistical_status"])

    def to_manifest(self) -> dict[str, Any]:
        document = json.loads(self._document_bytes)
        if not isinstance(document, dict):  # defensive; constructor enforces this
            raise LeadLagError("calculation manifest root must be an object")
        return document

    @property
    def manifest(self) -> dict[str, Any]:
        return self.to_manifest()


@dataclass(frozen=True)
class _Series:
    lineage: Mapping[str, Any]
    values: Mapping[str, float]
    calendar: tuple[str, ...]


@dataclass(frozen=True)
class _Design:
    x: np.ndarray
    y: np.ndarray
    base_positions: np.ndarray
    end_positions: np.ndarray
    base_sessions: tuple[str, ...]
    expected_observations: int


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise LeadLagError("calculation value is not finite canonical JSON") from exc


def _domain_digest(domain: str, value: Any) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\0" + _canonical_json(value)).hexdigest()


def _calculation_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise LeadLagError(f"{label} must be boolean")
    return value


def _calculation_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LeadLagError(f"{label} must be an integer >= {minimum}")
    return value


def _calculation_number(
    value: Any,
    label: str,
    *,
    nullable: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None and nullable:
        return None
    if not _finite_number(value):
        raise LeadLagError(f"{label} must be finite")
    number = float(value)
    if minimum is not None and number < minimum:
        raise LeadLagError(f"{label} is below its allowed range")
    if maximum is not None and number > maximum:
        raise LeadLagError(f"{label} is above its allowed range")
    return number


def _validate_bootstrap_probability(
    value: float, repetitions: int, label: str
) -> None:
    scaled = value * (repetitions + 1)
    if (
        value <= 0.0
        or not math.isclose(scaled, round(scaled), rel_tol=0.0, abs_tol=1e-10)
    ):
        raise LeadLagError(f"{label} is not on the bootstrap probability grid")


def _same_design_coefficients_are_consistent(
    total: float | None,
    left: float | None,
    right: float | None,
) -> bool:
    """Check additive coefficients, including the constant-outcome null shape."""

    values = (left, right, total)
    missing = sum(value is None for value in values)
    if missing == len(values):
        return True
    if missing > 1:
        return False
    normalized = tuple(0.0 if value is None else float(value) for value in values)
    left_value, right_value, total_value = normalized
    scale = max(*(abs(value) for value in normalized), 1.0)
    tolerance = max(1e-12, 1e-10 * scale)
    return abs(total_value - (left_value + right_value)) <= tolerance


def _same_design_residual_norms_are_consistent(
    total_mse: float | None,
    left_mse: float | None,
    right_mse: float | None,
) -> bool:
    """Check additive residual norms, including a constant-outcome null shape."""

    values = (left_mse, right_mse, total_mse)
    missing = sum(value is None for value in values)
    if missing == len(values):
        return True
    if missing > 1:
        return False
    left_value, right_value, total_value = (
        0.0 if value is None else float(value) for value in values
    )
    total_norm = math.sqrt(total_value)
    left_norm = math.sqrt(left_value)
    right_norm = math.sqrt(right_value)
    scale = max(total_norm, left_norm, right_norm, 1.0)
    tolerance = max(1e-12, 1e-10 * scale)
    return bool(
        abs(left_norm - right_norm) - tolerance
        <= total_norm
        <= left_norm + right_norm + tolerance
    )


def _calculation_digest(value: Any, label: str) -> str:
    return _digest(value, label)


def _calculation_session_or_none(value: Any, label: str) -> str | None:
    return None if value is None else _session(value, label)


def _calculation_string_array(
    value: Any, label: str, *, non_empty: bool = False
) -> list[str]:
    if not isinstance(value, list) or (non_empty and not value):
        raise LeadLagError(f"{label} must be an array")
    result = []
    for position, item in enumerate(value):
        if not isinstance(item, str):
            raise LeadLagError(f"{label}[{position}] must be text")
        result.append(item)
    return result


def _calculation_object_array(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise LeadLagError(f"{label} must be an array")
    if any(not isinstance(item, dict) for item in value):
        raise LeadLagError(f"{label} must contain only objects")
    return value


def _validate_calculation_method(document: dict[str, Any]) -> None:
    method = _exact_fields(document.get("method"), _METHOD_FIELDS, "calculation.method")
    expected_method = {
        "name": METHOD_NAME,
        "version": METHOD_VERSION,
        "estimator": "ols_cumulative_local_projection",
        "inference": "stationary_block_residual_bootstrap",
        "multiplicity": "family_wide_benjamini_hochberg",
        "family_scope": "one_locked_driver_target_endpoint_45_tests",
        "publication_requires_external_global_family_adjustment": True,
        "method_families": [
            {"family_id": family_id, "role": role}
            for family_id, role in _METHOD_FAMILIES.items()
        ],
    }
    families = _calculation_object_array(
        method["method_families"], "calculation.method.method_families"
    )
    for position, item in enumerate(families):
        _exact_fields(
            item,
            _METHOD_FAMILY_FIELDS,
            f"calculation.method.method_families[{position}]",
        )
    if method != expected_method:
        raise LeadLagError("calculation method does not match the fixed method contract")
    supplied = _calculation_digest(document.get("method_sha256"), "method_sha256")
    if supplied != _domain_digest(METHOD_HASH_DOMAIN, method):
        raise LeadLagError("calculation method_sha256 does not match")


def _validate_calculation_config(document: dict[str, Any]) -> LeadLagConfig:
    config = document.get("config")
    if not isinstance(config, dict):
        raise LeadLagError("calculation.config must be an object")
    expected_fields = frozenset(field.name for field in dataclasses.fields(LeadLagConfig))
    _exact_fields(config, expected_fields, "calculation.config")
    raw_controls = _calculation_object_array(
        config["required_controls"], "calculation.config.required_controls"
    )
    controls: list[ControlSpec] = []
    for position, item in enumerate(raw_controls):
        _exact_fields(
            item,
            _CONTROL_SPEC_FIELDS,
            f"calculation.config.required_controls[{position}]",
        )
        try:
            controls.append(ControlSpec(**item))
        except TypeError as exc:  # exact fields should make this defensive only
            raise LeadLagError("calculation required control is invalid") from exc
    if (
        not isinstance(config["windows"], list)
        or not isinstance(config["horizons"], list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in [*config["windows"], *config["horizons"]]
        )
    ):
        raise LeadLagError("calculation config windows and horizons must be arrays")
    try:
        normalized = LeadLagConfig(
            **{
                **config,
                "windows": tuple(config["windows"]),
                "horizons": tuple(config["horizons"]),
                "required_controls": tuple(controls),
            }
        )
    except TypeError as exc:
        raise LeadLagError("calculation config values are invalid") from exc
    if _config_payload(normalized) != config:
        raise LeadLagError("calculation config is not canonical")
    supplied = _calculation_digest(document.get("config_sha256"), "config_sha256")
    if supplied != _domain_digest(CONFIG_HASH_DOMAIN, config):
        raise LeadLagError("calculation config_sha256 does not match")
    return normalized


def _validate_calculation_lineage(
    value: Any,
    label: str,
    *,
    expected_fields: frozenset[str] = _LINEAGE_FIELDS,
) -> dict[str, Any]:
    record = _exact_fields(value, expected_fields, label)
    for key in (
        "series_id",
        "asset_id",
        "market",
        "calendar",
        "currency",
        "price_basis",
    ):
        _plain_text(record[key], f"{label}.{key}")
    if record["return_kind"] not in {
        "source_return",
        "control_return",
        "target_gap",
        "target_intraday",
    }:
        raise LeadLagError(f"{label}.return_kind is invalid")
    if record["timing_domain"] not in _TIMING_DOMAINS:
        raise LeadLagError(f"{label}.timing_domain is invalid")
    if record["return_basis"] != "log" or record["status"] != "accepted":
        raise LeadLagError(f"{label} is not an accepted log-return lineage")
    first = _session(record["first_observation_session"], f"{label}.first_observation_session")
    last = _session(record["last_observation_session"], f"{label}.last_observation_session")
    if first > last:
        raise LeadLagError(f"{label} observation sessions are reversed")
    _calculation_int(record["observation_count"], f"{label}.observation_count", 1)
    for key in (
        "consensus_output_sha256",
        "manifest_bytes_sha256",
        "bars_sha256",
        "calendar_sessions_sha256",
        "returns_sha256",
    ):
        _calculation_digest(record[key], f"{label}.{key}")
    if expected_fields == _CONTROL_LINEAGE_FIELDS:
        _plain_text(record["role"], f"{label}.role")
        _calculation_digest(record["definition_sha256"], f"{label}.definition_sha256")
    return record


def _validate_calculation_session_map(value: Any) -> dict[str, Any]:
    label = "calculation.inputs.session_map"
    record = _exact_fields(value, _SESSION_MAP_FIELDS, label)
    source_market = _plain_text(record["source_market"], f"{label}.source_market")
    target_market = _plain_text(record["target_market"], f"{label}.target_market")
    schedules: dict[str, tuple[VenueSession, ...]] = {}
    for key in ("source_schedule", "target_schedule"):
        rows = _calculation_object_array(record[key], f"{label}.{key}")
        if not rows:
            raise LeadLagError(f"{label}.{key} must not be empty")
        parsed = []
        for position, row in enumerate(rows):
            _exact_fields(row, _VENUE_SESSION_FIELDS, f"{label}.{key}[{position}]")
            parsed.append(VenueSession(row["session"], row["open_at"], row["close_at"]))
        schedules[key] = tuple(parsed)
    rebuilt = build_verified_session_map(
        source_market,
        target_market,
        schedules["source_schedule"],
        schedules["target_schedule"],
    )
    for key, fields in (
        ("links", _SESSION_LINK_FIELDS),
        ("excluded_collisions", _EXCLUDED_COLLISION_FIELDS),
        ("unmatched_source_closes", _UNMATCHED_CLOSE_FIELDS),
    ):
        rows = _calculation_object_array(record[key], f"{label}.{key}")
        for position, row in enumerate(rows):
            _exact_fields(row, fields, f"{label}.{key}[{position}]")
            if "reason_codes" in row:
                reasons = _calculation_string_array(
                    row["reason_codes"], f"{label}.{key}[{position}].reason_codes", non_empty=True
                )
                if reasons != sorted(set(reasons)):
                    raise LeadLagError(f"{label}.{key}[{position}].reason_codes are not canonical")
    expected = {
        "source_market": rebuilt.source_market,
        "target_market": rebuilt.target_market,
        "source_schedule": [dataclasses.asdict(item) for item in rebuilt.source_schedule],
        "target_schedule": [dataclasses.asdict(item) for item in rebuilt.target_schedule],
        "source_schedule_sha256": rebuilt.source_schedule_sha256,
        "target_schedule_sha256": rebuilt.target_schedule_sha256,
        "rule": "latest_completed_source_close_to_next_actual_target_open",
        "links": [dataclasses.asdict(item) for item in rebuilt.links],
        "excluded_collisions": list(rebuilt.excluded_collisions),
        "unmatched_source_closes": list(rebuilt.unmatched_source_closes),
        "session_map_sha256": rebuilt.session_map_sha256,
    }
    if record != expected:
        raise LeadLagError("calculation session_map does not match its schedules")
    return record


def _validate_calculation_hashed_records(
    value: Any,
    label: str,
    fields: frozenset[str],
    domain: str,
) -> list[dict[str, Any]]:
    wrapper = _exact_fields(value, _HASHED_RECORDS_FIELDS, label)
    records = _calculation_object_array(wrapper["records"], f"{label}.records")
    for position, record in enumerate(records):
        _exact_fields(record, fields, f"{label}.records[{position}]")
    supplied = _calculation_digest(wrapper["sha256"], f"{label}.sha256")
    if supplied != _domain_digest(domain, records):
        raise LeadLagError(f"{label}.sha256 does not match its records")
    return records


def _validate_calculation_inputs(
    document: dict[str, Any],
    config: LeadLagConfig,
    estimation: str,
    evaluation: str,
) -> dict[str, Any]:
    inputs = _exact_fields(document.get("inputs"), _INPUT_FIELDS, "calculation.inputs")
    driver = _validate_calculation_lineage(inputs["driver"], "calculation.inputs.driver")
    target_gap = _validate_calculation_lineage(inputs["target_gap"], "calculation.inputs.target_gap")
    target_intraday = _validate_calculation_lineage(
        inputs["target_intraday"], "calculation.inputs.target_intraday"
    )
    expected_kinds = (
        (driver, "source_return", "source_close"),
        (target_gap, "target_gap", "target_session"),
        (target_intraday, "target_intraday", "target_session"),
    )
    if any(item["return_kind"] != kind or item["timing_domain"] != timing for item, kind, timing in expected_kinds):
        raise LeadLagError("calculation primary input return roles are invalid")
    if driver["series_id"] != config.primary_driver_series_id:
        raise LeadLagError("calculation driver differs from config")
    if target_gap["series_id"] != config.primary_target_series_id:
        raise LeadLagError("calculation target differs from config")
    if tuple(driver[key] for key in ("asset_id", "market", "calendar")) == tuple(
        target_gap[key] for key in ("asset_id", "market", "calendar")
    ):
        raise LeadLagError(
            "calculation driver and target must be different economic series"
        )
    shared_target = {
        "series_id",
        "asset_id",
        "market",
        "calendar",
        "currency",
        "price_basis",
        "status",
        "last_observation_session",
        "consensus_output_sha256",
        "manifest_bytes_sha256",
        "bars_sha256",
        "calendar_sessions_sha256",
    }
    if any(target_gap[key] != target_intraday[key] for key in shared_target):
        raise LeadLagError("calculation target lineages do not share one accepted manifest")
    controls = _calculation_object_array(inputs["controls"], "calculation.inputs.controls")
    normalized_controls = [
        _validate_calculation_lineage(
            item,
            f"calculation.inputs.controls[{position}]",
            expected_fields=_CONTROL_LINEAGE_FIELDS,
        )
        for position, item in enumerate(controls)
    ]
    required_controls = sorted(config.required_controls, key=lambda item: item.series_id)
    if [item["series_id"] for item in normalized_controls] != [
        item.series_id for item in required_controls
    ]:
        raise LeadLagError("calculation controls differ from config")
    for item, spec in zip(normalized_controls, required_controls):
        if (
            item["return_kind"] != "control_return"
            or item["timing_domain"] != spec.timing_domain
            or item["role"] != spec.role
            or item["definition_sha256"] != spec.definition_sha256
        ):
            raise LeadLagError("calculation control binding differs from config")
    session_map = _validate_calculation_session_map(inputs["session_map"])
    if session_map["source_market"] != driver["market"] or session_map["target_market"] != target_gap["market"]:
        raise LeadLagError("calculation session_map markets differ from inputs")
    source_schedule = [item["session"] for item in session_map["source_schedule"]]
    target_schedule = [item["session"] for item in session_map["target_schedule"]]
    target_session_set = set(target_schedule)
    if estimation not in target_session_set or evaluation not in target_session_set:
        raise LeadLagError("calculation sessions are not scheduled target sessions")

    def lineage_sessions(
        lineage: Mapping[str, Any], schedule: Sequence[str], lineage_label: str
    ) -> tuple[str, ...]:
        try:
            first_position = schedule.index(lineage["first_observation_session"])
            last_position = schedule.index(lineage["last_observation_session"])
        except ValueError as exc:
            raise LeadLagError(
                f"{lineage_label} is not covered by its venue schedule"
            ) from exc
        expected = tuple(schedule[first_position : last_position + 1])
        if (
            not expected
            or len(expected) != lineage["observation_count"]
            or expected[0] != lineage["first_observation_session"]
            or expected[-1] != lineage["last_observation_session"]
        ):
            raise LeadLagError(
                f"{lineage_label} observation range is inconsistent"
            )
        return expected

    driver_sessions = lineage_sessions(
        driver, source_schedule, "calculation.inputs.driver"
    )
    gap_sessions = lineage_sessions(
        target_gap, target_schedule, "calculation.inputs.target_gap"
    )
    intraday_sessions = lineage_sessions(
        target_intraday, target_schedule, "calculation.inputs.target_intraday"
    )
    if (
        target_intraday["last_observation_session"] != estimation
        or target_gap["last_observation_session"] != estimation
        or len(intraday_sessions) != len(gap_sessions) + 1
        or intraday_sessions[1:] != gap_sessions
    ):
        raise LeadLagError("calculation target return ranges are inconsistent")
    if not driver_sessions or driver["market"] != session_map["source_market"]:
        raise LeadLagError("calculation driver schedule binding is inconsistent")
    for position, item in enumerate(normalized_controls):
        schedule = (
            source_schedule
            if item["timing_domain"] == "source_close"
            else target_schedule
        )
        expected_market = (
            session_map["source_market"]
            if item["timing_domain"] == "source_close"
            else session_map["target_market"]
        )
        if item["market"] != expected_market:
            raise LeadLagError(
                f"calculation.inputs.controls[{position}] market is inconsistent"
            )
        lineage_sessions(
            item, schedule, f"calculation.inputs.controls[{position}]"
        )
        if (
            item["timing_domain"] == "target_previous_close"
            and item["last_observation_session"] > estimation
        ):
            raise LeadLagError(
                f"calculation.inputs.controls[{position}] extends past estimation"
            )
    manifest_hashes = [
        driver["manifest_bytes_sha256"],
        target_gap["manifest_bytes_sha256"],
        *(item["manifest_bytes_sha256"] for item in normalized_controls),
    ]
    if len(manifest_hashes) != len(set(manifest_hashes)):
        raise LeadLagError("calculation input manifests are not distinct")
    if any(
        item["bars_sha256"] in {driver["bars_sha256"], target_gap["bars_sha256"]}
        for item in normalized_controls
    ):
        raise LeadLagError("calculation control clones primary accepted bars")
    invalidations = _validate_calculation_hashed_records(
        inputs["invalidations"],
        "calculation.inputs.invalidations",
        _INVALIDATION_FIELDS,
        INVALIDATION_HASH_DOMAIN,
    )
    for position, item in enumerate(invalidations):
        label = f"calculation.inputs.invalidations.records[{position}]"
        if not isinstance(item["code"], str) or _INVALIDATION_RE.fullmatch(item["code"]) is None:
            raise LeadLagError(f"{label}.code is invalid")
        effective = _session(item["effective_session"], f"{label}.effective_session")
        resolved = _calculation_session_or_none(item["resolved_session"], f"{label}.resolved_session")
        if effective not in target_session_set or (
            resolved is not None
            and (resolved not in target_session_set or resolved <= effective)
        ):
            raise LeadLagError(f"{label}.resolved_session is invalid")
        if item["driver_series_id"] != driver["series_id"] or item["target_series_id"] != target_gap["series_id"]:
            raise LeadLagError(f"{label} is scoped to another association")
        _calculation_digest(item["evidence_sha256"], f"{label}.evidence_sha256")
    if invalidations != sorted(
        invalidations,
        key=lambda item: (item["effective_session"], item["code"], item["evidence_sha256"]),
    ):
        raise LeadLagError("calculation invalidations are not canonical")
    if len({(item["effective_session"], item["code"]) for item in invalidations}) != len(invalidations):
        raise LeadLagError("calculation invalidations are duplicated")
    diagnostics = _validate_calculation_hashed_records(
        inputs["auxiliary_diagnostics"],
        "calculation.inputs.auxiliary_diagnostics",
        _AUXILIARY_RECORD_FIELDS,
        DIAGNOSTIC_HASH_DOMAIN,
    )
    if diagnostics != sorted(diagnostics, key=lambda item: item["family_id"]):
        raise LeadLagError("calculation auxiliary diagnostics are not canonical")
    seen_families: set[str] = set()
    for position, item in enumerate(diagnostics):
        label = f"calculation.inputs.auxiliary_diagnostics.records[{position}]"
        family_id = item["family_id"]
        if family_id not in _AUXILIARY_FAMILIES or family_id in seen_families:
            raise LeadLagError(f"{label}.family_id is invalid")
        seen_families.add(family_id)
        if item["role"] != _METHOD_FAMILIES[family_id] or item["status"] not in _AUXILIARY_STATUSES:
            raise LeadLagError(f"{label} role or status is invalid")
        as_of = _session(item["as_of_session"], f"{label}.as_of_session")
        if as_of not in target_session_set or as_of > evaluation:
            raise LeadLagError(f"{label} is not point-in-time available")
        _calculation_digest(item["artifact_sha256"], f"{label}.artifact_sha256")
        _calculation_digest(item["artifact_bytes_sha256"], f"{label}.artifact_bytes_sha256")
        intraday = _calculation_bool(item["validated_intraday_data"], f"{label}.validated_intraday_data")
        sessions = _calculation_int(item["intraday_session_count"], f"{label}.intraday_session_count")
        observations = _calculation_int(item["intraday_observation_count"], f"{label}.intraday_observation_count")
        if family_id == "microstructure":
            if not intraday or sessions < config.microstructure_min_sessions or observations < config.microstructure_min_observations:
                raise LeadLagError(f"{label} lacks validated intraday coverage")
        elif intraday or sessions or observations:
            raise LeadLagError(f"{label} has reserved intraday coverage")
    return inputs


def _overall_statistical_status(counts: Mapping[str, int], total: int) -> str:
    if counts["accepted"]:
        return "accepted_associations"
    if counts["invalidated"] == total:
        return "invalidated"
    if counts["expired"] == total:
        return "expired"
    if counts["quarantined"] == total:
        return "quarantined"
    return "descriptive_only"


def _validate_calculation_result(
    item: Any,
    position: int,
    *,
    config: LeadLagConfig,
    evaluation: str,
    target_observation_sessions: Sequence[str],
    available_base_sessions: frozenset[str],
    active_invalidations: Sequence[Mapping[str, Any]],
    auxiliary_falsified: bool,
) -> tuple[int, str, int, bool]:
    label = f"calculation.results[{position}]"
    result = _exact_fields(item, _RESULT_FIELDS, label)
    window = _calculation_int(result["window_sessions"], f"{label}.window_sessions", 1)
    horizon = _calculation_int(result["horizon_sessions"], f"{label}.horizon_sessions", 1)
    outcome = result["outcome"]
    if window not in REQUIRED_WINDOWS or horizon not in REQUIRED_HORIZONS or outcome not in OUTCOMES:
        raise LeadLagError(f"{label} family member identity is invalid")
    first = _calculation_session_or_none(result["base_first_session"], f"{label}.base_first_session")
    last = _calculation_session_or_none(result["base_last_session"], f"{label}.base_last_session")
    expected = _calculation_int(result["expected_observations"], f"{label}.expected_observations")
    usable = _calculation_int(result["usable_observations"], f"{label}.usable_observations")
    required = _calculation_int(result["required_observations"], f"{label}.required_observations", 1)
    parameter_count = _calculation_int(result["parameter_count"], f"{label}.parameter_count", 1)
    estimation_position = len(target_observation_sessions) - 1
    start_position = estimation_position - window + 1
    if start_position < 0:
        expected_base_sessions: tuple[str, ...] = ()
        expected_count = 0
    else:
        candidate_base_sessions = tuple(
            target_observation_sessions[
                start_position : estimation_position - horizon + 2
            ]
        )
        expected_count = max(0, window - horizon + 1)
        expected_base_sessions = tuple(
            session
            for session in candidate_base_sessions
            if session in available_base_sessions
        )
    base_position_lookup = {
        session: position
        for position, session in enumerate(target_observation_sessions)
    }
    base_positions = [
        base_position_lookup[session] for session in expected_base_sessions
    ]
    end_positions = [position + horizon - 1 for position in base_positions]
    adjacent_end_position = estimation_position - config.adjacent_shift_sessions
    adjacent_start_position = adjacent_end_position - window + 1
    if adjacent_start_position < 0:
        adjacent_usable = 0
    else:
        adjacent_usable = sum(
            session in available_base_sessions
            for session in target_observation_sessions[
                adjacent_start_position : adjacent_end_position - horizon + 2
            ]
        )
    required_train = max(
        config.oos_min_train,
        config.observations_per_parameter * parameter_count,
    )
    maximum_oos_predictions = sum(
        sum(
            prior_end < base_position - config.oos_embargo_sessions
            for prior_end in end_positions[:position]
        )
        >= required_train
        for position, base_position in enumerate(base_positions)
    )
    if (
        expected != expected_count
        or usable != len(expected_base_sessions)
        or parameter_count != 3 + len(config.required_controls)
        or required
        != max(
            config.min_observations,
            config.observations_per_parameter * parameter_count,
        )
    ):
        raise LeadLagError(f"{label} observation counts are inconsistent")
    expected_first = expected_base_sessions[0] if expected_base_sessions else None
    expected_last = expected_base_sessions[-1] if expected_base_sessions else None
    if first != expected_first or last != expected_last:
        raise LeadLagError(f"{label} observation sessions are inconsistent")
    driver = _calculation_number(result["driver_coefficient"], f"{label}.driver_coefficient", nullable=True)
    standardized = _calculation_number(result["standardized_driver_coefficient"], f"{label}.standardized_driver_coefficient", nullable=True)
    raw_probability = _calculation_number(result["raw_probability"], f"{label}.raw_probability", minimum=0.0, maximum=1.0)
    assert raw_probability is not None
    _validate_bootstrap_probability(
        raw_probability, config.bootstrap_repetitions, f"{label}.raw_probability"
    )
    q_value = _calculation_number(result["fdr_q_value"], f"{label}.fdr_q_value", minimum=0.0, maximum=1.0)
    interval = result["bootstrap_interval_95"]
    if interval is not None:
        if not isinstance(interval, list) or len(interval) != 2:
            raise LeadLagError(f"{label}.bootstrap_interval_95 must be a two-number array or null")
        lower = _calculation_number(interval[0], f"{label}.bootstrap_interval_95[0]")
        upper = _calculation_number(interval[1], f"{label}.bootstrap_interval_95[1]")
        if lower is not None and upper is not None and lower > upper:
            raise LeadLagError(f"{label}.bootstrap_interval_95 is reversed")
    adjacent = _exact_fields(result["adjacent_window_drift_check"], _ADJACENT_FIELDS, f"{label}.adjacent_window_drift_check")
    if _calculation_int(adjacent["shift_sessions"], f"{label}.adjacent_window_drift_check.shift_sessions", 1) != config.adjacent_shift_sessions:
        raise LeadLagError(f"{label} adjacent shift differs from config")
    adjacent_coefficient = _calculation_number(adjacent["driver_coefficient"], f"{label}.adjacent_window_drift_check.driver_coefficient", nullable=True)
    same_direction = _calculation_bool(adjacent["same_direction"], f"{label}.adjacent_window_drift_check.same_direction")
    if same_direction != bool(driver is not None and adjacent_coefficient is not None and driver * adjacent_coefficient > 0):
        raise LeadLagError(f"{label} adjacent direction is inconsistent")
    oos = _exact_fields(result["oos_comparison"], _OOS_FIELDS, f"{label}.oos_comparison")
    oos_passed = _calculation_bool(oos["passed"], f"{label}.oos_comparison.passed")
    if _calculation_int(oos["embargo_sessions"], f"{label}.oos_comparison.embargo_sessions", 1) != config.oos_embargo_sessions:
        raise LeadLagError(f"{label} OOS embargo differs from config")
    predictions = _calculation_int(oos["prediction_count"], f"{label}.oos_comparison.prediction_count")
    full_mse = _calculation_number(oos["full_mse"], f"{label}.oos_comparison.full_mse", nullable=True, minimum=0.0)
    null_mse = _calculation_number(oos["null_mse"], f"{label}.oos_comparison.null_mse", nullable=True, minimum=0.0)
    improvement = _calculation_number(oos["relative_improvement"], f"{label}.oos_comparison.relative_improvement", nullable=True)
    if predictions > maximum_oos_predictions:
        raise LeadLagError(f"{label} OOS prediction_count is inconsistent")
    if any(value is None for value in (full_mse, null_mse, improvement)):
        if (
            any(value is not None for value in (full_mse, null_mse, improvement))
            or oos_passed
            or predictions >= config.oos_min_predictions
        ):
            raise LeadLagError(f"{label} OOS nullability is inconsistent")
    else:
        if predictions < config.oos_min_predictions:
            raise LeadLagError(f"{label} OOS prediction_count is inconsistent")
        derived_improvement = 0.0 if null_mse <= 0 else 1.0 - full_mse / null_mse
        if not math.isclose(improvement, derived_improvement, rel_tol=1e-12, abs_tol=1e-15):
            raise LeadLagError(f"{label} OOS improvement is inconsistent")
        expected_pass = predictions >= config.oos_min_predictions and null_mse > 0 and improvement >= config.oos_min_relative_improvement
        if oos_passed != expected_pass:
            raise LeadLagError(f"{label} OOS pass is inconsistent")
    structural = _exact_fields(result["structural_break"], _STRUCTURAL_BREAK_FIELDS, f"{label}.structural_break")
    structural_passed = _calculation_bool(structural["passed"], f"{label}.structural_break.passed")
    if structural["method"] != "stationary_block_bootstrap_fixed_split":
        raise LeadLagError(f"{label} structural-break method is invalid")
    split_observation = _calculation_int(
        structural["split_observation"],
        f"{label}.structural_break.split_observation",
    )
    if split_observation != usable // 2:
        raise LeadLagError(f"{label} structural-break split is inconsistent")
    probability = _calculation_number(structural["probability"], f"{label}.structural_break.probability", nullable=True, minimum=0.0, maximum=1.0)
    first_coefficient = _calculation_number(structural["first_coefficient"], f"{label}.structural_break.first_coefficient", nullable=True)
    second_coefficient = _calculation_number(structural["second_coefficient"], f"{label}.structural_break.second_coefficient", nullable=True)
    if any(value is None for value in (probability, first_coefficient, second_coefficient)):
        if any(value is not None for value in (probability, first_coefficient, second_coefficient)) or structural_passed:
            raise LeadLagError(f"{label} structural-break nullability is inconsistent")
    else:
        _validate_bootstrap_probability(
            probability,
            config.bootstrap_repetitions,
            f"{label}.structural_break.probability",
        )
        expected_structural_pass = bool(
            probability >= config.break_alpha
            and first_coefficient * second_coefficient > 0
            and driver is not None
            and driver * first_coefficient > 0
        )
        if structural_passed != expected_structural_pass:
            raise LeadLagError(f"{label} structural-break pass is inconsistent")
    decay = _exact_fields(result["effect_decay"], _EFFECT_DECAY_FIELDS, f"{label}.effect_decay")
    if decay["basis"] != "marginal_cumulative_horizon_response":
        raise LeadLagError(f"{label} effect-decay basis is invalid")
    half_life = _calculation_number(decay["estimated_half_life_sessions"], f"{label}.effect_decay.estimated_half_life_sessions", nullable=True, minimum=0.0)
    expiry = _exact_fields(result["expiry"], _EXPIRY_FIELDS, f"{label}.expiry")
    expiry_half_life = _calculation_number(expiry["estimated_half_life_sessions"], f"{label}.expiry.estimated_half_life_sessions", nullable=True, minimum=0.0)
    if half_life != expiry_half_life or expiry["evaluation_session"] != evaluation:
        raise LeadLagError(f"{label} expiry binding is inconsistent")
    expiry_sessions = {
        key: _calculation_session_or_none(expiry[key], f"{label}.expiry.{key}")
        for key in ("recalibration_session", "half_life_session", "maximum_validity_session", "expiry_session")
    }
    expired = _calculation_bool(expiry["expired"], f"{label}.expiry.expired")
    not_expired = _calculation_bool(expiry["not_expired_at_historical_evaluation"], f"{label}.expiry.not_expired_at_historical_evaluation")
    if expiry_sessions["expiry_session"] is None:
        if any(value is not None for value in expiry_sessions.values()) or expired or not_expired:
            raise LeadLagError(f"{label} unidentified expiry is inconsistent")
    else:
        candidates = [expiry_sessions[key] for key in ("recalibration_session", "half_life_session", "maximum_validity_session")]
        if any(value is None for value in candidates) or expiry_sessions["expiry_session"] != min(candidates):
            raise LeadLagError(f"{label} expiry session is inconsistent")
        if expired != (evaluation > expiry_sessions["expiry_session"]) or not_expired != (not expired):
            raise LeadLagError(f"{label} expiry state is inconsistent")
    primary_gates = _exact_fields(result["primary_gates"], _PRIMARY_GATE_FIELDS, f"{label}.primary_gates")
    if any(not isinstance(value, bool) for value in primary_gates.values()):
        raise LeadLagError(f"{label}.primary_gates must be boolean")
    numerical_outputs = (driver, standardized, interval)
    if usable < required:
        if any(value is not None for value in numerical_outputs):
            raise LeadLagError(f"{label} has estimates without a sufficient sample")
        identified = False
    elif all(value is not None for value in numerical_outputs):
        identified = True
    elif all(value is None for value in numerical_outputs):
        identified = False
    else:
        raise LeadLagError(f"{label} model-identification outputs are inconsistent")
    if identified and (
        (driver == 0.0) != (standardized == 0.0)
        or (driver != 0.0 and standardized != 0.0 and driver * standardized < 0.0)
    ):
        raise LeadLagError(f"{label} standardized coefficient sign is inconsistent")
    expected_adjacent_estimate = identified and adjacent_usable >= required
    if (adjacent_coefficient is not None) != expected_adjacent_estimate:
        raise LeadLagError(f"{label} adjacent coefficient nullability is inconsistent")
    if not identified and (
        raw_probability != 1.0
        or adjacent_coefficient is not None
        or same_direction
        or predictions != 0
        or any(value is not None for value in (full_mse, null_mse, improvement))
        or structural_passed
        or any(
            value is not None
            for value in (probability, first_coefficient, second_coefficient)
        )
    ):
        raise LeadLagError(f"{label} unidentified-model failure shape is inconsistent")
    expected_gates = {
        "sample_size": identified,
        "bootstrap_interval": bool(interval is not None and (interval[0] > 0 or interval[1] < 0)),
        "family_wide_fdr": bool(q_value <= config.fdr_alpha),
        "standardized_magnitude": bool(standardized is not None and abs(standardized) >= config.min_standardized_coefficient),
        "purged_embargoed_oos": oos_passed,
        "adjacent_window_drift": same_direction,
        "no_detected_structural_break": structural_passed,
        "not_expired_at_historical_evaluation": not_expired,
        "preregistered_controls": bool(config.required_controls),
        "global_search_family_registered": bool(config.global_search_family_sha256),
        "preregistration_bound": bool(config.preregistration_sha256),
    }
    if primary_gates != expected_gates:
        raise LeadLagError(f"{label}.primary_gates are inconsistent")
    context_gates = _exact_fields(result["context_gates"], _CONTEXT_GATE_FIELDS, f"{label}.context_gates")
    expected_context = {
        "no_active_point_in_time_invalidation": not bool(active_invalidations),
        "no_auxiliary_falsification": not auxiliary_falsified,
    }
    if context_gates != expected_context:
        raise LeadLagError(f"{label}.context_gates are inconsistent")
    primary_status = "accepted" if all(expected_gates.values()) else (
        "quarantined" if not identified or expiry_sessions["expiry_session"] is None
        else ("expired" if expired else "descriptive_only")
    )
    if result["historical_primary_status"] not in _PRIMARY_STATUSES or result["historical_primary_status"] != primary_status:
        raise LeadLagError(f"{label}.historical_primary_status is inconsistent")
    statistical_status = "invalidated" if active_invalidations else (
        "descriptive_only" if primary_status == "accepted" and auxiliary_falsified else primary_status
    )
    if result["statistical_status"] not in _RESULT_STATUSES or result["statistical_status"] != statistical_status:
        raise LeadLagError(f"{label}.statistical_status is inconsistent")
    if result["status"] != "ABSTAIN" or result["publication_status"] != "ABSTAIN":
        raise LeadLagError("calculation result publication posture must remain ABSTAIN")
    reasons = _calculation_string_array(result["reason_codes"], f"{label}.reason_codes", non_empty=True)
    expected_reasons: set[str] = {"current_shock_not_applied"}
    if usable < required:
        expected_reasons.add("insufficient_observations")
    elif not identified:
        expected_reasons.add("model_not_identified")
    reason_by_gate = {
        "bootstrap_interval": "bootstrap_interval_includes_zero",
        "family_wide_fdr": "family_wide_fdr_not_passed",
        "standardized_magnitude": "standardized_magnitude_below_threshold",
        "purged_embargoed_oos": "oos_null_not_beaten",
        "adjacent_window_drift": "adjacent_window_drift_not_confirmed",
        "no_detected_structural_break": "structural_break_detected",
        "preregistered_controls": "required_control_contract_empty",
        "global_search_family_registered": "global_search_family_unregistered",
        "preregistration_bound": "preregistration_unbound",
    }
    if identified:
        expected_reasons.update(
            reason
            for gate, reason in reason_by_gate.items()
            if not expected_gates[gate]
        )
    if not not_expired:
        expected_reasons.add("expired" if expired else "expiry_not_identified")
    if active_invalidations:
        expected_reasons.update(
            "invalidation_" + item["code"]
            for item in active_invalidations
        )
    if auxiliary_falsified:
        expected_reasons.add("auxiliary_falsification")
    if reasons != sorted(expected_reasons):
        raise LeadLagError(f"{label}.reason_codes are inconsistent")
    return window, outcome, horizon, identified


def _validate_calculation_document(value: Any) -> dict[str, Any]:
    """Copy and strictly validate one detached historical result document."""

    if not isinstance(value, Mapping):
        raise LeadLagError("calculation manifest root must be a mapping")
    try:
        document = json.loads(_canonical_json(value))
    except json.JSONDecodeError as exc:  # pragma: no cover - encoder output is JSON
        raise LeadLagError("calculation manifest cannot be detached") from exc
    if not isinstance(document, dict):
        raise LeadLagError("calculation manifest root must be an object")
    _exact_fields(document, _CALCULATION_TOP_FIELDS, "calculation manifest")
    if document.get("schema") != SCHEMA:
        raise LeadLagError("calculation manifest schema is invalid")
    if document.get("artifact_type") != "historical_calibration":
        raise LeadLagError("calculation manifest is not a historical calibration")
    if document.get("status") != "ABSTAIN" or document.get(
        "publication_status"
    ) != "ABSTAIN":
        raise LeadLagError("calculation publication posture must remain ABSTAIN")
    if document.get("current_shock_application") is not None:
        raise LeadLagError("historical calibration cannot contain a current shock")
    if document.get("scope") != "conditional_lead_lag_association":
        raise LeadLagError("calculation manifest scope is invalid")
    if document.get("provenance_boundary") != (
        "calculation_integrity_relative_to_upstream_accepted_manifests_and_schedules"
    ):
        raise LeadLagError("calculation manifest provenance boundary is invalid")
    _validate_calculation_method(document)
    runtime = _exact_fields(document.get("runtime"), _RUNTIME_FIELDS, "calculation.runtime")
    for key in _RUNTIME_FIELDS:
        _calculation_digest(runtime[key], f"calculation.runtime.{key}")
    if _calculation_digest(document.get("runtime_sha256"), "runtime_sha256") != _domain_digest(RUNTIME_HASH_DOMAIN, runtime):
        raise LeadLagError("calculation runtime_sha256 does not match")
    config = _validate_calculation_config(document)
    estimation = _session(document.get("estimation_session"), "calculation.estimation_session")
    evaluation = _session(document.get("evaluation_session"), "calculation.evaluation_session")
    if evaluation < estimation:
        raise LeadLagError("calculation evaluation precedes estimation")
    inputs = _validate_calculation_inputs(
        document, config, estimation, evaluation
    )
    family = _exact_fields(document.get("family"), _FAMILY_FIELDS, "calculation.family")
    if family != {
        "study_family_id": config.study_family_id,
        "primary_driver_series_id": config.primary_driver_series_id,
        "primary_target_series_id": config.primary_target_series_id,
        "windows": list(config.windows),
        "horizons": list(config.horizons),
        "outcomes": list(OUTCOMES),
        "test_count": len(REQUIRED_WINDOWS) * len(REQUIRED_HORIZONS) * len(OUTCOMES),
    }:
        raise LeadLagError("calculation family does not match the fixed config")
    results = document.get("results")
    if not isinstance(results, list) or len(results) != family["test_count"]:
        raise LeadLagError("calculation manifest results are invalid")
    invalidations = inputs["invalidations"]["records"]
    active_invalidations = [
        item
        for item in invalidations
        if (
        item["effective_session"] <= evaluation
        and (item["resolved_session"] is None or evaluation < item["resolved_session"])
        )
    ]
    auxiliary_falsified = any(
        item["status"] == "falsifies"
        for item in inputs["auxiliary_diagnostics"]["records"]
    )
    target_gap = inputs["target_gap"]
    target_intraday = inputs["target_intraday"]
    target_schedule = [
        item["session"] for item in inputs["session_map"]["target_schedule"]
    ]
    intraday_first = target_schedule.index(
        target_intraday["first_observation_session"]
    )
    intraday_last = target_schedule.index(
        target_intraday["last_observation_session"]
    )
    target_observation_sessions = tuple(
        target_schedule[intraday_first : intraday_last + 1]
    )
    gap_first = target_schedule.index(target_gap["first_observation_session"])
    gap_last = target_schedule.index(target_gap["last_observation_session"])
    gap_sessions = set(target_schedule[gap_first : gap_last + 1])
    linked_targets = {
        item["target_session"]
        for item in inputs["session_map"]["links"]
    }
    driver = inputs["driver"]
    source_schedule = [
        item["session"] for item in inputs["session_map"]["source_schedule"]
    ]
    driver_first = source_schedule.index(driver["first_observation_session"])
    driver_last = source_schedule.index(driver["last_observation_session"])
    driver_sessions = set(source_schedule[driver_first : driver_last + 1])
    link_source_by_target = {
        item["target_session"]: item["source_session"]
        for item in inputs["session_map"]["links"]
    }
    control_session_sets: list[tuple[str, set[str]]] = []
    for control in inputs["controls"]:
        schedule = (
            source_schedule
            if control["timing_domain"] == "source_close"
            else target_schedule
        )
        first_position = schedule.index(control["first_observation_session"])
        last_position = schedule.index(control["last_observation_session"])
        control_session_sets.append(
            (
                control["timing_domain"],
                set(schedule[first_position : last_position + 1]),
            )
        )
    available_base_sessions: set[str] = set()
    for position, session in enumerate(target_observation_sessions):
        if position == 0 or session not in linked_targets:
            continue
        source_session = link_source_by_target[session]
        if source_session not in driver_sessions:
            continue
        previous_session = target_observation_sessions[position - 1]
        if previous_session not in gap_sessions:
            continue
        if all(
            (
                source_session
                if timing_domain == "source_close"
                else previous_session
            )
            in sessions
            for timing_domain, sessions in control_session_sets
        ):
            available_base_sessions.add(session)
    member_records = [
        _validate_calculation_result(
            item,
            position,
            config=config,
            evaluation=evaluation,
            target_observation_sessions=target_observation_sessions,
            available_base_sessions=frozenset(available_base_sessions),
            active_invalidations=active_invalidations,
            auxiliary_falsified=auxiliary_falsified,
        )
        for position, item in enumerate(results)
    ]
    member_keys = [item[:3] for item in member_records]
    expected_keys = [
        (window, outcome, horizon)
        for window in REQUIRED_WINDOWS
        for outcome in OUTCOMES
        for horizon in REQUIRED_HORIZONS
    ]
    if member_keys != expected_keys:
        raise LeadLagError("calculation results do not exactly cover the fixed family")

    results_by_key = {
        (item["window_sessions"], item["outcome"], item["horizon_sessions"]): item
        for item in results
    }
    identified_by_key = {
        member[:3]: member[3] for member in member_records
    }
    for window in REQUIRED_WINDOWS:
        for horizon in REQUIRED_HORIZONS:
            outcome_results = {
                outcome: results_by_key[(window, outcome, horizon)]
                for outcome in OUTCOMES
            }
            structural_shapes = {
                tuple(
                    outcome_results[outcome]["structural_break"][key] is None
                    for key in (
                        "probability",
                        "first_coefficient",
                        "second_coefficient",
                    )
                )
                for outcome in OUTCOMES
                if identified_by_key[(window, outcome, horizon)]
            }
            if len(structural_shapes) > 1:
                raise LeadLagError(
                    "calculation same-design structural-break nullability "
                    "is inconsistent"
                )
            oos_prediction_counts = {
                int(outcome_results[outcome]["oos_comparison"]["prediction_count"])
                for outcome in OUTCOMES
                if identified_by_key[(window, outcome, horizon)]
            }
            if len(oos_prediction_counts) > 1:
                raise LeadLagError(
                    "calculation same-design OOS prediction counts are inconsistent"
                )
            standardized_scale_variances = {}
            standardized_scale_unknown = False
            for outcome in OUTCOMES:
                outcome_result = outcome_results[outcome]
                coefficient = outcome_result["driver_coefficient"]
                standardized = outcome_result["standardized_driver_coefficient"]
                identified = identified_by_key[(window, outcome, horizon)]
                if not identified:
                    standardized_scale_variances[outcome] = 0.0
                elif coefficient == 0.0 and standardized == 0.0:
                    standardized_scale_unknown = True
                    standardized_scale_variances[outcome] = None
                else:
                    assert coefficient is not None and standardized is not None
                    ratio = float(coefficient) / float(standardized)
                    standardized_scale_variances[outcome] = ratio * ratio
            if (
                not standardized_scale_unknown
                and not _same_design_residual_norms_are_consistent(
                    standardized_scale_variances["total"],
                    standardized_scale_variances["gap"],
                    standardized_scale_variances["intraday"],
                )
            ):
                raise LeadLagError(
                    "calculation same-design standardized coefficient scales "
                    "are inconsistent"
                )
            for mse_field in ("full_mse", "null_mse"):
                mse_values = {
                    outcome: outcome_results[outcome]["oos_comparison"][mse_field]
                    for outcome in OUTCOMES
                }
                if not _same_design_residual_norms_are_consistent(
                    mse_values["total"],
                    mse_values["gap"],
                    mse_values["intraday"],
                ):
                    raise LeadLagError(
                        "calculation same-design OOS residual norm "
                        f"{mse_field} is inconsistent"
                    )
            additive_coefficients = (
                (
                    "driver_coefficient",
                    lambda item: item["driver_coefficient"],
                ),
                (
                    "adjacent_window_drift_check.driver_coefficient",
                    lambda item: item["adjacent_window_drift_check"][
                        "driver_coefficient"
                    ],
                ),
                (
                    "structural_break.first_coefficient",
                    lambda item: item["structural_break"]["first_coefficient"],
                ),
                (
                    "structural_break.second_coefficient",
                    lambda item: item["structural_break"]["second_coefficient"],
                ),
            )
            for field, extract in additive_coefficients:
                gap = extract(outcome_results["gap"])
                intraday = extract(outcome_results["intraday"])
                total = extract(outcome_results["total"])
                if not _same_design_coefficients_are_consistent(
                    total,
                    gap,
                    intraday,
                ):
                    raise LeadLagError(
                        "calculation same-design outcome coefficient "
                        f"{field} is inconsistent"
                    )

    adjusted = _bh_adjust([float(item["raw_probability"]) for item in results])
    if [float(result["fdr_q_value"]) for result in results] != adjusted:
        raise LeadLagError("calculation family-wide FDR values are inconsistent")

    for window in REQUIRED_WINDOWS:
        for outcome in OUTCOMES:
            group = [
                result
                for result in results
                if result["window_sessions"] == window
                and result["outcome"] == outcome
            ]
            group.sort(key=lambda item: item["horizon_sessions"])
            coefficients = [
                item["driver_coefficient"]
                if identified_by_key[
                    (window, outcome, item["horizon_sessions"])
                ]
                else None
                for item in group
            ]
            expected_half_life = _effect_half_life(coefficients)
            expected_expiry = _expiry(
                target_schedule,
                estimation,
                evaluation,
                expected_half_life,
                config,
            )
            for item in group:
                if item["effect_decay"] != {
                    "basis": "marginal_cumulative_horizon_response",
                    "estimated_half_life_sessions": expected_half_life,
                }:
                    raise LeadLagError(
                        "calculation effect-decay values are inconsistent"
                    )
                if item["expiry"] != expected_expiry:
                    raise LeadLagError(
                        "calculation expiry does not match the target schedule"
                    )
    summary = _exact_fields(document.get("summary"), _SUMMARY_FIELDS, "calculation.summary")
    if summary.get("publication_status") != "ABSTAIN":
        raise LeadLagError("calculation summary publication posture must remain ABSTAIN")
    counts = _exact_fields(
        summary.get("statistical_counts"),
        frozenset(_RESULT_STATUSES),
        "calculation.summary.statistical_counts",
    )
    expected_counts = {
        status: sum(item["statistical_status"] == status for item in results)
        for status in _RESULT_STATUSES
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()) or counts != expected_counts:
        raise LeadLagError("calculation summary statistical_counts do not match results")
    expected_overall = _overall_statistical_status(expected_counts, len(results))
    if document.get("statistical_status") not in _OVERALL_STATUSES or document.get("statistical_status") != expected_overall:
        raise LeadLagError("calculation overall statistical_status does not match results")
    output_sha256 = document.get("output_sha256")
    if not isinstance(output_sha256, str) or _DIGEST_RE.fullmatch(output_sha256) is None:
        raise LeadLagError("calculation output_sha256 is invalid")
    unsigned = dict(document)
    unsigned.pop("output_sha256")
    if _domain_digest(OUTPUT_HASH_DOMAIN, unsigned) != output_sha256:
        raise LeadLagError("calculation output_sha256 does not match its manifest")
    return document


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _plain_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or _TEXT_RE.fullmatch(value) is None:
        raise LeadLagError(f"{label} must be a bounded identifier")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise LeadLagError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _session(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise LeadLagError(f"{label} must be an ISO calendar date")
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise LeadLagError(f"{label} must be an ISO calendar date") from exc
    if parsed.isoformat() != value:
        raise LeadLagError(f"{label} must be a canonical ISO calendar date")
    return value


def _timestamp(value: Any, label: str) -> tuple[dt.datetime, str]:
    if not isinstance(value, str) or _RFC3339_RE.fullmatch(value) is None:
        raise LeadLagError(f"{label} must be an RFC 3339 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise LeadLagError(f"{label} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LeadLagError(f"{label} must carry a UTC offset")
    utc = parsed.astimezone(dt.timezone.utc)
    timespec = "microseconds" if utc.microsecond else "seconds"
    rendered = utc.isoformat(timespec=timespec).replace("+00:00", "Z")
    return utc, rendered


def _ordered_sessions(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise LeadLagError(f"{label} must be a session array")
    result = tuple(_session(value, f"{label}[{position}]") for position, value in enumerate(values))
    if not result:
        raise LeadLagError(f"{label} must not be empty")
    if result != tuple(sorted(set(result))):
        raise LeadLagError(f"{label} must be strictly sorted and unique")
    return result


def session_calendar_sha256(sessions: Sequence[str]) -> str:
    """Return the consensus calendar digest after strict session validation."""

    ordered = _ordered_sessions(sessions, "calendar_sessions")
    return _domain_digest(CALENDAR_HASH_DOMAIN, list(ordered))


def validated_returns_sha256(
    sessions: Sequence[str], values: Sequence[float]
) -> str:
    """Return the canonical digest for one session-aligned return vector."""

    ordered = _ordered_sessions(sessions, "sessions")
    if isinstance(values, (str, bytes)) or len(values) != len(ordered):
        raise LeadLagError("values length must match sessions")
    records: list[dict[str, Any]] = []
    for position, (session, value) in enumerate(zip(ordered, values)):
        if not _finite_number(value):
            raise LeadLagError(f"values[{position}] must be finite")
        number = float(value)
        records.append({"session": session, "value": 0.0 if number == 0 else number})
    return _domain_digest(RETURNS_HASH_DOMAIN, records)


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_MANIFEST_BYTES:
        raise LeadLagError(f"{label} must be bounded non-empty bytes")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LeadLagError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise LeadLagError(f"{label} contains non-finite JSON number {value}")

    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise LeadLagError(f"{label} is not strict JSON") from exc
    if not isinstance(document, dict):
        raise LeadLagError(f"{label} root must be an object")
    return document


def _exact_fields(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise LeadLagError(f"{label} fields do not match the strict contract")
    return value


def _consensus_series(request: ValidatedReturnSeries) -> _Series:
    """Strictly validate one accepted consensus manifest and derive returns."""

    if not isinstance(request, ValidatedReturnSeries):
        raise LeadLagError("every return input must be ValidatedReturnSeries")
    series_id = _plain_text(request.series_id, "series_id")
    if request.return_kind not in {
        "source_return",
        "control_return",
        "target_gap",
        "target_intraday",
    }:
        raise LeadLagError(f"{series_id}.return_kind is unsupported")
    if request.timing_domain not in _TIMING_DOMAINS:
        raise LeadLagError(f"{series_id}.timing_domain is unsupported")
    expected_timing = {
        "source_return": {"source_close"},
        "control_return": {"source_close", "target_previous_close"},
        "target_gap": {"target_session"},
        "target_intraday": {"target_session"},
    }[request.return_kind]
    if request.timing_domain not in expected_timing:
        raise LeadLagError(
            f"{series_id}.timing_domain conflicts with return_kind"
        )
    document = _strict_json(
        request.consensus_manifest_bytes, f"{series_id}.consensus_manifest_bytes"
    )
    _exact_fields(document, _CONSENSUS_TOP_FIELDS, f"{series_id}.manifest")
    if document.get("schema") != CONSENSUS_SCHEMA or document.get("status") != "accepted":
        raise LeadLagError(f"{series_id} requires an accepted consensus manifest")
    identity = _exact_fields(
        document.get("identity"), _IDENTITY_FIELDS, f"{series_id}.identity"
    )
    for key in _IDENTITY_FIELDS:
        _plain_text(identity.get(key), f"{series_id}.identity.{key}")
    if identity["interval"] != "1d":
        raise LeadLagError(f"{series_id} consensus interval must be '1d'")
    calendar = document.get("calendar")
    if not isinstance(calendar, dict) or set(calendar) != {
        "name",
        "sessions",
        "sessions_sha256",
    }:
        raise LeadLagError(f"{series_id}.calendar fields are invalid")
    if calendar.get("name") != identity["calendar"]:
        raise LeadLagError(f"{series_id}.calendar name differs from identity")
    sessions = _ordered_sessions(calendar.get("sessions"), f"{series_id}.calendar.sessions")
    calendar_digest = _digest(
        calendar.get("sessions_sha256"), f"{series_id}.calendar.sessions_sha256"
    )
    if calendar_digest != session_calendar_sha256(sessions):
        raise LeadLagError(f"{series_id} calendar digest does not match")
    raw_bars = document.get("bars")
    if not isinstance(raw_bars, list) or len(raw_bars) < 2:
        raise LeadLagError(f"{series_id}.bars must contain at least two records")
    bars: list[dict[str, float | str]] = []
    for position, raw in enumerate(raw_bars):
        bar = _exact_fields(raw, _BAR_FIELDS, f"{series_id}.bars[{position}]")
        date = _session(bar.get("date"), f"{series_id}.bars[{position}].date")
        values = {}
        for column in ("open", "high", "low", "close", "volume"):
            value = bar.get(column)
            if not _finite_number(value):
                raise LeadLagError(f"{series_id}.bars[{position}].{column} must be finite")
            values[column] = float(value)
        if (
            values["open"] <= 0
            or values["high"] <= 0
            or values["low"] <= 0
            or values["close"] <= 0
            or values["volume"] < 0
            or values["high"] < max(values["open"], values["low"], values["close"])
            or values["low"] > min(values["open"], values["high"], values["close"])
        ):
            raise LeadLagError(f"{series_id}.bars[{position}] has impossible geometry")
        if date not in set(sessions):
            raise LeadLagError(f"{series_id}.bars[{position}] is not an actual session")
        bars.append({"date": date, **values})
    dates = tuple(str(bar["date"]) for bar in bars)
    if dates != tuple(sorted(set(dates))):
        raise LeadLagError(f"{series_id}.bars must be sorted and unique")
    if dates != sessions:
        raise LeadLagError(
            f"{series_id}.accepted bars must exactly cover its declared calendar"
        )
    bars_digest = _digest(document.get("bars_sha256"), f"{series_id}.bars_sha256")
    if bars_digest != _domain_digest(BARS_HASH_DOMAIN, bars):
        raise LeadLagError(f"{series_id}.bars_sha256 does not match")
    supplied_output = _digest(
        document.get("output_sha256"), f"{series_id}.output_sha256"
    )
    unsealed = {key: value for key, value in document.items() if key != "output_sha256"}
    if supplied_output != _domain_digest(CONSENSUS_OUTPUT_HASH_DOMAIN, unsealed):
        raise LeadLagError(f"{series_id}.output_sha256 does not match")

    derived_sessions: list[str] = []
    derived_values: list[float] = []
    if request.return_kind in {"target_gap", "target_intraday"}:
        for position, bar in enumerate(bars):
            if request.return_kind == "target_gap":
                if position == 0:
                    continue
                value = math.log(float(bar["open"]) / float(bars[position - 1]["close"]))
            else:
                value = math.log(float(bar["close"]) / float(bar["open"]))
            derived_sessions.append(str(bar["date"]))
            derived_values.append(value)
    else:
        for position in range(1, len(bars)):
            derived_sessions.append(str(bars[position]["date"]))
            derived_values.append(
                math.log(float(bars[position]["close"]) / float(bars[position - 1]["close"]))
            )
    return_digest = validated_returns_sha256(derived_sessions, derived_values)
    manifest_bytes_sha256 = _sha256_bytes(request.consensus_manifest_bytes)
    lineage = {
        "series_id": series_id,
        "asset_id": identity["symbol"],
        "market": identity["market"],
        "calendar": identity["calendar"],
        "currency": identity["currency"],
        "price_basis": identity["price_basis"],
        "return_kind": request.return_kind,
        "timing_domain": request.timing_domain,
        "return_basis": "log",
        "status": "accepted",
        "first_observation_session": derived_sessions[0],
        "last_observation_session": derived_sessions[-1],
        "observation_count": len(derived_sessions),
        "consensus_output_sha256": supplied_output,
        "manifest_bytes_sha256": manifest_bytes_sha256,
        "bars_sha256": bars_digest,
        "calendar_sessions_sha256": calendar_digest,
        "returns_sha256": return_digest,
    }
    return _Series(
        lineage=lineage,
        values=dict(zip(derived_sessions, derived_values)),
        calendar=sessions,
    )


def _normalize_schedule(
    schedule: Sequence[VenueSession], label: str
) -> tuple[tuple[VenueSession, ...], str]:
    if isinstance(schedule, (str, bytes)) or not schedule:
        raise LeadLagError(f"{label} must be a non-empty venue-session array")
    normalized: list[VenueSession] = []
    prior_close: dt.datetime | None = None
    for position, raw in enumerate(schedule):
        if not isinstance(raw, VenueSession):
            raise LeadLagError(f"{label}[{position}] must be VenueSession")
        session = _session(raw.session, f"{label}[{position}].session")
        opened, open_text = _timestamp(raw.open_at, f"{label}[{position}].open_at")
        closed, close_text = _timestamp(raw.close_at, f"{label}[{position}].close_at")
        if opened >= closed:
            raise LeadLagError(f"{label}[{position}] open must precede close")
        if prior_close is not None and opened <= prior_close:
            raise LeadLagError(f"{label} sessions must be strictly non-overlapping")
        prior_close = closed
        normalized.append(VenueSession(session, open_text, close_text))
    labels = tuple(item.session for item in normalized)
    if labels != tuple(sorted(set(labels))):
        raise LeadLagError(f"{label} labels must be strictly sorted and unique")
    records = [dataclasses.asdict(item) for item in normalized]
    return tuple(normalized), _domain_digest(SCHEDULE_HASH_DOMAIN, records)


def build_verified_session_map(
    source_market: str,
    target_market: str,
    source_schedule: Sequence[VenueSession],
    target_schedule: Sequence[VenueSession],
) -> VerifiedSessionMap:
    """Map each completed close to the next actual open, excluding collisions.

    If a target-market holiday causes multiple source closes to map to the same
    target open, only the latest completed source close is retained.  Earlier
    closes are explicit exclusions; no return is carried or aggregated.
    """

    source_market = _plain_text(source_market, "source_market")
    target_market = _plain_text(target_market, "target_market")
    sources, source_digest = _normalize_schedule(source_schedule, "source_schedule")
    targets, target_digest = _normalize_schedule(target_schedule, "target_schedule")
    target_times = [(_timestamp(item.open_at, "target_open_at")[0], item) for item in targets]
    candidates: dict[str, list[tuple[dt.datetime, VenueSession, VenueSession]]] = {}
    for source in sources:
        source_close = _timestamp(source.close_at, "source_close_at")[0]
        target = next((item for opened, item in target_times if opened > source_close), None)
        if target is None:
            continue
        candidates.setdefault(target.session, []).append((source_close, source, target))
    links: list[SessionLink] = []
    excluded: list[dict[str, Any]] = []
    for target_session in sorted(candidates):
        group = sorted(candidates[target_session], key=lambda item: (item[0], item[1].session))
        retained = group[-1]
        for _, source, target in group[:-1]:
            excluded.append(
                {
                    "source_session": source.session,
                    "source_close_at": source.close_at,
                    "target_session": target.session,
                    "target_open_at": target.open_at,
                    "reason_codes": ["target_session_collision"],
                }
            )
        _, source, target = retained
        links.append(
            SessionLink(
                source_session=source.session,
                source_close_at=source.close_at,
                target_session=target.session,
                target_open_at=target.open_at,
            )
        )
    matched_sources = {item.source_session for item in links}
    excluded_sources = {item["source_session"] for item in excluded}
    unmatched = [
        {
            "source_session": item.session,
            "source_close_at": item.close_at,
            "reason_codes": ["no_later_target_open"],
        }
        for item in sources
        if item.session not in matched_sources and item.session not in excluded_sources
    ]
    if not links:
        raise LeadLagError("venue schedules produce no completed-close links")
    payload = {
        "source_market": source_market,
        "target_market": target_market,
        "source_schedule_sha256": source_digest,
        "target_schedule_sha256": target_digest,
        "rule": "latest_completed_source_close_to_next_actual_target_open",
        "links": [dataclasses.asdict(item) for item in links],
        "excluded_collisions": excluded,
        "unmatched_source_closes": unmatched,
    }
    return VerifiedSessionMap(
        source_market=source_market,
        target_market=target_market,
        source_schedule=sources,
        target_schedule=targets,
        links=tuple(links),
        excluded_collisions=tuple(excluded),
        unmatched_source_closes=tuple(unmatched),
        source_schedule_sha256=source_digest,
        target_schedule_sha256=target_digest,
        session_map_sha256=_domain_digest(SESSION_MAP_HASH_DOMAIN, payload),
    )


def _verified_map(value: VerifiedSessionMap) -> VerifiedSessionMap:
    if not isinstance(value, VerifiedSessionMap):
        raise LeadLagError("session_map must be VerifiedSessionMap")
    rebuilt = build_verified_session_map(
        value.source_market,
        value.target_market,
        value.source_schedule,
        value.target_schedule,
    )
    if value != rebuilt:
        raise LeadLagError("session_map does not match its bound venue schedules")
    return rebuilt


def _require_contiguous_schedule_slice(
    calendar: Sequence[str], schedule: Sequence[VenueSession], label: str
) -> None:
    labels = tuple(item.session for item in schedule)
    try:
        first = labels.index(calendar[0])
        last = labels.index(calendar[-1])
    except (ValueError, IndexError) as exc:
        raise LeadLagError(f"{label} is not covered by its venue schedule") from exc
    if tuple(calendar) != labels[first : last + 1]:
        raise LeadLagError(f"{label} omits an actual scheduled session")


def _config_payload(config: LeadLagConfig) -> dict[str, Any]:
    payload = dataclasses.asdict(config)
    payload["windows"] = list(config.windows)
    payload["horizons"] = list(config.horizons)
    payload["required_controls"] = [
        dataclasses.asdict(item) for item in config.required_controls
    ]
    return payload


def _normalize_runtime(runtime: MethodRuntime) -> dict[str, str]:
    if not isinstance(runtime, MethodRuntime):
        raise LeadLagError("runtime must be MethodRuntime")
    return {
        "code_sha256": _digest(runtime.code_sha256, "runtime.code_sha256"),
        "environment_sha256": _digest(
            runtime.environment_sha256, "runtime.environment_sha256"
        ),
    }


def _normalize_diagnostics(
    diagnostics: Sequence[AuxiliaryDiagnostic],
    target_calendar: Sequence[str],
    evaluation_session: str,
    config: LeadLagConfig,
) -> tuple[list[dict[str, Any]], str, bool]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    target_set = set(target_calendar)
    falsified = False
    for position, item in enumerate(diagnostics):
        if not isinstance(item, AuxiliaryDiagnostic):
            raise LeadLagError(
                f"auxiliary_diagnostics[{position}] must be AuxiliaryDiagnostic"
            )
        if item.family_id not in _AUXILIARY_FAMILIES:
            raise LeadLagError(
                f"auxiliary_diagnostics[{position}].family_id is not auxiliary"
            )
        if item.family_id in seen:
            raise LeadLagError("each auxiliary family may appear at most once")
        seen.add(item.family_id)
        if item.status not in _AUXILIARY_STATUSES:
            raise LeadLagError(
                f"auxiliary_diagnostics[{position}].status is invalid"
            )
        as_of = _session(
            item.as_of_session,
            f"auxiliary_diagnostics[{position}].as_of_session",
        )
        if as_of not in target_set or as_of > evaluation_session:
            raise LeadLagError("auxiliary diagnostic is not point-in-time available")
        artifact_sha256 = _digest(
            item.artifact_sha256,
            f"auxiliary_diagnostics[{position}].artifact_sha256",
        )
        if (
            not isinstance(item.artifact_bytes, bytes)
            or not item.artifact_bytes
            or len(item.artifact_bytes) > MAX_MANIFEST_BYTES
        ):
            raise LeadLagError("auxiliary diagnostic artifact bytes are invalid")
        artifact_bytes_sha256 = _sha256_bytes(item.artifact_bytes)
        artifact = _strict_json(
            item.artifact_bytes,
            f"auxiliary_diagnostics[{position}].artifact_bytes",
        )
        expected_fields = {
            "schema",
            "family_id",
            "status",
            "as_of_session",
            "validated_intraday_data",
            "intraday_session_count",
            "intraday_observation_count",
            "output_sha256",
        }
        if set(artifact) != expected_fields:
            raise LeadLagError("auxiliary diagnostic artifact fields are invalid")
        if artifact.get("schema") != "quantkit.semiconductor-method-diagnostic.v1":
            raise LeadLagError("auxiliary diagnostic artifact schema is invalid")
        supplied_semantic = _digest(
            artifact.get("output_sha256"),
            "auxiliary diagnostic artifact output_sha256",
        )
        unsealed_artifact = {
            key: value for key, value in artifact.items() if key != "output_sha256"
        }
        if supplied_semantic != _domain_digest(
            DIAGNOSTIC_HASH_DOMAIN, unsealed_artifact
        ):
            raise LeadLagError("auxiliary diagnostic semantic digest does not match")
        if artifact_sha256 != supplied_semantic:
            raise LeadLagError("auxiliary diagnostic expected semantic digest differs")
        bound_fields = {
            "family_id": item.family_id,
            "status": item.status,
            "as_of_session": as_of,
            "validated_intraday_data": item.validated_intraday_data,
            "intraday_session_count": item.intraday_session_count,
            "intraday_observation_count": item.intraday_observation_count,
        }
        if any(artifact.get(key) != value for key, value in bound_fields.items()):
            raise LeadLagError("auxiliary diagnostic fields differ from artifact")
        if (
            not isinstance(item.validated_intraday_data, bool)
            or isinstance(item.intraday_session_count, bool)
            or not isinstance(item.intraday_session_count, int)
            or item.intraday_session_count < 0
            or isinstance(item.intraday_observation_count, bool)
            or not isinstance(item.intraday_observation_count, int)
            or item.intraday_observation_count < 0
        ):
            raise LeadLagError("auxiliary diagnostic coverage counts are invalid")
        if item.family_id == "microstructure":
            if (
                not item.validated_intraday_data
                or item.intraday_session_count < config.microstructure_min_sessions
                or item.intraday_observation_count
                < config.microstructure_min_observations
            ):
                raise LeadLagError(
                    "microstructure diagnostic lacks validated intraday coverage"
                )
        elif (
            item.validated_intraday_data
            or item.intraday_session_count
            or item.intraday_observation_count
        ):
            raise LeadLagError(
                "intraday coverage fields are reserved for microstructure"
            )
        record = {
            "family_id": item.family_id,
            "role": _METHOD_FAMILIES[item.family_id],
            "status": item.status,
            "as_of_session": as_of,
            "artifact_sha256": artifact_sha256,
            "artifact_bytes_sha256": artifact_bytes_sha256,
            "validated_intraday_data": item.validated_intraday_data,
            "intraday_session_count": item.intraday_session_count,
            "intraday_observation_count": item.intraday_observation_count,
        }
        normalized.append(record)
        falsified = falsified or item.status == "falsifies"
    normalized.sort(key=lambda item: item["family_id"])
    return (
        normalized,
        _domain_digest(DIAGNOSTIC_HASH_DOMAIN, normalized),
        falsified,
    )


def _seed(base: int, *parts: Any) -> int:
    payload = _canonical_json([base, *parts])
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise LeadLagError("regression arrays are malformed")
    if len(y) <= x.shape[1] or np.linalg.matrix_rank(x) != x.shape[1]:
        raise LeadLagError("regression design is rank deficient")
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    sse = float(residuals @ residuals)
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(residuals)):
        raise LeadLagError("regression produced a non-finite result")
    return beta, residuals, sse


def _stationary_indices(
    length: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    result = np.empty(length, dtype=int)
    position = 0
    while position < length:
        start = int(rng.integers(0, length))
        run = int(rng.geometric(1.0 / min(block_length, length)))
        take = min(run, length - position)
        result[position : position + take] = (start + np.arange(take)) % length
        position += take
    return result


def _bootstrap_inference(
    x: np.ndarray,
    y: np.ndarray,
    config: LeadLagConfig,
    seed: int,
) -> tuple[float, float, float]:
    beta, residuals, _ = _ols(x, y)
    null_x = np.delete(x, 1, axis=1)
    null_beta, null_residuals, _ = _ols(null_x, y)
    residuals = residuals - residuals.mean()
    null_residuals = null_residuals - null_residuals.mean()
    fitted = x @ beta
    null_fitted = null_x @ null_beta
    rng = np.random.default_rng(seed)
    estimates = np.empty(config.bootstrap_repetitions, dtype=float)
    null_estimates = np.empty(config.bootstrap_repetitions, dtype=float)
    for position in range(config.bootstrap_repetitions):
        indices = _stationary_indices(
            len(y), config.bootstrap_block_length, rng
        )
        estimates[position] = _ols(x, fitted + residuals[indices])[0][1]
        null_estimates[position] = _ols(
            x, null_fitted + null_residuals[indices]
        )[0][1]
    lower = float(np.quantile(estimates, config.interval_alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - config.interval_alpha / 2.0))
    probability = float(
        (1 + np.count_nonzero(np.abs(null_estimates) >= abs(float(beta[1]))))
        / (config.bootstrap_repetitions + 1)
    )
    return lower, upper, probability


def _standardized_coefficient(x: np.ndarray, y: np.ndarray, coefficient: float) -> float:
    x_scale = float(np.std(x[:, 1], ddof=1))
    y_scale = float(np.std(y, ddof=1))
    if x_scale <= 0 or y_scale <= 0:
        raise LeadLagError("standardized coefficient has zero scale")
    return float(coefficient * x_scale / y_scale)


def _oos_comparison(
    design: _Design, config: LeadLagConfig
) -> dict[str, Any]:
    x, y = design.x, design.y
    null_x = np.delete(x, 1, axis=1)
    required_train = max(config.oos_min_train, config.observations_per_parameter * x.shape[1])
    full_errors: list[float] = []
    null_errors: list[float] = []
    for test_position in range(len(y)):
        eligible = np.flatnonzero(
            design.end_positions[:test_position]
            < design.base_positions[test_position] - config.oos_embargo_sessions
        )
        if len(eligible) < required_train:
            continue
        try:
            full_beta, _, _ = _ols(x[eligible], y[eligible])
            null_beta, _, _ = _ols(null_x[eligible], y[eligible])
        except LeadLagError:
            continue
        full_error = float(y[test_position] - x[test_position] @ full_beta)
        null_error = float(y[test_position] - null_x[test_position] @ null_beta)
        full_errors.append(full_error * full_error)
        null_errors.append(null_error * null_error)
    if len(full_errors) < config.oos_min_predictions:
        return {
            "passed": False,
            "embargo_sessions": config.oos_embargo_sessions,
            "prediction_count": len(full_errors),
            "full_mse": None,
            "null_mse": None,
            "relative_improvement": None,
        }
    full_mse = float(np.mean(full_errors))
    null_mse = float(np.mean(null_errors))
    improvement = 0.0 if null_mse <= 0 else float(1.0 - full_mse / null_mse)
    return {
        "passed": bool(
            null_mse > 0 and improvement >= config.oos_min_relative_improvement
        ),
        "embargo_sessions": config.oos_embargo_sessions,
        "prediction_count": len(full_errors),
        "full_mse": full_mse,
        "null_mse": null_mse,
        "relative_improvement": improvement,
    }


def _break_diagnostic(
    x: np.ndarray, y: np.ndarray, config: LeadLagConfig, seed: int
) -> dict[str, Any]:
    parameter_count = x.shape[1]
    split = len(y) // 2
    if split <= parameter_count or len(y) - split <= parameter_count:
        return {
            "passed": False,
            "method": "stationary_block_bootstrap_fixed_split",
            "split_observation": split,
            "probability": None,
            "first_coefficient": None,
            "second_coefficient": None,
        }
    try:
        pooled_beta, residuals, _ = _ols(x, y)
        first_beta, _, _ = _ols(x[:split], y[:split])
        second_beta, _, _ = _ols(x[split:], y[split:])
    except LeadLagError:
        return {
            "passed": False,
            "method": "stationary_block_bootstrap_fixed_split",
            "split_observation": split,
            "probability": None,
            "first_coefficient": None,
            "second_coefficient": None,
        }
    observed_change = abs(float(second_beta[1]) - float(first_beta[1]))
    residuals = residuals - residuals.mean()
    fitted = x @ pooled_beta
    rng = np.random.default_rng(seed)
    boot_changes = np.empty(config.bootstrap_repetitions, dtype=float)
    for position in range(config.bootstrap_repetitions):
        indices = _stationary_indices(
            len(y), config.bootstrap_block_length, rng
        )
        synthetic = fitted + residuals[indices]
        try:
            first = _ols(x[:split], synthetic[:split])[0]
            second = _ols(x[split:], synthetic[split:])[0]
            boot_changes[position] = abs(float(second[1]) - float(first[1]))
        except LeadLagError:
            boot_changes[position] = math.inf
    probability = float(
        (1 + np.count_nonzero(boot_changes >= observed_change))
        / (config.bootstrap_repetitions + 1)
    )
    same_direction = bool(
        float(first_beta[1]) * float(second_beta[1]) > 0
        and float(pooled_beta[1]) * float(first_beta[1]) > 0
    )
    return {
        "passed": bool(probability >= config.break_alpha and same_direction),
        "method": "stationary_block_bootstrap_fixed_split",
        "split_observation": split,
        "probability": probability,
        "first_coefficient": float(first_beta[1]),
        "second_coefficient": float(second_beta[1]),
    }


def _effect_half_life(cumulative_coefficients: Sequence[float | None]) -> float | None:
    """Estimate decay from marginal horizon responses, never driver persistence."""

    if len(cumulative_coefficients) != len(REQUIRED_HORIZONS) or any(
        coefficient is None for coefficient in cumulative_coefficients
    ):
        return None
    cumulative = np.asarray(cumulative_coefficients, dtype=float)
    marginal = np.diff(np.concatenate([[0.0], cumulative]))
    base = abs(float(marginal[0]))
    if not math.isfinite(base) or base <= 1e-12:
        return None
    usable = [
        (horizon, abs(float(value)) / base)
        for horizon, value in zip(REQUIRED_HORIZONS, marginal)
        if abs(float(value)) > 1e-12
    ]
    if len(usable) < 2:
        return 0.0
    x = np.asarray([horizon - 1 for horizon, _ in usable], dtype=float)
    y = np.log(np.asarray([ratio for _, ratio in usable], dtype=float))
    slope = float(np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1])
    if not math.isfinite(slope) or slope >= 0:
        return None
    return float(math.log(2.0) / -slope)


def _expiry(
    target_calendar: Sequence[str],
    estimation_session: str,
    evaluation_session: str,
    half_life: float | None,
    config: LeadLagConfig,
) -> dict[str, Any]:
    position = target_calendar.index(estimation_session)
    half_life_steps = (
        None
        if half_life is None
        else min(
            max(1, int(math.ceil(2.0 * half_life))),
            config.max_validity_sessions,
        )
    )
    required_steps = max(
        config.recalibration_sessions,
        config.max_validity_sessions,
        half_life_steps or 0,
    )
    if position + required_steps >= len(target_calendar) or half_life_steps is None:
        return {
            "not_expired_at_historical_evaluation": False,
            "estimated_half_life_sessions": half_life,
            "recalibration_session": None,
            "half_life_session": None,
            "maximum_validity_session": None,
            "expiry_session": None,
            "evaluation_session": evaluation_session,
            "expired": False,
        }
    candidates = {
        "recalibration_session": target_calendar[
            position + config.recalibration_sessions
        ],
        "half_life_session": target_calendar[position + half_life_steps],
        "maximum_validity_session": target_calendar[
            position + config.max_validity_sessions
        ],
    }
    expiry_session = min(candidates.values())
    expired = evaluation_session > expiry_session
    return {
        "not_expired_at_historical_evaluation": not expired,
        "estimated_half_life_sessions": half_life,
        **candidates,
        "expiry_session": expiry_session,
        "evaluation_session": evaluation_session,
        "expired": expired,
    }


def _build_design(
    *,
    window: int,
    horizon: int,
    outcome: str,
    end_position: int,
    target_calendar: Sequence[str],
    target_gap: Mapping[str, float],
    target_intraday: Mapping[str, float],
    links_by_target: Mapping[str, str],
    driver: Mapping[str, float],
    controls: Sequence[_Series],
) -> _Design:
    start_position = end_position - window + 1
    rows: list[list[float]] = []
    outcomes: list[float] = []
    base_positions: list[int] = []
    end_positions: list[int] = []
    base_sessions: list[str] = []
    if start_position < 0:
        return _Design(
            np.empty((0, 3 + len(controls))),
            np.empty(0),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            (),
            0,
        )
    expected = max(0, window - horizon + 1)
    for base_position in range(start_position, end_position - horizon + 2):
        if base_position == 0:
            continue
        base_session = target_calendar[base_position]
        source_session = links_by_target.get(base_session)
        if source_session is None or source_session not in driver:
            continue
        previous_session = target_calendar[base_position - 1]
        control_values: list[float] = []
        control_missing = False
        for control in controls:
            control_session = (
                source_session
                if control.lineage["timing_domain"] == "source_close"
                else previous_session
            )
            if control_session not in control.values:
                control_missing = True
                break
            control_values.append(control.values[control_session])
        if control_missing:
            continue
        if previous_session not in target_gap or previous_session not in target_intraday:
            continue
        future_sessions = target_calendar[base_position : base_position + horizon]
        if any(
            session not in target_gap or session not in target_intraday
            for session in future_sessions
        ):
            continue
        if outcome == "gap":
            dependent = sum(target_gap[session] for session in future_sessions)
        elif outcome == "intraday":
            dependent = sum(target_intraday[session] for session in future_sessions)
        else:
            dependent = sum(
                target_gap[session] + target_intraday[session]
                for session in future_sessions
            )
        own_lag = target_gap[previous_session] + target_intraday[previous_session]
        rows.append([1.0, driver[source_session], *control_values, own_lag])
        outcomes.append(float(dependent))
        base_positions.append(base_position)
        end_positions.append(base_position + horizon - 1)
        base_sessions.append(base_session)
    width = 3 + len(controls)
    return _Design(
        np.asarray(rows, dtype=float).reshape((-1, width)),
        np.asarray(outcomes, dtype=float),
        np.asarray(base_positions, dtype=int),
        np.asarray(end_positions, dtype=int),
        tuple(base_sessions),
        expected,
    )


def _bh_adjust(probabilities: Sequence[float]) -> list[float]:
    count = len(probabilities)
    order = sorted(range(count), key=lambda position: (probabilities[position], position))
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        position = order[rank_index]
        rank = rank_index + 1
        running = min(running, float(probabilities[position]) * count / rank)
        adjusted[position] = min(1.0, running)
    return adjusted


def _normalize_invalidations(
    invalidations: Sequence[InvalidationFlag],
    target_calendar: Sequence[str],
    driver_series_id: str,
    target_series_id: str,
) -> tuple[list[dict[str, str]], str]:
    normalized: list[dict[str, str]] = []
    for position, item in enumerate(invalidations):
        if not isinstance(item, InvalidationFlag):
            raise LeadLagError(f"invalidations[{position}] must be InvalidationFlag")
        if not isinstance(item.code, str) or _INVALIDATION_RE.fullmatch(item.code) is None:
            raise LeadLagError(f"invalidations[{position}].code is invalid")
        effective = _session(
            item.effective_session, f"invalidations[{position}].effective_session"
        )
        if effective not in set(target_calendar):
            raise LeadLagError("invalidation effective_session is not a target session")
        item_driver = _plain_text(
            item.driver_series_id,
            f"invalidations[{position}].driver_series_id",
        )
        item_target = _plain_text(
            item.target_series_id,
            f"invalidations[{position}].target_series_id",
        )
        if item_driver != driver_series_id or item_target != target_series_id:
            raise LeadLagError("invalidation is scoped to a different association")
        resolved: str | None = None
        if item.resolved_session is not None:
            resolved = _session(
                item.resolved_session,
                f"invalidations[{position}].resolved_session",
            )
            if resolved not in set(target_calendar) or resolved <= effective:
                raise LeadLagError("invalidation resolved_session is invalid")
        normalized.append(
            {
                "code": item.code,
                "effective_session": effective,
                "resolved_session": resolved,
                "driver_series_id": item_driver,
                "target_series_id": item_target,
                "evidence_sha256": _digest(
                    item.evidence_sha256,
                    f"invalidations[{position}].evidence_sha256",
                ),
            }
        )
    normalized.sort(key=lambda item: (item["effective_session"], item["code"], item["evidence_sha256"]))
    if len({(item["effective_session"], item["code"]) for item in normalized}) != len(normalized):
        raise LeadLagError("invalidations must not repeat a code on one session")
    return normalized, _domain_digest(
        INVALIDATION_HASH_DOMAIN, normalized
    )


def build_semiconductor_lead_lag(
    driver: ValidatedReturnSeries,
    target_gap: ValidatedReturnSeries,
    target_intraday: ValidatedReturnSeries,
    controls: Sequence[ValidatedReturnSeries],
    session_map: VerifiedSessionMap,
    *,
    estimation_session: str,
    runtime: MethodRuntime,
    evaluation_session: str | None = None,
    auxiliary_diagnostics: Sequence[AuxiliaryDiagnostic] = (),
    invalidations: Sequence[InvalidationFlag] = (),
    config: LeadLagConfig | None = None,
) -> LeadLagCalculation:
    """Build the fixed conditional lead-lag family from admitted return inputs.

    The function raises on contract or lineage drift.  Statistical non-passes
    remain explicit ``descriptive_only`` results; unavailable model/expiry
    requirements remain ``quarantined``.  No result is a recommendation.
    """

    policy = config or LeadLagConfig()
    estimation = _session(estimation_session, "estimation_session")
    evaluation = _session(
        evaluation_session or estimation_session, "evaluation_session"
    )
    if evaluation < estimation:
        raise LeadLagError("evaluation_session must not precede estimation_session")
    runtime_record = _normalize_runtime(runtime)
    verified_map = _verified_map(session_map)

    normalized_driver = _consensus_series(driver)
    normalized_gap = _consensus_series(target_gap)
    normalized_intraday = _consensus_series(target_intraday)
    if driver.return_kind != "source_return":
        raise LeadLagError("driver must derive source_return")
    if target_gap.return_kind != "target_gap":
        raise LeadLagError("target_gap must derive target_gap")
    if target_intraday.return_kind != "target_intraday":
        raise LeadLagError("target_intraday must derive target_intraday")
    if (
        target_gap.series_id != target_intraday.series_id
        or target_gap.consensus_manifest_bytes
        != target_intraday.consensus_manifest_bytes
    ):
        raise LeadLagError(
            "target gap and intraday must derive from identical manifest bytes"
        )
    if driver.series_id != policy.primary_driver_series_id:
        raise LeadLagError("driver does not match the preregistered primary endpoint")
    if target_gap.series_id != policy.primary_target_series_id:
        raise LeadLagError("target does not match the preregistered primary endpoint")
    driver_identity = tuple(
        normalized_driver.lineage[key]
        for key in ("asset_id", "market", "calendar")
    )
    target_identity = tuple(
        normalized_gap.lineage[key]
        for key in ("asset_id", "market", "calendar")
    )
    if driver_identity == target_identity:
        raise LeadLagError("driver and target must be different economic series")
    if normalized_gap.calendar != normalized_intraday.calendar:
        raise LeadLagError("target gap and intraday calendars differ")

    normalized_controls = [_consensus_series(item) for item in controls]
    if any(item.return_kind != "control_return" for item in controls):
        raise LeadLagError("every control must derive control_return")
    normalized_controls.sort(key=lambda item: str(item.lineage["series_id"]))
    required_specs = sorted(policy.required_controls, key=lambda item: item.series_id)
    actual_control_ids = [str(item.lineage["series_id"]) for item in normalized_controls]
    required_control_ids = [item.series_id for item in required_specs]
    if actual_control_ids != required_control_ids:
        raise LeadLagError("controls do not match the preregistered control set")
    for item, spec in zip(normalized_controls, required_specs):
        if item.lineage["timing_domain"] != spec.timing_domain:
            raise LeadLagError("control timing differs from its preregistered role")
    all_manifest_hashes = [
        normalized_driver.lineage["manifest_bytes_sha256"],
        normalized_gap.lineage["manifest_bytes_sha256"],
        *(
            item.lineage["manifest_bytes_sha256"]
            for item in normalized_controls
        ),
    ]
    if len(set(all_manifest_hashes)) != len(all_manifest_hashes):
        raise LeadLagError("driver, target, and controls require distinct manifests")
    reserved_bars_hashes = {
        normalized_driver.lineage["bars_sha256"],
        normalized_gap.lineage["bars_sha256"],
    }
    if any(
        item.lineage["bars_sha256"] in reserved_bars_hashes
        for item in normalized_controls
    ):
        raise LeadLagError("control cannot clone driver or target accepted bars")

    source_schedule_labels = {item.session for item in verified_map.source_schedule}
    target_schedule_labels = tuple(
        item.session for item in verified_map.target_schedule
    )
    target_schedule_set = set(target_schedule_labels)
    if verified_map.source_market != normalized_driver.lineage["market"]:
        raise LeadLagError("session map source market differs from driver")
    if verified_map.target_market != normalized_gap.lineage["market"]:
        raise LeadLagError("session map target market differs from target")
    if not set(normalized_driver.calendar) <= source_schedule_labels:
        raise LeadLagError("driver calendar is not covered by the source schedule")
    if not set(normalized_gap.calendar) <= target_schedule_set:
        raise LeadLagError("target calendar is not covered by the target schedule")
    _require_contiguous_schedule_slice(
        normalized_driver.calendar,
        verified_map.source_schedule,
        "driver calendar",
    )
    _require_contiguous_schedule_slice(
        normalized_gap.calendar,
        verified_map.target_schedule,
        "target observation calendar",
    )
    if estimation not in normalized_gap.values:
        raise LeadLagError("estimation_session requires a completed target bar")
    if estimation != max(normalized_gap.values):
        raise LeadLagError("target manifest contains observations after estimation")
    if estimation not in target_schedule_set or evaluation not in target_schedule_set:
        raise LeadLagError("estimation and evaluation must be scheduled target sessions")

    target_open_lookup = {
        item.session: _timestamp(item.open_at, "target_schedule.open_at")[0]
        for item in verified_map.target_schedule
    }
    source_close_lookup = {
        item.session: _timestamp(item.close_at, "source_schedule.close_at")[0]
        for item in verified_map.source_schedule
    }
    estimation_open = target_open_lookup[estimation]
    used_source_sessions = {
        item.source_session
        for item in verified_map.links
        if item.target_session <= estimation
    }
    if any(
        source_close_lookup[session] >= estimation_open
        for session in normalized_driver.calendar
        if session in used_source_sessions
    ):
        raise LeadLagError("driver manifest contains data unavailable at target open")
    for item in normalized_controls:
        if item.lineage["timing_domain"] == "source_close":
            if item.lineage["market"] != verified_map.source_market:
                raise LeadLagError(
                    "source-close control requires its own verified venue map"
                )
            if not set(item.calendar) <= source_schedule_labels:
                raise LeadLagError(
                    "source-close control calendar is not covered by source schedule"
                )
            _require_contiguous_schedule_slice(
                item.calendar,
                verified_map.source_schedule,
                "source-close control calendar",
            )
            if any(
                source_close_lookup[session] >= estimation_open
                for session in item.calendar
                if session in used_source_sessions
            ):
                raise LeadLagError(
                    "source-close control contains data unavailable at target open"
                )
        else:
            if item.lineage["market"] != verified_map.target_market:
                raise LeadLagError(
                    "target-close control market differs from target schedule"
                )
            if not set(item.calendar) <= target_schedule_set:
                raise LeadLagError(
                    "target-close control calendar is not covered by target schedule"
                )
            _require_contiguous_schedule_slice(
                item.calendar,
                verified_map.target_schedule,
                "target-close control calendar",
            )
            if max(item.values) > estimation:
                raise LeadLagError("target-close control contains post-estimation data")

    links = [
        item
        for item in verified_map.links
        if item.target_session <= estimation
    ]
    links_by_target = {
        item.target_session: item.source_session for item in links
    }
    if len(links_by_target) != len(links):
        raise LeadLagError("verified session map is not one-to-one")
    target_calendar = normalized_gap.calendar
    estimation_position = target_calendar.index(estimation)
    normalized_invalidations, invalidations_sha256 = _normalize_invalidations(
        invalidations,
        target_schedule_labels,
        driver.series_id,
        target_gap.series_id,
    )
    active_invalidations = [
        item
        for item in normalized_invalidations
        if item["effective_session"] <= evaluation
        and (
            item["resolved_session"] is None
            or evaluation < item["resolved_session"]
        )
    ]
    normalized_diagnostics, diagnostics_sha256, auxiliary_falsified = (
        _normalize_diagnostics(
            auxiliary_diagnostics,
            target_schedule_labels,
            evaluation,
            policy,
        )
    )

    # Pass one: estimate every locked specification without using its peers.
    pending: list[dict[str, Any]] = []
    for window in policy.windows:
        for outcome in OUTCOMES:
            for horizon in policy.horizons:
                design = _build_design(
                    window=window,
                    horizon=horizon,
                    outcome=outcome,
                    end_position=estimation_position,
                    target_calendar=target_calendar,
                    target_gap=normalized_gap.values,
                    target_intraday=normalized_intraday.values,
                    links_by_target=links_by_target,
                    driver=normalized_driver.values,
                    controls=normalized_controls,
                )
                parameter_count = design.x.shape[1]
                required_observations = max(
                    policy.min_observations,
                    policy.observations_per_parameter * parameter_count,
                )
                reasons: set[str] = set()
                identified = len(design.y) >= required_observations
                if not identified:
                    reasons.add("insufficient_observations")
                coefficient: float | None = None
                standardized: float | None = None
                interval: list[float] | None = None
                raw_probability = 1.0
                adjacent_coefficient: float | None = None
                adjacent_pass = False
                oos: dict[str, Any] = {
                    "passed": False,
                    "embargo_sessions": policy.oos_embargo_sessions,
                    "prediction_count": 0,
                    "full_mse": None,
                    "null_mse": None,
                    "relative_improvement": None,
                }
                break_result: dict[str, Any] = {
                    "passed": False,
                    "method": "stationary_block_bootstrap_fixed_split",
                    "split_observation": len(design.y) // 2,
                    "probability": None,
                    "first_coefficient": None,
                    "second_coefficient": None,
                }
                if identified:
                    try:
                        beta, _, _ = _ols(design.x, design.y)
                        coefficient = float(beta[1])
                        standardized = _standardized_coefficient(
                            design.x, design.y, coefficient
                        )
                        lower, upper, raw_probability = _bootstrap_inference(
                            design.x,
                            design.y,
                            policy,
                            _seed(
                                policy.random_seed,
                                "inference",
                                window,
                                outcome,
                                horizon,
                            ),
                        )
                        interval = [lower, upper]
                        oos = _oos_comparison(design, policy)
                        break_result = _break_diagnostic(
                            design.x,
                            design.y,
                            policy,
                            _seed(
                                policy.random_seed,
                                "break",
                                window,
                                outcome,
                                horizon,
                            ),
                        )
                        adjacent_design = _build_design(
                            window=window,
                            horizon=horizon,
                            outcome=outcome,
                            end_position=(
                                estimation_position
                                - policy.adjacent_shift_sessions
                            ),
                            target_calendar=target_calendar,
                            target_gap=normalized_gap.values,
                            target_intraday=normalized_intraday.values,
                            links_by_target=links_by_target,
                            driver=normalized_driver.values,
                            controls=normalized_controls,
                        )
                        if len(adjacent_design.y) >= required_observations:
                            adjacent_coefficient = float(
                                _ols(adjacent_design.x, adjacent_design.y)[0][1]
                            )
                            adjacent_pass = bool(
                                coefficient * adjacent_coefficient > 0
                            )
                    except LeadLagError:
                        identified = False
                        reasons.add("model_not_identified")
                        coefficient = None
                        standardized = None
                        interval = None
                        raw_probability = 1.0
                        adjacent_coefficient = None
                        adjacent_pass = False
                        oos = {
                            "passed": False,
                            "embargo_sessions": policy.oos_embargo_sessions,
                            "prediction_count": 0,
                            "full_mse": None,
                            "null_mse": None,
                            "relative_improvement": None,
                        }
                        break_result = {
                            "passed": False,
                            "method": "stationary_block_bootstrap_fixed_split",
                            "split_observation": len(design.y) // 2,
                            "probability": None,
                            "first_coefficient": None,
                            "second_coefficient": None,
                        }
                pending.append(
                    {
                        "window_sessions": window,
                        "outcome": outcome,
                        "horizon_sessions": horizon,
                        "base_first_session": (
                            design.base_sessions[0] if design.base_sessions else None
                        ),
                        "base_last_session": (
                            design.base_sessions[-1] if design.base_sessions else None
                        ),
                        "expected_observations": design.expected_observations,
                        "usable_observations": len(design.y),
                        "required_observations": required_observations,
                        "parameter_count": parameter_count,
                        "driver_coefficient": coefficient,
                        "standardized_driver_coefficient": standardized,
                        "bootstrap_interval_95": interval,
                        "raw_probability": raw_probability,
                        "fdr_q_value": 1.0,
                        "adjacent_window_drift_check": {
                            "shift_sessions": policy.adjacent_shift_sessions,
                            "driver_coefficient": adjacent_coefficient,
                            "same_direction": adjacent_pass,
                        },
                        "oos_comparison": oos,
                        "structural_break": break_result,
                        "_identified": identified,
                        "_reason_codes": reasons,
                    }
                )

    # Pass two: family inference, horizon-response decay, expiry, and gates.
    adjusted = _bh_adjust([float(item["raw_probability"]) for item in pending])
    half_lives: dict[tuple[int, str], float | None] = {}
    for window in policy.windows:
        for outcome in OUTCOMES:
            members = [
                item
                for item in pending
                if item["window_sessions"] == window and item["outcome"] == outcome
            ]
            members.sort(key=lambda item: item["horizon_sessions"])
            half_lives[(window, outcome)] = _effect_half_life(
                [item["driver_coefficient"] for item in members]
            )

    results: list[dict[str, Any]] = []
    has_control_contract = bool(required_specs)
    for item, q_value in zip(pending, adjusted):
        item["fdr_q_value"] = q_value
        reasons = set(item.pop("_reason_codes"))
        identified = bool(item.pop("_identified"))
        half_life = half_lives[(item["window_sessions"], item["outcome"])]
        expiry = _expiry(
            target_schedule_labels,
            estimation,
            evaluation,
            half_life,
            policy,
        )
        item["effect_decay"] = {
            "basis": "marginal_cumulative_horizon_response",
            "estimated_half_life_sessions": half_life,
        }
        item["expiry"] = expiry
        interval = item["bootstrap_interval_95"]
        standardized = item["standardized_driver_coefficient"]
        primary_gates = {
            "sample_size": identified,
            "bootstrap_interval": bool(
                interval is not None and (interval[0] > 0 or interval[1] < 0)
            ),
            "family_wide_fdr": bool(q_value <= policy.fdr_alpha),
            "standardized_magnitude": bool(
                standardized is not None
                and abs(float(standardized))
                >= policy.min_standardized_coefficient
            ),
            "purged_embargoed_oos": bool(item["oos_comparison"]["passed"]),
            "adjacent_window_drift": bool(
                item["adjacent_window_drift_check"]["same_direction"]
            ),
            "no_detected_structural_break": bool(
                item["structural_break"]["passed"]
            ),
            "not_expired_at_historical_evaluation": bool(
                expiry["not_expired_at_historical_evaluation"]
            ),
            "preregistered_controls": has_control_contract,
            "global_search_family_registered": bool(
                policy.global_search_family_sha256
            ),
            "preregistration_bound": bool(policy.preregistration_sha256),
        }
        reason_by_gate = {
            "bootstrap_interval": "bootstrap_interval_includes_zero",
            "family_wide_fdr": "family_wide_fdr_not_passed",
            "standardized_magnitude": "standardized_magnitude_below_threshold",
            "purged_embargoed_oos": "oos_null_not_beaten",
            "adjacent_window_drift": "adjacent_window_drift_not_confirmed",
            "no_detected_structural_break": "structural_break_detected",
            "preregistered_controls": "required_control_contract_empty",
            "global_search_family_registered": "global_search_family_unregistered",
            "preregistration_bound": "preregistration_unbound",
        }
        if identified:
            for gate, reason in reason_by_gate.items():
                if not primary_gates[gate]:
                    reasons.add(reason)
        if not expiry["not_expired_at_historical_evaluation"]:
            reasons.add("expired" if expiry["expired"] else "expiry_not_identified")
        primary_status = (
            "accepted"
            if all(primary_gates.values())
            else (
                "quarantined"
                if not identified or expiry["expiry_session"] is None
                else ("expired" if expiry["expired"] else "descriptive_only")
            )
        )
        for invalidation in active_invalidations:
            reasons.add("invalidation_" + invalidation["code"])
        if auxiliary_falsified:
            reasons.add("auxiliary_falsification")
        if active_invalidations:
            status = "invalidated"
        elif primary_status == "accepted" and auxiliary_falsified:
            status = "descriptive_only"
        else:
            status = primary_status
        item["primary_gates"] = primary_gates
        item["context_gates"] = {
            "no_active_point_in_time_invalidation": not active_invalidations,
            "no_auxiliary_falsification": not auxiliary_falsified,
        }
        item["historical_primary_status"] = primary_status
        item["statistical_status"] = status
        reasons.add("current_shock_not_applied")
        # The conventional status field is deliberately the safe publication
        # posture.  Historical estimator state has its own explicitly named
        # field so a generic consumer cannot mistake calibration for a view.
        item["status"] = "ABSTAIN"
        item["publication_status"] = "ABSTAIN"
        item["reason_codes"] = sorted(reasons)
        results.append(item)

    counts = {
        status: sum(item["statistical_status"] == status for item in results)
        for status in _RESULT_STATUSES
    }
    overall_status = _overall_statistical_status(counts, len(results))

    config_payload = _config_payload(policy)
    config_sha256 = _domain_digest(CONFIG_HASH_DOMAIN, config_payload)
    method = {
        "name": METHOD_NAME,
        "version": METHOD_VERSION,
        "estimator": "ols_cumulative_local_projection",
        "inference": "stationary_block_residual_bootstrap",
        "multiplicity": "family_wide_benjamini_hochberg",
        "family_scope": "one_locked_driver_target_endpoint_45_tests",
        "publication_requires_external_global_family_adjustment": True,
        "method_families": [
            {"family_id": family_id, "role": role}
            for family_id, role in _METHOD_FAMILIES.items()
        ],
    }
    method_sha256 = _domain_digest(METHOD_HASH_DOMAIN, method)
    runtime_sha256 = _domain_digest(RUNTIME_HASH_DOMAIN, runtime_record)
    control_inputs = []
    for item, spec in zip(normalized_controls, required_specs):
        control_inputs.append(
            {
                **dict(item.lineage),
                "role": spec.role,
                "definition_sha256": spec.definition_sha256,
            }
        )
    session_map_record = {
        "source_market": verified_map.source_market,
        "target_market": verified_map.target_market,
        "source_schedule": [
            dataclasses.asdict(item) for item in verified_map.source_schedule
        ],
        "target_schedule": [
            dataclasses.asdict(item) for item in verified_map.target_schedule
        ],
        "source_schedule_sha256": verified_map.source_schedule_sha256,
        "target_schedule_sha256": verified_map.target_schedule_sha256,
        "rule": "latest_completed_source_close_to_next_actual_target_open",
        "links": [dataclasses.asdict(item) for item in verified_map.links],
        "excluded_collisions": list(verified_map.excluded_collisions),
        "unmatched_source_closes": list(verified_map.unmatched_source_closes),
        "session_map_sha256": verified_map.session_map_sha256,
    }
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "ABSTAIN",
        "statistical_status": overall_status,
        "artifact_type": "historical_calibration",
        "publication_status": "ABSTAIN",
        "current_shock_application": None,
        "scope": "conditional_lead_lag_association",
        "provenance_boundary": (
            "calculation_integrity_relative_to_upstream_accepted_manifests_and_schedules"
        ),
        "method": method,
        "method_sha256": method_sha256,
        "runtime": runtime_record,
        "runtime_sha256": runtime_sha256,
        "config": config_payload,
        "config_sha256": config_sha256,
        "estimation_session": estimation,
        "evaluation_session": evaluation,
        "inputs": {
            "driver": dict(normalized_driver.lineage),
            "target_gap": dict(normalized_gap.lineage),
            "target_intraday": dict(normalized_intraday.lineage),
            "controls": control_inputs,
            "session_map": session_map_record,
            "invalidations": {
                "records": normalized_invalidations,
                "sha256": invalidations_sha256,
            },
            "auxiliary_diagnostics": {
                "records": normalized_diagnostics,
                "sha256": diagnostics_sha256,
            },
        },
        "family": {
            "study_family_id": policy.study_family_id,
            "primary_driver_series_id": policy.primary_driver_series_id,
            "primary_target_series_id": policy.primary_target_series_id,
            "windows": list(policy.windows),
            "horizons": list(policy.horizons),
            "outcomes": list(OUTCOMES),
            "test_count": len(results),
        },
        "results": results,
        "summary": {
            "publication_status": "ABSTAIN",
            "statistical_counts": counts,
        },
    }
    document["output_sha256"] = _domain_digest(OUTPUT_HASH_DOMAIN, document)
    _canonical_json(document)
    return LeadLagCalculation(document)


__all__ = [
    "SCHEMA",
    "METHOD_NAME",
    "METHOD_VERSION",
    "LeadLagError",
    "ValidatedReturnSeries",
    "SessionLink",
    "VenueSession",
    "VerifiedSessionMap",
    "AuxiliaryDiagnostic",
    "MethodRuntime",
    "ControlSpec",
    "InvalidationFlag",
    "LeadLagConfig",
    "LeadLagCalculation",
    "session_calendar_sha256",
    "validated_returns_sha256",
    "build_verified_session_map",
    "build_semiconductor_lead_lag",
]

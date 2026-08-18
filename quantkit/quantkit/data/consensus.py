"""Deterministic, provider-neutral consensus for captured daily OHLCV bars.

The module is deliberately offline.  Callers capture provider frames elsewhere,
declare their economic identity explicitly, and pass those immutable inputs to
``build_ohlcv_consensus``.  The result separates publishable bars from audit
diagnostics and exposes a canonical JSON-compatible manifest.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = OHLCV_COLUMNS[:4]
MANIFEST_SCHEMA = "quantkit.ohlcv-consensus.v1"


class OHLCVConsensusError(ValueError):
    """Raised when the consensus contract cannot be evaluated safely."""


@dataclass(frozen=True)
class OHLCVIdentity:
    """Research identity shared by every source in one consensus call."""

    symbol: str
    market: str
    interval: str
    calendar: str
    currency: str


@dataclass(frozen=True)
class OHLCVSecurityIdentity:
    """Security identity asserted independently by each provider capture."""

    symbol: str
    market: str
    currency: str


@dataclass(frozen=True)
class OHLCVSource:
    """One immutable provider capture and its explicit economic metadata."""

    provider: str
    independence_group: str
    price_basis: str
    volume_unit: str
    frame: pd.DataFrame = field(repr=False, compare=False)
    identity: OHLCVSecurityIdentity | None = None
    input_artifact_sha256: str | None = None
    source_id: str | None = None


@dataclass(frozen=True)
class ConsensusTolerance:
    """Symmetric relative-spread bands for prices and volume.

    Values at or below ``pass`` are clean.  Values above ``pass`` and at or
    below ``warning`` remain publishable with a warning.  Values above
    ``warning`` cannot support a consensus pair.  ``quarantine`` distinguishes
    a quarantined outlier from an extreme invalid outlier in diagnostics; both
    are excluded from voting.
    """

    price_pass: float = 0.001
    price_warning: float = 0.005
    price_quarantine: float = 0.02
    volume_pass: float = 0.02
    volume_warning: float = 0.10
    volume_quarantine: float = 0.30

    def __post_init__(self) -> None:
        for label, values in (
            (
                "price",
                (self.price_pass, self.price_warning, self.price_quarantine),
            ),
            (
                "volume",
                (self.volume_pass, self.volume_warning, self.volume_quarantine),
            ),
        ):
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                raise OHLCVConsensusError(f"{label} tolerances must be finite numbers")
            low, warning, quarantine = (float(value) for value in values)
            if not 0 <= low <= warning < quarantine:
                raise OHLCVConsensusError(
                    f"{label} tolerances must satisfy 0 <= pass <= warning < quarantine"
                )


@dataclass(frozen=True)
class OHLCVConsensusResult:
    """Accepted bars plus immutable-by-copy diagnostics and manifest access."""

    _accepted_bars: pd.DataFrame = field(repr=False, compare=False)
    _diagnostics: tuple[dict[str, Any], ...] = field(repr=False, compare=False)
    _manifest: dict[str, Any] = field(repr=False, compare=False)

    @property
    def status(self) -> str:
        return str(self._manifest["status"])

    @property
    def accepted_bars(self) -> pd.DataFrame:
        """Return a detached copy of the accepted canonical bars."""

        return self._accepted_bars.copy(deep=True)

    @property
    def diagnostics(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(item) for item in self._diagnostics)

    @property
    def manifest(self) -> dict[str, Any]:
        return self.to_manifest()

    def to_manifest(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible audit manifest."""

        return copy.deepcopy(self._manifest)


@dataclass(frozen=True)
class _Row:
    values: Mapping[str, float]


@dataclass(frozen=True)
class _NormalizedSource:
    source_id: str
    provider: str
    independence_group: str
    price_basis: str
    volume_unit: str
    identity: Mapping[str, str]
    input_artifact_sha256: str
    frame_sha256: str
    rows: Mapping[str, _Row]
    invalid_rows: Mapping[str, tuple[str, ...]]
    source_reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class _Vote:
    values: Mapping[str, float]
    source_ids: tuple[str, ...]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(domain: str, value: Any) -> str:
    payload = domain.encode("ascii") + b"\0" + _canonical_json(value)
    return hashlib.sha256(payload).hexdigest()


def _tag_scalar(value: Any) -> dict[str, Any]:
    """Encode common pandas scalars without lossy string coercion."""

    if value is None:
        return {"type": "none"}
    if value is pd.NA:
        return {"type": "pd.NA"}
    if value is pd.NaT:
        return {"type": "pd.NaT"}
    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return {"type": "pd.NaT"}
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, dt.datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, dt.date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, dt.timedelta):
        return {
            "type": "timedelta",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "+inf" if value > 0 else "-inf"
        else:
            encoded = value.hex()
        return {"type": "float", "value": encoded}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [_tag_scalar(item) for item in value]}
    raise OHLCVConsensusError(
        "source frame contains a scalar that cannot be hashed deterministically: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _hash_original_frame(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame):
        raise OHLCVConsensusError("each source frame must be a pandas DataFrame")
    if isinstance(frame.index, pd.MultiIndex):
        index_dtypes = [str(level.dtype) for level in frame.index.levels]
        index_names = [_tag_scalar(name) for name in frame.index.names]
    else:
        index_dtypes = [str(frame.index.dtype)]
        index_names = [_tag_scalar(frame.index.name)]
    payload = {
        "schema": "quantkit.ohlcv-source-frame.v1",
        "index": {
            "dtypes": index_dtypes,
            "names": index_names,
            "values": [_tag_scalar(value) for value in frame.index.tolist()],
        },
        "columns": [
            {
                "label": _tag_scalar(frame.columns[position]),
                "dtype": str(frame.dtypes.iloc[position]),
                "values": [
                    _tag_scalar(value) for value in frame.iloc[:, position].tolist()
                ],
            }
            for position in range(frame.shape[1])
        ],
    }
    return _digest("quantkit.ohlcv-source-frame.v1", payload)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OHLCVConsensusError(f"{label} must be a non-empty string")
    return value.strip()


def _session_label(value: Any) -> str | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(timestamp):
        return None
    # Daily labels represent a venue session.  Removing the timezone preserves
    # that label; UTC conversion could shift it to a different calendar date.
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.date().isoformat()


def _row_reason_codes(values: Mapping[str, Any]) -> tuple[str, ...]:
    parsed: dict[str, float] = {}
    for column in OHLCV_COLUMNS:
        value = values.get(column)
        if isinstance(value, bool):
            return ("non_finite_ohlcv",)
        try:
            parsed[column] = float(value)
        except (TypeError, ValueError, OverflowError):
            return ("non_finite_ohlcv",)
    reasons: list[str] = []
    if any(not math.isfinite(parsed[column]) for column in OHLCV_COLUMNS):
        reasons.append("non_finite_ohlcv")
    if any(parsed[column] <= 0 for column in PRICE_COLUMNS):
        reasons.append("non_positive_price")
    if parsed["volume"] < 0:
        reasons.append("negative_volume")
    if not reasons and not (
        parsed["low"] <= min(parsed["open"], parsed["close"])
        and max(parsed["open"], parsed["close"]) <= parsed["high"]
    ):
        reasons.append("invalid_ohlc_geometry")
    return tuple(reasons)


def _normalize_source(
    source: OHLCVSource,
    source_id: str,
    security_identity: Mapping[str, str],
    artifact_sha256: str,
) -> _NormalizedSource:
    frame = source.frame
    frame_sha256 = _hash_original_frame(frame)
    working = frame.copy(deep=True)

    normalized_columns: dict[str, int] = {}
    duplicate_columns: set[str] = set()
    for position, label in enumerate(working.columns):
        normalized = str(label).strip().casefold().replace(" ", "_")
        if normalized in OHLCV_COLUMNS:
            if normalized in normalized_columns:
                duplicate_columns.add(normalized)
            normalized_columns[normalized] = position

    missing = sorted(set(OHLCV_COLUMNS) - set(normalized_columns))
    column_failure = bool(missing or duplicate_columns)
    source_reasons: set[str] = set()
    if missing:
        source_reasons.add("missing_required_columns")
    if duplicate_columns:
        source_reasons.add("duplicate_required_columns")
    if working.empty:
        source_reasons.add("empty_source")

    labels: list[str | None] = [_session_label(value) for value in working.index]
    counts: dict[str, int] = {}
    for label in labels:
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    if any(count > 1 for count in counts.values()):
        source_reasons.add("duplicate_session")
    if any(label is None for label in labels):
        source_reasons.add("invalid_session_label")

    rows: dict[str, _Row] = {}
    invalid_rows: dict[str, tuple[str, ...]] = {}
    for row_position, label in enumerate(labels):
        if label is None:
            continue
        reasons: set[str] = set()
        if counts[label] > 1:
            reasons.add("duplicate_session")
        if column_failure:
            reasons.update(source_reasons & {
                "missing_required_columns",
                "duplicate_required_columns",
            })
        values: dict[str, Any] = {}
        if not column_failure:
            values = {
                column: working.iloc[row_position, normalized_columns[column]]
                for column in OHLCV_COLUMNS
            }
            reasons.update(_row_reason_codes(values))
        if reasons:
            invalid_rows[label] = tuple(sorted(reasons))
            continue
        parsed = {column: float(values[column]) for column in OHLCV_COLUMNS}
        rows[label] = _Row(values=parsed)

    return _NormalizedSource(
        source_id=source_id,
        provider=source.provider.strip(),
        independence_group=source.independence_group.strip(),
        price_basis=source.price_basis.strip(),
        volume_unit=source.volume_unit.strip(),
        identity=dict(security_identity),
        input_artifact_sha256=artifact_sha256,
        frame_sha256=frame_sha256,
        rows=rows,
        invalid_rows=invalid_rows,
        source_reason_codes=tuple(sorted(source_reasons)),
    )


def _relative_spread(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    low = min(values)
    high = max(values)
    denominator = max(abs(low), abs(high))
    if denominator == 0:
        return 0.0
    return abs(high - low) / denominator


def _field_threshold(
    column: str, tolerance: ConsensusTolerance, band: str
) -> float:
    prefix = "volume" if column == "volume" else "price"
    return float(getattr(tolerance, f"{prefix}_{band}"))


def _votes_compatible(votes: Sequence[_Vote], tolerance: ConsensusTolerance) -> bool:
    if len(votes) < 2:
        return True
    return all(
        _relative_spread([vote.values[column] for vote in votes])
        <= _field_threshold(column, tolerance, "warning")
        for column in OHLCV_COLUMNS
    )


def _comparison_band(
    votes: Sequence[_Vote], tolerance: ConsensusTolerance
) -> str:
    """Classify a multi-field comparison using the strictest field band."""

    if all(
        _relative_spread([vote.values[column] for vote in votes])
        <= _field_threshold(column, tolerance, "pass")
        for column in OHLCV_COLUMNS
    ):
        return "pass"
    if _votes_compatible(votes, tolerance):
        return "warning"
    if all(
        _relative_spread([vote.values[column] for vote in votes])
        <= _field_threshold(column, tolerance, "quarantine")
        for column in OHLCV_COLUMNS
    ):
        return "quarantine"
    return "beyond_quarantine"


def _median_vote(votes: Sequence[_Vote]) -> _Vote:
    return _Vote(
        values={
            column: float(statistics.median(vote.values[column] for vote in votes))
            for column in OHLCV_COLUMNS
        },
        source_ids=tuple(
            sorted({source_id for vote in votes for source_id in vote.source_ids})
        ),
    )


def _source_manifest(source: _NormalizedSource) -> dict[str, Any]:
    if source.rows and not source.invalid_rows and not source.source_reason_codes:
        status = "ready"
    elif source.rows:
        status = "partial"
    elif source.invalid_rows:
        status = "invalid"
    else:
        status = "empty"
    return {
        "source_id": source.source_id,
        "provider": source.provider,
        "independence_group": source.independence_group,
        "price_basis": source.price_basis,
        "volume_unit": source.volume_unit,
        "identity": dict(source.identity),
        "input_artifact_sha256": source.input_artifact_sha256,
        "frame_sha256": source.frame_sha256,
        "status": status,
        "reason_codes": list(source.source_reason_codes),
    }


def _empty_bars() -> pd.DataFrame:
    frame = pd.DataFrame(columns=OHLCV_COLUMNS, dtype=float)
    frame.index = pd.DatetimeIndex([], name="date")
    return frame


def build_ohlcv_consensus(
    identity: OHLCVIdentity,
    sources: Sequence[OHLCVSource],
    *,
    sessions: Sequence[str | dt.date | pd.Timestamp],
    tolerance: ConsensusTolerance | None = None,
) -> OHLCVConsensusResult:
    """Build a fail-closed daily OHLCV consensus from captured provider frames.

    Votes are hierarchical: compatible duplicate captures first collapse to one
    provider vote, compatible providers then collapse to one independence-group
    vote, and only independence groups participate in the final two-of-three
    decision.  An ambiguous bridge cluster is quarantined rather than resolved
    by source order.
    """

    if not isinstance(identity, OHLCVIdentity):
        raise OHLCVConsensusError("identity must be an OHLCVIdentity")
    identity_values = {
        "symbol": _required_text(identity.symbol, "identity.symbol"),
        "market": _required_text(identity.market, "identity.market"),
        "interval": _required_text(identity.interval, "identity.interval"),
        "calendar": _required_text(identity.calendar, "identity.calendar"),
        "currency": _required_text(identity.currency, "identity.currency"),
    }
    if identity_values["interval"] != "1d":
        raise OHLCVConsensusError("only the 1d interval is supported")
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        raise OHLCVConsensusError("sessions must be an explicit sequence")
    calendar_sessions: list[str] = []
    for position, value in enumerate(sessions):
        if isinstance(value, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise OHLCVConsensusError(
                f"sessions[{position}] must use the YYYY-MM-DD date format"
            )
        label = _session_label(value)
        if label is None:
            raise OHLCVConsensusError(f"sessions[{position}] is not a valid session date")
        if isinstance(value, (str, dt.date)) and not isinstance(value, dt.datetime):
            pass
        elif isinstance(value, pd.Timestamp) and value == value.normalize():
            pass
        else:
            raise OHLCVConsensusError(
                f"sessions[{position}] must be a date without an intraday time"
            )
        calendar_sessions.append(label)
    if not calendar_sessions:
        raise OHLCVConsensusError("sessions must contain at least one actual session")
    if calendar_sessions != sorted(set(calendar_sessions)):
        raise OHLCVConsensusError("sessions must be unique and strictly increasing")
    calendar_set = set(calendar_sessions)
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise OHLCVConsensusError("sources must be a sequence")
    if not sources:
        raise OHLCVConsensusError("at least two independent source groups are required")
    policy = tolerance or ConsensusTolerance()
    if not isinstance(policy, ConsensusTolerance):
        raise OHLCVConsensusError("tolerance must be a ConsensusTolerance")

    prepared: list[tuple[OHLCVSource, str, dict[str, str], str]] = []
    source_ids: set[str] = set()
    provider_groups: dict[str, str] = {}
    bases: set[str] = set()
    volume_units: set[str] = set()
    groups: set[str] = set()
    for position, source in enumerate(sources):
        if not isinstance(source, OHLCVSource):
            raise OHLCVConsensusError(f"sources[{position}] must be an OHLCVSource")
        provider = _required_text(source.provider, f"sources[{position}].provider")
        group = _required_text(
            source.independence_group, f"sources[{position}].independence_group"
        )
        basis = _required_text(source.price_basis, f"sources[{position}].price_basis")
        unit = _required_text(source.volume_unit, f"sources[{position}].volume_unit")
        if not isinstance(source.identity, OHLCVSecurityIdentity):
            raise OHLCVConsensusError(
                f"sources[{position}].identity must be an OHLCVSecurityIdentity"
            )
        source_identity = {
            "symbol": _required_text(
                source.identity.symbol, f"sources[{position}].identity.symbol"
            ),
            "market": _required_text(
                source.identity.market, f"sources[{position}].identity.market"
            ),
            "currency": _required_text(
                source.identity.currency, f"sources[{position}].identity.currency"
            ),
        }
        expected_security_identity = {
            key: identity_values[key] for key in ("symbol", "market", "currency")
        }
        if source_identity != expected_security_identity:
            raise OHLCVConsensusError(
                f"sources[{position}] security identity does not match consensus identity"
            )
        artifact_sha256 = source.input_artifact_sha256
        if not isinstance(artifact_sha256, str) or not artifact_sha256:
            raise OHLCVConsensusError(
                f"sources[{position}].input_artifact_sha256 is required"
            )
        artifact_sha256 = artifact_sha256.casefold()
        if len(artifact_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in artifact_sha256
        ):
            raise OHLCVConsensusError(
                f"sources[{position}].input_artifact_sha256 must be a SHA-256 digest"
            )
        source_id = _required_text(
            source.source_id if source.source_id is not None else provider,
            f"sources[{position}].source_id",
        )
        if source_id in source_ids:
            raise OHLCVConsensusError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        prior_group = provider_groups.setdefault(provider, group)
        if prior_group != group:
            raise OHLCVConsensusError(
                f"provider {provider!r} cannot belong to multiple independence groups"
            )
        bases.add(basis)
        volume_units.add(unit)
        groups.add(group)
        prepared.append((source, source_id, source_identity, artifact_sha256))

    if len(groups) < 2:
        raise OHLCVConsensusError("at least two independent source groups are required")
    if len(groups) > 3:
        raise OHLCVConsensusError("two-of-three consensus supports at most three groups")
    if len(bases) != 1:
        raise OHLCVConsensusError("price bases are incompatible")
    if len(volume_units) != 1:
        raise OHLCVConsensusError("volume units are incompatible")

    normalized = sorted(
        (
            _normalize_source(source, source_id, source_identity, artifact_sha256)
            for source, source_id, source_identity, artifact_sha256 in prepared
        ),
        key=lambda item: item.source_id,
    )
    all_dates = sorted(
        calendar_set
        | {
            date
            for source in normalized
            for date in (*source.rows.keys(), *source.invalid_rows.keys())
        }
    )
    accepted_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for date in all_dates:
        source_decisions: dict[str, dict[str, Any]] = {}
        for source in normalized:
            if date in source.invalid_rows:
                source_decisions[source.source_id] = {
                    "source_id": source.source_id,
                    "provider": source.provider,
                    "independence_group": source.independence_group,
                    "status": "invalid",
                    "reason_codes": list(source.invalid_rows[date]),
                }
            elif date not in source.rows:
                source_decisions[source.source_id] = {
                    "source_id": source.source_id,
                    "provider": source.provider,
                    "independence_group": source.independence_group,
                    "status": "missing",
                    "reason_codes": ["source_missing"],
                }
            else:
                source_decisions[source.source_id] = {
                    "source_id": source.source_id,
                    "provider": source.provider,
                    "independence_group": source.independence_group,
                    "status": "pending",
                    "reason_codes": [],
                }

        if date not in calendar_set:
            for source_decision in source_decisions.values():
                if source_decision["status"] == "pending":
                    source_decision["status"] = "excluded"
                    source_decision["reason_codes"] = ["outside_actual_session_calendar"]
            diagnostics.append(
                {
                    "date": date,
                    "status": "quarantined",
                    "reason_codes": ["outside_actual_session_calendar"],
                    "supporting_groups": [],
                    "sources": [
                        source_decisions[key] for key in sorted(source_decisions)
                    ],
                    "fields": {
                        column: {
                            "status": "quarantined",
                            "consensus": None,
                            "relative_spread": None,
                            "contributing_sources": [],
                            "excluded_sources": sorted(source_ids),
                        }
                        for column in OHLCV_COLUMNS
                    },
                }
            )
            continue

        group_votes: dict[str, _Vote] = {}
        for group in sorted(groups):
            group_sources = [
                source for source in normalized if source.independence_group == group
            ]
            provider_votes: list[_Vote] = []
            for provider in sorted({source.provider for source in group_sources}):
                provider_sources = [
                    source
                    for source in group_sources
                    if source.provider == provider and date in source.rows
                ]
                if not provider_sources:
                    continue
                votes = [
                    _Vote(
                        values=source.rows[date].values,
                        source_ids=(source.source_id,),
                    )
                    for source in provider_sources
                ]
                if not _votes_compatible(votes, policy):
                    for source in provider_sources:
                        source_decisions[source.source_id]["status"] = "excluded"
                        source_decisions[source.source_id]["reason_codes"] = [
                            "provider_internal_disagreement"
                        ]
                    continue
                provider_votes.append(_median_vote(votes))
            if not provider_votes:
                continue
            if not _votes_compatible(provider_votes, policy):
                involved = {
                    source_id
                    for vote in provider_votes
                    for source_id in vote.source_ids
                }
                for source_id in involved:
                    source_decisions[source_id]["status"] = "excluded"
                    source_decisions[source_id]["reason_codes"] = [
                        "group_internal_disagreement"
                    ]
                continue
            group_votes[group] = _median_vote(provider_votes)

        supporting_groups: tuple[str, ...] = ()
        decision = "quarantined"
        reason_codes: set[str] = set()
        usable_groups = sorted(group_votes)
        if len(usable_groups) < 2:
            reason_codes.add("insufficient_independent_groups")
        elif len(usable_groups) == 2:
            pair = [group_votes[group] for group in usable_groups]
            if _votes_compatible(pair, policy):
                supporting_groups = tuple(usable_groups)
                decision = "accepted"
            else:
                reason_codes.add("independent_group_disagreement")
                reason_codes.add(f"{_comparison_band(pair, policy)}_band")
        else:
            compatible_pairs: list[tuple[str, str]] = []
            for left_position, left in enumerate(usable_groups):
                for right in usable_groups[left_position + 1 :]:
                    if _votes_compatible([group_votes[left], group_votes[right]], policy):
                        compatible_pairs.append((left, right))
            if len(compatible_pairs) == 3:
                supporting_groups = tuple(usable_groups)
                decision = "accepted"
            elif len(compatible_pairs) == 1:
                supporting_groups = compatible_pairs[0]
                decision = "warning"
                reason_codes.add("outlier_excluded")
            elif len(compatible_pairs) > 1:
                reason_codes.add("ambiguous_support_cluster")
            else:
                reason_codes.add("independent_group_disagreement")
                reason_codes.add(
                    f"{_comparison_band([group_votes[group] for group in usable_groups], policy)}_band"
                )

        accepted_vote: _Vote | None = None
        if supporting_groups:
            accepted_vote = _median_vote(
                [group_votes[group] for group in supporting_groups]
            )
            field_warning = False
            for column in OHLCV_COLUMNS:
                spread = _relative_spread(
                    [group_votes[group].values[column] for group in supporting_groups]
                )
                if spread > _field_threshold(column, policy, "pass"):
                    field_warning = True
            if field_warning:
                decision = "warning"
                reason_codes.add("field_tolerance_warning")

        contributing_sources: set[str] = set()
        if supporting_groups:
            contributing_sources = {
                source_id
                for group in supporting_groups
                for source_id in group_votes[group].source_ids
            }
        for source_id, source_decision in source_decisions.items():
            if source_decision["status"] != "pending":
                continue
            group = source_decision["independence_group"]
            if source_id in contributing_sources:
                source_decision["status"] = "contributed"
            else:
                source_decision["status"] = "excluded"
                if decision == "quarantined":
                    source_decision["reason_codes"] = ["session_quarantined"]
                elif group not in supporting_groups:
                    outlier_band = _comparison_band(
                        [group_votes[group], accepted_vote], policy
                    )
                    source_decision["reason_codes"] = sorted(
                        [f"{outlier_band}_band", "group_outlier"]
                    )
                else:
                    source_decision["reason_codes"] = ["not_in_group_vote"]

        nonclean_sources = any(
            item["status"] in {"missing", "invalid", "excluded"}
            for item in source_decisions.values()
        )
        if accepted_vote is not None and nonclean_sources:
            decision = "warning"
            reason_codes.add("incomplete_source_support")

        if accepted_vote is not None:
            record = {"date": date}
            for column in OHLCV_COLUMNS:
                value = float(accepted_vote.values[column])
                record[column] = 0.0 if value == 0 else value
            accepted_records.append(record)

        fields: dict[str, dict[str, Any]] = {}
        excluded_ids = sorted(set(source_ids) - contributing_sources)
        for column in OHLCV_COLUMNS:
            group_values = [
                group_votes[group].values[column] for group in supporting_groups
            ]
            spread = _relative_spread(group_values) if group_values else None
            if accepted_vote is None:
                field_status = "quarantined"
                consensus_value = None
            else:
                consensus_value = accepted_vote.values[column]
                field_status = (
                    "accepted"
                    if spread is not None
                    and spread <= _field_threshold(column, policy, "pass")
                    else "warning"
                )
            fields[column] = {
                "status": field_status,
                "consensus": consensus_value,
                "relative_spread": spread,
                "contributing_sources": sorted(contributing_sources),
                "excluded_sources": excluded_ids,
            }

        diagnostics.append(
            {
                "date": date,
                "status": decision,
                "reason_codes": sorted(reason_codes),
                "supporting_groups": list(supporting_groups),
                "sources": [source_decisions[key] for key in sorted(source_decisions)],
                "fields": fields,
            }
        )

    bars_sha256 = _digest("quantkit.ohlcv-accepted-bars.v1", accepted_records)
    if not accepted_records:
        status = "quarantined"
    elif any(item["status"] != "accepted" for item in diagnostics):
        status = "partial"
    else:
        status = "accepted"

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": status,
        "identity": {
            **identity_values,
            "price_basis": next(iter(bases)),
            "volume_unit": next(iter(volume_units)),
        },
        "policy": {
            "price_relative": {
                "pass": float(policy.price_pass),
                "warning": float(policy.price_warning),
                "quarantine": float(policy.price_quarantine),
            },
            "volume_relative": {
                "pass": float(policy.volume_pass),
                "warning": float(policy.volume_warning),
                "quarantine": float(policy.volume_quarantine),
            },
            "minimum_independent_groups": 2,
            "maximum_independent_groups": 3,
            "alignment": "observed_union_no_forward_fill",
            "vote_rule": "hierarchical_independence_group_2_of_3_median",
            "hash_domains": {
                "frame_sha256": "quantkit.ohlcv-source-frame.v1",
                "calendar_sessions_sha256": "quantkit.ohlcv-session-calendar.v1",
                "bars_sha256": "quantkit.ohlcv-accepted-bars.v1",
                "output_sha256": "quantkit.ohlcv-consensus-manifest.v1",
            },
        },
        "calendar": {
            "name": identity_values["calendar"],
            "sessions": calendar_sessions,
            "sessions_sha256": _digest(
                "quantkit.ohlcv-session-calendar.v1", calendar_sessions
            ),
        },
        "bars": accepted_records,
        "bars_sha256": bars_sha256,
        "sources": [_source_manifest(source) for source in normalized],
        "diagnostics": diagnostics,
    }
    manifest["output_sha256"] = _digest(
        "quantkit.ohlcv-consensus-manifest.v1", manifest
    )

    if accepted_records:
        accepted_bars = pd.DataFrame.from_records(accepted_records)
        accepted_bars["date"] = pd.to_datetime(accepted_bars["date"])
        accepted_bars = accepted_bars.set_index("date")[list(OHLCV_COLUMNS)]
    else:
        accepted_bars = _empty_bars()
    return OHLCVConsensusResult(
        _accepted_bars=accepted_bars,
        _diagnostics=tuple(copy.deepcopy(diagnostics)),
        _manifest=copy.deepcopy(manifest),
    )


__all__ = [
    "ConsensusTolerance",
    "MANIFEST_SCHEMA",
    "OHLCVConsensusError",
    "OHLCVConsensusResult",
    "OHLCVIdentity",
    "OHLCVSecurityIdentity",
    "OHLCVSource",
    "build_ohlcv_consensus",
]

"""Dependency-free validation helpers for GALAHAD research artifacts.

The JSON Schema documents are authoritative.  This module implements only the
small, deliberately constrained keyword subset used by those contracts, plus
point-in-time and filesystem checks that JSON Schema cannot express safely.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sysconfig
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_SCHEMA_CANDIDATES = (
    Path(__file__).resolve().parent / "schemas",
    Path(sysconfig.get_path("data")) / "researchkit" / "schemas",
)
_KNOWN_SCHEMAS = {
    "source-artifact": "source-artifact.schema.json",
    "calculation-artifact": "calculation-artifact.schema.json",
    "research-artifact": "research-artifact.schema.json",
}
_SCHEMA_DIRECTORIES = {"1.0.0": "v1", "2.0.0": "v2"}
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class ContractError(ValueError):
    """Report one or more contract violations."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable UTF-8 encoding."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(["$: value must contain only finite JSON-compatible data"]) from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def artifact_sha256(document: Mapping[str, Any]) -> str:
    """Hash an artifact envelope, excluding its content-derived ID.

    This is deliberately distinct from a SourceArtifact's snapshot-byte
    ``content_hash`` and a CalculationArtifact's result-only ``output_hash``.
    """

    if not isinstance(document, Mapping):
        raise ContractError(["$: expected object"])
    envelope = dict(document)
    envelope.pop("artifact_id", None)
    return canonical_sha256(envelope)


def seal_artifact(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with a content-derived artifact ID.

    Omit ``artifact_id`` from the hash input so repeated sealing is idempotent.
    The artifact type determines the human-readable ID prefix.
    """

    if not isinstance(document, Mapping):
        raise ContractError(["$: expected object"])
    sealed = dict(document)
    sealed.pop("artifact_id", None)
    prefixes = {"source": "src", "calculation": "calc", "research": "res"}
    try:
        prefix = prefixes[str(sealed["artifact_type"])]
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(["$.artifact_type: expected source, calculation, or research"]) from exc
    try:
        digest = artifact_sha256(sealed).split(":", 1)[1]
    except ContractError as exc:
        raise ContractError(["$: artifact must contain only finite JSON-compatible values"]) from exc
    sealed["artifact_id"] = f"{prefix}_{digest}"
    return sealed


def validate_artifact(document: Mapping[str, Any], contract: str | None = None) -> None:
    """Validate an artifact against its JSON contract and semantic invariants."""

    if not isinstance(document, Mapping):
        raise ContractError(["$: expected object"])
    selected = contract or str(document.get("artifact_type", ""))
    aliases = {"source": "source-artifact", "calculation": "calculation-artifact", "research": "research-artifact"}
    selected = aliases.get(selected, selected)
    if selected not in _KNOWN_SCHEMAS:
        raise ContractError([f"$: unknown artifact contract {selected!r}"])

    version = str(document.get("schema_version", ""))
    schema_directory = _SCHEMA_DIRECTORIES.get(version)
    if schema_directory is None:
        raise ContractError([f"$.schema_version: unsupported artifact version {version!r}"])
    schema_path = next(
        (
            directory / schema_directory / _KNOWN_SCHEMAS[selected]
            for directory in _SCHEMA_CANDIDATES
            if (directory / schema_directory / _KNOWN_SCHEMAS[selected]).is_file()
        ),
        None,
    )
    if schema_path is None:
        raise ContractError([f"$: installed schema is missing for {selected}"])
    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)

    errors: list[str] = []
    _validate_node(document, schema, "$", errors)
    if not errors:
        _validate_semantics(document, selected, errors)
    if errors:
        raise ContractError(errors)


def verify_source_snapshot(document: Mapping[str, Any], repository_root: str | Path) -> Path:
    """Verify that a SourceArtifact points to an in-root file with the stated hash."""

    validate_artifact(document, "source-artifact")
    try:
        root = Path(repository_root).resolve()
        snapshot = (root / str(document["snapshot_path"])).resolve()
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(["$.snapshot_path: cannot resolve snapshot path"]) from exc
    if root != snapshot and root not in snapshot.parents:
        raise ContractError(["$.snapshot_path: resolves outside repository_root"])
    try:
        if not snapshot.is_file():
            raise ContractError(["$.snapshot_path: snapshot file does not exist"])
        actual = "sha256:" + hashlib.sha256(snapshot.read_bytes()).hexdigest()
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(["$.snapshot_path: cannot read snapshot bytes"]) from exc
    if actual != document["content_hash"]:
        raise ContractError(["$.content_hash: does not match snapshot bytes"])
    return snapshot


def validate_artifact_graph(
    documents: Iterable[Mapping[str, Any]],
    *,
    root_artifact_ids: Sequence[str] | None = None,
    repository_root: str | Path | None = None,
    require_all_reachable: bool = False,
) -> dict[str, Any]:
    """Validate a strict v2 artifact graph and every referenced JSON Pointer.

    The caller supplies the complete local evidence pack.  References never
    trigger acquisition.  Source snapshot pointers are resolved only when a
    repository root is supplied, so replay remains explicit and offline.
    """

    if isinstance(documents, (str, bytes, Mapping)):
        raise ContractError(["$: artifact graph must be an iterable of documents"])
    try:
        documents = tuple(documents)
    except TypeError as exc:
        raise ContractError(["$: artifact graph must be an iterable of documents"]) from exc
    if not documents:
        raise ContractError(["$: artifact graph must contain at least one document"])
    errors: list[str] = []
    index: dict[str, Mapping[str, Any]] = {}
    for position, document in enumerate(documents):
        try:
            validate_artifact(document)
        except ContractError as exc:
            errors.extend(f"$[{position}]{message[1:]}" for message in exc.errors)
            continue
        if document.get("schema_version") != "2.0.0":
            errors.append(f"$[{position}].schema_version: graph validation requires 2.0.0")
        artifact_id = str(document["artifact_id"])
        if artifact_id in index:
            errors.append(f"$[{position}].artifact_id: duplicate artifact ID")
        else:
            index[artifact_id] = document
    if errors:
        raise ContractError(errors)
    if repository_root is None and any(
        document["artifact_type"] == "source" for document in index.values()
    ):
        raise ContractError(["$.repository_root: required to verify SourceArtifact snapshots"])

    roots_supplied = root_artifact_ids is not None
    if not isinstance(require_all_reachable, bool):
        errors.append("$.require_all_reachable: expected boolean")
    if isinstance(root_artifact_ids, (str, bytes, Mapping)):
        errors.append("$.root_artifact_ids: expected an iterable of artifact IDs, not a string")
        roots = []
    elif root_artifact_ids is not None:
        try:
            roots = list(root_artifact_ids)
        except TypeError:
            errors.append("$.root_artifact_ids: expected an iterable of artifact IDs")
            roots = []
    else:
        roots = [
            artifact_id
            for artifact_id, document in index.items()
            if document["artifact_type"] == "research"
        ]
    if not roots and not roots_supplied:
        roots = list(index)
    valid_roots: list[str] = []
    for position, root in enumerate(roots):
        if not isinstance(root, str) or not root:
            errors.append(
                f"$.root_artifact_ids[{position}]: expected a non-empty artifact ID string"
            )
            continue
        valid_roots.append(root)
    roots = valid_roots
    if roots_supplied and not roots:
        errors.append("$.root_artifact_ids: must contain at least one artifact ID")
    if len(roots) != len(set(roots)):
        errors.append("$.root_artifact_ids: values must be unique")
    for root in roots:
        if root not in index:
            errors.append(f"$.root_artifact_ids: missing artifact {root}")

    snapshot_paths: dict[str, Path] = {}
    snapshots: dict[str, Any] = {}
    references: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for artifact_id, document in index.items():
        if repository_root is not None and document["artifact_type"] == "source":
            try:
                snapshot_paths[artifact_id] = verify_source_snapshot(document, repository_root)
            except ContractError as exc:
                detail = str(exc)
                errors.append(f"$[{artifact_id}].snapshot: {detail}")
        refs = _document_references(document)
        references[artifact_id] = refs
    snapshot_targets = {
        str(reference["artifact_id"])
        for refs in references.values()
        for _, reference in refs
        if reference.get("target") == "snapshot"
    }
    snapshot_targets.update(
        str(reference["artifact_id"])
        for document in index.values()
        if document["artifact_type"] == "calculation"
        and document["calculation_type"] == "sec-companyfacts-normalization"
        for reference in document["inputs"]
        if reference.get("role") in {"subject", "availability"}
    )
    for artifact_id in snapshot_targets:
        snapshot_path = snapshot_paths.get(artifact_id)
        if snapshot_path is None:
            continue
        try:
            snapshots[artifact_id] = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"$[{artifact_id}].snapshot: referenced snapshot is not JSON ({exc})")
    for artifact_id, document in index.items():
        refs = references[artifact_id]
        for path, reference in refs:
            _validate_reference(path, reference, index, snapshots, errors)
        if document["artifact_type"] == "calculation":
            _validate_calculation_graph_fields(document, index, snapshots, errors)
            _validate_calculation_reconciliation(document, index, snapshots, errors)
        elif document["artifact_type"] == "research":
            _validate_research_graph_fields(document, index, errors)

    reachable: set[str] = set()
    active: set[str] = set()
    for root in roots:
        if root not in index or root in reachable:
            continue
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            artifact_id, exiting = stack.pop()
            if artifact_id not in index:
                continue
            if exiting:
                active.discard(artifact_id)
                reachable.add(artifact_id)
                continue
            if artifact_id in reachable:
                continue
            if artifact_id in active:
                errors.append(f"$.graph: cycle detected at {artifact_id}")
                continue
            active.add(artifact_id)
            stack.append((artifact_id, True))
            for _, reference in reversed(references.get(artifact_id, [])):
                child = str(reference["artifact_id"])
                if child in active:
                    errors.append(f"$.graph: cycle detected at {child}")
                elif child not in reachable:
                    stack.append((child, False))
    orphans = sorted(set(index) - reachable)
    if require_all_reachable and orphans:
        errors.append("$.graph: unreachable artifacts: " + ", ".join(orphans))
    if errors:
        raise ContractError(errors)
    return {
        "status": "pass",
        "artifact_count": len(index),
        "root_artifact_ids": roots,
        "reachable_artifact_ids": sorted(reachable),
        "orphan_artifact_ids": orphans,
        "snapshots_verified": len(snapshot_paths),
    }


def _document_references(
    document: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    kind = document["artifact_type"]
    if kind == "calculation":
        refs = [
            (f"$.inputs[{index}]", reference)
            for index, reference in enumerate(document["inputs"])
        ]
        refs.extend(
            (f"$.lineage[{index}]", reference)
            for index, reference in enumerate(document["lineage"])
        )
        refs.extend(
            (f"$.lineage[{index}].availability_ref", record["availability_ref"])
            for index, record in enumerate(document["lineage"])
            if "availability_ref" in record
        )
        return refs
    if kind == "research":
        return [
            (f"$.evidence_refs[{index}]", reference)
            for index, reference in enumerate(document["evidence_refs"])
        ]
    return []


def _validate_reference(
    path: str,
    reference: Mapping[str, Any],
    index: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Any],
    errors: list[str],
) -> None:
    artifact_id = str(reference["artifact_id"])
    target = index.get(artifact_id)
    if target is None:
        errors.append(f"{path}.artifact_id: dangling reference {artifact_id}")
        return
    expected_hash = artifact_sha256(target)
    if reference["artifact_hash"] != expected_hash:
        errors.append(f"{path}.artifact_hash: does not match referenced artifact")
    pointer = reference.get("json_pointer")
    if pointer is None:
        return
    target_kind = reference.get("target")
    if target_kind == "snapshot":
        if target["artifact_type"] != "source":
            errors.append(f"{path}.target: snapshot target must be a SourceArtifact")
            return
        if artifact_id not in snapshots:
            errors.append(f"{path}.target: snapshot replay requires repository_root and JSON bytes")
            return
        pointer_target = snapshots[artifact_id]
    else:
        pointer_target = target
    try:
        _resolve_json_pointer(pointer_target, str(pointer))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        errors.append(f"{path}.json_pointer: does not resolve ({exc})")


def _validate_calculation_graph_fields(
    document: Mapping[str, Any],
    index: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Any],
    errors: list[str],
) -> None:
    input_ids = {str(reference["artifact_id"]) for reference in document["inputs"]}
    input_roles = {
        str(reference["artifact_id"]): str(reference["role"])
        for reference in document["inputs"]
    }
    if not any(reference["role"] == "subject" for reference in document["inputs"]):
        errors.append("$.inputs: at least one subject input is required")
    lineage_ids: set[str] = set()
    for position, record in enumerate(document["lineage"]):
        lineage_ids.add(str(record["artifact_id"]))
        if "availability_ref" in record:
            lineage_ids.add(str(record["availability_ref"]["artifact_id"]))
        try:
            parameter_value = _resolve_json_pointer(document, str(record["parameter_pointer"]))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f"$.lineage[{position}].parameter_pointer: does not resolve ({exc})")
            continue
        try:
            reported_value = _resolve_reference_value(record, index, snapshots)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f"$.lineage[{position}].json_pointer: cannot read reported value ({exc})")
            continue
        if not _json_values_equal(reported_value, record["reported_value"]):
            errors.append(f"$.lineage[{position}].reported_value: differs from cited source value")
        try:
            normalized_value = _apply_normalization(record["reported_value"], record["normalization"])
        except (TypeError, ValueError) as exc:
            errors.append(f"$.lineage[{position}].normalization: {exc}")
        else:
            if not _json_values_equal(normalized_value, parameter_value):
                errors.append(f"$.lineage[{position}]: normalized reported_value does not match parameter")
        available_at = record.get("available_at")
        if available_at is not None and _parse_datetime(str(available_at)) > _parse_datetime(str(document["as_of_cutoff"])):
            errors.append(f"$.lineage[{position}].available_at: later than calculation cutoff")
        availability_ref = record.get("availability_ref")
        if availability_ref is not None and available_at is not None:
            try:
                cited = _resolve_reference_value(availability_ref, index, snapshots)
                cited_time = _parse_datetime(str(cited))
                declared_time = _parse_datetime(str(available_at))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                errors.append(f"$.lineage[{position}].availability_ref: cannot read an RFC 3339 instant ({exc})")
            else:
                if cited_time != declared_time:
                    errors.append(f"$.lineage[{position}].available_at: differs from availability_ref")
        source_id = str(record["artifact_id"])
        source = index.get(source_id)
        required_role = _required_lineage_role(
            str(document["calculation_type"]), str(record["parameter_pointer"])
        )
        if required_role is not None and input_roles.get(source_id) != required_role:
            errors.append(
                f"$.lineage[{position}]: parameter requires an input with role {required_role}"
            )
        if source is not None and source.get("source_kind") == "sec-companyfacts":
            for key in (
                "accession",
                "available_at",
                "availability_ref",
                "reported_concept",
                "reported_unit",
                "amendment_sequence",
                "restatement_status",
            ):
                if key not in record:
                    errors.append(f"$.lineage[{position}].{key}: required for SEC CompanyFacts")
            if availability_ref is not None:
                availability_source = index.get(str(availability_ref["artifact_id"]))
                if availability_source is not None and availability_source.get("source_kind") != "sec-submissions":
                    errors.append(f"$.lineage[{position}].availability_ref: must cite SEC Submissions")
                else:
                    try:
                        fact_pointer = str(record["json_pointer"])
                        concept, unit = _parse_sec_companyfacts_pointer(fact_pointer)
                        fact_object_pointer = fact_pointer.rsplit("/", 1)[0]
                        fact_object = _resolve_json_pointer(snapshots[str(record["artifact_id"])], fact_object_pointer)
                        availability_pointer = str(availability_ref["json_pointer"])
                        if not re.fullmatch(
                            r"/filings/recent/acceptanceDateTime/(?:0|[1-9][0-9]*)",
                            availability_pointer,
                        ):
                            raise ValueError(
                                "availability_ref must cite filings/recent/acceptanceDateTime/<index>"
                            )
                        availability_parent = availability_pointer.rsplit("/", 2)[0]
                        accession_pointer = availability_parent + "/accessionNumber/" + availability_pointer.rsplit("/", 1)[1]
                        cited_accession = _resolve_json_pointer(
                            snapshots[str(availability_ref["artifact_id"])], accession_pointer
                        )
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        errors.append(f"$.lineage[{position}]: cannot verify SEC accession join ({exc})")
                    else:
                        if not isinstance(fact_object, Mapping) or fact_object.get("accn") != record.get("accession"):
                            errors.append(f"$.lineage[{position}].accession: differs from CompanyFacts fact")
                        if cited_accession != record.get("accession"):
                            errors.append(f"$.lineage[{position}].accession: differs from Submissions filing")
                        if concept != record.get("reported_concept"):
                            errors.append(f"$.lineage[{position}].reported_concept: differs from CompanyFacts pointer")
                        if unit != record.get("reported_unit"):
                            errors.append(f"$.lineage[{position}].reported_unit: differs from CompanyFacts pointer")
                        if availability_source is not None:
                            fact_entity = _artifact_entity_id(source)
                            availability_entity = _artifact_entity_id(availability_source)
                            if fact_entity != availability_entity or fact_entity != document["entity_id"]:
                                errors.append(f"$.lineage[{position}].availability_ref: SEC sources have a different entity")
                            try:
                                fact_cik = str(snapshots[source_id]["cik"]).lstrip("0")
                                submissions_cik = str(
                                    snapshots[str(availability_ref["artifact_id"])]["cik"]
                                ).lstrip("0")
                                envelope_cik = _sec_entity_cik(str(document["entity_id"]))
                            except (KeyError, TypeError, ValueError) as exc:
                                errors.append(f"$.lineage[{position}]: cannot verify SEC CIK join ({exc})")
                            else:
                                if fact_cik != submissions_cik or fact_cik != envelope_cik:
                                    errors.append(f"$.lineage[{position}]: SEC CIK differs across sources and calculation")
    if document["calculation_type"] == "sec-companyfacts-normalization":
        assumption_ids = {
            artifact_id
            for artifact_id, role in input_roles.items()
            if role == "assumption"
        }
        if lineage_ids != assumption_ids:
            errors.append(
                "$.lineage: SEC normalization lineage must exactly match assumption inputs"
            )
        _validate_sec_normalization_inputs(document, index, snapshots, errors)
    elif lineage_ids != input_ids:
        errors.append("$.lineage: referenced artifact IDs must exactly match $.inputs inventory")
    _validate_lineage_coverage(document, errors)

    cutoff = _parse_datetime(str(document["as_of_cutoff"]))
    created = _parse_datetime(str(document["created_at"]))
    if created < cutoff:
        errors.append("$.created_at: must not precede as_of_cutoff")
    for input_ref in document["inputs"]:
        upstream = index.get(str(input_ref["artifact_id"]))
        if upstream is None:
            continue
        upstream_cutoff = upstream.get("as_of_cutoff")
        if upstream_cutoff is not None and _parse_datetime(str(upstream_cutoff)) > cutoff:
            errors.append(f"$.inputs: upstream {upstream['artifact_id']} cutoff is later")
        upstream_created = upstream.get("created_at") or upstream.get("retrieved_at")
        if upstream_created is not None and _parse_datetime(str(upstream_created)) > created:
            errors.append(f"$.inputs: upstream {upstream['artifact_id']} was created later")
        if input_ref["role"] == "subject" and _artifact_entity_id(upstream) != document["entity_id"]:
            errors.append(f"$.inputs: subject input {upstream['artifact_id']} has a different entity")


def _validate_lineage_coverage(document: Mapping[str, Any], errors: list[str]) -> None:
    kind = document["calculation_type"]
    parameters = document["parameters"]
    required: set[str] = set()
    expected_keys: set[str]
    if kind == "sec-companyfacts-normalization":
        expected_keys = {"periods", "scale", "currency", "unit", "period_basis"}
        required.update(
            {
                "/parameters/scale",
                "/parameters/currency",
                "/parameters/unit",
                "/parameters/period_basis",
            }
        )
        for period_index, period in enumerate(parameters.get("periods", [])):
            if not isinstance(period, Mapping):
                continue
            required.update(
                f"/parameters/periods/{period_index}/{field}"
                for field in ("period_end", "accession", "form", "fy", "fp")
            )
            fields = period.get("fields", {})
            if isinstance(fields, Mapping):
                for field, selector in fields.items():
                    if isinstance(selector, Mapping):
                        required.update(
                            f"/parameters/periods/{period_index}/fields/{field}/{selector_field}"
                            for selector_field in selector
                        )
    elif kind == "financial_statement_analysis":
        expected_keys = {"dataset", "reconciliation_tolerance"}
        required.add("/parameters/reconciliation_tolerance")
        fields = (
            "period_end",
            "currency",
            "unit",
            "period_basis",
            "revenue",
            "cost_of_revenue",
            "operating_expenses",
            "operating_income",
            "net_income",
            "cash_from_operations",
            "capital_expenditure",
            "total_assets",
            "total_liabilities",
            "total_equity",
            "cash",
            "debt",
        )
        dataset = parameters.get("dataset", [])
        if not isinstance(dataset, list):
            errors.append("$.parameters.dataset: expected array")
            dataset = []
        expected_fields = set(fields)
        for position, row in enumerate(dataset):
            if not isinstance(row, Mapping):
                errors.append(f"$.parameters.dataset[{position}]: expected object")
                continue
            _validate_exact_parameter_fields(
                row,
                expected_fields,
                f"$.parameters.dataset[{position}]",
                errors,
            )
            required.update(f"/parameters/dataset/{position}/{field}" for field in fields)
    elif kind == "dcf-valuation":
        expected_keys = {
            "forecasts",
            "wacc",
            "terminal_growth",
            "net_debt",
            "non_operating_assets",
            "diluted_shares",
            "wacc_values",
            "terminal_growth_values",
        }
        required.update(
            f"/parameters/{field}"
            for field in (
                "wacc",
                "terminal_growth",
                "net_debt",
                "non_operating_assets",
                "diluted_shares",
            )
        )
        for axis in ("wacc_values", "terminal_growth_values"):
            values = parameters.get(axis, [])
            if not isinstance(values, list):
                errors.append(f"$.parameters.{axis}: expected array")
                values = []
            required.update(f"/parameters/{axis}/{position}" for position, _ in enumerate(values))
        forecasts = parameters.get("forecasts", [])
        if not isinstance(forecasts, list):
            errors.append("$.parameters.forecasts: expected array")
            forecasts = []
        forecast_fields = {"period", "unlevered_fcf", "discount_period"}
        for position, row in enumerate(forecasts):
            if not isinstance(row, Mapping):
                errors.append(f"$.parameters.forecasts[{position}]: expected object")
                continue
            _validate_exact_parameter_fields(
                row,
                forecast_fields,
                f"$.parameters.forecasts[{position}]",
                errors,
            )
            required.update(
                f"/parameters/forecasts/{position}/{field}"
                for field in forecast_fields
            )
    elif kind == "comps-valuation":
        expected_keys = {
            "peers",
            "subject_metric",
            "basis",
            "diluted_shares",
            "net_debt",
            "non_operating_assets",
            "statistic",
            "explicit_multiple",
        }
        required.update(
            f"/parameters/{field}"
            for field in expected_keys - {"peers"}
        )
        peers = parameters.get("peers", [])
        if not isinstance(peers, list):
            errors.append("$.parameters.peers: expected array")
            peers = []
        for position, peer in enumerate(peers):
            if not isinstance(peer, Mapping):
                errors.append(f"$.parameters.peers[{position}]: expected object")
                continue
            if peer.get("included") is True:
                peer_fields = {"name", "included", "numerator", "denominator"}
            elif peer.get("included") is False:
                peer_fields = {"name", "included", "exclusion_reason"}
            else:
                peer_fields = {"name", "included"}
            _validate_exact_parameter_fields(
                peer,
                peer_fields,
                f"$.parameters.peers[{position}]",
                errors,
            )
            required.update(
                f"/parameters/peers/{position}/{field}" for field in peer_fields
            )
    else:
        errors.append(f"$.calculation_type: no strict lineage contract for {kind}")
        return
    if set(parameters) != expected_keys:
        missing_keys = sorted(expected_keys - set(parameters))
        extra_keys = sorted(set(parameters) - expected_keys)
        if missing_keys:
            errors.append("$.parameters: missing method parameters: " + ", ".join(missing_keys))
        if extra_keys:
            errors.append("$.parameters: unexpected method parameters: " + ", ".join(extra_keys))
    pointers = [str(record["parameter_pointer"]) for record in document["lineage"]]
    if len(pointers) != len(set(pointers)):
        errors.append("$.lineage: parameter_pointer values must be unique")
    actual = set(pointers)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        if missing:
            errors.append("$.lineage: missing parameter coverage: " + ", ".join(missing))
        if extra:
            errors.append("$.lineage: unexpected parameter coverage: " + ", ".join(extra))


def _validate_exact_parameter_fields(
    value: Mapping[str, Any],
    expected: set[str],
    path: str,
    errors: list[str],
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"{path}: missing fields: " + ", ".join(missing))
    if extra:
        errors.append(f"{path}: unexpected fields: " + ", ".join(extra))


def _validate_research_graph_fields(
    document: Mapping[str, Any],
    index: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> None:
    cutoff = _parse_datetime(str(document["as_of_cutoff"]))
    created = _parse_datetime(str(document["created_at"]))
    for position, reference in enumerate(document["evidence_refs"]):
        upstream = index.get(str(reference["artifact_id"]))
        if upstream is None:
            continue
        if _artifact_entity_id(upstream) != document["entity_id"]:
            errors.append(f"$.evidence_refs[{position}]: evidence has a different entity")
        upstream_cutoff = upstream.get("as_of_cutoff")
        if upstream_cutoff is not None and _parse_datetime(str(upstream_cutoff)) > cutoff:
            errors.append(f"$.evidence_refs[{position}]: evidence cutoff is later")
        upstream_created = upstream.get("created_at") or upstream.get("retrieved_at")
        if upstream_created is not None and _parse_datetime(str(upstream_created)) > created:
            errors.append(f"$.evidence_refs[{position}]: evidence was created later")


def _validate_sec_normalization_inputs(
    document: Mapping[str, Any],
    index: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Any],
    errors: list[str],
) -> None:
    roles: dict[str, list[str]] = {}
    for reference in document["inputs"]:
        roles.setdefault(str(reference["role"]), []).append(str(reference["artifact_id"]))
    if len(roles.get("subject", [])) != 1:
        errors.append("$.inputs: SEC normalization requires exactly one subject CompanyFacts source")
        return
    if len(roles.get("availability", [])) != 1:
        errors.append("$.inputs: SEC normalization requires exactly one availability Submissions source")
        return
    companyfacts_id = roles["subject"][0]
    submissions_id = roles["availability"][0]
    companyfacts_artifact = index.get(companyfacts_id)
    submissions_artifact = index.get(submissions_id)
    if companyfacts_artifact is None or submissions_artifact is None:
        return
    if companyfacts_artifact.get("source_kind") != "sec-companyfacts":
        errors.append("$.inputs: subject must be a sec-companyfacts SourceArtifact")
    if submissions_artifact.get("source_kind") != "sec-submissions":
        errors.append("$.inputs: availability must be a sec-submissions SourceArtifact")
    entities = {
        _artifact_entity_id(companyfacts_artifact),
        _artifact_entity_id(submissions_artifact),
        str(document["entity_id"]),
    }
    if len(entities) != 1:
        errors.append("$.inputs: SEC normalization sources have a different entity")
    if companyfacts_id not in snapshots or submissions_id not in snapshots:
        return
    try:
        companyfacts_cik = str(snapshots[companyfacts_id]["cik"]).lstrip("0")
        submissions_cik = str(snapshots[submissions_id]["cik"]).lstrip("0")
        entity_cik = _sec_entity_cik(str(document["entity_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"$.inputs: cannot verify SEC CIK join ({exc})")
    else:
        if companyfacts_cik != submissions_cik or companyfacts_cik != entity_cik:
            errors.append("$.inputs: SEC CIK differs across sources and calculation")

    try:
        from .sec import normalize_sec_companyfacts

        normalized = normalize_sec_companyfacts(
            snapshots[companyfacts_id],
            snapshots[submissions_id],
            document["parameters"],
            entity_id=str(document["entity_id"]),
            as_of_cutoff=str(document["as_of_cutoff"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"$.parameters: cannot normalize frozen SEC snapshots ({exc})")
        return

    provenance = {
        str(item["parameter_pointer"]): item for item in normalized["provenance"]
    }
    parameters = document["parameters"]
    for period_index, period in enumerate(parameters.get("periods", [])):
        if not isinstance(period, Mapping):
            continue
        for field in period.get("fields", {}):
            pointer = f"/periods/{period_index}/fields/{field}"
            item = provenance.get(pointer)
            if item is None:
                errors.append(f"$.parameters{pointer}: missing deterministic SEC provenance")
                continue
            accession = str(period.get("accession", ""))
            if item["accession"] != accession:
                errors.append(f"$.parameters{pointer}: accession differs from selected fact")
            if _parse_datetime(str(item["accepted_at"])) > _parse_datetime(
                str(document["as_of_cutoff"])
            ):
                errors.append(f"$.parameters{pointer}: filing acceptance is later than cutoff")


def _validate_calculation_reconciliation(
    document: Mapping[str, Any],
    index: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Any],
    errors: list[str],
) -> None:
    kind = str(document["calculation_type"])
    method = str(document["method"]["name"])
    result = document["result"]
    try:
        if document["warnings"] != result["warnings"]:
            raise ValueError("artifact warnings differ from result warnings")
        if (kind, method) == (
            "sec-companyfacts-normalization",
            "researchkit.normalize_sec_companyfacts",
        ):
            from .sec import normalize_sec_companyfacts

            companyfacts_ref = next(
                reference
                for reference in document["inputs"]
                if reference.get("role") == "subject"
            )
            submissions_ref = next(
                reference
                for reference in document["inputs"]
                if reference.get("role") == "availability"
            )
            companyfacts_id = str(companyfacts_ref["artifact_id"])
            submissions_id = str(submissions_ref["artifact_id"])
            replayed = normalize_sec_companyfacts(
                snapshots[companyfacts_id],
                snapshots[submissions_id],
                document["parameters"],
                entity_id=str(document["entity_id"]),
                as_of_cutoff=str(document["as_of_cutoff"]),
            )
            if document["method"]["version"] != replayed["method_version"]:
                raise ValueError("method.version differs from result.method_version")
            if canonical_sha256(replayed) != canonical_sha256(result):
                raise ValueError("result differs from deterministic parameter replay")
            residual = 0.0
        elif (kind, method) == ("financial_statement_analysis", "researchkit.analyze_financial_statements"):
            from .financials import analyze_financial_statements

            if document["method"]["version"] != result["method_version"]:
                raise ValueError("method.version differs from result.method_version")
            if result["method"] != "normalized-three-statement-analysis":
                raise ValueError("result.method is unsupported")
            residual = 0.0
            periods = result["periods"]
            if not periods:
                raise ValueError("result.periods must not be empty")
            for period in periods:
                balance_residual = (
                    float(period["total_assets"])
                    - float(period["total_liabilities"])
                    - float(period["total_equity"])
                )
                operating_residual = float(period["operating_income"]) - (
                    float(period["revenue"])
                    - float(period["cost_of_revenue"])
                    - float(period["operating_expenses"])
                )
                if not math.isclose(float(period["balance_sheet_residual"]), balance_residual, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError("result balance_sheet_residual is inconsistent")
                if not math.isclose(float(period["operating_income_residual"]), operating_residual, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError("result operating_income_residual is inconsistent")
                residual = max(
                    residual,
                    abs(balance_residual) / max(abs(float(period["total_assets"])), 1.0),
                    abs(operating_residual) / max(abs(float(period["revenue"])), 1.0),
                )
            expected_result_status = "pass" if residual <= float(result["reconciliation"]["tolerance"]) else "fail"
            if result["reconciliation"]["status"] != expected_result_status:
                raise ValueError("result.reconciliation.status is inconsistent")
            expected_warnings = ["reconciliation_failure"] if expected_result_status == "fail" else []
            if result["warnings"] != expected_warnings:
                raise ValueError("result.warnings is inconsistent")
            replayed = analyze_financial_statements(
                document["parameters"]["dataset"],
                {
                    "reconciliation_tolerance": document["parameters"][
                        "reconciliation_tolerance"
                    ]
                },
            )
            if canonical_sha256(replayed) != canonical_sha256(result):
                raise ValueError("result differs from deterministic parameter replay")
        elif (kind, method) == ("dcf-valuation", "researchkit.dcf_valuation"):
            from .valuation import dcf_sensitivity, dcf_valuation

            base = result["base"]
            if document["method"]["version"] != base["method_version"]:
                raise ValueError("method.version differs from result.base.method_version")
            if base["method"] != "unlevered-fcf-perpetuity-growth":
                raise ValueError("result.base.method is unsupported")
            if result.get("warnings") != base["warnings"]:
                raise ValueError("result.warnings differs from base warnings")
            parameters = document["parameters"]
            replayed_base = dcf_valuation(
                parameters["forecasts"],
                wacc=parameters["wacc"],
                terminal_growth=parameters["terminal_growth"],
                net_debt=parameters["net_debt"],
                non_operating_assets=parameters["non_operating_assets"],
                diluted_shares=parameters["diluted_shares"],
            )
            replayed_sensitivity = dcf_sensitivity(
                parameters["forecasts"],
                wacc_values=parameters["wacc_values"],
                terminal_growth_values=parameters["terminal_growth_values"],
                net_debt=parameters["net_debt"],
                non_operating_assets=parameters["non_operating_assets"],
                diluted_shares=parameters["diluted_shares"],
            )
            replayed = {
                "method_version": replayed_base["method_version"],
                "base": replayed_base,
                "sensitivity": replayed_sensitivity,
                "warnings": replayed_base["warnings"],
            }
            if canonical_sha256(replayed) != canonical_sha256(result):
                raise ValueError("result differs from deterministic parameter replay")
            wacc_values = parameters["wacc_values"]
            growth_values = parameters["terminal_growth_values"]
            if parameters["wacc"] != wacc_values[len(wacc_values) // 2]:
                raise ValueError("base wacc must equal the sensitivity-axis center")
            if parameters["terminal_growth"] != growth_values[len(growth_values) // 2]:
                raise ValueError("base terminal_growth must equal the sensitivity-axis center")
            residual = max(
                _relative_residual(
                    float(base["enterprise_value"]),
                    float(base["pv_forecast_cash_flows"]) + float(base["pv_terminal_value"]),
                ),
                _relative_residual(
                    float(base["equity_value"]),
                    float(base["enterprise_value"]) - float(base["net_debt"]) + float(base["non_operating_assets"]),
                ),
                _relative_residual(
                    float(base["value_per_share"]),
                    float(base["equity_value"]) / float(base["diluted_shares"]),
                ),
            )
        elif (kind, method) == ("comps-valuation", "researchkit.comps_valuation"):
            from .valuation import comps_valuation

            if document["method"]["version"] != result["method_version"]:
                raise ValueError("method.version differs from result.method_version")
            parameters = document["parameters"]
            replayed = comps_valuation(
                parameters["peers"],
                subject_metric=parameters["subject_metric"],
                basis=parameters["basis"],
                diluted_shares=parameters["diluted_shares"],
                net_debt=parameters["net_debt"],
                non_operating_assets=parameters["non_operating_assets"],
                statistic=parameters["statistic"],
                explicit_multiple=parameters["explicit_multiple"],
            )
            if canonical_sha256(replayed) != canonical_sha256(result):
                raise ValueError("result differs from deterministic parameter replay")
            residual = 0.0
            for translated in (result["selected"], *result["range"].values()):
                indicated = float(translated["multiple"]) * float(
                    result["subject_metric"]
                )
                if result["basis"] == "enterprise":
                    residual = max(
                        residual,
                        _relative_residual(
                            float(translated["enterprise_value"]), indicated
                        ),
                        _relative_residual(
                            float(translated["equity_value"]),
                            float(translated["enterprise_value"])
                            - float(result["net_debt"])
                            + float(result["non_operating_assets"]),
                        ),
                        _relative_residual(
                            float(translated["value_per_share"]),
                            float(translated["equity_value"])
                            / float(result["diluted_shares"]),
                        ),
                    )
                else:
                    residual = max(
                        residual,
                        _relative_residual(
                            float(translated["equity_value"]), indicated
                        ),
                        _relative_residual(
                            float(translated["value_per_share"]),
                            float(translated["equity_value"])
                            / float(result["diluted_shares"]),
                        ),
                    )
        else:
            errors.append(f"$.method: no strict reconciliation rule for {kind}/{method}")
            return
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append(f"$.result: cannot recompute method-specific reconciliation ({exc})")
        return
    declared = document["reconciliation"]
    tolerance = float(declared["tolerance"])
    if not math.isclose(float(declared["residual"]), residual, rel_tol=0.0, abs_tol=1e-15):
        errors.append("$.reconciliation.residual: differs from method-specific recomputation")
    expected_status = "pass" if residual <= tolerance else "fail"
    if declared["status"] != expected_status:
        errors.append("$.reconciliation.status: differs from method-specific recomputation")


def _relative_residual(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(actual), abs(expected), 1.0)


def _apply_normalization(value: Any, normalization: Mapping[str, Any]) -> Any:
    operation = normalization["operation"]
    if operation == "identity":
        if "factor" in normalization:
            raise ValueError("identity must not declare factor")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric normalization requires a JSON number")
    if operation == "negate":
        if "factor" in normalization:
            raise ValueError("negate must not declare factor")
        return -value
    if operation == "scale":
        factor = normalization.get("factor")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            raise ValueError(f"{operation} requires numeric factor")
        return value * factor
    raise ValueError(f"unsupported operation {operation!r}")


def _json_values_equal(left: Any, right: Any) -> bool:
    if _is_number(left) and _is_number(right):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
        except OverflowError:
            return left == right
    return left == right


def _resolve_reference_value(
    reference: Mapping[str, Any],
    index: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Any],
) -> Any:
    artifact_id = str(reference["artifact_id"])
    if reference.get("target") == "snapshot":
        target = snapshots[artifact_id]
    else:
        target = index[artifact_id]
    return _resolve_json_pointer(target, str(reference["json_pointer"]))


def _artifact_entity_id(document: Mapping[str, Any]) -> str | None:
    if document.get("artifact_type") == "source":
        identifiers = document.get("identifiers")
        return str(identifiers.get("entity_id")) if isinstance(identifiers, Mapping) else None
    value = document.get("entity_id")
    return str(value) if value is not None else None


def _required_lineage_role(kind: str, parameter_pointer: str) -> str | None:
    if kind == "sec-companyfacts-normalization":
        return "assumption"
    if kind == "financial_statement_analysis":
        return "subject" if parameter_pointer.startswith("/parameters/dataset/") else "assumption"
    if kind == "dcf-valuation":
        if parameter_pointer == "/parameters/net_debt":
            return "subject"
        return "assumption"
    if kind == "comps-valuation":
        if parameter_pointer.startswith("/parameters/peers/"):
            return "comparison"
        if parameter_pointer in {
            "/parameters/subject_metric",
            "/parameters/diluted_shares",
            "/parameters/net_debt",
            "/parameters/non_operating_assets",
        }:
            return "subject"
        return "assumption"
    return None


def _parse_sec_companyfacts_pointer(pointer: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"/facts/(?:dei|us-gaap)/([^/]+)/units/([^/]+)/(?:0|[1-9][0-9]*)/val",
        pointer,
    )
    if match is None:
        raise ValueError("CompanyFacts pointer must identify facts/<taxonomy>/<concept>/units/<unit>/<index>/val")
    return match.group(1), match.group(2)


def _sec_entity_cik(entity_id: str) -> str:
    match = re.fullmatch(r"SEC:CIK(\d+)", entity_id)
    if match is None:
        raise ValueError("entity_id must use SEC:CIK<digits>")
    return match.group(1).lstrip("0")


def _resolve_json_pointer(document: Any, pointer: str) -> Any:
    if not isinstance(pointer, str):
        raise ValueError("must be an RFC 6901 string")
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError("must be an RFC 6901 pointer")
    current = document
    for raw_part in pointer[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", raw_part):
            raise ValueError("pointer contains an invalid escape")
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            current = current[part]
        elif isinstance(current, list):
            if not part.isdigit() or (part != "0" and part.startswith("0")):
                raise ValueError("array index is invalid")
            position = int(part)
            if position >= len(current):
                raise IndexError("array index is out of range")
            current = current[position]
        else:
            raise TypeError("pointer traverses a scalar")
    return current


def _validate_node(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    expected = schema.get("type")
    if expected is not None and not _matches_type(value, expected):
        errors.append(f"{path}: expected {_describe_type(expected)}")
        return

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}")

    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            errors.append(f"{path}: string is shorter than minLength")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(str(pattern), value) is None:
            errors.append(f"{path}: does not match required pattern")
        if schema.get("format") == "date-time":
            try:
                parsed = _parse_datetime(value)
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise ValueError("timezone required")
            except ValueError:
                errors.append(f"{path}: expected an RFC 3339 date-time with offset")

    if _is_number(value):
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            errors.append(f"{path}: number must be finite")
            return
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: must be < {schema['exclusiveMaximum']}")

    if isinstance(value, Mapping):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key}: required property is missing")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key}: additional property is not allowed")
        for key, child_schema in properties.items():
            if key in value:
                _validate_node(value[key], child_schema, f"{path}.{key}", errors)

    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            errors.append(f"{path}: array is shorter than minItems")
        if schema.get("uniqueItems"):
            try:
                encoded = [
                    json.dumps(item, sort_keys=True, separators=(",", ":"))
                    for item in value
                ]
            except (TypeError, ValueError):
                errors.append(f"{path}: array items must be JSON-compatible")
            else:
                if len(encoded) != len(set(encoded)):
                    errors.append(f"{path}: array items must be unique")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_node(item, item_schema, f"{path}[{index}]", errors)


def _validate_semantics(document: Mapping[str, Any], selected: str, errors: list[str]) -> None:
    expected_prefix = {"source-artifact": "src", "calculation-artifact": "calc", "research-artifact": "res"}[selected]
    try:
        expected_id = seal_artifact(document)["artifact_id"]
    except ContractError as exc:
        errors.extend(exc.errors)
        return
    if document.get("artifact_id") != expected_id:
        errors.append(f"$.artifact_id: expected content-derived ID with {expected_prefix}_ prefix")

    if selected == "source-artifact":
        published_value = document["source_published_at"]
        published = _parse_datetime(str(published_value)) if published_value is not None else None
        available = _parse_datetime(str(document["source_available_at"]))
        retrieved = _parse_datetime(str(document["retrieved_at"]))
        cutoff = _parse_datetime(str(document["as_of_cutoff"]))
        if published is not None and published > available:
            errors.append("$.source_published_at: must not be later than source_available_at")
        if available > cutoff:
            errors.append("$.source_available_at: later than as_of_cutoff (look-ahead)")
        if available > retrieved:
            errors.append("$.retrieved_at: must not precede source_available_at")
        snapshot = Path(str(document["snapshot_path"]))
        if snapshot.is_absolute() or ".." in snapshot.parts:
            errors.append("$.snapshot_path: must be a normalized repository-relative path")
        if document["schema_version"] == "2.0.0":
            basis = document["availability_basis"]
            if basis == "observed-at-retrieval":
                if published is not None:
                    errors.append("$.source_published_at: must be null when availability is first observed at retrieval")
                if available != retrieved:
                    errors.append("$.source_available_at: must equal retrieved_at when availability_basis is observed-at-retrieval")
            elif published is None:
                errors.append("$.source_published_at: required for provider-publication or filing-acceptance")
            timezone = str(document["timezone"])
            try:
                ZoneInfo(timezone)
            except (ValueError, ZoneInfoNotFoundError):
                errors.append("$.timezone: expected an IANA timezone name")
            accessions = document["identifiers"]["accessions"]
            if document["source_kind"] == "filing" and len(accessions) != 1:
                errors.append("$.identifiers.accessions: a filing source must identify exactly one accession")
            if basis == "filing-acceptance" and document["source_kind"] != "filing":
                errors.append("$.availability_basis: filing-acceptance requires source_kind filing")
            if document["source_kind"] in {"sec-companyfacts", "sec-submissions"}:
                if document["currency"] is not None or document["unit"] is not None or document["adjustment_basis"] is not None:
                    errors.append("$: SEC aggregate responses must not claim one currency, unit, or adjustment basis")
                if basis != "observed-at-retrieval":
                    errors.append("$.availability_basis: mutable SEC aggregates must use observed-at-retrieval")

    if selected == "calculation-artifact":
        artifact_ids = [item["artifact_id"] for item in document["inputs"]]
        if len(artifact_ids) != len(set(artifact_ids)):
            errors.append("$.inputs: artifact_id values must be unique")
        expected_output = canonical_sha256(document["result"])
        if document["output_hash"] != expected_output:
            errors.append("$.output_hash: does not match canonical result hash")
        reconciliation = document["reconciliation"]
        expected_status = "pass" if abs(float(reconciliation["residual"])) <= float(reconciliation["tolerance"]) else "fail"
        if reconciliation["status"] != expected_status:
            errors.append("$.reconciliation.status: inconsistent with residual and tolerance")
        if document["schema_version"] == "2.0.0":
            created = _parse_datetime(str(document["created_at"]))
            cutoff = _parse_datetime(str(document["as_of_cutoff"]))
            if created < cutoff:
                errors.append("$.created_at: must not precede as_of_cutoff")

    if selected == "research-artifact":
        created = _parse_datetime(str(document["created_at"]))
        cutoff = _parse_datetime(str(document["as_of_cutoff"]))
        if created < cutoff:
            errors.append("$.created_at: must not precede as_of_cutoff")
        if document["claim_type"] == "judgment":
            if not document["alternatives"]:
                errors.append("$.alternatives: a judgment must state at least one alternative")
            if not document["invalidation"]:
                errors.append("$.invalidation: a judgment must state invalidation conditions")


def _matches_type(value: Any, expected: str | Sequence[str]) -> bool:
    candidates = [expected] if isinstance(expected, str) else list(expected)
    return any(
        {
            "object": isinstance(value, Mapping),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "number": _is_number(value),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "null": value is None,
        }.get(candidate, False)
        for candidate in candidates
    )


def _describe_type(expected: str | Sequence[str]) -> str:
    return expected if isinstance(expected, str) else " or ".join(expected)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_datetime(value: str) -> datetime:
    if _RFC3339_RE.fullmatch(value) is None:
        raise ValueError("RFC 3339 date-time required")
    if value[11:13] == "24":
        raise ValueError("RFC 3339 hour must be between 00 and 23")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone required")
    return parsed

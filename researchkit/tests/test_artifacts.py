import hashlib
import json
from pathlib import Path

import pytest

from researchkit.artifacts import (
    ContractError,
    canonical_sha256,
    seal_artifact,
    validate_artifact,
    verify_source_snapshot,
)


def _source(tmp_path: Path) -> dict:
    snapshot = tmp_path / "fixtures" / "filing.json"
    snapshot.parent.mkdir(exist_ok=True)
    snapshot.write_text('{"revenue":100}', encoding="utf-8")
    document = {
        "schema_version": "1.0.0",
        "artifact_type": "source",
        "provider": "Example Exchange",
        "canonical_uri": "https://example.test/filing/1",
        "query": "entity=EXAMPLE",
        "source_published_at": "2025-02-01T12:00:00Z",
        "source_available_at": "2025-02-01T12:05:00Z",
        "retrieved_at": "2025-02-02T09:00:00Z",
        "as_of_cutoff": "2025-02-01T23:59:59Z",
        "identifiers": {"entity_id": "EXAMPLE", "security_id": "XNYS:EXM", "accession": None},
        "timezone": "UTC",
        "currency": "USD",
        "unit": "USD millions",
        "adjustment_basis": "reported",
        "rights_tag": "synthetic-redistributable",
        "content_hash": "sha256:" + hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "snapshot_path": str(snapshot.relative_to(tmp_path)),
    }
    return seal_artifact(document)


def test_source_round_trip_and_snapshot_verification(tmp_path: Path) -> None:
    document = _source(tmp_path)
    validate_artifact(document)
    assert verify_source_snapshot(document, tmp_path).name == "filing.json"
    assert seal_artifact(document) == document


def test_source_rejects_lookahead_and_tampering(tmp_path: Path) -> None:
    document = _source(tmp_path)
    document["as_of_cutoff"] = "2025-01-31T23:59:59Z"
    document = seal_artifact(document)
    with pytest.raises(ContractError, match="look-ahead"):
        validate_artifact(document)

    document = _source(tmp_path)
    (tmp_path / document["snapshot_path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ContractError, match="snapshot bytes"):
        verify_source_snapshot(document, tmp_path)


def test_research_judgment_requires_alternative_and_invalidation() -> None:
    document = seal_artifact({
        "schema_version": "1.0.0",
        "artifact_type": "research",
        "research_type": "sector_structure",
        "subject": "Example sector",
        "claim": "Supplier concentration may constrain margins.",
        "claim_type": "judgment",
        "evidence_refs": [{"artifact_id": "src_" + "a" * 64, "json_pointer": "/suppliers/0"}],
        "confidence": 0.6,
        "alternatives": [],
        "invalidation": [],
        "authoring_model": "test-model",
        "created_at": "2025-02-02T10:00:00Z",
        "as_of_cutoff": "2025-02-01T23:59:59Z",
    })
    with pytest.raises(ContractError, match="judgment"):
        validate_artifact(document)


def test_research_creation_must_not_precede_cutoff() -> None:
    document = seal_artifact({
        "schema_version": "1.0.0",
        "artifact_type": "research",
        "research_type": "fact_check",
        "subject": "Example",
        "claim": "A source existed by the cutoff.",
        "claim_type": "fact",
        "evidence_refs": [{"artifact_id": "src_" + "a" * 64, "json_pointer": "/value"}],
        "confidence": 1.0,
        "alternatives": [],
        "invalidation": [],
        "authoring_model": "test-model",
        "created_at": "2025-01-31T10:00:00Z",
        "as_of_cutoff": "2025-02-01T23:59:59Z",
    })
    with pytest.raises(ContractError, match="must not precede"):
        validate_artifact(document)


def test_calculation_hashes_result_and_checks_unique_inputs() -> None:
    result = {"enterprise_value": 125.0}
    base = {
        "schema_version": "1.0.0",
        "artifact_type": "calculation",
        "calculation_type": "dcf",
        "inputs": [{"artifact_id": "src_" + "a" * 64, "content_hash": "sha256:" + "b" * 64}],
        "method": {"name": "test", "version": "1.0.0"},
        "parameters": {},
        "code_commit": "abcdef0",
        "environment": {"python": "3.14.6", "lock_hash": "sha256:" + "c" * 64},
        "result": result,
        "output_hash": canonical_sha256(result),
        "warnings": [],
        "reconciliation": {"status": "pass", "residual": 0.0, "tolerance": 1e-9},
        "created_at": "2025-02-02T10:00:00Z",
    }
    document = seal_artifact(base)
    validate_artifact(document)
    document["output_hash"] = "sha256:" + "d" * 64
    document = seal_artifact(document)
    with pytest.raises(ContractError, match="canonical result hash"):
        validate_artifact(document)

    base["reconciliation"] = {"status": "pass", "residual": 1.0, "tolerance": 0.1}
    document = seal_artifact(base)
    with pytest.raises(ContractError, match="inconsistent"):
        validate_artifact(document)


def test_schema_files_are_valid_json() -> None:
    schema_dir = Path(__file__).parents[1] / "researchkit" / "schemas" / "v1"
    names = {path.name for path in schema_dir.glob("*.schema.json")}
    assert names == {
        "source-artifact.schema.json",
        "calculation-artifact.schema.json",
        "research-artifact.schema.json",
    }
    for path in schema_dir.glob("*.schema.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["$schema"].endswith("2020-12/schema")


def test_sealing_rejects_nonfinite_and_non_json_values_as_contract_errors() -> None:
    with pytest.raises(ContractError, match="finite JSON-compatible"):
        seal_artifact({"artifact_type": "research", "value": float("nan")})
    with pytest.raises(ContractError, match="finite JSON-compatible"):
        seal_artifact({"artifact_type": "research", "value": {1, 2}})


def test_datetime_format_rejects_space_separator(tmp_path: Path) -> None:
    document = _source(tmp_path)
    document["retrieved_at"] = "2025-02-02 09:00:00+00:00"
    document = seal_artifact(document)
    with pytest.raises(ContractError, match="RFC 3339"):
        validate_artifact(document)

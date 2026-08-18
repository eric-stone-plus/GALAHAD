import copy
import hashlib
import json
import socket
from pathlib import Path

import pytest

from researchkit.artifacts import (
    ContractError,
    artifact_sha256,
    canonical_sha256,
    seal_artifact,
    validate_artifact,
    validate_artifact_graph,
)
from researchkit.financials import analyze_financial_statements
from researchkit.sec import normalize_sec_companyfacts
from researchkit.valuation import comps_valuation, dcf_sensitivity, dcf_valuation


CUTOFF = "2026-08-14T10:02:13Z"
CREATED = "2026-08-14T10:05:00Z"
ACCESSION = "0000320193-25-000079"
ENTITY = "SEC:CIK0000320193"
ACCEPTED = "2025-10-31T10:01:26Z"
FIELDS = (
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
CONCEPTS = {
    "revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "cost_of_revenue": "CostOfGoodsAndServicesSold",
    "operating_expenses": "OperatingExpenses",
    "operating_income": "OperatingIncomeLoss",
    "net_income": "NetIncomeLoss",
    "cash_from_operations": "NetCashProvidedByUsedInOperatingActivities",
    "capital_expenditure": "PaymentsToAcquirePropertyPlantAndEquipment",
    "total_assets": "Assets",
    "total_liabilities": "Liabilities",
    "total_equity": "StockholdersEquity",
    "cash": "CashAndCashEquivalentsAtCarryingValue",
    "debt": "LongTermDebt",
}
RAW_VALUES = {
    "2024-09-28": {
        "revenue": 391035000000,
        "cost_of_revenue": 210352000000,
        "operating_expenses": 57467000000,
        "operating_income": 123216000000,
        "net_income": 93736000000,
        "cash_from_operations": 118254000000,
        "capital_expenditure": 9447000000,
        "total_assets": 364980000000,
        "total_liabilities": 308030000000,
        "total_equity": 56950000000,
        "cash": 29943000000,
        "debt": 96662000000,
    },
    "2025-09-27": {
        "revenue": 416161000000,
        "cost_of_revenue": 220960000000,
        "operating_expenses": 62151000000,
        "operating_income": 133050000000,
        "net_income": 112010000000,
        "cash_from_operations": 111482000000,
        "capital_expenditure": 12715000000,
        "total_assets": 359241000000,
        "total_liabilities": 285508000000,
        "total_equity": 73733000000,
        "cash": 35934000000,
        "debt": 90678000000,
    },
}


def _write_json(root: Path, name: str, value: object) -> str:
    path = root / name
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source(
    root: Path,
    *,
    name: str,
    source_kind: str,
    value: object,
    retrieved_at: str,
) -> dict:
    return seal_artifact(
        {
            "schema_version": "2.0.0",
            "artifact_type": "source",
            "source_kind": source_kind,
            "provider": "U.S. Securities and Exchange Commission",
            "canonical_uri": f"https://data.sec.gov/{name}",
            "query": "CIK0000320193",
            "source_published_at": None,
            "source_available_at": retrieved_at,
            "availability_basis": "observed-at-retrieval",
            "retrieved_at": retrieved_at,
            "as_of_cutoff": CUTOFF,
            "identifiers": {
                "entity_id": ENTITY,
                "security_id": "XNAS:AAPL",
                "accessions": [ACCESSION],
            },
            "snapshot_scope": {
                "representation": "complete-response",
                "description": "Minimal frozen response retaining the exact SEC JSON shape used by the test.",
            },
            "timezone": "UTC",
            "currency": None,
            "unit": None,
            "adjustment_basis": None,
            "rights_tag": "SEC.gov public filing data; redistribution review required",
            "content_hash": _write_json(root, name, value),
            "snapshot_path": name,
        }
    )


def _artifact_ref(artifact: dict, role: str) -> dict:
    return {
        "artifact_id": artifact["artifact_id"],
        "artifact_hash": artifact_sha256(artifact),
        "role": role,
    }


def _evidence_ref(artifact: dict, pointer: str) -> dict:
    return {
        "artifact_id": artifact["artifact_id"],
        "artifact_hash": artifact_sha256(artifact),
        "target": "artifact",
        "json_pointer": pointer,
    }


def _lineage(artifact: dict, parameter: str, pointer: str, value: object) -> dict:
    return {
        "parameter_pointer": parameter,
        "artifact_id": artifact["artifact_id"],
        "artifact_hash": artifact_sha256(artifact),
        "target": "artifact",
        "json_pointer": pointer,
        "reported_value": value,
        "normalization": {"operation": "identity"},
    }


def _calculation(
    kind: str,
    method: str,
    parameters: dict,
    result: dict,
    inputs: list[dict],
    lineage: list[dict],
) -> dict:
    return seal_artifact(
        {
            "schema_version": "2.0.0",
            "artifact_type": "calculation",
            "calculation_type": kind,
            "entity_id": ENTITY,
            "as_of_cutoff": CUTOFF,
            "inputs": inputs,
            "lineage": lineage,
            "method": {"name": method, "version": result["method_version"]},
            "parameters": parameters,
            "code_commit": "00fa68b",
            "environment": {
                "python": "3.14.6",
                "lock_hash": "sha256:" + "a" * 64,
            },
            "result": result,
            "output_hash": canonical_sha256(result),
            "warnings": result["warnings"],
            "reconciliation": {
                "status": "pass",
                "metric": "max-relative-residual",
                "residual": 0.0,
                "tolerance": 1e-12,
            },
            "created_at": CREATED,
        }
    )


def _research(kind: str, claim: str, evidence: list[dict], payload: dict) -> dict:
    return seal_artifact(
        {
            "schema_version": "2.0.0",
            "artifact_type": "research",
            "research_type": kind,
            "entity_id": ENTITY,
            "subject": "Apple Inc.",
            "claim": claim,
            "claim_type": "judgment",
            "evidence_refs": evidence,
            "payload": payload,
            "confidence": 0.8,
            "alternatives": ["A different explicit policy may produce a different conditional result."],
            "invalidation": ["Invalidate when any cited input, cutoff, or replay check fails."],
            "authoring_model": "GALAHAD offline graph fixture",
            "created_at": CREATED,
            "as_of_cutoff": CUTOFF,
        }
    )


def _sec_snapshots() -> tuple[dict, dict, dict]:
    facts: dict = {"cik": 320193, "entityName": "Apple Inc.", "facts": {"us-gaap": {}}}
    starts = {"2024-09-28": "2023-10-01", "2025-09-27": "2024-09-29"}
    instant = {"total_assets", "total_liabilities", "total_equity", "cash", "debt"}
    for field, concept in CONCEPTS.items():
        entries = []
        for period_end, values in RAW_VALUES.items():
            entries.append(
                {
                    "start": None if field in instant else starts[period_end],
                    "end": period_end,
                    "val": values[field],
                    "accn": ACCESSION,
                    "fy": 2025,
                    "fp": "FY",
                    "form": "10-K",
                    "frame": None,
                }
            )
        facts["facts"]["us-gaap"][concept] = {
            "label": concept,
            "description": concept,
            "units": {"USD": entries},
        }
    submissions = {
        "cik": "0000320193",
        "entityType": "operating",
        "filings": {
            "recent": {
                "accessionNumber": [ACCESSION],
                "acceptanceDateTime": ["2025-10-31T10:01:26.000Z"],
            }
        },
    }
    periods = []
    for period_end in RAW_VALUES:
        periods.append(
            {
                "period_end": period_end,
                "accession": ACCESSION,
                "form": "10-K",
                "fy": 2025,
                "fp": "FY",
                "fields": {
                    field: {
                        "taxonomy": "us-gaap",
                        "concept": concept,
                        "unit": "USD",
                        "start": None if field in instant else starts[period_end],
                    }
                    for field, concept in CONCEPTS.items()
                },
            }
        )
    parameters = {
        "periods": periods,
        "scale": 1e-6,
        "currency": "USD",
        "unit": "USD millions",
        "period_basis": "FY (52 weeks)",
    }
    return facts, submissions, parameters


def _normalization_policy(parameters: dict) -> dict:
    return _research(
        "sec-normalization-policy",
        "Use the exact accession, concept, unit, period, form, and fiscal selectors recorded in the payload.",
        [
            {
                "artifact_id": "src_" + "0" * 64,
                "artifact_hash": "sha256:" + "0" * 64,
                "target": "artifact",
                "json_pointer": "/schema_version",
            }
        ],
        parameters,
    )


def _graph(root: Path) -> tuple[list[dict], dict]:
    facts, submissions, normalize_parameters = _sec_snapshots()
    companyfacts = _source(
        root,
        name="companyfacts.json",
        source_kind="sec-companyfacts",
        value=facts,
        retrieved_at="2026-08-14T10:01:27Z",
    )
    submissions_source = _source(
        root,
        name="submissions.json",
        source_kind="sec-submissions",
        value=submissions,
        retrieved_at=CUTOFF,
    )
    policy = _normalization_policy(normalize_parameters)
    policy["evidence_refs"] = [_evidence_ref(companyfacts, "/artifact_id")]
    policy = seal_artifact(policy)

    normalize_lineage = []
    for field in ("scale", "currency", "unit", "period_basis"):
        normalize_lineage.append(
            _lineage(policy, f"/parameters/{field}", f"/payload/{field}", normalize_parameters[field])
        )
    for period_index, period in enumerate(normalize_parameters["periods"]):
        for field in ("period_end", "accession", "form", "fy", "fp"):
            normalize_lineage.append(
                _lineage(
                    policy,
                    f"/parameters/periods/{period_index}/{field}",
                    f"/payload/periods/{period_index}/{field}",
                    period[field],
                )
            )
        for statement_field, selector in period["fields"].items():
            for selector_field, value in selector.items():
                normalize_lineage.append(
                    _lineage(
                        policy,
                        f"/parameters/periods/{period_index}/fields/{statement_field}/{selector_field}",
                        f"/payload/periods/{period_index}/fields/{statement_field}/{selector_field}",
                        value,
                    )
                )
    normalized_result = normalize_sec_companyfacts(
        facts,
        submissions,
        normalize_parameters,
        entity_id=ENTITY,
        as_of_cutoff=CUTOFF,
    )
    normalized = _calculation(
        "sec-companyfacts-normalization",
        "researchkit.normalize_sec_companyfacts",
        normalize_parameters,
        normalized_result,
        [
            _artifact_ref(companyfacts, "subject"),
            _artifact_ref(submissions_source, "availability"),
            _artifact_ref(policy, "assumption"),
        ],
        normalize_lineage,
    )

    tolerance = _research(
        "calculation-policy",
        "Use a one-part-per-million statement reconciliation tolerance.",
        [_evidence_ref(normalized, "/result/method_version")],
        {"reconciliation_tolerance": 1e-6},
    )
    dataset = normalized_result["dataset"]
    financial_parameters = {"dataset": dataset, "reconciliation_tolerance": 1e-6}
    financial_result = analyze_financial_statements(dataset)
    financial_lineage = []
    for period_index, period in enumerate(dataset):
        for field, value in period.items():
            financial_lineage.append(
                _lineage(
                    normalized,
                    f"/parameters/dataset/{period_index}/{field}",
                    f"/result/dataset/{period_index}/{field}",
                    value,
                )
            )
    financial_lineage.append(
        _lineage(
            tolerance,
            "/parameters/reconciliation_tolerance",
            "/payload/reconciliation_tolerance",
            1e-6,
        )
    )
    financial = _calculation(
        "financial_statement_analysis",
        "researchkit.analyze_financial_statements",
        financial_parameters,
        financial_result,
        [_artifact_ref(normalized, "subject"), _artifact_ref(tolerance, "assumption")],
        financial_lineage,
    )

    forecasts = [
        {"period": f"FY{year}", "unlevered_fcf": value, "discount_period": float(year - 2025)}
        for year, value in zip(range(2026, 2031), [101730.0, 104782.0, 107925.0, 111163.0, 114498.0])
    ]
    dcf_parameters = {
        "forecasts": forecasts,
        "wacc": 0.09,
        "terminal_growth": 0.025,
        "net_debt": financial_result["periods"][1]["net_debt"],
        "non_operating_assets": 0.0,
        "diluted_shares": 15204.137,
        "wacc_values": [0.085, 0.09, 0.095],
        "terminal_growth_values": [0.02, 0.025, 0.03],
    }
    assumptions_payload = copy.deepcopy(dcf_parameters)
    assumptions_payload.pop("net_debt")
    assumptions = _research(
        "dcf-smoke-assumptions",
        "Forecast, discount-rate, growth, asset, and share values are transparent test assumptions.",
        [_evidence_ref(financial, "/result/periods/1/free_cash_flow")],
        assumptions_payload,
    )
    base = dcf_valuation(
        forecasts,
        wacc=dcf_parameters["wacc"],
        terminal_growth=dcf_parameters["terminal_growth"],
        net_debt=dcf_parameters["net_debt"],
        non_operating_assets=dcf_parameters["non_operating_assets"],
        diluted_shares=dcf_parameters["diluted_shares"],
    )
    sensitivity = dcf_sensitivity(
        forecasts,
        wacc_values=dcf_parameters["wacc_values"],
        terminal_growth_values=dcf_parameters["terminal_growth_values"],
        net_debt=dcf_parameters["net_debt"],
        non_operating_assets=dcf_parameters["non_operating_assets"],
        diluted_shares=dcf_parameters["diluted_shares"],
    )
    dcf_result = {
        "method_version": base["method_version"],
        "base": base,
        "sensitivity": sensitivity,
        "warnings": base["warnings"],
    }
    dcf_lineage = []
    for row_index, row in enumerate(forecasts):
        for field, value in row.items():
            dcf_lineage.append(
                _lineage(
                    assumptions,
                    f"/parameters/forecasts/{row_index}/{field}",
                    f"/payload/forecasts/{row_index}/{field}",
                    value,
                )
            )
    for field in ("wacc", "terminal_growth", "non_operating_assets", "diluted_shares"):
        dcf_lineage.append(
            _lineage(assumptions, f"/parameters/{field}", f"/payload/{field}", dcf_parameters[field])
        )
    for axis in ("wacc_values", "terminal_growth_values"):
        for position, value in enumerate(dcf_parameters[axis]):
            dcf_lineage.append(
                _lineage(
                    assumptions,
                    f"/parameters/{axis}/{position}",
                    f"/payload/{axis}/{position}",
                    value,
                )
            )
    dcf_lineage.append(
        _lineage(financial, "/parameters/net_debt", "/result/periods/1/net_debt", dcf_parameters["net_debt"])
    )
    dcf = _calculation(
        "dcf-valuation",
        "researchkit.dcf_valuation",
        dcf_parameters,
        dcf_result,
        [_artifact_ref(financial, "subject"), _artifact_ref(assumptions, "assumption")],
        dcf_lineage,
    )
    conclusion = _research(
        "offline-e2e-result",
        "The conditional DCF replays from SEC-shaped frozen facts and explicit test assumptions.",
        [
            _evidence_ref(financial, "/result/reconciliation/status"),
            _evidence_ref(dcf, "/result/base/value_per_share"),
        ],
        {"release_state": "READY_FOR_SYSTEM_TEST_ONLY"},
    )
    return [
        companyfacts,
        submissions_source,
        policy,
        normalized,
        tolerance,
        financial,
        assumptions,
        dcf,
        conclusion,
    ], conclusion


def _reseal_graph(graph: list[dict]) -> tuple[list[dict], dict]:
    """Reseal an acyclic graph in list order and update downstream references."""

    old_to_new: dict[str, dict] = {}
    rebuilt = []
    for document in graph:
        candidate = copy.deepcopy(document)
        for collection in ("inputs", "lineage", "evidence_refs"):
            for reference in candidate.get(collection, []):
                replacement = old_to_new.get(str(reference["artifact_id"]))
                if replacement is not None:
                    reference["artifact_id"] = replacement["artifact_id"]
                    reference["artifact_hash"] = artifact_sha256(replacement)
                availability = reference.get("availability_ref")
                if availability is not None:
                    replacement = old_to_new.get(str(availability["artifact_id"]))
                    if replacement is not None:
                        availability["artifact_id"] = replacement["artifact_id"]
                        availability["artifact_hash"] = artifact_sha256(replacement)
        old_id = str(document["artifact_id"])
        candidate = seal_artifact(candidate)
        old_to_new[old_id] = candidate
        rebuilt.append(candidate)
    return rebuilt, rebuilt[-1]


def test_v2_sec_shaped_graph_replays_offline(monkeypatch, tmp_path: Path) -> None:
    def denied(*_args, **_kwargs):
        raise AssertionError("network access is forbidden during graph validation")

    graph, root = _graph(tmp_path)
    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    report = validate_artifact_graph(
        graph,
        root_artifact_ids=[root["artifact_id"]],
        repository_root=tmp_path,
        require_all_reachable=True,
    )
    assert report == {
        "status": "pass",
        "artifact_count": 9,
        "root_artifact_ids": [root["artifact_id"]],
        "reachable_artifact_ids": sorted(item["artifact_id"] for item in graph),
        "orphan_artifact_ids": [],
        "snapshots_verified": 2,
    }
    assert graph[5]["result"]["trends"][0]["revenue_growth"] == pytest.approx(0.06425511782832749)
    assert graph[5]["result"]["periods"][1]["net_debt"] == 54744.0


def test_v2_graph_rejects_unsealed_tampering_first(tmp_path: Path) -> None:
    graph, root = _graph(tmp_path)
    graph[-1]["evidence_refs"][0]["artifact_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ContractError, match="content-derived ID"):
        validate_artifact_graph(graph, root_artifact_ids=[root["artifact_id"]], repository_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda graph: graph[-1]["evidence_refs"][0].update(json_pointer="/result/missing"), "does not resolve"),
        (lambda graph: graph[5]["lineage"][0].update(reported_value="wrong"), "reported_value"),
        (lambda graph: graph[5]["lineage"].pop(), "missing parameter coverage"),
        (lambda graph: graph[5]["result"]["periods"][0].update(revenue=1.0), "output_hash"),
        (lambda graph: graph[7]["parameters"].update(untraced_extra="secret"), "unexpected method parameters"),
        (lambda graph: graph[7]["lineage"].append(copy.deepcopy(graph[7]["lineage"][0])), "parameter_pointer values must be unique"),
        (lambda graph: graph[7]["lineage"][0]["normalization"].update(factor=999), "must not declare factor"),
    ],
)
def test_v2_graph_rejects_resealed_adversarial_mutations(tmp_path: Path, mutation, message: str) -> None:
    graph, _ = _graph(tmp_path)
    mutation(graph)
    graph, root = _reseal_graph(graph)
    with pytest.raises(ContractError, match=message):
        validate_artifact_graph(graph, root_artifact_ids=[root["artifact_id"]], repository_root=tmp_path)


def test_v2_graph_rejects_a_sealed_reference_with_wrong_artifact_hash(tmp_path: Path) -> None:
    graph, _ = _graph(tmp_path)
    graph[-1]["evidence_refs"][0]["artifact_hash"] = "sha256:" + "0" * 64
    graph[-1] = seal_artifact(graph[-1])
    with pytest.raises(ContractError, match="artifact_hash"):
        validate_artifact_graph(
            graph,
            root_artifact_ids=[graph[-1]["artifact_id"]],
            repository_root=tmp_path,
        )


def test_v2_graph_rejects_wrong_sec_entity_and_cik(tmp_path: Path) -> None:
    graph, _ = _graph(tmp_path)
    graph[1]["identifiers"]["entity_id"] = "SEC:CIK0000789019"
    graph, root = _reseal_graph(graph)
    with pytest.raises(ContractError, match="different entity"):
        validate_artifact_graph(graph, root_artifact_ids=[root["artifact_id"]], repository_root=tmp_path)

    graph, _ = _graph(tmp_path)
    submissions_path = tmp_path / graph[1]["snapshot_path"]
    submissions = json.loads(submissions_path.read_text(encoding="utf-8"))
    submissions["cik"] = "0000789019"
    graph[1]["content_hash"] = _write_json(tmp_path, graph[1]["snapshot_path"], submissions)
    graph, root = _reseal_graph(graph)
    with pytest.raises(ContractError, match="SEC CIK"):
        validate_artifact_graph(graph, root_artifact_ids=[root["artifact_id"]], repository_root=tmp_path)


def test_v2_graph_rejects_post_cutoff_acceptance(tmp_path: Path) -> None:
    graph, _ = _graph(tmp_path)
    submissions_path = tmp_path / graph[1]["snapshot_path"]
    submissions = json.loads(submissions_path.read_text(encoding="utf-8"))
    submissions["filings"]["recent"]["acceptanceDateTime"][0] = "2026-08-14T10:02:14Z"
    graph[1]["content_hash"] = _write_json(tmp_path, graph[1]["snapshot_path"], submissions)
    graph, root = _reseal_graph(graph)
    with pytest.raises(ContractError, match="accepted after the cutoff"):
        validate_artifact_graph(graph, root_artifact_ids=[root["artifact_id"]], repository_root=tmp_path)


def test_v2_graph_requires_repository_and_rejects_tampered_snapshot(tmp_path: Path) -> None:
    graph, root = _graph(tmp_path)
    with pytest.raises(ContractError, match="repository_root"):
        validate_artifact_graph(graph, root_artifact_ids=[root["artifact_id"]])
    (tmp_path / graph[0]["snapshot_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="snapshot bytes"):
        validate_artifact_graph(graph, root_artifact_ids=[root["artifact_id"]], repository_root=tmp_path)


def test_v2_graph_handles_empty_generator_and_bad_public_arguments() -> None:
    with pytest.raises(ContractError, match="at least one"):
        validate_artifact_graph(item for item in [])
    with pytest.raises(ContractError, match="iterable of documents"):
        validate_artifact_graph("not-a-graph")
    document = seal_artifact(
        {
            "schema_version": "2.0.0",
            "artifact_type": "research",
            "research_type": "test",
            "entity_id": ENTITY,
            "subject": "Apple Inc.",
            "claim": "Test claim.",
            "claim_type": "fact",
            "evidence_refs": [
                {
                    "artifact_id": "res_" + "0" * 64,
                    "artifact_hash": "sha256:" + "0" * 64,
                    "target": "artifact",
                    "json_pointer": "/claim",
                }
            ],
            "payload": {},
            "confidence": 1.0,
            "alternatives": [],
            "invalidation": [],
            "authoring_model": "test",
            "created_at": CREATED,
            "as_of_cutoff": CUTOFF,
        }
    )
    with pytest.raises(ContractError, match="root_artifact_ids"):
        validate_artifact_graph([document], root_artifact_ids=document["artifact_id"])
    with pytest.raises(ContractError, match="at least one artifact ID"):
        validate_artifact_graph([document], root_artifact_ids=[])
    with pytest.raises(ContractError, match="non-empty artifact ID string"):
        validate_artifact_graph([document], root_artifact_ids=[None])
    with pytest.raises(ContractError, match="require_all_reachable"):
        validate_artifact_graph([document], require_all_reachable=1)


def test_v2_artifact_rejects_invalid_timezone_and_huge_number(tmp_path: Path) -> None:
    graph, _ = _graph(tmp_path)
    source = copy.deepcopy(graph[0])
    source["timezone"] = "/"
    with pytest.raises(ContractError, match="IANA"):
        validate_artifact(seal_artifact(source))
    research = copy.deepcopy(graph[2])
    research["confidence"] = 10**400
    with pytest.raises(ContractError, match="finite"):
        validate_artifact(seal_artifact(research))


def test_sec_normalization_keeps_provenance_aligned_after_period_sorting() -> None:
    facts, submissions, parameters = _sec_snapshots()
    parameters["periods"].reverse()
    result = normalize_sec_companyfacts(
        facts,
        submissions,
        parameters,
        entity_id=ENTITY,
        as_of_cutoff=CUTOFF,
    )
    assert [row["period_end"] for row in result["dataset"]] == [
        "2024-09-28",
        "2025-09-27",
    ]
    for record in result["provenance"]:
        _, _, result_index, field = record["result_pointer"].split("/")
        assert result["dataset"][int(result_index)][field] == pytest.approx(
            record["reported_value"] * record["scale"]
        )


@pytest.mark.parametrize(
    ("artifact_position", "parameter_path"),
    [
        (5, ("dataset", 0)),
        (7, ("forecasts", 0)),
    ],
)
def test_v2_graph_rejects_ignored_nested_method_parameters(
    tmp_path: Path, artifact_position: int, parameter_path: tuple[str, int]
) -> None:
    graph, _ = _graph(tmp_path)
    collection, position = parameter_path
    graph[artifact_position]["parameters"] = copy.deepcopy(
        graph[artifact_position]["parameters"]
    )
    graph[artifact_position]["parameters"][collection][position]["ignored"] = "secret"
    graph, root = _reseal_graph(graph)
    with pytest.raises(ContractError, match="unexpected fields: ignored"):
        validate_artifact_graph(
            graph,
            root_artifact_ids=[root["artifact_id"]],
            repository_root=tmp_path,
        )


def test_v2_comps_artifact_has_strict_replay(tmp_path: Path) -> None:
    graph, _ = _graph(tmp_path)
    subject = graph[5]
    policy = _research(
        "comps-policy",
        "Use the preregistered median selection and supplied normalized peer set.",
        [_evidence_ref(subject, "/result/periods/1/revenue")],
        {"basis": "enterprise", "statistic": "median", "explicit_multiple": None},
    )
    peers = [
        {"name": "A", "included": True, "numerator": 100.0, "denominator": 10.0},
        {"name": "B", "included": True, "numerator": 180.0, "denominator": 15.0},
        {"name": "C", "included": True, "numerator": 240.0, "denominator": 20.0},
    ]
    parameters = {
        "peers": peers,
        "subject_metric": 100.0,
        "basis": "enterprise",
        "diluted_shares": 10.0,
        "net_debt": 20.0,
        "non_operating_assets": 5.0,
        "statistic": "median",
        "explicit_multiple": None,
    }
    result = comps_valuation(
        peers,
        subject_metric=100.0,
        basis="enterprise",
        diluted_shares=10.0,
        net_debt=20.0,
        non_operating_assets=5.0,
    )
    peer_source = _research(
        "normalized-comparison-set",
        "The local peer inputs are approved normalized comparison values.",
        [_evidence_ref(subject, "/result/periods/1/revenue")],
        {"peers": peers},
    )
    lineage = []
    for position, peer in enumerate(peers):
        for field, value in peer.items():
            lineage.append(
                _lineage(
                    peer_source,
                    f"/parameters/peers/{position}/{field}",
                    f"/payload/peers/{position}/{field}",
                    value,
                )
            )
    for field in ("subject_metric", "diluted_shares", "net_debt", "non_operating_assets"):
        pointer = "/result/periods/1/revenue" if field == "subject_metric" else "/result/periods/1/net_debt"
        value = parameters[field]
        if field == "subject_metric":
            record = _lineage(subject, f"/parameters/{field}", pointer, subject["result"]["periods"][1]["revenue"])
            record["normalization"] = {"operation": "scale", "factor": value / record["reported_value"]}
        else:
            assumption_values = {field: value}
            assumption = _research(
                f"comps-{field}",
                f"Use the supplied {field} test assumption.",
                [_evidence_ref(subject, "/result/periods/1/net_debt")],
                assumption_values,
            )
            graph.append(assumption)
            record = _lineage(assumption, f"/parameters/{field}", f"/payload/{field}", value)
            # These are subject bridge fields; a same-entity research artifact may carry them.
        lineage.append(record)
    for field in ("basis", "statistic", "explicit_multiple"):
        lineage.append(_lineage(policy, f"/parameters/{field}", f"/payload/{field}", parameters[field]))
    inputs = [_artifact_ref(peer_source, "comparison"), _artifact_ref(subject, "subject"), _artifact_ref(policy, "assumption")]
    inputs.extend(_artifact_ref(item, "subject") for item in graph[9:])
    calculation = _calculation(
        "comps-valuation",
        "researchkit.comps_valuation",
        parameters,
        result,
        inputs,
        lineage,
    )
    full_graph = [*graph[:9], peer_source, policy, *graph[9:], calculation]
    report = validate_artifact_graph(
        full_graph,
        root_artifact_ids=[calculation["artifact_id"]],
        repository_root=tmp_path,
        require_all_reachable=False,
    )
    assert report["status"] == "pass"


def test_v1_artifact_remains_supported() -> None:
    fixture = Path(__file__).parent / "fixtures" / "acme" / "source-artifact.json"
    validate_artifact(json.loads(fixture.read_text(encoding="utf-8")))

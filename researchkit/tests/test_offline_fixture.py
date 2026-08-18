import json
import socket
from pathlib import Path

from researchkit.artifacts import canonical_sha256, seal_artifact, validate_artifact, verify_source_snapshot
from researchkit.financials import analyze_financial_statements
from researchkit.valuation import comps_valuation, dcf_sensitivity, dcf_valuation


def test_wave_a_core_path_uses_frozen_fixture_with_network_denied(monkeypatch) -> None:
    def denied(*_args, **_kwargs):
        raise AssertionError("network access is forbidden in core fixture tests")

    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    component_root = Path(__file__).parents[1]
    artifact = json.loads(
        (component_root / "tests/fixtures/acme/source-artifact.json").read_text(encoding="utf-8")
    )
    validate_artifact(artifact)
    snapshot_path = verify_source_snapshot(artifact, component_root)
    fixture = json.loads(snapshot_path.read_text(encoding="utf-8"))
    input_ref = {"artifact_id": artifact["artifact_id"], "content_hash": artifact["content_hash"]}

    def calculation(kind, method, parameters, result, input_lineage, residual=0.0, tolerance=1e-9):
        document = seal_artifact({
            "schema_version": "1.0.0",
            "artifact_type": "calculation",
            "calculation_type": kind,
            "inputs": [input_ref],
            "method": {"name": method, "version": result["method_version"]},
            "parameters": {**parameters, "input_lineage": input_lineage},
            "code_commit": "abcdef0",
            "environment": {"python": "3.11", "lock_hash": "sha256:" + "a" * 64},
            "result": result,
            "output_hash": canonical_sha256(result),
            "warnings": result["warnings"],
            "reconciliation": {
                "status": "pass" if abs(residual) <= tolerance else "fail",
                "residual": residual,
                "tolerance": tolerance,
            },
            "created_at": "2026-02-02T12:00:00Z",
        })
        validate_artifact(document)
        return document

    statements_input = [
        {**row, "currency": fixture["currency"], "unit": fixture["unit"], "period_basis": fixture["period_basis"]}
        for row in fixture["statements"]
    ]
    statements = analyze_financial_statements(statements_input)
    assert statements["reconciliation"]["status"] == "pass"
    statement_artifact = calculation(
        "financial_statement_analysis",
        "researchkit.analyze_financial_statements",
        {"dataset": statements_input, "reconciliation_tolerance": statements["reconciliation"]["tolerance"]},
        statements,
        {
            f"{period_index}.{field}": {"artifact_id": artifact["artifact_id"], "json_pointer": f"/statements/{period_index}/{field}"}
            for period_index, row in enumerate(fixture["statements"])
            for field in row
        },
    )

    dcf_input = fixture["dcf"]
    forecasts = dcf_input["forecasts"]
    dcf = dcf_valuation(
        forecasts, wacc=dcf_input["wacc"], terminal_growth=dcf_input["terminal_growth"],
        net_debt=dcf_input["net_debt"], non_operating_assets=dcf_input["non_operating_assets"],
        diluted_shares=dcf_input["diluted_shares"],
    )
    sensitivity = dcf_sensitivity(
        forecasts, wacc_values=dcf_input["wacc_values"], terminal_growth_values=dcf_input["terminal_growth_values"],
        net_debt=dcf_input["net_debt"], non_operating_assets=dcf_input["non_operating_assets"],
        diluted_shares=dcf_input["diluted_shares"],
    )
    assert dcf["value_per_share"] > 0
    assert len(sensitivity) == 9
    dcf_result = {"method_version": dcf["method_version"], "base": dcf, "sensitivity": sensitivity, "warnings": dcf["warnings"]}
    dcf_artifact = calculation(
        "dcf-valuation", "researchkit.dcf_valuation", dcf_input, dcf_result,
        {
            f"dcf.{field}": {"artifact_id": artifact["artifact_id"], "json_pointer": f"/dcf/{field}"}
            for field in dcf_input
        },
    )

    comps_input = fixture["comps"]
    comps = comps_valuation(
        comps_input["peers"], subject_metric=comps_input["subject_metric"], basis=comps_input["basis"],
        diluted_shares=comps_input["diluted_shares"], net_debt=comps_input["net_debt"],
        non_operating_assets=comps_input["non_operating_assets"],
    )
    assert comps["distribution"]["median"] == 10
    comps_artifact = calculation(
        "comps-valuation", "researchkit.comps_valuation", comps_input, comps,
        {
            f"comps.{field}": {"artifact_id": artifact["artifact_id"], "json_pointer": f"/comps/{field}"}
            for field in comps_input
        },
    )
    assert {statement_artifact["artifact_type"], dcf_artifact["artifact_type"], comps_artifact["artifact_type"]} == {"calculation"}

    research_artifact = seal_artifact({
        "schema_version": "1.0.0",
        "artifact_type": "research",
        "research_type": "valuation_interpretation",
        "subject": fixture["entity"],
        "claim": "The base DCF is conditional on the supplied normalized forecast and discount assumptions.",
        "claim_type": "judgment",
        "evidence_refs": [
            {"artifact_id": dcf_artifact["artifact_id"], "json_pointer": "/result/base/value_per_share"},
            {"artifact_id": statement_artifact["artifact_id"], "json_pointer": "/result/reconciliation/status"},
        ],
        "confidence": 0.6,
        "alternatives": ["A trading-comparables interpretation may imply a different conditional range."],
        "invalidation": ["Invalidate if any source input, unit, cutoff, or equity-bridge item fails audit."],
        "authoring_model": "synthetic-test-model",
        "created_at": "2026-02-02T12:00:00Z",
        "as_of_cutoff": artifact["as_of_cutoff"],
    })
    validate_artifact(research_artifact)

"""Auditable research contracts and deterministic finance calculations."""

from .artifacts import (
    ContractError,
    artifact_sha256,
    canonical_sha256,
    seal_artifact,
    validate_artifact,
    validate_artifact_graph,
    verify_source_snapshot,
)
from .financials import analyze_financial_statements
from .sec import normalize_sec_companyfacts
from .valuation import comps_valuation, dcf_sensitivity, dcf_valuation

__all__ = [
    "ContractError",
    "analyze_financial_statements",
    "artifact_sha256",
    "canonical_sha256",
    "comps_valuation",
    "dcf_sensitivity",
    "dcf_valuation",
    "normalize_sec_companyfacts",
    "seal_artifact",
    "validate_artifact",
    "validate_artifact_graph",
    "verify_source_snapshot",
]

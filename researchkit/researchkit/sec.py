"""Deterministic normalization for frozen SEC CompanyFacts snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any


METHOD_VERSION = "1.0.0"
_STATEMENT_FIELDS = (
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


def normalize_sec_companyfacts(
    companyfacts: Mapping[str, Any],
    submissions: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    entity_id: str,
    as_of_cutoff: str,
) -> dict[str, Any]:
    """Select exact filing facts and emit normalized statement periods.

    The caller supplies a frozen CompanyFacts response, the matching frozen
    Submissions response, and an explicit selection policy.  No network access,
    taxonomy inference, fallback concept selection, or period imputation occurs.
    """

    if not isinstance(companyfacts, Mapping) or not isinstance(submissions, Mapping):
        raise ValueError("companyfacts and submissions must be JSON objects")
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a JSON object")
    expected = {"periods", "scale", "currency", "unit", "period_basis"}
    if set(parameters) != expected:
        missing = sorted(expected - set(parameters))
        extra = sorted(set(parameters) - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError("invalid normalization parameters: " + "; ".join(details))

    cik = _entity_cik(entity_id)
    companyfacts_cik = str(companyfacts.get("cik", "")).lstrip("0")
    submissions_cik = str(submissions.get("cik", "")).lstrip("0")
    if not cik or companyfacts_cik != cik or submissions_cik != cik:
        raise ValueError("SEC CIK differs across entity, CompanyFacts, and Submissions")

    scale = _finite_number("scale", parameters["scale"])
    if scale <= 0:
        raise ValueError("scale must be positive")
    currency = _nonempty_string("currency", parameters["currency"])
    if len(currency) != 3 or currency != currency.upper():
        raise ValueError("currency must be a three-letter uppercase code")
    unit = _nonempty_string("unit", parameters["unit"])
    period_basis = _nonempty_string("period_basis", parameters["period_basis"])
    raw_periods = parameters["periods"]
    if not isinstance(raw_periods, Sequence) or isinstance(raw_periods, (str, bytes)):
        raise ValueError("periods must be an array")
    if len(raw_periods) < 2:
        raise ValueError("periods must contain at least two selections")

    selected_periods: list[
        tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]
    ] = []
    seen_periods: set[str] = set()
    for period_index, raw_period in enumerate(raw_periods):
        if not isinstance(raw_period, Mapping):
            raise ValueError(f"period {period_index} must be an object")
        period_keys = {"period_end", "accession", "form", "fy", "fp", "fields"}
        if set(raw_period) != period_keys:
            raise ValueError(f"period {period_index} must contain exactly {sorted(period_keys)}")
        period_end = _iso_date("period_end", raw_period["period_end"])
        if period_end in seen_periods:
            raise ValueError("period_end selections must be unique")
        seen_periods.add(period_end)
        accession = _nonempty_string("accession", raw_period["accession"])
        form = _nonempty_string("form", raw_period["form"])
        fiscal_period = _nonempty_string("fp", raw_period["fp"])
        fiscal_year = raw_period["fy"]
        if isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int):
            raise ValueError("fy must be an integer")
        accepted_at = _accepted_at(submissions, accession)
        if _parse_instant(accepted_at) > _parse_instant(as_of_cutoff):
            raise ValueError(f"accession {accession} was accepted after the cutoff")

        fields = raw_period["fields"]
        if not isinstance(fields, Mapping) or set(fields) != set(_STATEMENT_FIELDS):
            raise ValueError(
                f"period {period_index} fields must contain exactly {sorted(_STATEMENT_FIELDS)}"
            )
        row: dict[str, Any] = {
            "period_end": period_end,
            "currency": currency,
            "unit": unit,
            "period_basis": period_basis,
        }
        period_provenance: list[tuple[str, dict[str, Any]]] = []
        for field in _STATEMENT_FIELDS:
            selector = fields[field]
            if not isinstance(selector, Mapping):
                raise ValueError(f"selector {period_index}.{field} must be an object")
            selector_keys = {"taxonomy", "concept", "unit", "start"}
            if set(selector) != selector_keys:
                raise ValueError(
                    f"selector {period_index}.{field} must contain exactly {sorted(selector_keys)}"
                )
            taxonomy = _nonempty_string("taxonomy", selector["taxonomy"])
            concept = _nonempty_string("concept", selector["concept"])
            fact_unit = _nonempty_string("fact unit", selector["unit"])
            start = selector["start"]
            if start is not None:
                start = _iso_date("start", start)
            facts = _fact_array(companyfacts, taxonomy, concept, fact_unit)
            matches = [
                (index, fact)
                for index, fact in enumerate(facts)
                if isinstance(fact, Mapping)
                and fact.get("accn") == accession
                and fact.get("form") == form
                and fact.get("fy") == fiscal_year
                and fact.get("fp") == fiscal_period
                and fact.get("end") == period_end
                and fact.get("start") == start
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"selector {period_index}.{field} resolved {len(matches)} facts; expected exactly one"
                )
            fact_index, fact = matches[0]
            raw_value = _finite_number(f"fact {period_index}.{field}", fact.get("val"))
            row[field] = raw_value * scale
            period_provenance.append(
                (
                    field,
                    {
                        "parameter_pointer": (
                            f"/periods/{period_index}/fields/{field}"
                        ),
                        "json_pointer": (
                            f"/facts/{taxonomy}/{concept}/units/{fact_unit}/"
                            f"{fact_index}/val"
                        ),
                        "taxonomy": taxonomy,
                        "concept": concept,
                        "reported_unit": fact_unit,
                        "reported_value": raw_value,
                        "scale": scale,
                        "accession": accession,
                        "accepted_at": accepted_at,
                    },
                )
            )
        selected_periods.append((row, period_provenance))

    selected_periods.sort(key=lambda item: item[0]["period_end"])
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for result_index, (row, period_provenance) in enumerate(selected_periods):
        rows.append(row)
        for field, record in period_provenance:
            provenance.append(
                {
                    **record,
                    "result_pointer": f"/dataset/{result_index}/{field}",
                }
            )
    return {
        "method": "sec-companyfacts-exact-selection",
        "method_version": METHOD_VERSION,
        "entity_id": entity_id,
        "dataset": rows,
        "provenance": provenance,
        "warnings": [],
    }


def _fact_array(
    companyfacts: Mapping[str, Any], taxonomy: str, concept: str, unit: str
) -> Sequence[Any]:
    try:
        facts = companyfacts["facts"][taxonomy][concept]["units"][unit]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing CompanyFacts path {taxonomy}/{concept}/{unit}") from exc
    if not isinstance(facts, Sequence) or isinstance(facts, (str, bytes)):
        raise ValueError(f"CompanyFacts path {taxonomy}/{concept}/{unit} must be an array")
    return facts


def _accepted_at(submissions: Mapping[str, Any], accession: str) -> str:
    try:
        recent = submissions["filings"]["recent"]
        accessions = recent["accessionNumber"]
        accepted = recent["acceptanceDateTime"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Submissions recent filing arrays are missing") from exc
    if not isinstance(accessions, list) or not isinstance(accepted, list):
        raise ValueError("Submissions recent filing fields must be arrays")
    if len(accessions) != len(accepted):
        raise ValueError("Submissions accession and acceptance arrays differ in length")
    positions = [index for index, value in enumerate(accessions) if value == accession]
    if len(positions) != 1:
        raise ValueError(f"accession {accession} resolved {len(positions)} Submissions rows")
    value = accepted[positions[0]]
    if not isinstance(value, str):
        raise ValueError("Submissions acceptanceDateTime must be a string")
    _parse_instant(value)
    return value


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return number


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _iso_date(name: str, value: Any) -> str:
    text = _nonempty_string(name, value)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD") from exc
    if parsed.strftime("%Y-%m-%d") != text:
        raise ValueError(f"{name} must use YYYY-MM-DD")
    return text


def _parse_instant(value: str) -> datetime:
    if not isinstance(value, str) or "T" not in value:
        raise ValueError("RFC 3339 instant required")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("RFC 3339 instant required") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("RFC 3339 offset required")
    return parsed


def _entity_cik(entity_id: str) -> str:
    prefix = "SEC:CIK"
    if not isinstance(entity_id, str) or not entity_id.startswith(prefix):
        raise ValueError("entity_id must use SEC:CIK<digits>")
    digits = entity_id[len(prefix) :]
    if not digits.isdigit():
        raise ValueError("entity_id must use SEC:CIK<digits>")
    return digits.lstrip("0")

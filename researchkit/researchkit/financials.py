"""Deterministic financial-statement reconciliation and ratio calculations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any


_REQUIRED_FIELDS = (
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


def analyze_financial_statements(
    dataset: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconcile normalized periods and calculate transparent operating metrics.

    Accept positive capital expenditure as a cash use.  Reject mixed currencies,
    units, period bases, missing fields, duplicate period ends, and nonfinite
    values instead of imputing financial data.
    """

    settings = dict(config or {})
    unknown_settings = sorted(set(settings) - {"reconciliation_tolerance"})
    if unknown_settings:
        raise ValueError(f"unsupported config fields: {', '.join(unknown_settings)}")
    raw_tolerance = settings.get("reconciliation_tolerance", 1e-6)
    if isinstance(raw_tolerance, bool) or not isinstance(raw_tolerance, (int, float)):
        raise ValueError("reconciliation_tolerance must be a JSON number, not boolean or string")
    tolerance = float(raw_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("reconciliation_tolerance must be finite and non-negative")
    if len(dataset) < 2:
        raise ValueError("dataset must contain at least two comparable normalized periods")

    rows: list[dict[str, Any]] = []
    seen_periods: set[str] = set()
    currency: str | None = None
    unit: str | None = None
    period_basis: str | None = None
    for index, raw in enumerate(dataset):
        if not isinstance(raw, Mapping):
            raise ValueError(f"period {index} must be an object")
        for key in ("period_end", "currency", "unit", "period_basis", *_REQUIRED_FIELDS):
            if key not in raw:
                raise ValueError(f"period {index} is missing {key}")
        raw_period_end = raw["period_end"]
        period_end = raw_period_end if isinstance(raw_period_end, str) else ""
        if not period_end or period_end in seen_periods:
            raise ValueError("period_end values must be non-empty and unique")
        try:
            parsed_period_end = date.fromisoformat(period_end)
        except ValueError as exc:
            raise ValueError(f"period {index} period_end must be an ISO date") from exc
        if parsed_period_end.isoformat() != period_end:
            raise ValueError(f"period {index} period_end must use YYYY-MM-DD")
        seen_periods.add(period_end)
        row_currency = raw["currency"] if isinstance(raw["currency"], str) else ""
        row_unit = raw["unit"] if isinstance(raw["unit"], str) else ""
        row_basis = raw["period_basis"] if isinstance(raw["period_basis"], str) else ""
        if len(row_currency) != 3 or not row_currency.isascii() or not row_currency.isalpha() or row_currency != row_currency.upper():
            raise ValueError(f"period {period_end} currency must be an uppercase ISO 4217 code")
        if not row_unit.strip() or not row_basis.strip():
            raise ValueError(f"period {period_end} unit and period_basis must be non-empty")
        row_unit = row_unit.strip()
        row_basis = row_basis.strip()
        if currency is None:
            currency, unit, period_basis = row_currency, row_unit, row_basis
        elif (row_currency, row_unit, row_basis) != (currency, unit, period_basis):
            raise ValueError("currency, unit, and period_basis must be normalized before analysis")

        numbers: dict[str, float] = {}
        for field in _REQUIRED_FIELDS:
            if isinstance(raw[field], bool) or not isinstance(raw[field], (int, float)):
                raise ValueError(f"period {period_end} field {field} must be a JSON number, not boolean or string")
            value = float(raw[field])
            if not math.isfinite(value):
                raise ValueError(f"period {period_end} field {field} must be finite")
            numbers[field] = value
        if numbers["revenue"] <= 0:
            raise ValueError(f"period {period_end} revenue must be positive")
        if numbers["capital_expenditure"] < 0:
            raise ValueError(f"period {period_end} capital_expenditure must use a positive cash-use sign")
        if numbers["total_assets"] <= 0:
            raise ValueError(f"period {period_end} total_assets must be positive")

        gross_profit = numbers["revenue"] - numbers["cost_of_revenue"]
        calculated_operating_income = gross_profit - numbers["operating_expenses"]
        balance_residual = numbers["total_assets"] - numbers["total_liabilities"] - numbers["total_equity"]
        operating_residual = numbers["operating_income"] - calculated_operating_income
        free_cash_flow = numbers["cash_from_operations"] - numbers["capital_expenditure"]
        scale = max(abs(numbers["total_assets"]), 1.0)
        balance_pass = abs(balance_residual) <= tolerance * scale
        operating_scale = max(abs(numbers["revenue"]), 1.0)
        operating_pass = abs(operating_residual) <= tolerance * operating_scale

        rows.append({
            "period_end": period_end,
            **numbers,
            "gross_profit": gross_profit,
            "calculated_operating_income": calculated_operating_income,
            "free_cash_flow": free_cash_flow,
            "net_debt": numbers["debt"] - numbers["cash"],
            "gross_margin": gross_profit / numbers["revenue"],
            "operating_margin": numbers["operating_income"] / numbers["revenue"],
            "net_margin": numbers["net_income"] / numbers["revenue"],
            "cash_conversion": _safe_ratio(numbers["cash_from_operations"], numbers["net_income"]),
            "liabilities_to_assets": numbers["total_liabilities"] / numbers["total_assets"],
            "balance_sheet_residual": balance_residual,
            "operating_income_residual": operating_residual,
            "balance_sheet_reconciles": balance_pass,
            "operating_income_reconciles": operating_pass,
        })

    rows.sort(key=lambda row: row["period_end"])
    trends: list[dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        trends.append({
            "from_period": previous["period_end"],
            "to_period": current["period_end"],
            "revenue_growth": current["revenue"] / previous["revenue"] - 1.0 if previous["revenue"] else None,
            "operating_margin_change": current["operating_margin"] - previous["operating_margin"],
            "free_cash_flow_change": current["free_cash_flow"] - previous["free_cash_flow"],
        })

    failures: list[str] = []
    for row in rows:
        if not row["balance_sheet_reconciles"]:
            failures.append(f"{row['period_end']}:balance_sheet")
        if not row["operating_income_reconciles"]:
            failures.append(f"{row['period_end']}:operating_income")

    result = {
        "method": "normalized-three-statement-analysis",
        "method_version": "1.0.0",
        "currency": currency,
        "unit": unit,
        "period_basis": period_basis,
        "periods": rows,
        "trends": trends,
        "reconciliation": {
            "status": "pass" if not failures else "fail",
            "tolerance": tolerance,
            "failures": failures,
        },
        "warnings": ["reconciliation_failure"] if failures else [],
    }
    _require_finite_result(result)
    return result


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0 else None


def _require_finite_result(value: Any, path: str = "result") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} is non-finite; input magnitudes overflowed the calculation")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_result(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            _require_finite_result(child, f"{path}[{index}]")

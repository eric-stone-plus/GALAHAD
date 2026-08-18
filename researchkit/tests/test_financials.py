import pytest

from researchkit.financials import analyze_financial_statements


def _period(period_end: str, revenue: float = 100.0) -> dict:
    return {
        "period_end": period_end,
        "currency": "USD",
        "unit": "USD millions",
        "period_basis": "FY",
        "revenue": revenue,
        "cost_of_revenue": revenue * 0.6,
        "operating_expenses": revenue * 0.2,
        "operating_income": revenue * 0.2,
        "net_income": revenue * 0.15,
        "cash_from_operations": revenue * 0.18,
        "capital_expenditure": revenue * 0.05,
        "total_assets": revenue * 2.0,
        "total_liabilities": revenue * 1.2,
        "total_equity": revenue * 0.8,
        "cash": revenue * 0.2,
        "debt": revenue * 0.5,
    }


def test_financials_reconcile_and_calculate_trends() -> None:
    result = analyze_financial_statements([_period("2024-12-31"), _period("2025-12-31", 110.0)])
    assert result["reconciliation"]["status"] == "pass"
    assert result["periods"][0]["gross_margin"] == pytest.approx(0.4)
    assert result["periods"][0]["free_cash_flow"] == pytest.approx(13.0)
    assert result["trends"][0]["revenue_growth"] == pytest.approx(0.1)


def test_financials_exposes_reconciliation_failures() -> None:
    row = _period("2025-12-31")
    row["total_equity"] = 70.0
    result = analyze_financial_statements([_period("2024-12-31"), row])
    assert result["reconciliation"]["status"] == "fail"
    assert result["warnings"] == ["reconciliation_failure"]


def test_financials_rejects_mixed_basis_and_sign_errors() -> None:
    second = _period("2025-12-31")
    second["currency"] = "EUR"
    with pytest.raises(ValueError, match="must be normalized"):
        analyze_financial_statements([_period("2024-12-31"), second])
    row = _period("2025-12-31")
    row["capital_expenditure"] = -5.0
    with pytest.raises(ValueError, match="positive cash-use sign"):
        analyze_financial_statements([_period("2024-12-31"), row])


def test_financials_rejects_bad_dates_currency_and_unknown_config() -> None:
    row = _period("2025/12/31")
    with pytest.raises(ValueError, match="ISO date"):
        analyze_financial_statements([_period("2024-12-31"), row])
    row = _period("2025-12-31")
    row["currency"] = "usd"
    with pytest.raises(ValueError, match="uppercase ISO 4217"):
        analyze_financial_statements([_period("2024-12-31"), row])
    with pytest.raises(ValueError, match="unsupported config"):
        analyze_financial_statements([_period("2024-12-31"), _period("2025-12-31")], {"impute": True})
    with pytest.raises(ValueError, match="JSON number"):
        analyze_financial_statements([_period("2024-12-31"), _period("2025-12-31")], {"reconciliation_tolerance": True})
    row = _period("2025-12-31")
    row["revenue"] = True
    with pytest.raises(ValueError, match="JSON number"):
        analyze_financial_statements([_period("2024-12-31"), row])
    row = _period("2025-12-31")
    row["revenue"] = "100"
    with pytest.raises(ValueError, match="JSON number"):
        analyze_financial_statements([_period("2024-12-31"), row])


def test_financials_requires_two_comparable_periods() -> None:
    with pytest.raises(ValueError, match="at least two"):
        analyze_financial_statements([_period("2025-12-31")])


def test_financials_rejects_finite_inputs_that_overflow_outputs() -> None:
    first = _period("2024-12-31")
    second = _period("2025-12-31")
    second["revenue"] = 1e308
    second["cost_of_revenue"] = -1e308
    with pytest.raises(ValueError, match="overflowed"):
        analyze_financial_statements([first, second])

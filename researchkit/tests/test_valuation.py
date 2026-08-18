import math

import pytest

from researchkit.valuation import comps_valuation, dcf_sensitivity, dcf_valuation


def test_one_period_dcf_matches_hand_calculation() -> None:
    result = dcf_valuation(
        [{"period": "2026", "unlevered_fcf": 10.0, "discount_period": 1.0}],
        wacc=0.10,
        terminal_growth=0.02,
        net_debt=20.0,
        non_operating_assets=5.0,
        diluted_shares=10.0,
    )
    expected_terminal = 10.0 * 1.02 / (0.10 - 0.02)
    expected_ev = 10.0 / 1.10 + expected_terminal / 1.10
    assert result["enterprise_value"] == pytest.approx(expected_ev)
    assert result["method_version"] == "1.0.0"
    assert result["equity_value"] == pytest.approx(expected_ev - 20.0 + 5.0)
    assert result["value_per_share"] == pytest.approx((expected_ev - 15.0) / 10.0)


def test_dcf_rejects_invalid_terminal_and_periods() -> None:
    forecast = [{"period": "2026", "unlevered_fcf": 10.0, "discount_period": 1.0}]
    with pytest.raises(ValueError, match="below wacc"):
        dcf_valuation(forecast, wacc=0.05, terminal_growth=0.05, net_debt=0, non_operating_assets=0, diluted_shares=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        dcf_valuation(
            forecast + [{"period": "2027", "unlevered_fcf": 11.0, "discount_period": 1.0}],
            wacc=0.10, terminal_growth=0.02, net_debt=0, non_operating_assets=0, diluted_shares=1,
        )
    with pytest.raises(ValueError, match="odd number"):
        dcf_sensitivity(
            [{"period": "2026", "unlevered_fcf": 10.0, "discount_period": 1.0}],
            wacc_values=[0.08, 0.10], terminal_growth_values=[0.01, 0.02, 0.03],
            net_debt=0, non_operating_assets=0, diluted_shares=1,
        )
    with pytest.raises(ValueError, match="symmetric"):
        dcf_sensitivity(
            [{"period": "2026", "unlevered_fcf": 10.0, "discount_period": 1.0}],
            wacc_values=[0.08, 0.09, 0.11], terminal_growth_values=[0.01, 0.02, 0.03],
            net_debt=0, non_operating_assets=0, diluted_shares=1,
        )
    with pytest.raises(ValueError, match="labels"):
        dcf_valuation(
            forecast + [{"period": "2026", "unlevered_fcf": 11.0, "discount_period": 2.0}],
            wacc=0.10, terminal_growth=0.02, net_debt=0, non_operating_assets=0, diluted_shares=1,
        )
    with pytest.raises(ValueError, match="JSON number"):
        dcf_valuation(forecast, wacc=True, terminal_growth=0.02, net_debt=0, non_operating_assets=0, diluted_shares=1)
    with pytest.raises(ValueError, match="JSON number"):
        dcf_valuation(forecast, wacc="0.10", terminal_growth=0.02, net_debt=0, non_operating_assets=0, diluted_shares=1)


def test_dcf_sensitivity_is_monotonic_and_marks_invalid_cells() -> None:
    grid = dcf_sensitivity(
        [{"period": "2026", "unlevered_fcf": 10.0, "discount_period": 1.0}],
        wacc_values=[0.08, 0.09, 0.10],
        terminal_growth_values=[0.02, 0.06, 0.10],
        net_debt=0,
        non_operating_assets=0,
        diluted_shares=1,
    )
    values = {(cell["wacc"], cell["terminal_growth"]): cell for cell in grid}
    assert values[(0.08, 0.02)]["value_per_share"] > values[(0.10, 0.02)]["value_per_share"]
    assert values[(0.10, 0.10)]["status"] == "invalid"
    with pytest.raises(ValueError, match="strictly increasing"):
        dcf_sensitivity(
            [{"period": "2026", "unlevered_fcf": 10.0, "discount_period": 1.0}],
            wacc_values=[0.10, 0.08], terminal_growth_values=[0.01, 0.02],
            net_debt=0, non_operating_assets=0, diluted_shares=1,
        )


def test_comps_quantiles_and_enterprise_bridge() -> None:
    peers = [
        {"name": "A", "included": True, "numerator": 80.0, "denominator": 10.0},
        {"name": "B", "included": True, "numerator": 100.0, "denominator": 10.0},
        {"name": "C", "included": True, "numerator": 120.0, "denominator": 10.0},
        {"name": "D", "included": False, "exclusion_reason": "different business mix"},
    ]
    result = comps_valuation(
        peers,
        subject_metric=20.0,
        basis="enterprise",
        diluted_shares=10.0,
        net_debt=30.0,
        non_operating_assets=10.0,
    )
    assert result["distribution"]["median"] == pytest.approx(10.0)
    assert result["selected"]["enterprise_value"] == pytest.approx(200.0)
    assert result["selected"]["equity_value"] == pytest.approx(180.0)
    assert result["selected"]["value_per_share"] == pytest.approx(18.0)


def test_comps_fails_closed_on_bad_peers_and_nonfinite_values() -> None:
    peers = [{"name": name, "included": True, "numerator": 10.0, "denominator": 1.0} for name in "ABC"]
    with pytest.raises(ValueError, match="subject_metric"):
        comps_valuation(peers, subject_metric=math.nan, basis="equity", diluted_shares=1)
    with pytest.raises(ValueError, match="positive numerator"):
        comps_valuation(peers[:2] + [{"name": "C", "included": True, "numerator": 10, "denominator": 0}], subject_metric=1, basis="equity", diluted_shares=1)
    with pytest.raises(ValueError, match="exclusion_reason"):
        comps_valuation(peers + [{"name": "D", "included": False}], subject_metric=1, basis="equity", diluted_shares=1)
    with pytest.raises(ValueError, match="unique"):
        comps_valuation(peers + [{"name": "A", "included": False, "exclusion_reason": "duplicate"}], subject_metric=1, basis="equity", diluted_shares=1)
    with pytest.raises(ValueError, match="only allowed"):
        comps_valuation(peers, subject_metric=1, basis="equity", diluted_shares=1, explicit_multiple=12)
    with pytest.raises(ValueError, match="exclusion_reason"):
        comps_valuation(peers + [{"name": "D", "included": False, "exclusion_reason": True}], subject_metric=1, basis="equity", diluted_shares=1)


def test_equity_comps_emit_json_safe_non_applicable_enterprise_value() -> None:
    peers = [
        {"name": name, "included": True, "numerator": multiple * 10.0, "denominator": 10.0}
        for name, multiple in zip("ABC", (8.0, 10.0, 12.0))
    ]
    result = comps_valuation(peers, subject_metric=20.0, basis="equity", diluted_shares=10.0)
    assert result["selected"]["enterprise_value"] is None
    assert result["selected"]["value_per_share"] == pytest.approx(20.0)


def test_valuation_rejects_finite_inputs_that_overflow_outputs() -> None:
    with pytest.raises(ValueError, match="overflowed"):
        dcf_valuation(
            [{"period": "Y1", "unlevered_fcf": 1e308, "discount_period": 1}],
            wacc=0.10, terminal_growth=0.09, net_debt=0, non_operating_assets=0, diluted_shares=1,
        )
    peers = [
        {"name": name, "included": True, "numerator": 1e308, "denominator": 1.0}
        for name in "ABC"
    ]
    with pytest.raises(ValueError, match="overflowed"):
        comps_valuation(peers, subject_metric=1e308, basis="enterprise", diluted_shares=1)
    with pytest.raises(ValueError, match="overflowed|overflow"):
        dcf_valuation(
            [{"period": "Y1", "unlevered_fcf": 1.0, "discount_period": 1e308}],
            wacc=0.10, terminal_growth=0.02, net_debt=0, non_operating_assets=0, diluted_shares=1,
        )


def test_sensitivity_rejects_shared_input_errors_before_cell_loop() -> None:
    with pytest.raises(ValueError, match="net_debt"):
        dcf_sensitivity(
            [{"period": "Y1", "unlevered_fcf": 10.0, "discount_period": 1.0}],
            wacc_values=[0.08, 0.09, 0.10], terminal_growth_values=[0.01, 0.02, 0.03],
            net_debt=math.nan, non_operating_assets=0, diluted_shares=1,
        )
    with pytest.raises(ValueError, match="shared discount periods"):
        dcf_sensitivity(
            [{"period": "Y1", "unlevered_fcf": 10.0, "discount_period": 1e308}],
            wacc_values=[0.08, 0.09, 0.10], terminal_growth_values=[0.01, 0.02, 0.03],
            net_debt=0, non_operating_assets=0, diluted_shares=1,
        )

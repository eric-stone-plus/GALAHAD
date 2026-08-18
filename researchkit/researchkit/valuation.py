"""Small deterministic valuation kernels used by GALAHAD skills."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


METHOD_VERSION = "1.0.0"


def dcf_valuation(
    forecasts: Sequence[Mapping[str, Any]],
    *,
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    non_operating_assets: float,
    diluted_shares: float,
) -> dict[str, Any]:
    """Calculate an unlevered FCF DCF with explicit discount periods."""

    wacc = _coerce_finite("wacc", wacc)
    terminal_growth = _coerce_finite("terminal_growth", terminal_growth)
    net_debt = _coerce_finite("net_debt", net_debt)
    non_operating_assets = _coerce_finite("non_operating_assets", non_operating_assets)
    diluted_shares = _coerce_finite("diluted_shares", diluted_shares)
    if not forecasts:
        raise ValueError("forecasts must contain at least one period")
    if wacc <= 0 or wacc >= 1:
        raise ValueError("wacc must be between 0 and 1")
    if terminal_growth <= -1 or terminal_growth >= wacc:
        raise ValueError("terminal_growth must be greater than -1 and below wacc")
    if diluted_shares <= 0:
        raise ValueError("diluted_shares must be positive")

    validated_forecasts = _normalize_forecasts(forecasts)
    normalized: list[dict[str, Any]] = []
    for row in validated_forecasts:
        label = row["period"]
        fcf = row["unlevered_fcf"]
        discount_period = row["discount_period"]
        try:
            discount_factor = (1.0 + wacc) ** discount_period
        except OverflowError as exc:
            raise ValueError("discount factor overflowed; discount_period magnitude is unsupported") from exc
        normalized.append({
            "period": label,
            "unlevered_fcf": fcf,
            "discount_period": discount_period,
            "present_value": fcf / discount_factor,
        })

    final_fcf = normalized[-1]["unlevered_fcf"]
    terminal_value = final_fcf * (1.0 + terminal_growth) / (wacc - terminal_growth)
    try:
        terminal_discount_factor = (1.0 + wacc) ** normalized[-1]["discount_period"]
    except OverflowError as exc:
        raise ValueError("terminal discount factor overflowed; discount_period magnitude is unsupported") from exc
    pv_terminal = terminal_value / terminal_discount_factor
    pv_forecast = sum(row["present_value"] for row in normalized)
    enterprise_value = pv_forecast + pv_terminal
    equity_value = enterprise_value - net_debt + non_operating_assets
    value_per_share = equity_value / diluted_shares
    terminal_share = pv_terminal / enterprise_value if enterprise_value else None

    warnings: list[str] = []
    if enterprise_value <= 0:
        warnings.append("enterprise_value_non_positive")
    if equity_value <= 0:
        warnings.append("equity_value_non_positive")
    if terminal_share is not None and math.isfinite(terminal_share) and terminal_share > 0.75:
        warnings.append("terminal_value_exceeds_75_percent_of_enterprise_value")

    result = {
        "method": "unlevered-fcf-perpetuity-growth",
        "method_version": METHOD_VERSION,
        "forecast": normalized,
        "wacc": wacc,
        "terminal_growth": terminal_growth,
        "pv_forecast_cash_flows": pv_forecast,
        "terminal_value_at_horizon": terminal_value,
        "pv_terminal_value": pv_terminal,
        "terminal_value_share": terminal_share,
        "enterprise_value": enterprise_value,
        "net_debt": net_debt,
        "non_operating_assets": non_operating_assets,
        "equity_value": equity_value,
        "diluted_shares": diluted_shares,
        "value_per_share": value_per_share,
        "warnings": warnings,
    }
    _require_finite_result(result)
    return result


def dcf_sensitivity(
    forecasts: Sequence[Mapping[str, Any]],
    *,
    wacc_values: Sequence[float],
    terminal_growth_values: Sequence[float],
    net_debt: float,
    non_operating_assets: float,
    diluted_shares: float,
) -> list[dict[str, Any]]:
    """Return a complete WACC/growth grid, marking invalid pairs explicitly."""

    if not wacc_values or not terminal_growth_values:
        raise ValueError("both sensitivity axes must be non-empty")
    normalized_wacc = _ordered_finite_axis(wacc_values, "wacc_values")
    normalized_growth = _ordered_finite_axis(terminal_growth_values, "terminal_growth_values")
    if len(normalized_wacc) < 3 or len(normalized_wacc) % 2 == 0:
        raise ValueError("wacc_values must contain an odd number of at least three values")
    if len(normalized_growth) < 3 or len(normalized_growth) % 2 == 0:
        raise ValueError("terminal_growth_values must contain an odd number of at least three values")
    _validate_symmetric_axis(normalized_wacc, "wacc_values")
    _validate_symmetric_axis(normalized_growth, "terminal_growth_values")
    validated_forecasts = _normalize_forecasts(forecasts)
    validated_net_debt = _coerce_finite("net_debt", net_debt)
    validated_assets = _coerce_finite("non_operating_assets", non_operating_assets)
    validated_shares = _coerce_finite("diluted_shares", diluted_shares)
    if validated_shares <= 0:
        raise ValueError("diluted_shares must be positive")
    _validate_discountability(validated_forecasts, min(normalized_wacc))
    cells: list[dict[str, Any]] = []
    for wacc in normalized_wacc:
        for growth in normalized_growth:
            try:
                result = dcf_valuation(
                    validated_forecasts,
                    wacc=wacc,
                    terminal_growth=growth,
                    net_debt=validated_net_debt,
                    non_operating_assets=validated_assets,
                    diluted_shares=validated_shares,
                )
            except ValueError as exc:
                cells.append({"wacc": wacc, "terminal_growth": growth, "status": "invalid", "reason": str(exc)})
            else:
                cells.append({
                    "wacc": wacc,
                    "terminal_growth": growth,
                    "status": "ok",
                    "value_per_share": result["value_per_share"],
                })
    _require_finite_result(cells)
    return cells


def comps_valuation(
    peers: Sequence[Mapping[str, Any]],
    *,
    subject_metric: float,
    basis: str,
    diluted_shares: float,
    net_debt: float = 0.0,
    non_operating_assets: float = 0.0,
    statistic: str = "median",
    explicit_multiple: float | None = None,
) -> dict[str, Any]:
    """Calculate peer multiples and translate the selected multiple to equity value."""

    subject_metric = _coerce_finite("subject_metric", subject_metric)
    diluted_shares = _coerce_finite("diluted_shares", diluted_shares)
    net_debt = _coerce_finite("net_debt", net_debt)
    non_operating_assets = _coerce_finite("non_operating_assets", non_operating_assets)
    if subject_metric <= 0:
        raise ValueError("subject_metric must be positive")
    if diluted_shares <= 0:
        raise ValueError("diluted_shares must be positive")
    if basis not in {"enterprise", "equity"}:
        raise ValueError("basis must be enterprise or equity")
    if statistic not in {"p25", "median", "p75", "explicit"}:
        raise ValueError("statistic must be p25, median, p75, or explicit")
    if statistic == "explicit" and explicit_multiple is None:
        raise ValueError("explicit_multiple is required for explicit selection")
    if explicit_multiple is not None and statistic != "explicit":
        raise ValueError("explicit_multiple is only allowed with statistic='explicit'")

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, peer in enumerate(peers):
        if not isinstance(peer, Mapping):
            raise ValueError(f"peer {index} must be an object")
        try:
            raw_name = peer["name"]
            name = raw_name if isinstance(raw_name, str) else ""
            use = peer["included"]
        except KeyError as exc:
            raise ValueError(f"peer {index} must provide name and included") from exc
        if not name or not isinstance(use, bool):
            raise ValueError(f"peer {index} has invalid name or included flag")
        if name in seen_names:
            raise ValueError(f"peer names must be unique: {name}")
        seen_names.add(name)
        if not use:
            raw_reason = peer.get("exclusion_reason")
            if not isinstance(raw_reason, str) or not raw_reason.strip():
                raise ValueError(f"excluded peer {name} requires exclusion_reason")
            reason = raw_reason.strip()
            excluded.append({"name": name, "reason": reason})
            continue
        try:
            numerator = _coerce_finite(f"included peer {name} numerator", peer["numerator"])
            denominator = _coerce_finite(f"included peer {name} denominator", peer["denominator"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"included peer {name} requires numeric numerator and denominator") from exc
        if numerator <= 0 or denominator <= 0:
            raise ValueError(f"included peer {name} requires positive numerator and denominator")
        included.append({
            "name": name,
            "numerator": numerator,
            "denominator": denominator,
            "multiple": numerator / denominator,
        })

    if len(included) < 3:
        raise ValueError("at least three included peers are required")
    multiples = sorted(row["multiple"] for row in included)
    distribution = {
        "count": len(multiples),
        "minimum": multiples[0],
        "p25": _quantile(multiples, 0.25),
        "median": _quantile(multiples, 0.50),
        "p75": _quantile(multiples, 0.75),
        "maximum": multiples[-1],
    }
    selected = _coerce_finite("explicit_multiple", explicit_multiple) if statistic == "explicit" else float(distribution[statistic])
    if selected <= 0:
        raise ValueError("selected multiple must be positive")

    def translate(multiple: float) -> dict[str, float | None]:
        indicated = multiple * subject_metric
        if basis == "enterprise":
            enterprise_value = indicated
            equity_value = enterprise_value - net_debt + non_operating_assets
        else:
            enterprise_value = None
            equity_value = indicated
        return {
            "multiple": multiple,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "value_per_share": equity_value / diluted_shares,
        }

    warnings: list[str] = []
    if excluded:
        warnings.append("peer_exclusions_present")
    if selected < distribution["p25"] or selected > distribution["p75"]:
        warnings.append("selected_multiple_outside_interquartile_range")

    result = {
        "method": "trading-comparables",
        "method_version": METHOD_VERSION,
        "basis": basis,
        "included_peers": included,
        "excluded_peers": excluded,
        "distribution": distribution,
        "selection": statistic,
        "selected": translate(selected),
        "range": {
            "p25": translate(distribution["p25"]),
            "median": translate(distribution["median"]),
            "p75": translate(distribution["p75"]),
        },
        "subject_metric": subject_metric,
        "net_debt": net_debt,
        "non_operating_assets": non_operating_assets,
        "diluted_shares": diluted_shares,
        "warnings": warnings,
    }
    _require_finite_result(result)
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _ordered_finite_axis(values: Sequence[float], name: str) -> list[float]:
    normalized: list[float] = []
    for value in values:
        try:
            number = _coerce_finite(name, value)
        except ValueError as exc:
            raise ValueError(f"{name} must contain finite JSON numbers") from exc
        normalized.append(number)
    if any(current <= previous for previous, current in zip(normalized, normalized[1:])):
        raise ValueError(f"{name} must be strictly increasing with unique values")
    return normalized


def _validate_symmetric_axis(values: Sequence[float], name: str) -> None:
    center = values[len(values) // 2]
    tolerance = 1e-12 * max(abs(center), 1.0)
    for lower, upper in zip(values, reversed(values)):
        if not math.isclose(center - lower, upper - center, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(f"{name} must be symmetric around its center value")


def _coerce_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number, not {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _normalize_forecasts(forecasts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate shared sensitivity inputs without evaluating a WACC/g pair."""

    if not forecasts:
        raise ValueError("forecasts must contain at least one period")
    normalized: list[dict[str, Any]] = []
    labels: set[str] = set()
    previous = 0.0
    for index, row in enumerate(forecasts):
        if not isinstance(row, Mapping):
            raise ValueError(f"forecast {index} must be an object")
        try:
            label = row["period"] if isinstance(row["period"], str) else ""
            fcf = _coerce_finite(f"forecast {index} unlevered_fcf", row["unlevered_fcf"])
            period = _coerce_finite(f"forecast {index} discount_period", row["discount_period"])
        except KeyError as exc:
            raise ValueError(f"forecast {index} must provide period, unlevered_fcf, and discount_period") from exc
        if not label or label in labels:
            raise ValueError("forecast period labels must be non-empty and unique")
        if period <= previous:
            raise ValueError("discount_period values must be positive and strictly increasing")
        labels.add(label)
        normalized.append({"period": label, "unlevered_fcf": fcf, "discount_period": period})
        previous = period
    return normalized


def _validate_discountability(forecasts: Sequence[Mapping[str, Any]], wacc: float) -> None:
    for row in forecasts:
        try:
            discount_factor = (1.0 + wacc) ** row["discount_period"]
        except OverflowError as exc:
            raise ValueError("discount factor overflowed; shared discount periods are unsupported") from exc
        if not math.isfinite(discount_factor):
            raise ValueError("discount factor is non-finite; shared discount periods are unsupported")


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

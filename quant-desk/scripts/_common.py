"""Shared helpers for quant_desk scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from quantkit.gates import COST_TIERS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
DATA = ROOT / "data"
STATE = ROOT / "state"
CFG_PATH = ROOT / "config.yaml"


def load_cfg() -> dict[str, Any]:
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}


def ensure_layout() -> None:
    for p in (OUT, DATA, STATE):
        p.mkdir(parents=True, exist_ok=True)


def resolve_fee_bps(cfg: dict[str, Any]) -> tuple[float, str | None]:
    """Per-side paper fee in bps, derived from the declared ``cost_tier``.

    Cost discipline (quiz2 r3): backtests assume a two-sided all-in friction
    from ``COST_TIERS`` (0.2%/0.4%/0.5%), never the legacy 「万五」 default.
    PaperBook charges per fill (single side), so the per-side fee is half the
    tier rate. Returns ``(fee_bps, cost_tier)``; ``cost_tier`` is None when
    only a legacy ``fee_bps`` key is configured (then fee_bps passes through).
    """
    tier = cfg.get("cost_tier")
    if tier is not None:
        if tier not in COST_TIERS:
            raise ValueError(f"unknown cost_tier: {tier!r} (use one of {sorted(COST_TIERS)})")
        return COST_TIERS[tier] * 10_000.0 / 2.0, str(tier)
    return float(cfg.get("fee_bps", 5.0)), None

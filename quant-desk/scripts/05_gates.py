#!/usr/bin/env python3
"""门控 go/no-go 评价：evaluate_gates → output/gate_report.json（fail-closed）。

  quant-python scripts/05_gates.py
  quant-python scripts/05_gates.py --metrics output/gate_metrics.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_layout, load_cfg  # noqa: E402
from _gates import run_gate_report  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="quant_desk 门控 go/no-go 评价")
    p.add_argument(
        "--metrics",
        default=None,
        help="外部证据指标 JSON（pbo/dsr/拥挤度/纸面/实盘等；缺省读 output/gate_metrics.json 若存在）",
    )
    args = p.parse_args()
    ensure_layout()
    report = run_gate_report(load_cfg(), extra_path=args.metrics)

    for g in report["gates"]:
        status = "GO" if g["passed"] else "NO-GO"
        line = f"[{status}] {g['gate']}"
        if g["missing"]:
            line += f" missing={g['missing']}"
        if g["failures"]:
            line += f" failures={g['failures']}"
        print(line)
    print(f"verdict: {report['verdict']}")
    print("gate_report → output/gate_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

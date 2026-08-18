#!/usr/bin/env python3
"""Daily routine: select → paper rebalance → evaluate → review."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def run(name: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / name)]
    print(f"\n>>> {name}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> int:
    run("01_select.py")
    run("02_trade.py")
    run("03_evaluate.py")
    run("04_review.py")
    run("05_gates.py")
    print("\nDaily run complete → output/review_latest.html + output/gate_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convenience launcher: quant-python scripts/run_paper.py [--source fixture]"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galahad_futures.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

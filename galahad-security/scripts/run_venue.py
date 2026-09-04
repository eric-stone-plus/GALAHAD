#!/usr/bin/env python3
"""Venue one-shot: execute today's decision on the Alpaca PAPER endpoint.

Loads latest daily bars from the Alpaca data API, runs the strategy
targets through the shared risk gate, submits market orders, fetches
resulting positions, and emits the same summary JSON + journal as the
offline paper engine (plus venue/reconciliation).

Refuses cleanly (non-zero exit, no traceback) when preconditions are
missing: ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET env vars,
risk.enable_alpaca_paper: true, risk.kill_switch: false.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galahad_security.cli import main

if __name__ == "__main__":
    argv = ["--engine", "alpaca_paper", *sys.argv[1:]]
    raise SystemExit(main(argv))

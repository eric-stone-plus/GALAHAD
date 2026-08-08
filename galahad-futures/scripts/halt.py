#!/usr/bin/env python3
"""HALT management for GALAHAD auto cycle.

Usage:
  quant-python scripts/halt.py status          # check HALT status
  quant-python scripts/halt.py on "reason"     # activate HALT
  quant-python scripts/halt.py off             # deactivate HALT
  quant-python scripts/halt.py history         # show HALT history

HALT file: futures/state/HALT
When present, auto_cycle.py refuses to run paper trades.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
HALT_FILE = STATE / "HALT"
HALT_LOG = STATE / "halt_log.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_action(action: str, reason: str = "") -> None:
    HALT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": utc_now(), "action": action, "reason": reason}
    with HALT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def status() -> None:
    if HALT_FILE.is_file():
        content = HALT_FILE.read_text(encoding="utf-8", errors="replace").strip()
        print(f"HALT ACTIVE: {content}")
    else:
        print("HALT INACTIVE — auto cycle can run")


def activate(reason: str) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    HALT_FILE.write_text(reason, encoding="utf-8")
    log_action("activate", reason)
    print(f"HALT ACTIVATED: {reason}")


def deactivate() -> None:
    if HALT_FILE.is_file():
        reason = HALT_FILE.read_text(encoding="utf-8", errors="replace").strip()
        HALT_FILE.unlink()
        log_action("deactivate", reason)
        print(f"HALT DEACTIVATED (was: {reason})")
    else:
        print("HALT was not active")


def history() -> None:
    if not HALT_LOG.is_file():
        print("No HALT history")
        return
    lines = HALT_LOG.read_text(encoding="utf-8").strip().split("\n")
    for line in lines[-20:]:
        try:
            e = json.loads(line)
            print(f"  {e['ts']} {e['action']}: {e.get('reason', '')}")
        except json.JSONDecodeError:
            pass


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: halt.py {status|on|off|history} [reason]")
        return 2

    cmd = sys.argv[1]
    if cmd == "status":
        status()
    elif cmd == "on":
        reason = sys.argv[2] if len(sys.argv) > 2 else "manual HALT"
        activate(reason)
    elif cmd == "off":
        deactivate()
    elif cmd == "history":
        history()
    else:
        print(f"Unknown command: {cmd}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

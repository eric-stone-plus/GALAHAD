"""Cycle HALT behavior — trade skipped, perception still runs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QP = Path.home() / ".local/bin/quant-python"
CYCLE = ROOT / "scripts" / "run_cycle.py"


def test_halt_skips_paper_but_writes_cycle(tmp_path):
    halt = tmp_path / "HALT"
    halt.write_text("test | unit halt\n", encoding="utf-8")
    ops = tmp_path / "ops"
    ops.mkdir()
    r = subprocess.run(
        [
            str(QP),
            str(CYCLE),
            "--offline",
            "--source",
            "fixture",
            "--halt-file",
            str(halt),
            "--ops-state",
            str(ops),
            "--json",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    data = json.loads(r.stdout)
    assert data["halted"] is True
    assert data["trade_status"] == "halted"
    assert data["paper"]["status"] == "halted"
    assert data["paper"].get("n_fills", 0) == 0
    # perception still populated
    assert data["perception"]["n_symbols"] >= 1 or data["perception"].get("fetch_error")
    assert (ops / "last_galahad_paper.json").is_file()
    paper = json.loads((ops / "last_galahad_paper.json").read_text())
    assert paper["status"] == "halted"


def test_no_halt_runs_paper(tmp_path):
    halt = tmp_path / "no_such_HALT"
    ops = tmp_path / "ops"
    ops.mkdir()
    r = subprocess.run(
        [
            str(QP),
            str(CYCLE),
            "--offline",
            "--source",
            "fixture",
            "--halt-file",
            str(halt),
            "--ops-state",
            str(ops),
            "--json",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    data = json.loads(r.stdout)
    assert data["halted"] is False
    assert data["paper"] is not None
    assert data["paper"].get("n_fills", 0) >= 1 or data["paper"].get("status")
    assert isinstance(data["paper"].get("final_equity"), (int, float))

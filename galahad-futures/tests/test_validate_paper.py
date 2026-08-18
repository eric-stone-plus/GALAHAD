"""Drive validate_paper entry on fixture — real shipped script path."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QP = Path.home() / ".local/bin/quant-python"
SCRIPT = ROOT / "scripts" / "validate_paper.py"


def test_validate_paper_writes_report():
    r = subprocess.run(
        [str(QP), str(SCRIPT), "--source", "fixture", "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    data = json.loads(r.stdout)
    assert data.get("status") in ("ok", "insufficient_returns")
    report = ROOT / "output" / "validation_report.json"
    assert report.is_file()
    disk = json.loads(report.read_text(encoding="utf-8"))
    assert disk.get("n_variants", 0) >= 1
    if disk.get("status") == "ok":
        assert "deflated_sharpe_ratio" in disk
        assert "policy_flags" in disk
        assert isinstance(disk["deflated_sharpe_ratio"], (int, float))

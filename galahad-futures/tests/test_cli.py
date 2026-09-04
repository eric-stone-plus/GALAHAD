"""CLI contract tests — ``--json`` keeps stdout machine-readable.

The live TradingNode banner and its event stream log to stdout; under
``--json`` those must be diverted to stderr so the summary channel stays
parseable (STAMMTISCH's galahad adapter reads stdout as JSON only).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_futures import cli


def _fake_session_noisy_stdout(**kwargs):
    print("GALAHAD-TESTNET-001.TradingNode: banner noise on stdout")
    print("more engine log lines", file=sys.stdout)
    return {
        "status": "ok",
        "mode": "paper",
        "engine": "paper",
        "engine_version": "test",
        "strategy": "dual_ma",
        "symbol": "BTCUSDT",
        "bars": 10,
        "source_used": "fixture",
        "data_note": "",
        "n_fills": 0,
        "n_risk_rejects": 0,
        "equity_curve_len": 0,
        "initial_equity": 10_000.0,
        "final_equity": 10_000.0,
        "sample_kind": "fixture",
        "total_funding": 0.0,
        "n_funding_events": 0,
        "invalidated": False,
        "max_drawdown": 0.0,
        "liquidated": False,
        "journal_path": None,
    }


def test_json_stdout_stays_parseable_when_engine_logs(monkeypatch, capsys):
    monkeypatch.setattr("galahad_futures.engine.run_paper_session", _fake_session_noisy_stdout)
    rc = cli.main(["--source", "fixture", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    # stdout is exactly one JSON document, despite the engine logging.
    summary = json.loads(captured.out)
    assert summary["status"] == "ok"
    # The diverted engine logs are preserved on stderr, not dropped.
    assert "banner noise on stdout" in captured.err


def test_non_json_keeps_human_report_on_stdout(monkeypatch, capsys):
    monkeypatch.setattr("galahad_futures.engine.run_paper_session", _fake_session_noisy_stdout)
    rc = cli.main(["--source", "fixture"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "GALAHAD Futures paper session" in captured.out


def test_json_error_path_restores_stdout(monkeypatch, capsys):
    def boom(**kwargs):
        raise RuntimeError("closed gate")

    monkeypatch.setattr("galahad_futures.engine.run_paper_session", boom)
    rc = cli.main(["--source", "fixture", "--json"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "error: closed gate" in captured.err
    # stdout restored after the error path: later prints land on stdout.
    print("post-restore")
    assert "post-restore" in capsys.readouterr().out

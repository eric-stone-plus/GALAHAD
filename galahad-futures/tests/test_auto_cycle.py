"""auto_cycle kill-switch and exit-code contracts.

Regression:
  - ``--halt-file`` must be honoured as the exact kill-switch path (it used
    to be reduced to a hard-coded ``HALT`` name inside its parent dir)
  - a successful ``--dry-run`` must exit 0, not fall into the generic
    "no paper runs → fetch failed" exit-2 path
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import auto_cycle  # noqa: E402


def _snapshot(status: str, symbols: dict | None = None) -> SimpleNamespace:
    symbols = symbols if symbols is not None else {}
    snap = SimpleNamespace(
        status=status,
        source="fixture" if symbols else "none",
        symbols=symbols,
        fetch_error=None if symbols else "no_prices",
    )
    snap.to_dict = lambda: {
        "status": snap.status,
        "source": snap.source,
        "symbols": sorted(snap.symbols),
        "fetch_error": snap.fetch_error,
    }
    return snap


def test_check_halt_uses_exact_path(tmp_path):
    flag = tmp_path / "kill.flag"
    flag.write_text("ops halt", encoding="utf-8")
    assert auto_cycle.check_halt(flag) == (True, "ops halt")
    # the sibling default HALT name must not answer for a custom path
    (tmp_path / "HALT").write_text("stale default", encoding="utf-8")
    assert auto_cycle.check_halt(tmp_path / "other.flag") == (False, "")
    assert auto_cycle.check_halt(tmp_path / "HALT") == (True, "stale default")


def test_halt_file_flag_stops_cycle(tmp_path, monkeypatch, capsys):
    ran = {"cycle": False}

    def _must_not_run(*args, **kwargs):
        ran["cycle"] = True
        raise AssertionError("cycle must not run when halted")

    monkeypatch.setattr(auto_cycle, "run_cycle", _must_not_run)
    state = tmp_path / "state"
    state.mkdir()
    flag = state / "kill.flag"
    flag.write_text("kill switch", encoding="utf-8")
    rc = auto_cycle.main(
        ["--state-dir", str(state), "--halt-file", str(flag), "--json"]
    )
    assert rc == 1
    assert ran["cycle"] is False
    assert "kill switch" in capsys.readouterr().out


def test_halt_custom_flag_absent_ignores_default_name(tmp_path, monkeypatch):
    # a default-named HALT in the state dir must be ignored when the
    # operator pointed --halt-file at a custom (absent) path: only that
    # exact path is the kill switch
    state = tmp_path / "state"
    state.mkdir()
    (state / "HALT").write_text("default-name halt", encoding="utf-8")
    called = {}

    def fake_run_cycle(cfg, *, state_dir, **kwargs):
        called["state_dir"] = state_dir
        return {
            "ts": "",
            "cycle_id": "",
            "halted": False,
            "perception": {"status": "ok", "n_symbols": 0},
            "paper_runs": [{"symbol": "BTCUSDT", "status": "ok"}],
            "summary": "",
        }

    monkeypatch.setattr(auto_cycle, "run_cycle", fake_run_cycle)
    rc = auto_cycle.main(
        ["--state-dir", str(state), "--halt-file", str(state / "kill.flag"), "--json"]
    )
    assert rc == 0
    assert called.get("state_dir") == state


def test_dry_run_success_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        auto_cycle, "build_snapshot", lambda **kw: _snapshot("ok", {"BTCUSDT": 60000.0})
    )
    rc = auto_cycle.main(["--state-dir", str(tmp_path), "--dry-run", "--json"])
    assert rc == 0


def test_dry_run_fetch_failed_still_exits_two(tmp_path, monkeypatch):
    monkeypatch.setattr(
        auto_cycle, "build_snapshot", lambda **kw: _snapshot("fetch_failed", {})
    )
    rc = auto_cycle.main(["--state-dir", str(tmp_path), "--dry-run", "--json"])
    assert rc == 2

"""CLI boundary tests — symbol validation and clean config-error refusal."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galahad_security.cli import main


def test_cli_rejects_traversal_symbols(capsys):
    # --symbols lands in data/fixtures|cache paths; "../../tmp/x" must be
    # refused at the CLI boundary, never reach the filesystem.
    rc = main(["--symbols", "../../tmp/x"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "invalid symbol" in err
    assert "Traceback" not in err


def test_cli_rejects_lowercase_and_special_chars(capsys):
    assert main(["--symbols", "AAPL,MS FT"]) == 2
    assert "invalid symbol" in capsys.readouterr().err
    assert main(["--symbols", "AA/PL"]) == 2
    assert "invalid symbol" in capsys.readouterr().err


def test_cli_accepts_legit_symbol_charset():
    # dots and dashes are legitimate (BRK.B, BTC-USD style tickers)
    from galahad_security.cli import _parse_symbols

    assert _parse_symbols("aapl, BRK.B, btc-usd ") == ["AAPL", "BRK.B", "BTC-USD"]


def test_cli_config_value_error_fails_clean(tmp_path, capsys):
    # Malformed config (here: unknown strategy) must refuse cleanly like the
    # RuntimeError path — exit 2, operator-facing error, no traceback dump.
    bad = tmp_path / "bad.yaml"
    bad.write_text("strategy:\n  name: not_a_strategy\n", encoding="utf-8")
    rc = main(["--config", str(bad), "--output-dir", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown strategy" in err
    assert "Traceback" not in err


def test_cli_malformed_derisk_ladder_fails_clean(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "risk:\n  kill_switch: false\n"
        "  derisk_ladder:\n    - {drawdown: 0.1, leverage_multiplier: 2.0}\n",
        encoding="utf-8",
    )
    rc = main(["--config", str(bad), "--output-dir", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "derisk_ladder" in err
    assert "Traceback" not in err

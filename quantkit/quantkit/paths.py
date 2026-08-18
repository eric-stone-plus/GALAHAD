"""Path helpers so project scripts never write into quant/."""

from __future__ import annotations

from pathlib import Path


def project_root(file: str | Path) -> Path:
    """Return the project slug root given any file under that project.

    Usage in a project script::

        ROOT = project_root(__file__)
        OUTPUT = ROOT / "output"
    """
    return Path(file).resolve().parent


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

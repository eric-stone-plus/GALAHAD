"""Path shim for tests/: lets the test modules import from scripts/.
quantkit is expected to be installed in the active environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

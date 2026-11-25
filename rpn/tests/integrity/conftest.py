"""Make the installed integrity kit (.integrity/controls.py, results_meta.py) importable by these tests."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RPN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / ".integrity"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

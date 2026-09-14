"""Shim: canonical launcher is ``node/start_all_nodes.py`` (training-node tree)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_target = Path(__file__).resolve().parent / "node" / "start_all_nodes.py"
sys.argv[0] = str(_target)
runpy.run_path(str(_target), run_name="__main__")

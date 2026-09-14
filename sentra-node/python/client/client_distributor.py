"""Dataset-owner entrypoint; implementation lives in ``sentra_client``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Monorepo dev: ``ml_training`` lives in ``../node`` (not installed as a wheel).
_node = Path(__file__).resolve().parent.parent / "node"
if _node.is_dir():
    np = str(_node)
    sys.path.insert(0, np)
    prev = (os.environ.get("PYTHONPATH") or "").strip()
    os.environ["PYTHONPATH"] = np if not prev else f"{np}{os.pathsep}{prev}"

from sentra_client.distributor import main

if __name__ == "__main__":
    main()

"""Shim: canonical entrypoint is ``client/client_distributor.py`` (dataset-owner tree)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_node = str(_root / "node")
_prev = (os.environ.get("PYTHONPATH") or "").strip()
os.environ["PYTHONPATH"] = _node if not _prev else f"{_node}{os.pathsep}{_prev}"
sys.path.insert(0, str(_root / "client"))


def _run() -> None:
    from sentra_client.distributor import main

    main()


if __name__ == "__main__":
    _run()

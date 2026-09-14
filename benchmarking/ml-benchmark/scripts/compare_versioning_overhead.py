#!/usr/bin/env python3
"""Compare training time and versioning overhead between two run_* directories."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_NODE = _REPO / "sentra-node" / "python" / "node"
if str(_NODE) not in sys.path:
    sys.path.insert(0, str(_NODE))

from ml_training.benchmark_stats import (  # noqa: E402
    parse_benchmark_lines,
    parse_legacy_timings,
    summarize_runtime_metrics_from_logs,
)


def _read_run(run_dir: Path) -> dict[str, str]:
    logs: dict[str, str] = {}
    for path in sorted(run_dir.glob("node_*.log")):
        m = re.match(r"node_(\d+)\.log$", path.name)
        if m:
            logs[int(m.group(1))] = path.read_text(encoding="utf-8", errors="replace")
    return logs


def _training_sec(node_logs: dict[int, str]) -> float | None:
    for text in node_logs.values():
        legacy = parse_legacy_timings(text)
        if "training_sec" in legacy:
            return float(legacy["training_sec"])
        for row in parse_benchmark_lines(text):
            if row.get("phase") == "training" and "wall_sec" in row:
                return float(row["wall_sec"])
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Versioning overhead vs baseline run")
    p.add_argument("baseline_run", type=Path, help="run_* without use-weight-versioning")
    p.add_argument("versioned_run", type=Path, help="run_* with use-weight-versioning")
    args = p.parse_args()

    base_logs = _read_run(args.baseline_run)
    ver_logs = _read_run(args.versioned_run)
    base_train = _training_sec(base_logs)
    ver_train = _training_sec(ver_logs)
    rt = summarize_runtime_metrics_from_logs(ver_logs, "")
    n1 = (rt.get("nodes") or {}).get(1) or {}
    ver_sec = float(n1.get("versioning_sec_total", 0) or 0)

    print(f"Baseline run:  {args.baseline_run}")
    print(f"Versioned run: {args.versioned_run}")
    if base_train:
        print(f"  Baseline training (s):   {base_train:.2f}")
    if ver_train:
        print(f"  Versioned training (s):  {ver_train:.2f}")
    if base_train and ver_train:
        delta = ver_train - base_train
        print(f"  Training delta (s):      {delta:+.2f} ({100.0 * delta / base_train:+.2f}%)")
    if ver_sec > 0:
        print(f"  versioning_sec_total:  {ver_sec:.2f}")
    if base_train and ver_sec > 0:
        print(f"  Versioning overhead %%:  {100.0 * ver_sec / base_train:.2f}% (vs baseline training)")
    if base_train and ver_train and ver_sec > 0:
        attrib = 100.0 * ver_sec / max(1e-9, ver_train - base_train) if ver_train > base_train else 0.0
        print(f"  (versioning as %% of train delta): {attrib:.2f}%")


if __name__ == "__main__":
    main()

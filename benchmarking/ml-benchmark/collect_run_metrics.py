#!/usr/bin/env python3
"""Parse a run_* log directory and print dissertation metrics table."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[2]
_NODE = _REPO / "sentra-node" / "python" / "node"
if str(_NODE) not in sys.path:
    sys.path.insert(0, str(_NODE))

from ml_training.benchmark_stats import (  # noqa: E402
    parse_benchmark_lines,
    parse_legacy_timings,
    summarize_runtime_metrics_from_logs,
)


def _read(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _training_sec(text: str) -> float | None:
    legacy = parse_legacy_timings(text)
    if "training_sec" in legacy:
        return float(legacy["training_sec"])
    for row in parse_benchmark_lines(text):
        if row.get("phase") == "training" and "wall_sec" in row:
            return float(row["wall_sec"])
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize SENTRA run_* logs for paper tables")
    p.add_argument("run_dir", type=str, help="Path to run_YYYYMMDD_HHMMSS directory")
    p.add_argument("--mnist-samples", type=int, default=10000)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--baseline-training-sec", type=float, default=0.0, help="Non-versioned training sec for versioning overhead %%")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    node_logs: Dict[int, str] = {}
    for path in sorted(run_dir.glob("node_*.log")):
        m = re.match(r"node_(\d+)\.log$", path.name)
        if m:
            node_logs[int(m.group(1))] = _read(path)
    client_log = _read(run_dir / "client_distributor.log")

    train_sec = None
    for text in node_logs.values():
        train_sec = _training_sec(text)
        if train_sec is not None:
            break

    batches_per_epoch = (int(args.mnist_samples) + int(args.batch_size) - 1) // int(args.batch_size)
    total_iters = batches_per_epoch * int(args.num_epochs)
    samples_total = int(args.mnist_samples) * int(args.num_epochs)

    print(f"Run directory: {run_dir}")
    acc = re.findall(r"Client Final Accuracy \(\d+ samples\):\s*([0-9.]+)%", client_log)
    if acc:
        print(f"  Client Final Accuracy: {acc[-1]}%")

    if train_sec:
        print(f"  Training time (s): {train_sec:.2f}")
        print(f"  Throughput (samples/s): {samples_total / train_sec:.4f}")
        if total_iters > 0:
            print(f"  Per-iteration latency (ms): {1000.0 * train_sec / total_iters:.2f}")
            print(f"  Mini-batch rate (batches/s): {total_iters / train_sec:.4f}")

    rt = summarize_runtime_metrics_from_logs(node_logs, client_log)
    nodes_rt = rt.get("nodes") or {}
    if nodes_rt:
        n1 = nodes_rt.get(1) or next(iter(nodes_rt.values()))
        print("\n  Runtime metrics (node 1 [BENCHMARK] phase=runtime_metrics):")
        for k in sorted(n1.keys()):
            if k not in ("role", "phase", "wall_sec"):
                print(f"    {k}: {n1[k]}")

    recovery: List[Dict[str, Any]] = rt.get("recovery") or []
    if recovery:
        print(f"\n  Recovery events: {len(recovery)}")
        for r in recovery:
            print(
                f"    node={r.get('node')} kind={r.get('phase')} "
                f"wall_sec={r.get('wall_sec')} n_active={r.get('n_active')} success={r.get('success')}"
            )

    loss_lines = []
    for nid, text in node_logs.items():
        for row in parse_benchmark_lines(text):
            if str(row.get("phase", "")).startswith("loss_snapshot"):
                loss_lines.append((nid, row))
    if loss_lines:
        print("\n  Loss snapshots around failure:")
        for nid, row in loss_lines:
            print(f"    node{nid}: {row}")
        with_loss = [
            (int(r.get("epoch", -1)), float(r["loss"]))
            for _, r in loss_lines
            if r.get("loss") not in (None, "")
        ]
        if len(with_loss) >= 2:
            before, after = with_loss[0][1], with_loss[-1][1]
            print(f"  Loss after failure (first→last snapshot): {before:.6f} → {after:.6f} (Δ {after - before:+.6f})")

    n_active_vals = sorted(
        {
            int(r.get("n_active", 0))
            for _, r in loss_lines
            if r.get("n_active") not in (None, "")
        }
    )
    if len(n_active_vals) >= 2:
        print(f"  Dropout degradation signal: n_active {n_active_vals[0]} → {n_active_vals[-1]}")

    if args.baseline_training_sec > 0 and nodes_rt:
        n1 = nodes_rt.get(1) or {}
        ver = float(n1.get("versioning_sec_total", 0) or 0)
        if ver > 0:
            pct = 100.0 * ver / float(args.baseline_training_sec)
            print(f"\n  Versioning overhead vs baseline training: {pct:.2f}% ({ver:.2f}s)")


if __name__ == "__main__":
    main()

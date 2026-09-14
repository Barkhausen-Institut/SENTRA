import os
import re
import sys
import time
import random
import subprocess
from pathlib import Path

import pytest
from testing.integration_harness import integration_timeout_seconds, wait_all_processes


@pytest.mark.integration
@pytest.mark.slow
def test_batched_mnist_does_not_logit_explode():
    """
    Regression test for batched secure MNIST:
    - launch 3 local nodes
    - run 2 epochs on a small subset
    - assert diagnostics remain bounded and no explosion warning appears

    Uses owner-only MNIST load to reduce Keras dataset lock contention (WSL /mnt/c).
    """
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "testing" / "tmp_bench"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Avoid clashing with other local runs.
    base_port = 10000 + random.randint(0, 1500)
    common_args = [
        "-u",
        "run_mnist_batched_secure.py",
        "--n-nodes",
        "3",
        "--t",
        "1",
        "--enable-network",
        "--distribute-dataset-shares",
        "--dataset-owner-node",
        "1",
        "--base-port",
        str(base_port),
        "--host",
        "localhost",
        "--batch-size",
        "64",
        "--num-epochs",
        "2",
        "--learning-rate",
        "0.05",
        "--mnist-samples",
        "128",
        "--loss-mode",
        "softmax",
        "--seed",
        "2026",
    ]

    procs = []
    try:
        for node_id in (1, 2, 3):
            out_path = log_dir / f"node{node_id}_stability.log"
            err_path = log_dir / f"node{node_id}_stability.err"
            with open(out_path, "w", encoding="utf-8") as out_f, open(err_path, "w", encoding="utf-8") as err_f:
                p = subprocess.Popen(
                    [sys.executable, *common_args, "--node-id", str(node_id)],
                    cwd=str(root),
                    stdout=out_f,
                    stderr=err_f,
                )
                procs.append(p)
            time.sleep(0.4)

        wait_all_processes(procs, timeout_sec=integration_timeout_seconds())

        for p in procs:
            assert p.returncode == 0, f"Node process failed with code {p.returncode}"

        node1_log = (log_dir / "node1_stability.log").read_text(encoding="utf-8", errors="ignore")
        node1_err = (log_dir / "node1_stability.err").read_text(encoding="utf-8", errors="ignore")

        assert "[WARN] Logit explosion detected" not in node1_log
        assert "Traceback" not in node1_log
        assert "Traceback" not in node1_err

        diag_matches = re.findall(
            r"Epoch\s+\d+\s+Diagnostics:\s+mean\|logit\|=([0-9eE+\-.]+),\s+max\|logit\|=([0-9eE+\-.]+)",
            node1_log,
        )
        assert diag_matches, "Missing diagnostics output in node 1 log"

        max_seen = max(float(max_v) for _, max_v in diag_matches)
        assert max_seen < 1.0e5, f"max|logit| too high: {max_seen}"

    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass

import random
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from testing.integration_harness import integration_timeout_seconds, wait_all_processes


@pytest.mark.integration
@pytest.mark.slow
def test_batched_mnist_accuracy_floor():
    """
    Regression test for batched secure MNIST learning quality.
    Runs a fixed 3-node setup and asserts final reported accuracy
    on node 1 is above a conservative floor.

    Uses owner-only MNIST load (--distribute-dataset-shares) to avoid three-way
    contention on Keras cache under WSL or slow disks.
    """
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "testing" / "tmp_bench"
    log_dir.mkdir(parents=True, exist_ok=True)

    base_port = 11500 + random.randint(0, 1200)
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
            out_path = log_dir / f"node{node_id}_accfloor.log"
            err_path = log_dir / f"node{node_id}_accfloor.err"
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

        node1_log = (log_dir / "node1_accfloor.log").read_text(encoding="utf-8", errors="ignore")
        node1_err = (log_dir / "node1_accfloor.err").read_text(encoding="utf-8", errors="ignore")

        assert "Traceback" not in node1_log
        assert "Traceback" not in node1_err

        matches = re.findall(r"Epoch\s+(\d+)\s+Test Accuracy:\s+([0-9eE+\-.]+)%", node1_log)
        assert matches, "No epoch accuracy lines found in node 1 log"
        last_epoch, last_acc_pct = matches[-1]
        last_acc = float(last_acc_pct) / 100.0

        # Conservative floor chosen from observed stable runs with this exact seed/config.
        assert last_acc >= 0.12, f"Final accuracy too low at epoch {last_epoch}: {last_acc:.4f}"

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

import random
import re
from pathlib import Path

import pytest
from testing.integration_harness import (
    integration_timeout_seconds,
    read_node_err,
    read_node_log,
    start_node_processes,
    terminate_processes,
    wait_all_processes,
)


@pytest.mark.integration
@pytest.mark.slow
def test_batched_stage_invariants():
    """
    Runs a tiny 3-node secure-batched MNIST job and checks stage probes to
    localize numerical instability:
    - logits probe should stay bounded
    - dz2 probe should be O(1) for CE softmax gradients
    - probs_est probe should be finite, near simplex

    Owner-only MNIST load avoids triple Keras cache contention (WSL /mnt/c).
    """
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "testing" / "tmp_bench"
    log_dir.mkdir(parents=True, exist_ok=True)

    base_port = 14000 + random.randint(0, 1000)
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
        "1",
        "--learning-rate",
        "0.0002",
        "--mnist-samples",
        "128",
        "--loss-mode",
        "softmax",
        "--seed",
        "2026",
        "--field-size",
        str(2**61 - 1),
        "--scale-factor",
        "65536",
        "--softmax-temperature",
        "2",
        "--exp-approx",
        "pade22",
        "--softmax-grad-mode",
        "opened_exact",
        "--grad-clip",
        "0.20",
        "--logit-clip",
        "3.0",
        "--no-abort-on-instability",
        "--debug-numerics",
        "--grad-norm-threshold",
        "1500",
        "--explode-logit-threshold",
        "1000000000",
    ]

    procs = []
    try:
        proc_entries = start_node_processes(
            root=root,
            log_dir=log_dir,
            common_args=common_args,
            node_ids=(1, 2, 3),
            log_prefix="stageinv",
            startup_stagger_s=0.4,
        )
        procs = [p for _, p in proc_entries]

        wait_all_processes(procs, timeout_sec=integration_timeout_seconds())
        for p in procs:
            assert p.returncode == 0, f"Node process failed with code {p.returncode}"

        node1_log = read_node_log(log_dir, node_id=1, log_prefix="stageinv")
        node1_err = read_node_err(log_dir, node_id=1, log_prefix="stageinv")
        assert "Traceback" not in node1_log
        assert "Traceback" not in node1_err

        m_logits = re.search(r"Epoch 1 Probe logits\[0\]\[:10\]: \[(.*?)\]", node1_log, re.DOTALL)
        m_dz2 = re.search(r"Epoch 1 Probe dz2\[0\]\[:10\]: \[(.*?)\]", node1_log, re.DOTALL)
        m_probs = re.search(
            r"Epoch 1 Probe probs_est stats: sum=([0-9eE+\-.]+), min=([0-9eE+\-.]+), max=([0-9eE+\-.]+), target_sum=([0-9eE+\-.]+)",
            node1_log,
        )
        assert m_logits and m_dz2 and m_probs, "Missing stage probes in node 1 log"

        logits_vals = [float(v.strip()) for v in m_logits.group(1).split(",")]
        dz2_vals = [float(v.strip()) for v in m_dz2.group(1).split(",")]
        probs_sum = float(m_probs.group(1))
        probs_min = float(m_probs.group(2))
        probs_max = float(m_probs.group(3))

        max_abs_logit_probe = max(abs(v) for v in logits_vals)
        max_abs_dz2_probe = max(abs(v) for v in dz2_vals)

        # Invariant expectations for stable CE/softmax region.
        assert max_abs_logit_probe < 100.0, f"logit probe exploded: {max_abs_logit_probe:.6f}"
        assert max_abs_dz2_probe < 5.0, f"dz2 probe exploded: {max_abs_dz2_probe:.6f}"
        assert 0.95 <= probs_sum <= 1.05, f"probs_est sum drifted: {probs_sum:.6f}"
        assert probs_min >= -0.05, f"probs_est min invalid: {probs_min:.6f}"
        assert probs_max <= 1.05, f"probs_est max invalid: {probs_max:.6f}"

    finally:
        terminate_processes([(i + 1, p) for i, p in enumerate(procs)], terminate_timeout_s=2.0)


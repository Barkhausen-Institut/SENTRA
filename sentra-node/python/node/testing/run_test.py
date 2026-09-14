import os
import subprocess
import sys
import time
from pathlib import Path

_NODE_ROOT = Path(__file__).resolve().parent.parent

nodes = 3
procs = []

for i in range(1, nodes + 1):
    cmd = [
        sys.executable,
        "-u",
        str(_NODE_ROOT / "run_mnist_batched_secure.py"),
        "--node-id", str(i),
        "--n-nodes", str(nodes),
        "--base-port", "9300",
        "--host", "localhost",
        "--batch-size", "64",
        "--num-epochs", "1",
        "--learning-rate", "0.05",
        "--t", "1",
        "--mnist-samples", "64",
        "--enable-network"
    ]
    
    with open(f"node_{i}_output.log", "w") as f_out, open(f"node_{i}_error.log", "w") as f_err:
        p = subprocess.Popen(cmd, cwd=str(_NODE_ROOT), stdout=f_out, stderr=f_err)
        procs.append(p)
    
print("Nodes started. Waiting for completion...")
for p in procs:
    p.wait()
print("All nodes completed.")

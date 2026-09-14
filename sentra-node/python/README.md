# SENTRA

Secure, distributed ML training over secret shares with MPC-style coordination.

This repository contains a practical SENTRA training stack for:
- Multi-node secure training
- Secret-shared matrix operations
- Secure/nonlinear approximations for classification (MNIST MLP)
- Reproducible experiments in both protocol-compliant and comparison modes

## Highlights

- Multi-node orchestration via `start_all_nodes.py`
- Batched secure MNIST path via `run_mnist_batched_secure.py`
- Tunable fixed-point configuration (`field-size`, `scale-factor`)
- Stability instrumentation (`max|logit|`, gradient norm, update stats)
- Two softmax-gradient modes:
  - `secure_approx`: protocol-compliant SENTRA path
  - `opened_exact`: accelerated comparison variant (not vanilla SENTRA)

## Repository Structure

- `ml_training/` — core secure training, communication, secret-sharing, MPC ops
- `start_all_nodes.py` — launcher for multi-node runs
- `run_mnist_batched_secure.py` — batched secure MNIST runner (default)
- `start_all_nodes.py` + `run_mnist_batched_secure.py` — canonical multi-node training path
- `client_distributor.py` — client-side dataset owner; loads MNIST, shares data to nodes
- `testing/` — regression/integration tests and smoke checks
- `local_adapters/` — in-memory adapters for local testing (`SentraTrainingNode`)
- `attestation/` — enclave attestation-related assets

## Architecture Diagram

```text
                           +-----------------------+
                           |   start_all_nodes.py  |
                           |  (process launcher)   |
                           +-----------+-----------+
                                       |
                -------------------------------------------------
                |                       |                       |
        +-------v--------+      +-------v--------+      +-------v--------+
        |    Node 1      |      |    Node 2      |      |    Node 3      |
        | batched runner |      | batched runner |      | batched runner |
        +-------+--------+      +-------+--------+      +-------+--------+
                |                       |                       |
                |<------ secure comm + sync/barriers --------->|
                |         (ml_training/secure_comm.py)         |
                |                       |                       |
                +-----------+-----------+-----------+-----------+
                            |                       |
                    +-------v--------+      +-------v--------+
                    | Secret Sharing |      | Secure MPC Ops |
                    | + Versioned KV |      | Beaver / ReLU  |
                    | (data, weights)|      | Softmax / Div  |
                    +-------+--------+      +-------+--------+
                            |                       |
                            +-----------+-----------+
                                        |
                               +--------v---------+
                               |  Training Loop   |
                               | fwd/bwd/update   |
                               +--------+---------+
                                        |
                               +--------v---------+
                               |  Eval + Metrics  |
                               | accuracy / loss  |
                               +------------------+
```

Data/weight path summary:
- Dataset and model parameters are represented as secret shares.
- Nodes coordinate each batch through network synchronization and MPC primitives.
- Output-layer gradient mode is selectable (`secure_approx` vs `opened_exact`).
- Epoch metrics are reconstructed only for reporting/evaluation.

## Prerequisites

- Python 3.10+
- Windows/WSL/Linux environment with multiple local processes allowed

Install dependencies:

```bash
pip install -r requirements.txt
```

Core packages: `tensorflow`, `numpy`, `pytest`. Optional: `openpyxl` (Excel export), `psutil` (memory stats in headless runs).

## Quick Start

### 1) SENTRA-Compliant Baseline (Recommended)

This mode keeps softmax gradient in the secure approximation path.

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9600 \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000 \
  --learning-rate 0.002 --loss-mode softmax \
  --field-size 2305843009213693951 \
  --scale-factor 65536 --softmax-temperature 2 \
  --exp-approx pade22 --softmax-grad-mode secure_approx \
  --grad-clip 0.50 --logit-clip 4.0 \
  --explode-logit-threshold 100 --loss-growth-threshold 5 \
  --grad-norm-threshold 1500 --no-abort-on-instability \
  --debug-numerics --seed 2026
```

### 2) Accelerated Comparison Variant

This mode uses opened exact output-layer softmax gradient, then re-shares it.
Use for benchmarking/ablation, not as vanilla SENTRA protocol claim.

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9700 \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000 \
  --learning-rate 0.002 --loss-mode softmax \
  --field-size 2305843009213693951 \
  --scale-factor 65536 --softmax-temperature 2 \
  --exp-approx pade22 --softmax-grad-mode opened_exact \
  --grad-clip 0.50 --logit-clip 4.0 \
  --explode-logit-threshold 100 --loss-growth-threshold 5 \
  --grad-norm-threshold 1500 --no-abort-on-instability \
  --debug-numerics --seed 2026
```

### 3) Plaintext Baseline Runner (Single Process)

Use this to benchmark the same `784→128→10` MLP without MPC/fixed-point overhead.
It accepts a superset of the batched secure flags, ignoring MPC-only options.

```bash
python3 run_mnist_plaintext.py \
  --num-epochs 1 --batch-size 32 --mnist-samples 200 \
  --learning-rate 0.01 --loss-mode softmax \
  --client-eval-samples 100
```

## Mode Semantics

| Mode | Security posture | Typical accuracy/runtime |
|---|---|---|
| `secure_approx` | Protocol-compliant SENTRA nonlinear path | More conservative, protocol-faithful |
| `opened_exact` | Opens output logits/targets for exact softmax gradient at opener, then re-shares | Better convergence/accuracy; comparison-only variant |

## Stability and Debugging

When `--debug-numerics` is enabled, node logs include:
- `Epoch X Diagnostics`: mean/max logit magnitude and entropy
- `Estimated Grad Norm`
- `Update Sample Stats`
- Probe vectors (`logits`, `dz2`, reconstructed probability estimate)

Healthy signs:
- `probs_est stats sum` near `1.0`
- `probs_est min >= 0`
- no rapid growth in `max|logit|`
- decreasing test loss over epochs

## Testing

Run core fast tests:

```bash
python -m pytest -q testing/test_forward_scaling.py testing/test_gradient_scaling.py
```

Run stage-invariant integration check:

```bash
python -m pytest -q testing/test_batched_stage_invariants.py -s
```

Full suite:

```bash
python3 -m pytest -q
```

## Export Runs to Excel

You can auto-record run parameters and parsed final metrics to an Excel file.
This uses headless mode so all node logs are captured under `logs/run_<timestamp>/`.

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9600 \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000 \
  --learning-rate 0.002 --loss-mode softmax \
  --field-size 2305843009213693951 --scale-factor 65536 \
  --softmax-temperature 2 --exp-approx pade22 \
  --softmax-grad-mode secure_approx \
  --record-results-xlsx logs/sentra_runs.xlsx
```

## Common Issues

### Ports already in use

Stop existing processes before restarting with a fresh `--base-port`:

- Linux/macOS: `pkill -f "start_all_nodes.py|run_mnist_batched_secure.py" || true`
- Windows: use Task Manager or `taskkill /F /IM python.exe` (closes all Python processes)

### Broken pipe / reset by peer

Usually one node exited early. Check per-node logs and ensure all nodes use identical flags.

### Slow integration tests on WSL (/mnt/c)

On WSL, especially when running from `/mnt/c/...`, multi-node integration tests can be slow.
You can raise the wall-clock budget with:

```bash
export SENTRA_INTEGRATION_TIMEOUT_SEC=1200
```

If possible, run from a Linux-native filesystem (e.g. `~/sentra2`) instead of `/mnt/c`.

### Packing safety bound (strict)

Packed MPC enforces the strict bound \(2(t + s - 1) < n_{active}\).
In particular, equality is **unsafe**.
The runner will cap `s` (packing factor) to a safe value automatically.

### Client-side sharing (recommended)

For real deployments, do not load raw MNIST on any training node.

1) Start nodes in receive-only mode:

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9600 --headless \
  --receive-dataset-shares-from-client --dataset-source-node-id 0 \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000
```

2) Start client distributor (input owner):

```bash
python3 client_distributor.py --client-node-id 0 --n-nodes 3 --base-port 9600 \
  --mnist-samples 10000
```

Or in one command (headless), auto-start the client:

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9600 --headless \
  --receive-dataset-shares-from-client --dataset-source-node-id 0 \
  --start-client-distributor --client-eval-after-training --client-eval-samples 100 \
  --client-test-samples 100 \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000
```

# SENTRA

Secure, distributed ML training over secret shares with MPC-style coordination.

This repository contains a practical SENTRA training stack for:
- Multi-node secure training
- Secret-shared matrix operations
- Secure/nonlinear approximations for classification (MNIST MLP)
- Reproducible experiments in both protocol-compliant and comparison modes

## Highlights

- Multi-node orchestration via `start_all_nodes.py`
- Batched secure MNIST path via `run_mnist_batched_secure.py`
- Tunable fixed-point configuration (`field-size`, `scale-factor`)
- Stability instrumentation (`max|logit|`, gradient norm, update stats)
- Two softmax-gradient modes:
  - `secure_approx`: protocol-compliant SENTRA path
  - `opened_exact`: accelerated comparison variant (not vanilla SENTRA)

## Repository Structure

- `ml_training/` — core secure training, communication, secret-sharing, MPC ops
- `start_all_nodes.py` — launcher for multi-node runs
- `run_mnist_batched_secure.py` — batched secure MNIST runner (default)
- `start_all_nodes.py` + `run_mnist_batched_secure.py` — canonical multi-node training path
- `client_distributor.py` — client-side dataset owner; loads MNIST, shares data to nodes
- `testing/` — regression/integration tests and smoke checks
- `local_adapters/` — in-memory adapters for local testing (`SentraTrainingNode`)
- `attestation/` — enclave attestation-related assets

## Architecture Diagram

```text
                           +-----------------------+
                           |   start_all_nodes.py  |
                           |  (process launcher)   |
                           +-----------+-----------+
                                       |
                -------------------------------------------------
                |                       |                       |
        +-------v--------+      +-------v--------+      +-------v--------+
        |    Node 1      |      |    Node 2      |      |    Node 3      |
        | batched runner |      | batched runner |      | batched runner |
        +-------+--------+      +-------+--------+      +-------+--------+
                |                       |                       |
                |<------ secure comm + sync/barriers --------->|
                |         (ml_training/secure_comm.py)         |
                |                       |                       |
                +-----------+-----------+-----------+-----------+
                            |                       |
                    +-------v--------+      +-------v--------+
                    | Secret Sharing |      | Secure MPC Ops |
                    | + Versioned KV |      | Beaver / ReLU  |
                    | (data, weights)|      | Softmax / Div  |
                    +-------+--------+      +-------+--------+
                            |                       |
                            +-----------+-----------+
                                        |
                               +--------v---------+
                               |  Training Loop   |
                               | fwd/bwd/update   |
                               +--------+---------+
                                        |
                               +--------v---------+
                               |  Eval + Metrics  |
                               | accuracy / loss  |
                               +------------------+
```

Data/weight path summary:
- Dataset and model parameters are represented as secret shares.
- Nodes coordinate each batch through network synchronization and MPC primitives.
- Output-layer gradient mode is selectable (`secure_approx` vs `opened_exact`).
- Epoch metrics are reconstructed only for reporting/evaluation.

## Prerequisites

- Python 3.10+
- Windows/WSL/Linux environment with multiple local processes allowed

Install dependencies:

```bash
pip install -r requirements.txt
```

Core packages: `tensorflow`, `numpy`, `pytest`. Optional: `openpyxl` (Excel export), `psutil` (memory stats in headless runs).

## Quick Start

### 1) SENTRA-Compliant Baseline (Recommended)

This mode keeps softmax gradient in the secure approximation path.

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9600 \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000 \
  --learning-rate 0.002 --loss-mode softmax \
  --field-size 2305843009213693951 \
  --scale-factor 65536 --softmax-temperature 2 \
  --exp-approx pade22 --softmax-grad-mode secure_approx \
  --grad-clip 0.50 --logit-clip 4.0 \
  --explode-logit-threshold 100 --loss-growth-threshold 5 \
  --grad-norm-threshold 1500 --no-abort-on-instability \
  --debug-numerics --seed 2026
```

### 2) Accelerated Comparison Variant

This mode uses opened exact output-layer softmax gradient, then re-shares it.
Use for benchmarking/ablation, not as vanilla SENTRA protocol claim.

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9700 \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000 \
  --learning-rate 0.002 --loss-mode softmax \
  --field-size 2305843009213693951 \
  --scale-factor 65536 --softmax-temperature 2 \
  --exp-approx pade22 --softmax-grad-mode opened_exact \
  --grad-clip 0.50 --logit-clip 4.0 \
  --explode-logit-threshold 100 --loss-growth-threshold 5 \
  --grad-norm-threshold 1500 --no-abort-on-instability \
  --debug-numerics --seed 2026
```

### 3) Plaintext Baseline Runner (Single Process)

Use this to benchmark the same `784→128→10` MLP without MPC/fixed-point overhead.
It accepts a superset of the batched secure flags, ignoring MPC-only options.

```bash
python3 run_mnist_plaintext.py \
  --num-epochs 1 --batch-size 32 --mnist-samples 200 \
  --learning-rate 0.01 --loss-mode softmax \
  --client-eval-samples 100
```

## Mode Semantics

| Mode | Security posture | Typical accuracy/runtime |
|---|---|---|
| `secure_approx` | Protocol-compliant SENTRA nonlinear path | More conservative, protocol-faithful |
| `opened_exact` | Opens output logits/targets for exact softmax gradient at opener, then re-shares | Better convergence/accuracy; comparison-only variant |

## Stability and Debugging

When `--debug-numerics` is enabled, node logs include:
- `Epoch X Diagnostics`: mean/max logit magnitude and entropy
- `Estimated Grad Norm`
- `Update Sample Stats`
- Probe vectors (`logits`, `dz2`, reconstructed probability estimate)

Healthy signs:
- `probs_est stats sum` near `1.0`
- `probs_est min >= 0`
- no rapid growth in `max|logit|`
- decreasing test loss over epochs

## Testing

Run core fast tests:

```bash
python -m pytest -q testing/test_forward_scaling.py testing/test_gradient_scaling.py
```

Run stage-invariant integration check:

```bash
python -m pytest -q testing/test_batched_stage_invariants.py -s
```

Or use the test runner: `python testing/run_tests.py --type fast`

Full suite (non-interactive by default, including auto-started stub nodes for failure detection):

```bash
python3 -m pytest -q
```

Notes on default test toggles:
- `SENTRA_RUN_INTERACTIVE=0` skips node failure detection.
- `SENTRA_RUN_SENTRA_NODE_LOGIC=0` skips SentraTrainingNode logic tests.
- `SENTRA_RUN_SOFTMAX_QUICK=0` skips the MPC softmax quick tests (these now spin up a 3-node harness).
- `SENTRA_STRICT_BASELINE=0` skips the secure-vs-plaintext tolerance check.

## Export Runs to Excel

You can auto-record run parameters and parsed final metrics to an Excel file.
This uses headless mode so all node logs are captured under `logs/run_<timestamp>/`.

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9600 \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000 \
  --learning-rate 0.002 --loss-mode softmax \
  --field-size 2305843009213693951 --scale-factor 65536 \
  --softmax-temperature 2 --exp-approx pade22 \
  --softmax-grad-mode secure_approx \
  --record-results-xlsx logs/sentra_runs.xlsx
```

Notes:
- `--record-results-xlsx` implies `--headless`.
- Requires `openpyxl` (included in `requirements.txt`).
- One row is appended per run (command, params, status, final accuracy/loss, log directory).
- Timing fields now include detailed metrics when available: `prover_time_sec`, `dataset_share_prep_time_sec`, `training_time_sec`, `client_distribution_time_sec`, `client_eval_time_sec`, `client_eval_upload_time_sec`.

## Evaluate Exported Model

If you export a reconstructed model (`.npz`), you can run plaintext inference on MNIST:

```bash
python3 testing/eval_exported_model.py --model-npz logs/final_model_run_20260306.npz --split test --samples 1000
```

## Benchmark Snapshot

Recent validated local results (3 nodes, MNIST batched secure path):

- **SENTRA-compliant mode** (`--softmax-grad-mode secure_approx`):
  - stable learning observed
  - representative run reached ~`79%` test accuracy by epoch 7 (100-sample eval)
- **Accelerated comparison mode** (`--softmax-grad-mode opened_exact`):
  - higher/faster convergence expected
  - representative run reached ~`80%` test accuracy by epoch 8 (100-sample eval)

Reference command family:
- `field-size`: `2305843009213693951` (`2^61 - 1`)
- `scale-factor`: `65536`
- `exp-approx`: `pade22`
- `softmax-temperature`: `2`
- `batch-size`: `64`
- `mnist-samples`: `10000`

Note: 100-sample evaluation is useful for fast iteration. For reporting, use larger evaluation sets (e.g., 1000+ or full test set).

## Common Issues

### Ports already in use

Stop existing processes before restarting with a fresh `--base-port`:

- Linux/macOS: `pkill -f "start_all_nodes.py|run_mnist_batched_secure.py" || true`
- Windows: use Task Manager or `taskkill /F /IM python.exe` (closes all Python processes)

### Broken pipe / reset by peer

Usually one node exited early. Check per-node logs and ensure all nodes use identical flags.

### Slow integration tests on WSL (/mnt/c)

On WSL, especially when running from `/mnt/c/...`, multi-node integration tests can be slow.
You can raise the wall-clock budget with:

```bash
export SENTRA_INTEGRATION_TIMEOUT_SEC=1200
```

If possible, run from a Linux-native filesystem (e.g. `~/sentra2`) instead of `/mnt/c`.

### Packing safety bound (strict)

Packed MPC enforces the strict bound \(2(t + s - 1) < n_{active}\).
In particular, equality is **unsafe**. Example: with `t=1`, `n_active=4`, `s=2` is invalid because \(2(1+2-1)=4\not<4\).
The runner will cap `s` (packing factor) to a safe value automatically.

### Client-side sharing (recommended)

For real deployments, do not load raw MNIST on any training node.

1) Start nodes in receive-only mode:

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9600 --headless \
  --receive-dataset-shares-from-client --dataset-source-node-id 0 \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000
```

2) Start client distributor (input owner):

```bash
python3 client_distributor.py --client-node-id 0 --n-nodes 3 --base-port 9600 \
  --mnist-samples 10000
```

Or in one command (headless), auto-start the client:

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9600 --headless \
  --receive-dataset-shares-from-client --dataset-source-node-id 0 \
  --start-client-distributor --client-eval-after-training --client-eval-samples 100 \
  --client-test-samples 100 \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000
```

### Full Recommended Command (Client-Side Sharing + Excel Recording)

Production-style run with client-side dataset sharing, post-training client evaluation, and Excel recording. Raw MNIST is loaded only by the client distributor; training nodes receive only secret shares.

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9600 \
  --num-epochs 4 --batch-size 64 --mnist-samples 10000 \
  --learning-rate 0.003 --loss-mode softmax \
  --field-size 2305843009213693951 \
  --scale-factor 65536 --softmax-temperature 2 \
  --exp-approx pade22 --softmax-grad-mode secure_approx \
  --grad-clip 0.50 --logit-clip 4.0 \
  --explode-logit-threshold 100 --loss-growth-threshold 5 \
  --grad-norm-threshold 1500 --no-abort-on-instability \
  --debug-numerics --seed 2026 \
  --receive-dataset-shares-from-client --dataset-source-node-id 0 \
  --start-client-distributor --client-eval-after-training \
  --client-eval-samples 100 --client-test-samples 100 \
  --record-results-xlsx logs/sentra_runs.xlsx
```

This command implies `--headless`; logs go to `logs/run_<timestamp>/`.

#### Parameter Reference

The launcher always invokes the batched secure MNIST runner (`run_mnist_batched_secure.py`). The `--batched` flag is deprecated and has no effect.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--n-nodes` | 3 | Number of training nodes |
| `--base-port` | 9600 | Base port (node i uses `base-port + i`) |
| `--dataset` | mnist | Dataset mode (MNIST only; default `mnist`) |
| `--num-epochs` | 4 | Training epochs |
| `--batch-size` | 64 | Mini-batch size |
| `--mnist-samples` | 10000 | MNIST training samples per node |
| `--learning-rate` | 0.003 | SGD learning rate |
| `--loss-mode` | softmax | Output gradient mode |
| `--field-size` | 2305843009213693951 | Finite field modulus (`2^61 - 1`) |
| `--scale-factor` | 65536 | Fixed-point scaling for MPC |
| `--softmax-temperature` | 2 | Softmax temperature (integer in fixed-point) |
| `--exp-approx` | pade22 | Secure exp approximation (Pade 2,2) |
| `--softmax-grad-mode` | secure_approx | SENTRA-compliant softmax gradient path |
| `--grad-clip` | 0.50 | Gradient clipping bound |
| `--logit-clip` | 4.0 | Logit clipping before exp approx |
| `--explode-logit-threshold` | 100 | Trigger for logit explosion checks |
| `--loss-growth-threshold` | 5 | Loss growth threshold for instability |
| `--grad-norm-threshold` | 1500 | Gradient norm threshold for instability |
| `--no-abort-on-instability` | flag | Continue training despite instability checks |
| `--debug-numerics` | flag | Log probes (logits, gradients, probs) |
| `--seed` | 2026 | Global RNG seed |
| `--receive-dataset-shares-from-client` | flag | Nodes receive shares from client; no raw data |
| `--dataset-source-node-id` | 0 | Client distributor node id |
| `--start-client-distributor` | flag | Auto-start `client_distributor.py` in headless mode |
| `--client-eval-after-training` | flag | Client reconstructs final accuracy from shares |
| `--client-eval-samples` | 100 | Samples for client-side accuracy evaluation |
| `--client-test-samples` | 100 | Test shares client sends to nodes |
| `--record-results-xlsx` | logs/sentra_runs.xlsx | Append run metrics to Excel (implies headless) |

### Keep raw MNIST on one node only (simulation only)

Use distributed dataset-share mode so non-owner nodes never load raw MNIST:

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9600 \
  --enable-network --distribute-dataset-shares --dataset-owner-node 1 \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000
```

In this mode, owner node `--dataset-owner-node` loads MNIST, creates Shamir shares, and sends each node only its local shares.

### Non-integer temperature error

In fixed-point mode, temperature is currently integer-only (`1`, `2`, ...).

## Reproducibility Checklist

For published runs, always record:
- full command line
- `seed`
- `field-size` and `scale-factor`
- `softmax-grad-mode`
- epoch-wise accuracy/loss/diagnostics

## Notes

- This README reflects the current batched MNIST secure path and tested commands.
- `start_all_nodes.py` always launches the batched secure runner; `--batched` is optional and deprecated.
- See `USAGE_GUIDE.md` for a concise runbook (quick start, log interpretation, common problems).

# SENTRA Multi-Node Guide

## Scope

SENTRA is research and benchmarking software for secure, distributed MNIST training. The current local multi-node path launches several Python processes on one Linux or WSL host. It is not a production deployment guide.

The canonical entrypoints are:

- `sentra-node/python/node/start_all_nodes.py` — starts and supervises all local node processes.
- `sentra-node/python/node/run_mnist_batched_secure.py` — per-node runner invoked by the launcher.

See `sentra-node/python/README.md` for the complete, current CLI reference.

## Install

Run this command from the repository root:

```powershell
python -m pip install -r sentra-node/python/node/requirements.txt
```

Local multi-process runs are supported primarily on Linux or WSL. Native Windows has not been established as an equivalent benchmark environment.

## Recommended local run

Run this command from the repository root:

```powershell
python sentra-node/python/node/start_all_nodes.py `
  --n-nodes 3 --base-port 9600 --headless `
  --receive-dataset-shares-from-client --start-client-distributor `
  --client-eval-after-training --client-eval-samples 100 --client-test-samples 100 `
  --num-epochs 1 --batch-size 64 --mnist-samples 128 `
  --learning-rate 0.002 --loss-mode softmax `
  --field-size 2305843009213693951 --scale-factor 65536 `
  --softmax-temperature 2 --exp-approx pade22 `
  --softmax-grad-mode secure_approx --seed 2026
```

On bash, replace PowerShell backticks with backslashes. `--headless` keeps orchestration in one terminal and writes per-node logs below `sentra-node/python/node/logs/`.

This client-owned-data mode sends secret shares to the training nodes. For a simpler local simulation in which node 1 loads MNIST and distributes shares, use:

```powershell
python sentra-node/python/node/start_all_nodes.py --n-nodes 3 --base-port 9600 --headless --distribute-dataset-shares --dataset-owner-node 1 --num-epochs 1 --batch-size 8 --mnist-samples 128
```

The owner-node mode is a simulation convenience; it does not model an independent data owner.

## Data

Set `MNIST_NPZ_PATH` to an MNIST `.npz` file or place the file where `sentra-node/python/node/ml_training/util.py` expects it. The Docker benchmark images use `/mnist.npz`.

## Node and threshold settings

- Each node receives a unique port: node `i` uses `base-port + i`.
- All processes must use the same node count, threshold, field, scale, and training arguments.
- The configured safety condition is `2 * (t + s) < n_active`; changing committee or adversary parameters requires checking the relevant protocol assumptions.
- The launcher currently targets local multi-process orchestration. Do not infer a supported cross-machine deployment merely by changing `--host`.

For direct debugging only, invoke `sentra-node/python/node/run_mnist_batched_secure.py` once per node with matching arguments and unique `--node-id` values. Prefer the launcher because it constructs consistent commands and manages logs and process lifetime.

## Verification

The pytest configuration is `sentra-node/python/pytest.ini`. Run tests from `sentra-node/python` so its configured import paths and test roots apply:

```powershell
Set-Location sentra-node/python
python -m pytest -q node/testing/test_start_all_nodes_cli.py
python -m pytest -q node/testing/test_softmax_quick.py -s
python -m pytest -q node/testing/test_batched_stage_invariants.py -s
```

The softmax and stage-invariant tests exercise real local multi-process/network paths. Some integration tests are slower and may require available ports and MNIST data.

## Troubleshooting

- **Port in use:** choose another `--base-port` and ensure no previous node processes remain.
- **Connection reset or broken pipe:** inspect every per-node log; one process usually exited before its peers.
- **Missing MNIST:** set `MNIST_NPZ_PATH` explicitly.
- **Numerical instability:** rerun with `--debug-numerics` and inspect epoch diagnostics. Do not treat a completed process as evidence that accuracy or security goals were met.

## Deployment limitations

SGX is not enabled by the local Python command above. SGX/Occlum benchmarks require compatible Intel SGX hardware, drivers, Occlum, and the Docker workflow documented in `benchmarking/ml-benchmark/README.md`.

The manifests under `sentra-deployment/` are a separate backend-assisted path. They require access to a private backend image and should not be presented as a generally reproducible or production-ready deployment.

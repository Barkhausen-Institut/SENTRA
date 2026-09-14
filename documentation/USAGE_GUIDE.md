# SENTRA Usage Guide

SENTRA is research and benchmarking software for secure multi-party MNIST training. Use `README.md` for the repository overview, `sentra-node/python/README.md` for detailed launcher flags, and `benchmarking/README.md` for benchmark workflows.

## Install

From the repository root:

```powershell
python -m pip install -r sentra-node/python/node/requirements.txt
```

Linux or WSL is the supported environment for local multi-process runs. Docker is used for packaged non-SGX and SGX benchmarks.

## Local multi-node training

The canonical launcher is `sentra-node/python/node/start_all_nodes.py`. It starts one `run_mnist_batched_secure.py` process per node.

From the repository root, run the client-owned-data workflow:

```powershell
python sentra-node/python/node/start_all_nodes.py `
  --n-nodes 3 --base-port 9600 --headless `
  --receive-dataset-shares-from-client --start-client-distributor `
  --client-eval-after-training --client-eval-samples 100 --client-test-samples 100 `
  --num-epochs 8 --batch-size 64 --mnist-samples 10000 `
  --learning-rate 0.002 --loss-mode softmax `
  --field-size 2305843009213693951 --scale-factor 65536 `
  --softmax-temperature 2 --exp-approx pade22 `
  --softmax-grad-mode secure_approx `
  --grad-clip 0.50 --logit-clip 4.0 `
  --explode-logit-threshold 100 --loss-growth-threshold 5 `
  --grad-norm-threshold 1500 --no-abort-on-instability `
  --debug-numerics --seed 2026
```

Use backslashes instead of PowerShell backticks on bash.

For a short owner-node simulation:

```powershell
python sentra-node/python/node/start_all_nodes.py --n-nodes 3 --base-port 9600 --headless --distribute-dataset-shares --dataset-owner-node 1 --num-epochs 1 --batch-size 8 --mnist-samples 128
```

`--distribute-dataset-shares` means one training node initially loads the raw dataset. Use the client distributor workflow when evaluating separation between the data owner and compute nodes.

## Training modes

- `--softmax-grad-mode secure_approx` uses the fixed-point secure approximation path and is the mode to use for SENTRA protocol experiments.
- `--softmax-grad-mode opened_exact` is an accelerated comparison/ablation path. Label results accordingly; it is not equivalent to the secure approximation.

The launcher is always batched. The legacy `--batched` flag is accepted but deprecated and has no effect.

## Data and logs

Set `MNIST_NPZ_PATH` to the MNIST `.npz` location when it is not found automatically.

Headless runs write logs below:

```text
sentra-node/python/node/logs/run_<timestamp>/
```

With `--debug-numerics`, inspect epoch accuracy/loss, logit magnitude, probability estimates, entropy, and instability warnings. Completion alone does not establish useful accuracy or protocol security.

## Tests

The active pytest configuration is `sentra-node/python/pytest.ini`. Run tests from `sentra-node/python`:

```powershell
Set-Location sentra-node/python
python -m pytest -q node/testing/test_forward_scaling.py node/testing/test_gradient_scaling.py
python -m pytest -q node/testing/test_batched_stage_invariants.py -s
```

The configured suites are under `node/testing/` and `node/tests/`. Integration tests can require free ports, MNIST data, and substantially more time than unit tests.

## Common problems

- **Port already in use:** terminate stale node processes or choose a new `--base-port`.
- **Broken pipe/reset by peer:** inspect all per-node logs and find the first process to fail.
- **Non-integer softmax temperature:** the fixed-point path currently expects an integer-valued temperature such as `1` or `2`.
- **Missing MNIST:** set `MNIST_NPZ_PATH` explicitly.
- **Unstable metrics:** use `--debug-numerics`, reduce the run size while diagnosing, and record all numerical parameters.

## Benchmark and deployment boundaries

For reproducible dissertation benchmarks, use the profiles and instructions in `benchmarking/README.md` and `benchmarking/ml-benchmark/README.md`.

SGX is optional and is not used by local Python runs. SGX/Occlum execution requires compatible Intel SGX hardware, drivers, Occlum, and the dedicated Docker build/run targets.

The manifests in `sentra-deployment/` are a separate backend-assisted workflow that requires a private backend image. They are not a self-contained public deployment path. No workflow in this guide should be described as production ready without an independent security review, operational hardening, and validation in the target environment.

## Reproducibility

Record the full command, source revision, seed, dataset/profile, node count, threshold, field size, scale factor, softmax mode, hardware/runtime environment, and per-epoch metrics for every reported result.

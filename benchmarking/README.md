# SENTRA benchmarking

Reproducible secure MNIST training benchmarks for the dissertation: **SENTRA (non-SGX Docker)**, **SENTRA (SGX/Occlum)**, and **CrypTen** on the same model and data split.

## Layout

| Path | Purpose |
|------|---------|
| **`ml-benchmark/`** | Docker/Compose orchestration, YAML profiles, SGX and non-SGX images, log collection. Start here for SENTRA runs. |
| **`crypten-benchmark/`** | CrypTen MPC baseline (784→128→10 MNIST), aligned to the SENTRA `with-client` profile. |
| **`sentra-node/`** | Packaged Sentra **node** images (Occlum/SGX and legacy NoSGX). Used by deployment; separate from the all-in-one ML benchmark container. |
| **`shared/`** | Shared MNIST subset helpers (`mnist_benchmark_subset.py`, `export_benchmark_indices.py`). |

**Deployment note:** `sentra-deployment/docker-compose.yaml` builds from **`benchmarking/sentra-node/`** Dockerfiles.

## Dissertation docs (measured results)

| Document | Contents |
|----------|----------|
| **`BENCHMARK_TIMING_COMPARISON.md`** | Run IDs, raw `[BENCHMARK]` lines, fair workflow phase mapping, SGX overhead, prover breakdown, supplementary runs (fault recovery, versioning). |
| **`SENTRA_EVALUATION.md`** | Dissertation draft: feature comparison, performance tables, analysis, copy-paste LaTeX. |

**Primary thesis profile:** `with-client` — 10,000 train samples, 10 epochs, batch 64, LR 0.002, seed 2026, 100 test samples.  
**Model:** MNIST MLP 784→128→10.  
**Shared row indices:** `ml-benchmark/assets/train_indices.npy` and `ml-benchmark/assets/test_indices.npy`.

## Quick start — SENTRA (Docker)

From `benchmarking/ml-benchmark/`:

```bash
make build-ml
make run-with-client              # non-SGX (primary thesis run)
make sgx-run-with-client          # SGX/Occlum (same config)
```

Logs land in the Docker volume at `/workspace/node/logs/run_<timestamp>/` (`node_*.log`, `client_distributor.log`).

Extract timings:

```bash
grep '\[BENCHMARK\]' /workspace/node/logs/run_YYYYMMDD_HHMMSS/*.log
make metrics-collect RUN_DIR=/workspace/node/logs/run_YYYYMMDD_HHMMSS
```

See **`ml-benchmark/README.md`** for SGX rebuild, runtime metrics fields, and fault-recovery runs.

## Quick start — CrypTen

From `benchmarking/crypten-benchmark/` (Linux or WSL):

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make run
```

Config: `configs/sentra-with-client.yaml`. See **`crypten-benchmark/README.md`** for fairness notes (no separate client; rank 0 holds data).

## YAML profiles (`ml-benchmark/configs/`)

| Config | Use |
|--------|-----|
| `with-client.yaml` | **Primary thesis run** — 3 nodes + client, 10k/10 epochs. |
| `with-client-2epoch.yaml` | Shorter sanity check (2 epochs, same 10k samples). |
| `with-client-versioned.yaml` | Per-epoch weight versioning overhead vs baseline. |
| `with-client-fault-5node.yaml` | 5-node dropout recovery (kill one node after Epoch 1 Batch 2). |
| `quick.yaml` | Fast smoke (256 samples, 1 epoch, node-owned data). |
| `standard.yaml` | Node-owned MNIST, no client distributor. |

Each config has matching `make run-*` and `make sgx-run-*` targets where applicable.

## Three-way comparison (what to report fairly)

SENTRA and CrypTen do **not** share the same pipeline stages:

| Stage | SENTRA | CrypTen |
|-------|--------|---------|
| Runtime / enclave init | Node cold start (~1 s non-SGX; ~250 s SGX) | — |
| Data to MPC | Client PSS distribution (~2–4 min) | Rank-0 load + encrypt (~1 s) |
| Secure training | Node `[BENCHMARK] phase=training` | `phase=training` |
| Post-train eval | Client `share_wait_sec` (not full `eval` wall) | `phase=eval` |

Use **`BENCHMARK_TIMING_COMPARISON.md` §3 and §13** for stacked-bar buckets and reporting rules. Do not add CrypTen “client distribution” or stack SENTRA client `eval` wall on top of training.

## Shared indices

Regenerate train/test row indices for a given sample count:

```bash
python benchmarking/shared/export_benchmark_indices.py \
  -o benchmarking/ml-benchmark/assets \
  --train-samples 10000 --test-samples 100
```

CrypTen reads the same `.npy` files via `sentra-with-client.yaml`.

## Further reading

- ML benchmark details: **`ml-benchmark/README.md`**
- CrypTen baseline: **`crypten-benchmark/README.md`**
- Sentra node images: **`sentra-node/README.md`**

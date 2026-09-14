# ML benchmark (Docker)

Occlum **base** image with full `sentra-node/python/node` + `client`, pinned pip deps, YAML profiles, and `start_all_nodes` orchestration.

## MNIST data prerequisite

The repository includes the Keras-format MNIST archive at
**`benchmarking/shared/mnist.npz`**. Both benchmark Dockerfiles copy it into
the image as **`/mnist.npz`**.

| Profile | Who loads MNIST |
|---------|-----------------|
| `quick.yaml` / `standard.yaml` | **Owner node** (node 1) from `/mnist.npz` |
| `with-client.yaml` | **Client only**; nodes receive secret shares (non-SGX and SGX all-in-one) |

- **Docker:** `MNIST_NPZ_PATH=/mnist.npz` in compose.
- **Rebuild after client/MNIST changes:** `docker compose build --no-cache ml-benchmark ml-benchmark-sgx`

## From this directory

```bash
make build-ml
make run-quick
```

Compose uses **repository root** as build context (`../..` from here).

## From repository root

```bash
docker compose -f benchmarking/ml-benchmark/docker-compose.yml build
docker compose -f benchmarking/ml-benchmark/docker-compose.yml run --rm ml-benchmark
```

## Local driver (no Docker)

```bash
python benchmarking/ml-benchmark/run_benchmark_from_config.py \
  -c benchmarking/ml-benchmark/configs/quick.yaml
```

## SGX with client (`make sgx-run-with-client`)

**Non-SGX and SGX all-in-one** both use **`with-client.yaml`** (default 3 nodes, client-side MNIST):

```bash
make run-with-client          # non-SGX
make sgx-run-with-client      # SGX (same config)
```


Logs: `/workspace/node/logs/run_<timestamp>/` (Docker volume `ml_benchmark_logs`). On failure, tail `node_*.log` and `client_distributor.log` there.

## Runtime metrics (dissertation)

When `SENTRA_RUNTIME_METRICS=1` (default in `docker-compose.yml`), each node logs a final line:

`[BENCHMARK] phase=runtime_metrics ...`

| Field | Meaning |
|-------|---------|
| `bytes_sent` / `bytes_recv` / `bytes_total` | Application wire volume (incl. framing) |
| `comm_mb_per_iter` | Total bytes ÷ secure training iterations |
| `cpu_avg_pct` / `cpu_peak_pct` | Process CPU% samples (`psutil`, interval from `SENTRA_RESOURCE_SAMPLE_INTERVAL_S`) |
| `rss_mb_peak` | Peak resident set (MiB) |
| `recovery_sec_total` | Sum of dropout/join recovery wall times |
| `versioning_sec_total` | KVS weight-versioning overhead (when enabled) |

**Collect after a run** (logs volume `ml_benchmark_logs`):

```bash
make metrics-collect RUN_DIR=/workspace/node/logs/run_YYYYMMDD_HHMMSS
```

**Versioning overhead:** run `make run-with-client` vs `make run-with-client-versioned`, then pass `--baseline-training-sec` from the non-versioned run.

**Failure / dropout:**

- `make run-with-client-fault-5node` / `make sgx-run-with-client-fault-5node` — 5 nodes, 512 samples, 5 epochs; kill node 3 after Epoch 1 Batch 2.
- Config: `configs/with-client-fault-5node.yaml`. Successful reference: `run_20260527_042846` (~430 s training, ~8 s recovery).

During training, kill one node (`kill -9 <pid>` inside the container). Check `recovery_*` lines in `node_*.log`. Timings and the dissertation metrics table are in `benchmarking/BENCHMARK_TIMING_COMPARISON.md` §§14–15.

Rebuild images after pulling metrics changes: `make build-ml` (and `make build-ml-sgx` for SGX).

## Other

- Packaged **Sentra node** images (Rust + Occlum package): `../sentra-node/README.md` and `make build-sentra-sgx` from this folder’s Makefile.

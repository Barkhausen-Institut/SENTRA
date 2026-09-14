# Testing Context-Based Multi-Node Communication

This guide uses tests and launchers that exist in the current repository. SENTRA is research/benchmark software; passing these checks is evidence for the exercised configuration, not a production security certification.

## Test environment

Install dependencies from the repository root:

```powershell
python -m pip install -r sentra-node/python/node/requirements.txt
```

Then make `sentra-node/python` the working directory:

```powershell
Set-Location sentra-node/python
```

The commands below intentionally use that directory because `pytest.ini` defines `node` and `client` import paths and the repository's test roots.

## Fast checks

Validate launcher argument handling and core share/reconstruction behavior:

```powershell
python -m pytest -q node/testing/test_start_all_nodes_cli.py
python -m pytest -q node/testing/test_secure_aggregation.py
python -m pytest -q node/testing/test_dpss_non_reconstructing.py
```

These checks do not by themselves prove a complete networked training run.

## Networked context test

Run the quick secure-softmax test:

```powershell
python -m pytest -q node/testing/test_softmax_quick.py -s
```

This test starts local MPC network participants, uses unique context prefixes, exchanges shares, and reconstructs selected softmax, loss, and gradient values. A pass checks the tested localhost path and its numerical assertions.

## End-to-end stage test

Run:

```powershell
python -m pytest -q node/testing/test_batched_stage_invariants.py -s
```

This integration test launches `node/run_mnist_batched_secure.py` processes with networking enabled and checks stage invariants. It may take longer than unit tests and requires free localhost ports and the test's runtime dependencies.

Other current integration coverage is available under `node/testing/`, including:

- `test_batched_stability_integration.py`
- `test_batched_accuracy_floor_integration.py`
- `test_batched_vs_plaintext_tolerance.py`
- `test_dropout_join_recovery.py`
- `test_node_failure_detection.py`

Run only the scenarios relevant to the claim being evaluated; several are intentionally expensive or environment-sensitive.

## Manual smoke run

From the repository root, launch all local nodes through the canonical orchestrator:

```powershell
python sentra-node/python/node/start_all_nodes.py --n-nodes 3 --base-port 9600 --headless --distribute-dataset-shares --dataset-owner-node 1 --num-epochs 1 --batch-size 8 --mnist-samples 128 --debug-numerics
```

For the stronger client-owned-data workflow:

```powershell
python sentra-node/python/node/start_all_nodes.py --n-nodes 3 --base-port 9600 --headless --receive-dataset-shares-from-client --start-client-distributor --client-eval-after-training --client-eval-samples 100 --client-test-samples 100 --num-epochs 1 --batch-size 64 --mnist-samples 128 --debug-numerics
```

Inspect the run directory reported by the launcher under `sentra-node/python/node/logs/`. Confirm:

- every node exits successfully;
- no peer reports a barrier, timeout, or connection error;
- context-specific operations do not collide or time out;
- numerical diagnostics remain finite;
- client evaluation completes when requested.

Exact log wording is not a stable API, so tests should assert behavior rather than copied console banners.

## Failure interpretation

- **Connection refused/reset:** a node exited early, a port is occupied, or a peer used inconsistent arguments.
- **Barrier/context timeout:** inspect all node logs for the first failure; the timeout is usually secondary.
- **Reconstruction failure:** verify the active committee and threshold satisfy the configured safety requirements.
- **Missing data:** set `MNIST_NPZ_PATH` to a valid MNIST `.npz` file.
- **Flaky port conflict:** rerun with a different `--base-port`; do not weaken assertions to hide the conflict.

## Limits of this verification

Local tests do not exercise SGX. SGX/Occlum requires compatible hardware and the workflow in `benchmarking/ml-benchmark/README.md`.

The separate `sentra-deployment/` manifests depend on a private backend image. They cannot be fully reproduced from the public/local Python workflow alone, and these tests do not validate that deployment path.

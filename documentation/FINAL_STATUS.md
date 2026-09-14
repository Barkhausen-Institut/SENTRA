# SENTRA Project Status

## Positioning

SENTRA is dissertation research and benchmarking software for secure, distributed MNIST training. The repository contains implemented MPC components, local multi-process tests, and non-SGX/SGX benchmark workflows. It should not be described as production ready or as independently security certified.

This document describes repository capabilities, not a claim that every configuration is currently running or that every security property has been formally verified.

## Current execution paths

- **Local orchestration:** `sentra-node/python/node/start_all_nodes.py`
- **Per-node secure MNIST runner:** `sentra-node/python/node/run_mnist_batched_secure.py`
- **Pytest configuration:** `sentra-node/python/pytest.ini`
- **Detailed local CLI guide:** `sentra-node/python/README.md`
- **Benchmark entrypoint:** `benchmarking/README.md`
- **SGX/non-SGX benchmark guide:** `benchmarking/ml-benchmark/README.md`

The launcher creates local node processes and invokes the batched secure runner. The client-owned-data flags can start a separate client distributor so compute nodes receive dataset shares rather than raw MNIST.

## Implemented and exercised areas

The codebase includes:

- Shamir and packed Shamir secret sharing;
- Beaver-triple-based secure arithmetic;
- secure comparison, division, and softmax approximation components;
- context-scoped network share exchange and reconstruction;
- batched MNIST MLP training;
- versioned KVS and weight-versioning paths;
- failure-detection, dropout/join, and proactive-refresh experiments;
- client-side dataset sharing and optional reconstruction of evaluation results;
- local unit and integration tests under `sentra-node/python/node/testing/` and `node/tests/`;
- Docker benchmark profiles for non-SGX, SGX/Occlum, and comparison with CrypTen.

Whether a feature is suitable for a reported result must be established by running the relevant test or benchmark profile at the source revision being evaluated. Existing historical measurements do not imply universal reliability, performance, or security.

## Recommended verification

Install from the repository root:

```powershell
python -m pip install -r sentra-node/python/node/requirements.txt
```

Run the configured tests from `sentra-node/python`:

```powershell
Set-Location sentra-node/python
python -m pytest -q node/testing/test_forward_scaling.py node/testing/test_gradient_scaling.py
python -m pytest -q node/testing/test_softmax_quick.py -s
python -m pytest -q node/testing/test_batched_stage_invariants.py -s
```

For a local smoke run from the repository root:

```powershell
python sentra-node/python/node/start_all_nodes.py --n-nodes 3 --base-port 9600 --headless --distribute-dataset-shares --dataset-owner-node 1 --num-epochs 1 --batch-size 8 --mnist-samples 128
```

Use the client distributor command in `USAGE_GUIDE.md` when testing the independent data-owner workflow.

## Security and operational caveats

- The local Python workflow does not run inside SGX.
- SGX/Occlum benchmarks require compatible Intel SGX hardware, drivers, Occlum, and the Docker workflow documented in `benchmarking/ml-benchmark/README.md`.
- Benchmark success is not equivalent to remote-attestation assurance, formal protocol verification, penetration testing, key-management review, or production hardening.
- Some comparison modes deliberately open values or relax the protocol path. In particular, `opened_exact` is an accelerated comparison/ablation mode and must not be reported as equivalent to `secure_approx`.
- Owner-node dataset distribution is a local simulation convenience; client-owned distribution is the relevant path when compute nodes must not initially load raw data.
- Fault-recovery and threshold behavior depend on committee size, active membership, and protocol assumptions; they should be reported only for tested configurations.
- Native Windows is not established as equivalent to the Linux/WSL multi-process benchmark environment.

## Deployment boundary

Core local training and the self-contained benchmark workflow do not require the legacy web demonstrator backend.

The manifests under `sentra-deployment/` are a separate backend-assisted path and require access to the private image:

```text
registry.tdp.trustworthy6g.net/tdp/sentra/sentra-backend:latest
```

Without registry access and the documented infrastructure prerequisites, that deployment path is not reproducible. The manifests should therefore not be used as evidence of a generally available production deployment.

## Documentation index

- `README.md` — repository overview and supported top-level workflows
- `documentation/USAGE_GUIDE.md` — concise local runbook
- `documentation/MULTI_NODE_GUIDE.md` — multi-node launcher and limitations
- `documentation/HOW_TO_TEST_CONTEXT.md` — current context/network verification
- `sentra-node/python/README.md` — detailed local CLI and training parameters
- `benchmarking/README.md` — dissertation benchmark index
- `benchmarking/ml-benchmark/README.md` — Docker, SGX, and non-SGX benchmark instructions

## Status summary

The repository provides a substantial research prototype and reproducible benchmark tooling for selected environments. Claims should remain scoped to the exact tested configuration, logs, source revision, and threat model. Further security review, deployment hardening, dependency management, observability, and target-environment validation would be required before production use.

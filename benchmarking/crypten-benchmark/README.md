# CrypTen benchmark (comparison with SENTRA)

[CrypTen](https://github.com/facebookresearch/CrypTen) is Facebook’s PyTorch-style **MPC** framework (`CrypTensor`). It is **archived** (read-only since May 2025) but still useful as a **software-MPC baseline** against SENTRA.

This folder runs a **784→128→10 MNIST MLP** with hyperparameters aligned to `benchmarking/ml-benchmark/configs/with-client.yaml`.

## What you are comparing

| System | Security model | Typical use in dissertation |
|--------|----------------|-----------------------------|
| **SENTRA non-SGX** | Custom MPC (Shamir/PSS, Beaver triples) | Your stack, no enclave |
| **SENTRA SGX** | Same MPC inside **Occlum/SGX** | Hardware TEE + MPC |
| **CrypTen** | **MPC only** (PyTorch-like); no SGX | External MPC baseline |

CrypTen does **not** replace SENTRA SGX: it measures “how does a mature MPC framework compare on the same model/dataset size,” not TEE overhead.

## Fairness notes (important)

| Aspect | SENTRA `with-client` | This CrypTen benchmark |
|--------|----------------------|-------------------------|
| Model | 784-128-10 MLP | Same topology |
| Train samples | 512 (config) | 512 |
| Epochs / batch / LR | 2 / 32 / 0.01 | Matched in YAML |
| Parties | `n-nodes` + **client** process | `world_size` MPC parties; **data at rank 0** |
| Loss | Secure softmax (fixed-point) | `cross_entropy` on encrypted tensors |
| Eval | 64 test samples, client reconstructs | 64 test samples, decrypt on rank 0 |

Accuracy and runtime will **not** match exactly: different protocols, fixed-point vs CrypTen’s arithmetic, and SENTRA’s secure softmax approximations.

## Requirements

- **Linux** (or WSL2 Linux). CrypTen does not support Windows natively.
- **Python 3.8–3.10** recommended (3.11+ may break old `crypten` wheels).
- **CPU** is fine; GPU optional if PyTorch sees CUDA.

## Install

```bash
cd benchmarking/crypten-benchmark
python3 -m venv .venv
source .venv/bin/activate
make install
```

**`sklearn` install error:** CrypTen 0.4.1 still declares the deprecated PyPI package `sklearn`. Use `make install` (not raw `pip install -r requirements-crypten.txt` alone): it installs `scikit-learn` first and sets `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` for the CrypTen step.

Manual equivalent:

```bash
pip install -r requirements-crypten.txt
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install crypten==0.4.1
```

If `pip install crypten` still fails, clone the repo and install from source:

```bash
git clone https://github.com/facebookresearch/CrypTen.git
cd CrypTen
pip install -r requirements.txt
pip install .
```

## Run

```bash
export MNIST_NPZ_PATH=/path/to/sentra/benchmarking/shared/mnist.npz
make run
# or
python3 launcher.py -c configs/sentra-with-client.yaml
```

Example output:

```text
Run summary:
  framework: CrypTen
  status: success
  final_accuracy_pct: ...
  end_to_end_sec: ...
  training_sec: ...
  eval_sec: ...
  world_size: 3
```

`[BENCHMARK]` lines use the same tag format as SENTRA for easy log parsing.

## Compare with SENTRA (same machine)

Run all three on **sentra-worker-3** (or one Linux host) and fill a table:

| Metric | SENTRA non-SGX | SENTRA SGX | CrypTen |
|--------|----------------|------------|---------|
| `end_to_end_sec` | from Run summary | from Run summary | from Run summary |
| `final_accuracy_pct` | client eval | client eval | rank-0 eval |
| `max_cold_start_sec` | orchestrator block | orchestrator block | N/A (see `data_load` + import) |
| `training_sec` | node table | node table | `training_sec` |

SENTRA commands (from `benchmarking/ml-benchmark`):

```bash
make run-with-client
make sgx-run-with-client
```

## Dissertation wording (suggested)

> We compare SENTRA against CrypTen [Knott et al., 2021], a PyTorch-based MPC framework, using an equivalent MNIST MLP (784-128-10) and matched training budget (512 samples, 2 epochs, batch size 32). CrypTen represents **software-only MPC**; SENTRA additionally reports **SGX/Occlum** deployment costs.

## References

- CrypTen repository: https://github.com/facebookresearch/CrypTen  
- Paper: *CrypTen: Secure Multi-Party Computation Meets Machine Learning* (arXiv:2109.00984)

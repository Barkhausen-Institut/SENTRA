# SENTRA Usage Guide

This guide is the practical runbook for the current SENTRA codebase.
For project overview and design notes, see `README.md`.

## 1. Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Recommended Run Modes

### A) SENTRA-compliant mode (`secure_approx`)

Use this when you need protocol-faithful results.

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

### B) Accelerated comparison mode (`opened_exact`)

Use this for benchmarking/ablation against the compliant mode.

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

## 3. How to Read Logs

Enable `--debug-numerics` for useful diagnostics.

Key lines:
- `Epoch X Test Accuracy`
- `Epoch X Test Loss`
- `Epoch X Diagnostics: mean|logit|, max|logit|, mean_entropy`
- `Epoch X Probe probs_est stats`

Healthy run indicators:
- `probs_est stats sum` near `1.0`
- `probs_est min >= 0`
- no rapid explosion in `max|logit|`
- loss trending down across epochs

## 4. Common Problems

### Port already in use

```bash
pkill -f "start_all_nodes.py|run_mnist_batched_secure.py" || true
```

Then re-run with a fresh `--base-port`.

### Broken pipe / reset by peer

One or more nodes exited early. Ensure all nodes run with identical arguments and check per-node logs.

### Non-integer softmax temperature

Fixed-point path currently expects integer temperature. Use values like `1` or `2`.

### Slow integration tests on WSL (/mnt/c)

If integration tests are timing out on WSL, increase the subprocess wait budget:

```bash
export SENTRA_INTEGRATION_TIMEOUT_SEC=1200
```

Running from a Linux-native filesystem (e.g. `~/sentra2`) is typically faster than `/mnt/c/...`.

### Packing safety bound (strict)

Packed MPC enforces the strict bound \(2(t + s - 1) < n_{active}\). Equality is unsafe.
If you see packing-safety failures, reduce `t`/`s`, or increase `n_active` (more live nodes).

## 5. Tests

Quick sanity tests:

```bash
python -m pytest -q testing/test_forward_scaling.py testing/test_gradient_scaling.py
```

Stage invariant integration test:

```bash
python -m pytest -q testing/test_batched_stage_invariants.py -s
```

## 6. Reproducibility Checklist

Always record:
- full command line
- seed
- field size and scale factor
- softmax gradient mode
- epoch-wise metrics (accuracy, loss, diagnostics)

## 7. Notes

- `secure_approx` is the SENTRA-compliant run mode.
- `opened_exact` should be labeled as a comparison/accelerated variant.
- Multi-node training is MNIST-only; you do not need `--batched` (it is deprecated and always on).
- Plaintext baselines: `run_mnist_plaintext.py` (flags aligned with the secure runner) or `testing/run_training_baseline.py` (minimal Keras trainer, useful for CI/smoke).

# SENTRA Usage Guide

This guide is the practical runbook for the current SENTRA codebase.
For project overview and design notes, see `README.md`.

## 1. Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Recommended Run Modes

### A) SENTRA-compliant mode (`secure_approx`)

Use this when you need protocol-faithful results.

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

### B) Accelerated comparison mode (`opened_exact`)

Use this for benchmarking/ablation against the compliant mode.

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

## 3. How to Read Logs

Enable `--debug-numerics` for useful diagnostics.

Key lines:
- `Epoch X Test Accuracy`
- `Epoch X Test Loss`
- `Epoch X Diagnostics: mean|logit|, max|logit|, mean_entropy`
- `Epoch X Probe probs_est stats`

Healthy run indicators:
- `probs_est stats sum` near `1.0`
- `probs_est min >= 0`
- no rapid explosion in `max|logit|`
- loss trending down across epochs

## 4. Common Problems

### Port already in use

```bash
pkill -f "start_all_nodes.py|run_mnist_batched_secure.py" || true
```

Then re-run with a fresh `--base-port`.

### Broken pipe / reset by peer

One or more nodes exited early. Ensure all nodes run with identical arguments and check per-node logs.

### Non-integer softmax temperature

Fixed-point path currently expects integer temperature. Use values like `1` or `2`.

### Slow integration tests on WSL (/mnt/c)

If integration tests are timing out on WSL, increase the subprocess wait budget:

```bash
export SENTRA_INTEGRATION_TIMEOUT_SEC=1200
```

Running from a Linux-native filesystem (e.g. `~/sentra2`) is typically faster than `/mnt/c/...`.

### Packing safety bound (strict)

Packed MPC enforces the strict bound \(2(t + s - 1) < n_{active}\). Equality is unsafe.
If you see packing-safety failures, reduce `t`/`s`, or increase `n_active` (more live nodes).

## 5. Tests

Quick sanity tests:

```bash
python -m pytest -q testing/test_forward_scaling.py testing/test_gradient_scaling.py
```

Stage invariant integration test:

```bash
python -m pytest -q testing/test_batched_stage_invariants.py -s
```

## 6. Reproducibility Checklist

Always record:
- full command line
- seed
- field size and scale factor
- softmax gradient mode
- epoch-wise metrics (accuracy, loss, diagnostics)

## 7. Notes

- `secure_approx` is the SENTRA-compliant run mode.
- `opened_exact` should be labeled as a comparison/accelerated variant.
- Multi-node training is MNIST-only; you do not need `--batched` (it is deprecated and always on).
- Plaintext baselines: `run_mnist_plaintext.py` (flags aligned with the secure runner) or `testing/run_training_baseline.py` (minimal Keras trainer, useful for CI/smoke).

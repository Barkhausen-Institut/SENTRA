"""
Plaintext MNIST MLP baseline runner.
Architecture: 784 -> 128 -> 10 with ReLU + softmax CE (or MSE).

Accepts a superset of the secure batched runner flags for drop-in benchmarking.
MPC/network-related flags are accepted but ignored (with a warning if set).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from ml_training.util import loadMNISTDataset


def _load_mnist(n_train: int | None, n_test: int | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = loadMNISTDataset()
    x_train = x_train.reshape(-1, 784).astype(np.float64) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float64) / 255.0
    if n_train is not None and n_train > 0:
        x_train = x_train[:n_train]
        y_train = y_train[:n_train]
    if n_test is not None and n_test > 0:
        x_test = x_test[:n_test]
        y_test = y_test[:n_test]
    return x_train, y_train, x_test, y_test


def _one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _softmax(logits: np.ndarray, temperature: float, logit_clip: float | None) -> np.ndarray:
    z = logits / max(1e-12, float(temperature))
    if logit_clip is not None and logit_clip > 0:
        z = np.clip(z, -float(logit_clip), float(logit_clip))
    zc = z - np.mean(z, axis=1, keepdims=True)
    ez = np.exp(zc)
    return ez / np.maximum(np.sum(ez, axis=1, keepdims=True), 1e-12)


def _evaluate(
    x: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    loss_mode: str,
    temperature: float,
    logit_clip: float | None,
    max_samples: int | None,
) -> Tuple[float, float]:
    if max_samples is not None and max_samples > 0:
        x = x[:max_samples]
        y = y[:max_samples]
    a1 = np.maximum(0.0, x @ w1.T + b1)
    logits = a1 @ w2.T + b2
    if str(loss_mode).lower() == "softmax":
        probs = _softmax(logits, temperature, logit_clip)
        pred = np.argmax(probs, axis=1)
        acc = float(np.mean(pred == y))
        y_oh = _one_hot(y, 10)
        loss = float(-np.mean(np.sum(y_oh * np.log(np.maximum(probs, 1e-12)), axis=1)))
        return acc, loss
    # MSE proxy
    y_oh = _one_hot(y, 10)
    diff = logits - y_oh
    loss = float(np.mean(np.sum(diff * diff, axis=1)))
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))
    return acc, loss


def _export_model(path: str, w1: np.ndarray, w2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> None:
    out_path = Path(path)
    if out_path.suffix.lower() != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, w1=w1, w2=w2, b1=b1, b2=b2)
    meta = {
        "format": "npz",
        "arrays": {"w1": list(w1.shape), "w2": list(w2.shape), "b1": list(b1.shape), "b2": list(b2.shape)},
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported model: {out_path}")
    print(f"Model metadata: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plaintext MNIST MLP baseline (784-128-10)")
    # Core training/eval flags
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mnist-samples", type=int, default=None, help="Number of train samples (default: full train)")
    parser.add_argument("--mnist-test-samples", type=int, default=None, help="Number of test samples (default: full test)")
    parser.add_argument("--client-test-samples", type=int, default=None, help="Alias for --mnist-test-samples")
    parser.add_argument("--client-eval-samples", type=int, default=100, help="Eval samples per epoch (0 = full test)")
    parser.add_argument("--loss-mode", choices=["mse", "softmax"], default="softmax")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--logit-clip", type=float, default=8.0)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--debug-numerics", action="store_true")
    parser.add_argument("--export-reconstructed-model", type=str, default="")

    # Secure/MPC-related flags (accepted but ignored)
    parser.add_argument("--node-id", type=int, default=1)
    parser.add_argument("--n-nodes", type=int, default=1)
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--enable-network", action="store_true")
    parser.add_argument("--field-size", type=int, default=2**32 - 5)
    parser.add_argument("--scale-factor", type=int, default=2**20)
    parser.add_argument("--exp-approx", choices=["taylor5", "pade22"], default="pade22")
    parser.add_argument("--softmax-grad-mode", choices=["secure_approx", "opened_exact"], default="secure_approx")
    parser.add_argument("--debug-division", action="store_true")
    parser.add_argument("--explode-logit-threshold", type=float, default=10.0)
    parser.add_argument("--loss-growth-threshold", type=float, default=5.0)
    parser.add_argument("--grad-norm-threshold", type=float, default=5.0)
    parser.add_argument("--no-abort-on-instability", action="store_true")
    parser.add_argument("--distribute-dataset-shares", action="store_true")
    parser.add_argument("--dataset-owner-node", type=int, default=1)
    parser.add_argument("--dataset-distribution-timeout", type=float, default=900.0)
    parser.add_argument("--receive-dataset-shares-from-client", action="store_true")
    parser.add_argument("--dataset-source-node-id", type=int, default=0)
    parser.add_argument("--client-eval-after-training", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--start-client-distributor", action="store_true")
    args = parser.parse_args()

    ignored_flags = []
    if args.enable_network or args.n_nodes != 1 or args.node_id != 1 or args.t != 1:
        ignored_flags.append("network/mpc topology")
    if args.distribute_dataset_shares or args.receive_dataset_shares_from_client or args.start_client_distributor:
        ignored_flags.append("dataset share distribution")
    if args.field_size != 2**32 - 5 or args.scale_factor != 2**20 or args.exp_approx != "pade22" or args.softmax_grad_mode != "secure_approx":
        ignored_flags.append("fixed-point/MPC numeric settings")
    if args.debug_division:
        ignored_flags.append("secure division debug")
    if args.headless or args.batched:
        ignored_flags.append("headless/batched flags")
    if ignored_flags:
        print(f"[WARN] Ignoring secure/MPC-only flags: {', '.join(sorted(set(ignored_flags)))}")

    np.random.seed(int(args.seed))

    n_test = args.client_test_samples if args.client_test_samples is not None else args.mnist_test_samples
    x_train, y_train, x_test, y_test = _load_mnist(args.mnist_samples, n_test)
    y_train_oh = _one_hot(y_train, 10)

    input_dim, hidden_dim, output_dim = 784, 128, 10
    w1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    w2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(output_dim, dtype=np.float64)

    print("Plaintext MNIST MLP baseline (784-128-10)")
    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.num_epochs}, LR: {args.learning_rate}")
    print(f"Loss mode: {args.loss_mode}, Softmax temp: {args.softmax_temperature}, Logit clip: {args.logit_clip}, Grad clip: {args.grad_clip}")

    eval_samples = int(args.client_eval_samples)
    if eval_samples <= 0:
        eval_samples = None

    acc_pre, loss_pre = _evaluate(
        x_test, y_test, w1, w2, b1, b2, args.loss_mode, args.softmax_temperature, args.logit_clip, 1
    )
    print(f"Pre-Train Test Accuracy: {acc_pre * 100:.2f}%")
    print(f"Pre-Train Test Loss: {loss_pre:.4f}")

    _t_train0 = time.time()
    for epoch in range(int(args.num_epochs)):
        lr = float(args.learning_rate) * (0.95 ** epoch)
        indices = np.arange(len(x_train))
        rng = np.random.default_rng(int(args.seed) + epoch)
        rng.shuffle(indices)

        if len(indices) == 0:
            break

        for start_idx in range(0, len(indices), int(args.batch_size)):
            batch_idx = indices[start_idx:start_idx + int(args.batch_size)]
            x_batch = x_train[batch_idx]
            y_batch = y_train_oh[batch_idx]

            z1 = x_batch @ w1.T + b1
            a1 = np.maximum(0.0, z1)
            logits = a1 @ w2.T + b2

            if args.debug_numerics and start_idx == 0:
                zc = logits - np.mean(logits, axis=1, keepdims=True)
                print(f"Epoch {epoch+1} Max Centered Logit: {np.max(zc):.4f}")

            if str(args.loss_mode).lower() == "softmax":
                probs = _softmax(logits, args.softmax_temperature, args.logit_clip)
                dz2 = (probs - y_batch)  # (B, 10)
            else:
                diff = logits - y_batch
                dz2 = diff / max(1, output_dim)
                dz2 = np.clip(dz2, -4.0, 4.0)

            batch_den = max(1, len(batch_idx))
            dw2 = (dz2.T @ a1) / batch_den
            db2 = np.sum(dz2, axis=0) / batch_den
            da1 = dz2 @ w2
            dz1 = da1 * (z1 > 0).astype(np.float64)
            dw1 = (dz1.T @ x_batch) / batch_den
            db1 = np.sum(dz1, axis=0) / batch_den

            if args.grad_clip is not None and float(args.grad_clip) > 0:
                clip = float(args.grad_clip)
                dw1 = np.clip(dw1, -clip, clip)
                dw2 = np.clip(dw2, -clip, clip)
                db1 = np.clip(db1, -clip, clip)
                db2 = np.clip(db2, -clip, clip)

            w1 -= lr * dw1
            b1 -= lr * db1
            w2 -= lr * dw2
            b2 -= lr * db2

        acc, loss = _evaluate(
            x_test, y_test, w1, w2, b1, b2,
            args.loss_mode, args.softmax_temperature, args.logit_clip, eval_samples
        )
        print(f"Epoch {epoch+1} Test Accuracy: {acc * 100:.2f}%")
        print(f"Epoch {epoch+1} Test Loss: {loss:.4f}")

    print(f"Training Time: {time.time() - _t_train0:.6f}s")

    if args.export_reconstructed_model:
        _export_model(args.export_reconstructed_model, w1, w2, b1, b2)


if __name__ == "__main__":
    main()

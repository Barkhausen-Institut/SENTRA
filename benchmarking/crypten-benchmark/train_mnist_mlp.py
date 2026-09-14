#!/usr/bin/env python3
"""
CrypTen MNIST MLP benchmark aligned with SENTRA ml-benchmark (with-client profile).

Architecture: 784 -> 128 (ReLU) -> 10, same topology as BatchedSecureMNISTMLP.
Runs as N MPC parties via multiprocess launcher (no SGX).

Reference: https://github.com/facebookresearch/CrypTen (archived May 2025).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MNIST = _REPO_ROOT / "benchmarking" / "shared" / "mnist.npz"
_DEFAULT_INDICES_DIR = _REPO_ROOT / "benchmarking" / "ml-benchmark" / "assets"


def _log_benchmark(*, phase: str, wall_sec: float, **extra: object) -> None:
    parts = [f"role=crypten", f"phase={phase}", f"wall_sec={wall_sec:.6f}"]
    for key, val in extra.items():
        parts.append(f"{key}={val}")
    print("[BENCHMARK] " + " ".join(parts), flush=True)


def _resolve_mnist_path(explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(f"MNIST file not found: {p}")
    for candidate in (
        os.environ.get("MNIST_NPZ_PATH", ""),
        "/mnist.npz",
        str(_DEFAULT_MNIST),
    ):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        f"mnist.npz not found. Set --mnist-npz or MNIST_NPZ_PATH. Tried {_DEFAULT_MNIST}"
    )


def _load_benchmark_row_indices(
    *,
    train_samples: int,
    test_samples: int,
    indices_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same rows as SENTRA client: first N train/test in mnist.npz order."""
    root = Path(indices_dir) if indices_dir else _DEFAULT_INDICES_DIR
    train_path = root / "train_indices.npy"
    test_path = root / "test_indices.npy"
    if train_path.is_file() and test_path.is_file():
        train_idx = np.load(train_path)
        test_idx = np.load(test_path)
    else:
        train_idx = np.arange(min(train_samples, 60_000), dtype=np.int64)
        test_idx = np.arange(min(test_samples, 10_000), dtype=np.int64)
    return train_idx, test_idx


def _load_mnist_subset(
    path: Path,
    *,
    train_samples: int,
    test_samples: int,
    seed: int,
    indices_dir: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del seed  # batch order uses torch.randperm(n_train) below
    data = np.load(path)
    x_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    y_train = data["y_train"].astype(np.int64)
    x_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    y_test = data["y_test"].astype(np.int64)

    train_idx, test_idx = _load_benchmark_row_indices(
        train_samples=train_samples,
        test_samples=test_samples,
        indices_dir=indices_dir,
    )

    return (
        torch.from_numpy(x_train[train_idx]),
        torch.from_numpy(y_train[train_idx]),
        torch.from_numpy(x_test[test_idx]),
        torch.from_numpy(y_test[test_idx]),
    )


class PlainMLP(nn.Module):
    """Plain PyTorch twin of SENTRA 784-128-10 MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def run_crypten_benchmark(args: argparse.Namespace) -> None:
    import crypten
    import crypten.nn as cnn
    from crypten import cryptensor

    rank = crypten.comm.get().get_rank()
    criterion = cnn.CrossEntropyLoss()
    t0 = time.time()

    torch.manual_seed(int(args.seed))
    train_cap = int(args.mnist_samples)
    test_cap = max(int(args.eval_samples), int(args.client_test_samples))

    x_train = y_train = x_test = y_test = None
    if rank == 0:
        x_train, y_train, x_test, y_test = _load_mnist_subset(
            _resolve_mnist_path(args.mnist_npz),
            train_samples=train_cap,
            test_samples=test_cap,
            seed=int(args.seed),
            indices_dir=str(getattr(args, "indices_dir", "") or ""),
        )
        if rank == 0:
            print(
                "CrypTen: MNIST subset matches SENTRA (first N rows / shared indices.npy)",
                flush=True,
            )
        _log_benchmark(
            phase="data_load",
            wall_sec=time.time() - t0,
            train_samples=int(x_train.size(0)),
        )

    n_train = train_cap
    n_test = test_cap

    plain = PlainMLP()
    model = cnn.from_pytorch(plain, torch.randn(1, 784))
    t_enc_model0 = time.time()
    model.encrypt(src=0)
    if rank == 0:
        _log_benchmark(phase="encrypt_model", wall_sec=time.time() - t_enc_model0)

    batch_size = int(args.batch_size)
    epochs = int(args.num_epochs)
    lr = float(args.learning_rate)

    t_train0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(n_train) if rank == 0 else None
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            if rank == 0:
                idx = perm[start:end]
                x_batch = x_train[idx]
                y_batch = F.one_hot(y_train[idx], num_classes=10).float()
            else:
                x_batch = torch.empty(end - start, 784)
                y_batch = torch.empty(end - start, 10)

            x_enc = cryptensor(x_batch, src=0)
            y_enc = cryptensor(y_batch, src=0)

            model.zero_grad()
            model.train()
            logits = model(x_enc)
            loss = criterion(logits, y_enc)
            loss.backward()
            model.update_parameters(lr)

        if rank == 0:
            print(f"CrypTen epoch {epoch + 1}/{epochs} completed", flush=True)

    train_sec = time.time() - t_train0
    if rank == 0:
        _log_benchmark(phase="training", wall_sec=train_sec, epochs=epochs)

    t_eval0 = time.time()
    model.eval()
    eval_samples = min(int(args.eval_samples), n_test)
    acc = 0.0
    if eval_samples > 0:
        if rank == 0:
            x_eval = x_test[:eval_samples]
            y_eval = y_test[:eval_samples]
        else:
            x_eval = torch.empty(eval_samples, 784)
            y_eval = torch.empty(eval_samples, dtype=torch.long)
        x_eval_enc = cryptensor(x_eval, src=0)
        logits = model(x_eval_enc)
        # Reveal is collective: every rank must call get_plain_text (dst=0 -> rank 0 only).
        logits_pt = logits.get_plain_text(dst=0)
        if rank == 0:
            preds = logits_pt.argmax(dim=1)
            labels = y_eval.long()
            correct = (preds == labels).sum().item()
            acc = 100.0 * float(correct) / float(eval_samples)
    eval_sec = time.time() - t_eval0

    total_sec = time.time() - t0
    if rank == 0:
        _log_benchmark(phase="eval", wall_sec=eval_sec, eval_samples=eval_samples)
        _log_benchmark(phase="total", wall_sec=total_sec)
        print("\nRun summary:")
        print("  framework: CrypTen")
        print("  status: success")
        print(f"  final_accuracy_pct: {acc:.2f}")
        print(f"  end_to_end_sec: {total_sec:.2f}")
        print(f"  training_sec: {train_sec:.2f}")
        print(f"  eval_sec: {eval_sec:.2f}")
        print(f"  world_size: {int(args.world_size)}")
        print(f"  mnist_samples: {n_train}")
        print(f"  batch_size: {batch_size}")
        print(f"  num_epochs: {epochs}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CrypTen MNIST MLP benchmark (SENTRA-aligned)")
    p.add_argument("--world-size", type=int, default=3, help="MPC parties (match SENTRA n-nodes)")
    p.add_argument("--mnist-npz", type=str, default="", help="Path to mnist.npz")
    p.add_argument("--mnist-samples", type=int, default=512)
    p.add_argument("--client-test-samples", type=int, default=64)
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--indices-dir",
        type=str,
        default="",
        help="Directory with train_indices.npy and test_indices.npy (SENTRA-aligned)",
    )
    p.add_argument(
        "--multiprocess",
        action="store_true",
        help="Launch world_size processes locally (set by launcher.py)",
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.multiprocess:
        print("Run: python launcher.py", file=sys.stderr)
        sys.exit(2)

    from multiprocess_launcher import MultiProcessLauncher

    launcher = MultiProcessLauncher(int(args.world_size), run_crypten_benchmark, args)
    wall0 = time.time()
    launcher.start()
    launcher.join()
    launcher.terminate()
    print(f"launcher_wall_sec: {time.time() - wall0:.2f}", flush=True)


if __name__ == "__main__":
    main()

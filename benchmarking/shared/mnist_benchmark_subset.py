"""Canonical MNIST subset for cross-framework benchmarks (SENTRA + CrypTen).

SENTRA client (`load_mnist_data`) uses the first N train/test rows from local
mnist.npz (not random subsampling). CrypTen must use the same indices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

DEFAULT_TRAIN_SAMPLES = 512
DEFAULT_TEST_SAMPLES = 64
DEFAULT_SEED = 2026  # training RNG (shuffle), not row selection


def benchmark_indices(
    train_samples: int = DEFAULT_TRAIN_SAMPLES,
    test_samples: int = DEFAULT_TEST_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Row indices into mnist.npz train/test arrays (SENTRA-compatible)."""
    return (
        np.arange(int(train_samples), dtype=np.int64),
        np.arange(int(test_samples), dtype=np.int64),
    )


def save_benchmark_indices(
    directory: str | Path,
    *,
    train_samples: int = DEFAULT_TRAIN_SAMPLES,
    test_samples: int = DEFAULT_TEST_SAMPLES,
) -> Tuple[Path, Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    train_idx, test_idx = benchmark_indices(train_samples, test_samples)
    train_path = out / "train_indices.npy"
    test_path = out / "test_indices.npy"
    np.save(train_path, train_idx)
    np.save(test_path, test_idx)
    return train_path, test_path


def load_benchmark_indices(directory: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(directory)
    train_path = root / "train_indices.npy"
    test_path = root / "test_indices.npy"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Missing {train_path} or {test_path}. "
            "Run: python benchmarking/shared/export_benchmark_indices.py"
        )
    return np.load(train_path), np.load(test_path)

#!/usr/bin/env python3
"""Write train_indices.npy / test_indices.npy for shared SENTRA/CrypTen benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from mnist_benchmark_subset import DEFAULT_TEST_SAMPLES, DEFAULT_TRAIN_SAMPLES, save_benchmark_indices


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "ml-benchmark" / "assets"),
    )
    p.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    p.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    args = p.parse_args()
    train_path, test_path = save_benchmark_indices(
        args.out_dir,
        train_samples=int(args.train_samples),
        test_samples=int(args.test_samples),
    )
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    main()

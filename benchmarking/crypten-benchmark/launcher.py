#!/usr/bin/env python3
"""Entry point: load YAML config and launch CrypTen multiprocess benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from train_mnist_mlp import build_arg_parser, run_crypten_benchmark


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch CrypTen MNIST benchmark")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "sentra-with-client.yaml"),
        help="YAML config (SENTRA-aligned hyperparameters)",
    )
    cli, _ = parser.parse_known_args()
    cfg = _load_config(Path(cli.config))

    argv = [
        "--multiprocess",
        "--world-size",
        str(int(cfg.get("world-size", cfg.get("n-nodes", 3)))),
        "--mnist-samples",
        str(int(cfg.get("mnist-samples", 512))),
        "--eval-samples",
        str(int(cfg.get("client-eval-samples", 64))),
        "--client-test-samples",
        str(int(cfg.get("client-test-samples", 64))),
        "--batch-size",
        str(int(cfg.get("batch-size", 32))),
        "--num-epochs",
        str(int(cfg.get("num-epochs", 2))),
        "--learning-rate",
        str(float(cfg.get("learning-rate", 0.01))),
        "--seed",
        str(int(cfg.get("seed", 2026))),
    ]
    mnist_npz = str(cfg.get("mnist-npz", "") or "").strip()
    if mnist_npz:
        argv.extend(["--mnist-npz", mnist_npz])
    indices_dir = str(cfg.get("indices-dir", "") or "").strip()
    if indices_dir:
        argv.extend(["--indices-dir", indices_dir])
    args = build_arg_parser().parse_args(argv)

    from multiprocess_launcher import MultiProcessLauncher

    print(f"Starting CrypTen benchmark (config={cli.config}, world_size={args.world_size})")
    launcher = MultiProcessLauncher(int(args.world_size), run_crypten_benchmark, args)
    launcher.start()
    launcher.join()
    launcher.terminate()


if __name__ == "__main__":
    main()

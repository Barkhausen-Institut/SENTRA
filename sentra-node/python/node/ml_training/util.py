"""Local MNIST loading (no Keras / network download)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

_DOCKER_PATH = Path("/mnist.npz")
_OCCLUM_KERAS_PATH = Path("/root/.keras/datasets/mnist.npz")


def _optional_home_keras_path() -> Path | None:
    """Keras cache under $HOME; skipped when HOME is unset (e.g. some Occlum runs)."""
    try:
        return Path.home() / ".keras" / "datasets" / "mnist.npz"
    except RuntimeError:
        return None


def _repo_resource_paths() -> list[Path]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "benchmarking" / "shared" / "mnist.npz"
        if candidate.is_file():
            return [candidate]
    return []


def _search_paths() -> list[Path]:
    paths = [_DOCKER_PATH, _OCCLUM_KERAS_PATH]
    home_keras = _optional_home_keras_path()
    if home_keras is not None:
        paths.append(home_keras)
    paths.extend(_repo_resource_paths())
    return paths


def resolve_mnist_npz_path() -> Path:
    override = os.environ.get("MNIST_NPZ_PATH", "").strip()
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"MNIST_NPZ_PATH is set but file not found: {p}")
        return p

    for p in _search_paths():
        if p.is_file():
            return p

    searched = ", ".join(str(c) for c in _search_paths())
    raise FileNotFoundError(
        "Local MNIST not found (downloads are disabled). "
        "Use Docker (mnist.npz baked at /mnist.npz), set MNIST_NPZ_PATH, "
        "or place benchmarking/shared/mnist.npz in the repo. "
        f"Searched: {searched}"
    )


def loadMNISTDataset() -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    """Load MNIST train/test (uint8, 28x28) from local mnist.npz only."""
    path = resolve_mnist_npz_path()
    data = np.load(path)
    x_train = np.asarray(data["x_train"])
    y_train = np.asarray(data["y_train"])
    x_test = np.asarray(data["x_test"])
    y_test = np.asarray(data["y_test"])
    return (x_train, y_train), (x_test, y_test)

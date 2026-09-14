"""
Run a local smoke test of SentraTrainingNode using local kvstore/mpc/coordination adapters.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

import numpy as np

# Ensure repo root is importable when running from testing/ directory.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Bind local adapters to generic module names before importing node runtime.
import local_adapters.coordination as _local_coordination
import local_adapters.kvstore as _local_kvstore
import local_adapters.mpc as _local_mpc

sys.modules["coordination"] = _local_coordination
sys.modules["kvstore"] = _local_kvstore
sys.modules["mpc"] = _local_mpc

from ml_training.sentra_training_node import (
    NodeIdentity,
    QuorumConfig,
    RetryConfig,
    SentraTrainingNode,
    TrainingConfig,
)
import kvstore


def seed_local_kvs(
    *,
    endpoint: str,
    identity: NodeIdentity,
    model_key: str,
    model_version_key: str,
    dataset_prefix: str,
    steps: int,
    batch_size: int,
    input_dim: int,
) -> None:
    client = kvstore.Client(endpoint, identity.mtls_credentials)
    client.authenticate({"enclave_id": identity.enclave_id, "epoch_token": identity.epoch_token, "jwt": identity.epoch_jwt})

    # Initial model at version 1 (multiclass linear head).
    initial_model = {
        "W": np.random.randn(input_dim, 10) * 0.01,
        "b": np.zeros((10,), dtype=np.float64),
    }
    ok = client.compare_and_set(
        key=model_key,
        old_version=0,
        new_value=initial_model,
        new_version=1,
        write_quorum=1,
        epoch_token=identity.epoch_token,
        jwt=identity.epoch_jwt,
    )
    if not ok["success"]:
        raise RuntimeError("failed to seed initial model")
    ok = client.compare_and_set(
        key=model_version_key,
        old_version=0,
        new_value=1,
        new_version=1,
        write_quorum=1,
        epoch_token=identity.epoch_token,
        jwt=identity.epoch_jwt,
    )
    if not ok["success"]:
        raise RuntimeError("failed to seed model version")

    # MNIST training batches (one-hot labels) as stand-in "shares".
    from tensorflow import keras

    from ml_training.util import loadMNISTDataset

    (x_train, y_train), _ = loadMNISTDataset()
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train.reshape(len(x_train), -1)[:, :input_dim]
    y_train_oh = np.eye(10, dtype=np.float64)[y_train.astype(np.int32)]

    max_samples = steps * batch_size
    x_train = x_train[:max_samples]
    y_train_oh = y_train_oh[:max_samples]

    for step in range(steps):
        start = step * batch_size
        end = start + batch_size
        x = x_train[start:end]
        y = y_train_oh[start:end]
        base = f"{dataset_prefix}/batch/{step}"
        in_key = f"{base}/inputs"
        tgt_key = f"{base}/targets"

        ok = client.compare_and_set(
            key=in_key,
            old_version=0,
            new_value=x.tolist(),
            new_version=1,
            write_quorum=1,
            epoch_token=identity.epoch_token,
            jwt=identity.epoch_jwt,
        )
        if not ok["success"]:
            raise RuntimeError(f"failed to seed {in_key}")
        ok = client.compare_and_set(
            key=tgt_key,
            old_version=0,
            new_value=y.tolist(),
            new_version=1,
            write_quorum=1,
            epoch_token=identity.epoch_token,
            jwt=identity.epoch_jwt,
        )
        if not ok["success"]:
            raise RuntimeError(f"failed to seed {tgt_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local smoke test for SentraTrainingNode")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--packing-factor", type=int, default=1)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    identity = NodeIdentity(
        enclave_id="local-enclave-1",
        epoch_token="local-epoch-token",
        epoch_jwt="local-epoch-jwt",
        mtls_credentials={"cert": "local", "key": "local"},
    )

    model_key = "model/params"
    model_version_key = "model/version"
    dataset_prefix = "dataset/train"

    seed_local_kvs(
        endpoint="local://kvs",
        identity=identity,
        model_key=model_key,
        model_version_key=model_version_key,
        dataset_prefix=dataset_prefix,
        steps=args.steps,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
    )

    node = SentraTrainingNode(
        kvs_endpoint="local://kvs",
        identity=identity,
        quorum=QuorumConfig(read_quorum=1, write_quorum=1),
        retry=RetryConfig(max_attempts=3),
        config=TrainingConfig(
            node_id=1,
            n_nodes=5,
            t=1,
            packing_factor=args.packing_factor,
            model_key=model_key,
            model_version_key=model_version_key,
            dataset_prefix=dataset_prefix,
            minibatch_size=args.batch_size,
            max_steps=args.steps,
            num_gpus=0,
            learning_rate=args.learning_rate,
        ),
    )

    node.train()

    # Evaluate on MNIST test subset and report classification accuracy locally.
    from ml_training.util import loadMNISTDataset

    _, (x_test, y_test) = loadMNISTDataset()
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(len(x_test), -1)[:, :args.input_dim]
    y_test = y_test.astype(np.int32)
    x_test = x_test[: args.test_samples]
    y_test = y_test[: args.test_samples]

    preds = node.evaluate(x_test.tolist())
    logits = np.asarray(preds, dtype=np.float64)
    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits[:, 0, :]
    if logits.ndim == 1:
        logits = logits.reshape(-1, 10)
    y_pred = np.argmax(logits, axis=1)
    accuracy = float(np.mean(y_pred == y_test[: len(y_pred)]))
    print(f"Local evaluation complete. Predictions returned: {len(preds)} samples")
    print(f"Local MNIST accuracy (proxy from evaluate outputs): {accuracy:.4f}")


if __name__ == "__main__":
    main()

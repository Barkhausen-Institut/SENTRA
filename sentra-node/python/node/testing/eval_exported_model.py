"""
Evaluate an exported reconstructed SENTRA MLP model on MNIST.

Expected NPZ arrays:
- w1: (784, 128)
- w2: (128, 10)
- b1: (128,)
- b2: (10,)
"""

import argparse
import numpy as np
from tensorflow import keras


def load_mnist(split: str, max_samples: int | None):
    from ml_training.util import loadMNISTDataset

    (x_train, y_train), (x_test, y_test) = loadMNISTDataset()
    if split == "train":
        x = x_train
        y = y_train
    else:
        x = x_test
        y = y_test
    x = x.reshape(-1, 784).astype("float32") / 255.0
    if max_samples is not None and max_samples > 0:
        x = x[:max_samples]
        y = y[:max_samples]
    return x, y


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.maximum(np.sum(exp_z, axis=1, keepdims=True), 1e-12)


def cross_entropy_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    one_hot = keras.utils.to_categorical(labels, probs.shape[1]).astype(np.float64)
    return float(-np.mean(np.sum(one_hot * np.log(np.maximum(probs, 1e-12)), axis=1)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate exported reconstructed MNIST MLP model")
    parser.add_argument("--model-npz", required=True, help="Path to exported model .npz")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--samples", type=int, default=0, help="Number of samples to evaluate (0 = full split)")
    args = parser.parse_args()

    data_limit = None if int(args.samples) <= 0 else int(args.samples)
    x, y = load_mnist(args.split, data_limit)

    model = np.load(args.model_npz)
    w1 = np.asarray(model["w1"], dtype=np.float64)
    w2 = np.asarray(model["w2"], dtype=np.float64)
    b1 = np.asarray(model["b1"], dtype=np.float64)
    b2 = np.asarray(model["b2"], dtype=np.float64)

    if w1.shape[0] != 784 or w2.shape[1] != 10 or b1.shape[0] != w1.shape[1] or b2.shape[0] != 10:
        raise ValueError(
            f"Unexpected model shapes: w1={w1.shape}, w2={w2.shape}, b1={b1.shape}, b2={b2.shape}"
        )

    hidden = np.maximum(0.0, x @ w1 + b1)
    logits = hidden @ w2 + b2
    probs = stable_softmax(logits)
    pred = np.argmax(probs, axis=1)

    acc = float(np.mean(pred == y))
    loss = cross_entropy_from_probs(probs, y)

    print(f"Model: {args.model_npz}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(x)}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Cross-Entropy Loss: {loss:.6f}")


if __name__ == "__main__":
    main()


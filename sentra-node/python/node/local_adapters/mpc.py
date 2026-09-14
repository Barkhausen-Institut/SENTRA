"""
Local PackedEngine adapter for SENTRA local testing.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class PackedEngine:
    """Minimal packed MPC runtime facade backed by plaintext numpy ops."""

    def __init__(self, num_gpus: int = 0, learning_rate: float = 0.01) -> None:
        self.num_gpus = num_gpus
        self.learning_rate = learning_rate
        self._last_inputs: np.ndarray | None = None

    def pack_shares(self, shares: Any, packing_factor: int = 1) -> np.ndarray:
        del packing_factor
        return np.asarray(shares, dtype=np.float64)

    def to_packed_shamir(self, shares: Any, packing_factor: int = 1) -> np.ndarray:
        return self.pack_shares(shares, packing_factor=packing_factor)

    def forward_pass(self, packed_inputs: np.ndarray, model_params: Dict[str, Any]) -> np.ndarray:
        W = np.asarray(model_params["W"], dtype=np.float64)
        b = np.asarray(model_params["b"], dtype=np.float64)
        x = np.asarray(packed_inputs, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self._last_inputs = x
        return x @ W + b

    def forward(self, packed_inputs: np.ndarray, model_params: Dict[str, Any]) -> np.ndarray:
        return self.forward_pass(packed_inputs, model_params)

    def infer(self, packed_inputs: np.ndarray, model_params: Dict[str, Any]) -> np.ndarray:
        return self.forward_pass(packed_inputs, model_params)

    def compute_gradients(
        self,
        predictions: np.ndarray,
        packed_targets: np.ndarray,
        model_params: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        del model_params
        y = np.asarray(packed_targets, dtype=np.float64)
        logits = np.asarray(predictions, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        if self._last_inputs is None:
            raise RuntimeError("forward_pass must be called before compute_gradients")
        x = self._last_inputs

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        m = max(1, x.shape[0])
        dlogits = (probs - y) / m
        dW = x.T @ dlogits
        db = np.sum(dlogits, axis=0)
        return {"dW": dW, "db": db}

    def backward_pass(
        self,
        predictions: np.ndarray,
        packed_targets: np.ndarray,
        model_params: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        return self.compute_gradients(predictions, packed_targets, model_params)

    def aggregate_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return gradients

    def secure_aggregate_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return gradients

    def update_model(self, model_params: Dict[str, Any], gradients: Dict[str, np.ndarray]) -> Dict[str, Any]:
        W = np.asarray(model_params["W"], dtype=np.float64)
        b = np.asarray(model_params["b"], dtype=np.float64)
        dW = np.asarray(gradients["dW"], dtype=np.float64)
        db = np.asarray(gradients["db"], dtype=np.float64)
        new_W = W - self.learning_rate * dW
        new_b = b - self.learning_rate * db
        return {"W": new_W, "b": new_b}

    def apply_gradients(self, model_params: Dict[str, Any], gradients: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return self.update_model(model_params, gradients)

    def dpss_reshare(self, reason: str = "") -> None:
        del reason
        return None


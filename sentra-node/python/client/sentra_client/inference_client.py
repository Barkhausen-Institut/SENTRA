"""
In-process secure MNIST inference API for the Sentra client.

Assumes training nodes are already running with matching MPC parameters.
Logits are reconstructed from secret-shared outputs; softmax "confidence"
is computed on the client from those logits (not inside MPC).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from tensorflow import keras

from ml_training.membership_epoch import MembershipEpochScope
from ml_training.util import loadMNISTDataset
from ml_training.packing_safety import get_max_safe_packing_factor
from ml_training.secret_sharing import PackedShamirSecretSharing, ShamirSecretSharing
from ml_training.secure_comm import create_mpc_network
from ml_training.topology import load_client_topology

from .distributor import (
    flatten_plain_weights_to_fixed_ints,
    load_plain_weights_from_npz,
    party_vectors_to_share_vectors,
    reconstruct_batched_logits_from_parties,
    reconstruct_logits_from_parties,
    share_vector_for_all_nodes_pss,
    wait_for_party_vectors,
    wait_for_vector_from_sender,
)


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    x = x - float(np.max(x))
    ex = np.exp(x)
    s = float(np.sum(ex))
    if s <= 0.0:
        return np.ones_like(x) / float(len(x))
    return ex / s


def _barrier_timeout(timeout_s: float) -> float:
    if float(timeout_s) > 0:
        return float(timeout_s)
    return 1e9


@dataclass
class ClassificationResult:
    """One digit class: class index and client-side softmax confidence from logits."""

    predictedValue: int
    likelihood: float


@dataclass
class InferenceResult:
    """Secure inference outcome: argmax digit, per-class confidences, raw logits."""

    m_predictedValue: int
    m_logits: List[float] = field(default_factory=list)
    m_classificationResults: List[ClassificationResult] = field(default_factory=list)


@dataclass
class SentraInferenceClientConfig:
    """Connection and crypto parameters; must match the running nodes."""

    weights_npz_path: str
    n_nodes: int = 3
    t: int = 1
    base_port: int = 8000
    host: str = "localhost"
    client_node_id: int = 0
    field_size: int = 2305843009213693951
    scale_factor: int = 65536
    seed: int = 42
    membership_epoch: int = 0
    eval_timeout: float = 0.0
    client_eval_batched_receive: bool = True
    barrier_timeout: float = 900.0
    topology_path: str = ""


class SentraInferenceClient:
    """
    Client-side MPC inference: secret-share one MNIST test image + weights,
    wait for nodes to evaluate, reconstruct logits, then softmax on client.
    """

    def __init__(self, config: SentraInferenceClientConfig) -> None:
        self._cfg = config
        if not str(config.weights_npz_path).strip():
            raise ValueError("weights_npz_path is required")

    def do_inference(self, input_image_index: int) -> InferenceResult:
        """
        Run secure inference on MNIST **test** sample ``input_image_index`` (0-based).

        Nodes must already be listening with matching ``field_size``, ``scale_factor``,
        ``t``, ``n_nodes``, ports, ``membership_epoch``, and must have been started with
        ``--receive-weights-shares-from-client`` (and inference flags) so they accept
        weight shares from this client before eval.
        """
        idx = int(input_image_index)
        if idx < 0:
            raise ValueError("input_image_index must be >= 0")

        (_, _), (x_test, _) = loadMNISTDataset()
        if idx >= int(len(x_test)):
            raise ValueError(f"input_image_index out of range: {idx} >= {len(x_test)}")

        x_vec = x_test[idx].reshape(784).astype(np.float32) / 255.0
        return self._run_one_sample(x_vec)

    def _run_one_sample(self, x_vec: np.ndarray) -> InferenceResult:
        cfg = self._cfg
        me = MembershipEpochScope(int(cfg.membership_epoch))
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        field_size = int(cfg.field_size)
        scale = int(cfg.scale_factor)
        if field_size <= 3:
            raise ValueError("field_size must be > 3")
        if scale <= 0:
            raise ValueError("scale_factor must be positive")

        n_train = 0
        n_test = 1
        feat_dim = 784
        cls_dim = 10

        x_train = np.zeros((0, 784), dtype=np.float32)
        y_train = np.zeros((0, 10), dtype=np.float32)
        x_test = np.asarray([np.asarray(x_vec, dtype=np.float32).reshape(784)], dtype=np.float32)
        y_test = keras.utils.to_categorical(np.asarray([0], dtype=np.int64), 10).astype(np.float32)

        if str(cfg.topology_path).strip():
            ct = load_client_topology(str(cfg.topology_path).strip())
            client_node_id = int(ct.party_id)
            n_nodes = int(ct.n_nodes)
            node_configs = ct.node_configs
            listen_port = int(ct.listen_port)
        else:
            client_node_id = int(cfg.client_node_id)
            n_nodes = int(cfg.n_nodes)
            node_configs = {i: {"host": cfg.host, "port": cfg.base_port + i} for i in range(1, n_nodes + 1)}
            listen_port = int(cfg.base_port + client_node_id)

        network = create_mpc_network(
            node_id=client_node_id,
            node_configs=node_configs,
            port=listen_port,
            membership_epoch=int(me.e),
        )
        shamir = ShamirSecretSharing(field_size)
        pss = PackedShamirSecretSharing(field_size)
        raw_packing = int(pss.max_packing_factor(int(n_nodes), int(cfg.t)))
        if raw_packing <= 0:
            raise ValueError(f"No valid PSS packing factor for n_nodes={n_nodes}, t={cfg.t}")
        n_active = int(n_nodes)
        pss_packing_factor = min(raw_packing, get_max_safe_packing_factor(int(cfg.t), n_active))

        npz_path = str(cfg.weights_npz_path).strip()
        w1, w2, b1, b2 = load_plain_weights_from_npz(npz_path)
        flat_mod_p = flatten_plain_weights_to_fixed_ints(
            w1=w1, w2=w2, b1=b1, b2=b2, scale=int(scale), p=int(field_size)
        )
        shamir_dist = ShamirSecretSharing(int(field_size))
        per_node = [[] for _ in range(int(n_nodes))]
        for secret in flat_mod_p:
            shares = shamir_dist.share(int(secret), int(n_nodes), int(cfg.t))
            for s in shares:
                per_node[int(s.node_id) - 1].append(int(s.y) % int(field_size))
        w_ctx = me.ctx("init_weights_from_npz/v1")
        for target in range(1, int(n_nodes) + 1):
            network.channel.send_vector(int(target), w_ctx, x=int(client_node_id), values=per_node[target - 1])

        meta_ctx = me.ctx("dataset/meta/v1")
        feat_stored = (int(feat_dim) + int(pss_packing_factor) - 1) // int(pss_packing_factor)
        cls_stored = (int(cls_dim) + int(pss_packing_factor) - 1) // int(pss_packing_factor)
        meta_payload = [n_train, n_test, feat_dim, cls_dim, 1, int(pss_packing_factor), int(feat_stored), int(cls_stored)]
        for target in range(1, n_nodes + 1):
            network.channel.send_vector(target, meta_ctx, x=target, values=meta_payload)

        for split_name, x_src, y_src in (("train", x_train, y_train), ("test", x_test, y_test)):
            n_split = int(len(x_src))
            for sample_idx in range(n_split):
                x_per_node = share_vector_for_all_nodes_pss(
                    x_src[sample_idx], n_nodes, cfg.t, field_size, pss, scale, int(pss_packing_factor)
                )
                y_per_node = share_vector_for_all_nodes_pss(
                    y_src[sample_idx], n_nodes, cfg.t, field_size, pss, scale, int(pss_packing_factor)
                )
                x_ctx = me.ctx(f"dataset/{split_name}/x/{sample_idx}")
                y_ctx = me.ctx(f"dataset/{split_name}/y/{sample_idx}")
                for target in range(1, n_nodes + 1):
                    network.channel.send_vector(target, x_ctx, x=target, values=x_per_node[target - 1])
                    network.channel.send_vector(target, y_ctx, x=target, values=y_per_node[target - 1])

        n_eval = 1
        eval_indices_vals = wait_for_vector_from_sender(
            network,
            me.ctx("client_eval_final/meta_indices"),
            sender_id=1,
            timeout_s=float(cfg.eval_timeout),
        )
        eval_indices = np.asarray(list(eval_indices_vals), dtype=np.int64)
        if eval_indices.size > n_eval:
            eval_indices = eval_indices[:n_eval]
        n_eval = int(eval_indices.size)

        p = int(field_size)
        n_classes = 10
        batched_receive = bool(getattr(cfg, "client_eval_batched_receive", True))
        if batched_receive:
            ctx = me.ctx("client_eval_final/logits/batched")
            by_sender = wait_for_party_vectors(network, ctx, int(cfg.t) + 1, float(cfg.eval_timeout))
            network.channel.clear_vector(ctx)
            share_vectors = party_vectors_to_share_vectors(by_sender)
            all_logits = reconstruct_batched_logits_from_parties(
                share_vectors, n_eval, n_classes, shamir, p
            )
            logits = all_logits[0] if all_logits else []
        else:
            for slot in range(n_eval):
                ctx = me.ctx(f"client_eval_final/logits/{slot}")
                by_sender = wait_for_party_vectors(network, ctx, int(cfg.t) + 1, float(cfg.eval_timeout))
                network.channel.clear_vector(ctx)
                share_vectors = party_vectors_to_share_vectors(by_sender)
                logits = reconstruct_logits_from_parties(share_vectors, n_classes, shamir, p)

        logits_np = np.asarray(logits, dtype=np.float64)
        probs = _stable_softmax(logits_np)
        pred = int(np.argmax(logits_np))
        results = [ClassificationResult(predictedValue=d, likelihood=float(probs[d])) for d in range(10)]

        network.barrier(me.barrier_tag("client_eval_final_done"), timeout=_barrier_timeout(float(cfg.barrier_timeout)))
        time.sleep(0.5)
        network.stop()

        return InferenceResult(
            m_predictedValue=pred,
            m_logits=[float(x) for x in logits_np.tolist()],
            m_classificationResults=results,
        )

    # Demonstrator-style name (camelCase)
    doInference = do_inference

"""
Batched Secure MNIST Training Script (MLP 784->128->10)
Uses SIMD matrix-matrix operations for major speedup.
"""

import sys
import os
import argparse
import numpy as np
import random
import time

_PROCESS_T0 = time.time()

from pathlib import Path
import json
from typing import Optional, Tuple, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.membership_epoch import (
    MembershipEpochScope,
    MutableMembershipEpochScope,
    record_membership_epoch_local_kvs,
)
from ml_training.packing_safety import check_packing_safety, PackingSafetyError, get_max_safe_packing_factor
from ml_training.mnist_mlp_batched import BatchedSecureMNISTMLP
from ml_training.beaver_triples import BeaverTripleGenerator, BeaverTriplePool, SecureMultiplier
from ml_training.secure_comparison import SecureComparator
from ml_training.secure_division import SecureDivider
from ml_training.secret_sharing import Share, ShamirSecretSharing, PackedShamirSecretSharing
from ml_training.packed_mpc_ops import PackedMPCOps
from ml_training.secure_softmax import SecureSoftmax
from ml_training.secure_comm import create_mpc_network
from ml_training.topology import load_node_topology
from ml_training.reconstruction import create_reconstruction_manager
from ml_training.beaver_triples import BeaverTripleDealerService
from ml_training.dpss_distributed import distributed_proactive_refresh_herzberg_vectorized
from ml_training.node_failure_detector import NodeFailureDetector
from ml_training.training_state_machine import TrainingStateMachine
from ml_training.batched_runtime_helpers import (
    handle_paused_training_recovery,
    run_epoch_eval_and_stability_checks,
    execute_training_batch,
)
from ml_training.sentra_kvs import put as kvs_put, get_batch as kvs_get_batch, key_data_sample_split
from ml_training.weight_versioning import put_weights_versioned
from ml_training.kvs import KVSCluster

from ml_training.util import loadMNISTDataset

from tensorflow import keras
def load_mnist_data(max_samples=None):
    (x_train, y_train), (x_test, y_test) = loadMNISTDataset()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    if max_samples:
        x_train = x_train[:max_samples]
        y_train = y_train[:max_samples]
        x_test = x_test[:max_samples]
        y_test = y_test[:max_samples]
        
    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train_oh), (x_test, y_test_oh)

def image_to_shares(image_flat, n_nodes, t, node_id, field_size, shamir, scale):
    shares_list = []
    for val in image_flat:
        val_int = int(val * scale) % field_size
        shares = shamir.share(val_int, n_nodes, t)
        shares_list.append(next(s for s in shares if s.node_id == node_id))
    return shares_list

def label_to_shares(label_oh, n_nodes, t, node_id, field_size, shamir, scale):
    shares_list = []
    for val in label_oh:
        val_int = int(val * scale) % field_size
        shares = shamir.share(val_int, n_nodes, t)
        shares_list.append(next(s for s in shares if s.node_id == node_id))
    return shares_list


def _share_vector_for_all_nodes(values, n_nodes, t, field_size, shamir, scale):
    per_node = [[] for _ in range(n_nodes)]
    for val in values:
        val_int = int(val * scale) % field_size
        shares = shamir.share(val_int, n_nodes, t)
        for s in shares:
            per_node[s.node_id - 1].append(int(s.y))
    return per_node

def _share_vector_for_all_nodes_pss(values, n_nodes, t, field_size, pss, scale, packing_factor):
    secrets = [int(v * scale) % field_size for v in values]
    chunks = pss.share_vector(secrets, n_nodes, t, packing_factor=int(packing_factor))
    per_node = [[] for _ in range(n_nodes)]
    for chunk in chunks:
        for s in chunk:
            per_node[int(s.node_id) - 1].append(int(s.y))
    return per_node


def _row_to_share_list(row_vals, node_id: int):
    return [Share(x=node_id, y=int(v), node_id=node_id) for v in row_vals]


def _packed_row_to_share_list(
    row_vals,
    *,
    node_id: int,
    original_len: int,
    packing_factor: int,
    packed_ops: PackedMPCOps,
    context_prefix: str,
    timeout_s: float = 120.0,
):
    if packed_ops is None:
        raise ValueError("packed_ops is required to unpack packed PSS rows")
    out = []
    remaining = int(original_len)
    for chunk_idx, packed_y in enumerate(row_vals):
        k_cur = min(int(packing_factor), max(0, remaining))
        if k_cur <= 0:
            break
        packed_share = Share(x=int(node_id), y=int(packed_y), node_id=int(node_id))
        lanes = packed_ops.unpack_packed_to_lane_shares(
            packed_share=packed_share,
            k=int(k_cur),
            context=f"{context_prefix}_c{chunk_idx}",
            timeout=float(timeout_s),
        )
        out.extend(lanes[:k_cur])
        remaining -= k_cur
    if len(out) != int(original_len):
        raise ValueError(
            f"Packed row unpack mismatch: expected {int(original_len)} shares, got {len(out)}"
        )
    return out


def _plaintext_vector_to_pss_lane_shares(
    values,
    *,
    node_id: int,
    n_nodes: int,
    t: int,
    field_size: int,
    scale: int,
    pss: PackedShamirSecretSharing,
    packing_factor: int,
    packed_ops: PackedMPCOps,
    context_prefix: str,
    timeout_s: float = 120.0,
):
    per_node_packed = _share_vector_for_all_nodes_pss(
        values, n_nodes, t, field_size, pss, scale, packing_factor
    )
    local_packed_row = per_node_packed[int(node_id) - 1]
    return _packed_row_to_share_list(
        local_packed_row,
        node_id=int(node_id),
        original_len=int(len(values)),
        packing_factor=int(packing_factor),
        packed_ops=packed_ops,
        context_prefix=context_prefix,
        timeout_s=float(timeout_s),
    )


def _build_lane_cache_from_packed_rows(
    *,
    packed_rows: np.ndarray,
    node_id: int,
    original_len: int,
    packing_factor: int,
    packed_ops: PackedMPCOps,
    context_prefix: str,
    timeout_s: float = 120.0,
    progress_every: int = 256,
) -> list:
    """
    Pre-unpack packed PSS rows into lane-share rows once, then reuse across epochs/evals.
    """
    cache = []
    total = int(len(packed_rows))
    for i in range(total):
        cache.append(
            _packed_row_to_share_list(
                packed_rows[int(i)],
                node_id=int(node_id),
                original_len=int(original_len),
                packing_factor=int(packing_factor),
                packed_ops=packed_ops,
                context_prefix=f"{context_prefix}_r{i}",
                timeout_s=float(timeout_s),
            )
        )
        if int(progress_every) > 0 and (i % int(progress_every) == 0):
            print(f"{context_prefix}: pre-unpacked row {i + 1}/{total}")
    return cache


def _get_or_unpack_cached_row(
    *,
    cache: Optional[dict],
    idx: int,
    packed_rows: np.ndarray,
    node_id: int,
    original_len: int,
    packing_factor: int,
    packed_ops: PackedMPCOps,
    context_prefix: str,
    timeout_s: float = 120.0,
) -> list:
    """
    Lazy packed-row unpack with memoization by sample index.
    """
    if cache is not None and int(idx) in cache:
        return cache[int(idx)]
    row = _packed_row_to_share_list(
        packed_rows[int(idx)],
        node_id=int(node_id),
        original_len=int(original_len),
        packing_factor=int(packing_factor),
        packed_ops=packed_ops,
        context_prefix=f"{context_prefix}_r{int(idx)}",
        timeout_s=float(timeout_s),
    )
    if cache is not None:
        cache[int(idx)] = row
    return row


def _get_or_unpack_cached_rows(
    *,
    cache: Optional[dict],
    indices: list,
    packed_rows: np.ndarray,
    node_id: int,
    original_len: int,
    packing_factor: int,
    packed_ops: PackedMPCOps,
    context_prefix: str,
    timeout_s: float = 120.0,
    metrics: Optional[dict] = None,
    metrics_key: str = "default",
) -> List[List[Share]]:
    """
    Batch version of lazy unpack: unpack only cache misses, in one amortized call.
    """
    idx_list = [int(i) for i in indices]
    out: List[Optional[List[Share]]] = [None] * len(idx_list)
    misses: List[int] = []
    miss_pos: List[int] = []
    for pos, idx in enumerate(idx_list):
        if cache is not None and idx in cache:
            out[pos] = cache[idx]
            if metrics is not None:
                m = metrics.setdefault(
                    metrics_key,
                    {"hits": 0, "misses": 0, "batch_calls": 0, "rows_unpacked": 0, "unpack_s": 0.0},
                )
                m["hits"] = int(m.get("hits", 0)) + 1
        else:
            misses.append(idx)
            miss_pos.append(pos)
    if misses:
        if metrics is not None:
            m = metrics.setdefault(
                metrics_key,
                {"hits": 0, "misses": 0, "batch_calls": 0, "rows_unpacked": 0, "unpack_s": 0.0},
            )
            m["misses"] = int(m.get("misses", 0)) + int(len(misses))
            m["batch_calls"] = int(m.get("batch_calls", 0)) + 1
        _t_unpack0 = time.time()
        miss_rows = [packed_rows[m] for m in misses]
        unpacked = packed_ops.unpack_packed_rows_to_lane_rows(
            packed_rows=miss_rows,
            original_len=int(original_len),
            packing_factor=int(packing_factor),
            context=f"{context_prefix}_batch",
            timeout=float(timeout_s),
        )
        if metrics is not None:
            m = metrics[metrics_key]
            m["rows_unpacked"] = int(m.get("rows_unpacked", 0)) + int(len(misses))
            m["unpack_s"] = float(m.get("unpack_s", 0.0)) + float(time.time() - _t_unpack0)
        for j, idx in enumerate(misses):
            row = unpacked[j]
            if cache is not None:
                cache[idx] = row
            out[miss_pos[j]] = row
    if any(r is None for r in out):
        raise ValueError("Batch lazy-unpack failed to populate all requested indices")
    return out  # type: ignore[return-value]


def _lazy_metrics_snapshot(metrics: Optional[dict], key: str) -> dict:
    m = (metrics or {}).get(key, {})
    return {
        "hits": int(m.get("hits", 0)),
        "misses": int(m.get("misses", 0)),
        "batch_calls": int(m.get("batch_calls", 0)),
        "rows_unpacked": int(m.get("rows_unpacked", 0)),
        "unpack_s": float(m.get("unpack_s", 0.0)),
    }


def _lazy_metrics_delta(metrics: Optional[dict], key: str, before: dict) -> dict:
    after = _lazy_metrics_snapshot(metrics, key)
    return {
        "hits": int(after["hits"] - int(before.get("hits", 0))),
        "misses": int(after["misses"] - int(before.get("misses", 0))),
        "batch_calls": int(after["batch_calls"] - int(before.get("batch_calls", 0))),
        "rows_unpacked": int(after["rows_unpacked"] - int(before.get("rows_unpacked", 0))),
        "unpack_s": float(after["unpack_s"] - float(before.get("unpack_s", 0.0))),
    }


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _prewarm_train_pss_unpack_cache(
    *,
    train_len: int,
    test_len: int = 0,
    args,
    network,
    me: MembershipEpochScope,
    train_x_shares,
    train_y_shares,
    test_x_shares=None,
    dataset_meta: dict,
    packed_ops: PackedMPCOps,
    train_x_lane_cache: dict,
    train_y_lane_cache: dict,
    test_x_lane_cache: Optional[dict] = None,
    lazy_unpack_metrics: dict,
    timeout_s: float = 900.0,
) -> None:
    """Unpack all train (and test X) PSS rows once before eval/training (avoids unpack races in Docker)."""
    original_len = int(dataset_meta.get("feat_dim", int(train_x_shares.shape[1])))
    cls_len = int(dataset_meta.get("cls_dim", int(train_y_shares.shape[1])))
    pss_k = int(dataset_meta.get("packing_factor", 1))
    all_train_idx = list(range(int(train_len)))
    if int(args.node_id) == 1:
        print(f"Pre-warming train PSS unpack cache ({train_len} samples)...")
    _get_or_unpack_cached_rows(
        cache=train_x_lane_cache,
        indices=all_train_idx,
        packed_rows=train_x_shares,
        node_id=int(args.node_id),
        original_len=original_len,
        packing_factor=pss_k,
        packed_ops=packed_ops,
        context_prefix=me.ctx("prewarm_train_x"),
        timeout_s=float(timeout_s),
        metrics=lazy_unpack_metrics,
        metrics_key="train_x",
    )
    _get_or_unpack_cached_rows(
        cache=train_y_lane_cache,
        indices=all_train_idx,
        packed_rows=train_y_shares,
        node_id=int(args.node_id),
        original_len=cls_len,
        packing_factor=pss_k,
        packed_ops=packed_ops,
        context_prefix=me.ctx("prewarm_train_y"),
        timeout_s=float(timeout_s),
        metrics=lazy_unpack_metrics,
        metrics_key="train_y",
    )
    if (
        int(test_len) > 0
        and test_x_shares is not None
        and test_x_lane_cache is not None
    ):
        all_test_idx = list(range(int(test_len)))
        if int(args.node_id) == 1:
            print(f"Pre-warming test PSS unpack cache ({test_len} samples)...")
        _get_or_unpack_cached_rows(
            cache=test_x_lane_cache,
            indices=all_test_idx,
            packed_rows=test_x_shares,
            node_id=int(args.node_id),
            original_len=original_len,
            packing_factor=pss_k,
            packed_ops=packed_ops,
            context_prefix=me.ctx("prewarm_test_x"),
            timeout_s=float(timeout_s),
            metrics=lazy_unpack_metrics,
            metrics_key="eval_x",
        )
    if network is not None:
        network.barrier(me.ctx("prewarm_train_unpack_done"), timeout=float(timeout_s))
    if int(args.node_id) == 1:
        print("Train/test PSS pre-warm complete.")


def _share_vector_for_all_nodes_pss(values, n_nodes, t, field_size, pss, scale, packing_factor):
    secrets = [int(v * scale) % field_size for v in values]
    chunks = pss.share_vector(secrets, n_nodes, t, packing_factor=int(packing_factor))
    per_node = [[] for _ in range(n_nodes)]
    for chunk in chunks:
        for s in chunk:
            per_node[int(s.node_id) - 1].append(int(s.y))
    return per_node


def _row_to_share_list(row_vals, node_id: int):
    return [Share(x=node_id, y=int(v), node_id=node_id) for v in row_vals]


def _packed_row_to_share_list(
    row_vals,
    *,
    node_id: int,
    original_len: int,
    packing_factor: int,
    packed_ops: PackedMPCOps,
    context_prefix: str,
    timeout_s: float = 120.0,
):
    if packed_ops is None:
        raise ValueError("packed_ops is required to unpack packed PSS rows")
    out = []
    remaining = int(original_len)
    for chunk_idx, packed_y in enumerate(row_vals):
        k_cur = min(int(packing_factor), max(0, remaining))
        if k_cur <= 0:
            break
        packed_share = Share(x=int(node_id), y=int(packed_y), node_id=int(node_id))
        lanes = packed_ops.unpack_packed_to_lane_shares(
            packed_share=packed_share,
            k=int(k_cur),
            context=f"{context_prefix}_c{chunk_idx}",
            timeout=float(timeout_s),
        )
        out.extend(lanes[:k_cur])
        remaining -= k_cur
    if len(out) != int(original_len):
        raise ValueError(
            f"Packed row unpack mismatch: expected {int(original_len)} shares, got {len(out)}"
        )
    return out


def _plaintext_vector_to_pss_lane_shares(
    values,
    *,
    node_id: int,
    n_nodes: int,
    t: int,
    field_size: int,
    scale: int,
    pss: PackedShamirSecretSharing,
    packing_factor: int,
    packed_ops: PackedMPCOps,
    context_prefix: str,
    timeout_s: float = 120.0,
):
    per_node_packed = _share_vector_for_all_nodes_pss(
        values, n_nodes, t, field_size, pss, scale, packing_factor
    )
    local_packed_row = per_node_packed[int(node_id) - 1]
    return _packed_row_to_share_list(
        local_packed_row,
        node_id=int(node_id),
        original_len=int(len(values)),
        packing_factor=int(packing_factor),
        packed_ops=packed_ops,
        context_prefix=context_prefix,
        timeout_s=float(timeout_s),
    )


def _build_lane_cache_from_packed_rows(
    *,
    packed_rows: np.ndarray,
    node_id: int,
    original_len: int,
    packing_factor: int,
    packed_ops: PackedMPCOps,
    context_prefix: str,
    timeout_s: float = 120.0,
    progress_every: int = 256,
) -> list:
    """
    Pre-unpack packed PSS rows into lane-share rows once, then reuse across epochs/evals.
    """
    cache = []
    total = int(len(packed_rows))
    for i in range(total):
        cache.append(
            _packed_row_to_share_list(
                packed_rows[int(i)],
                node_id=int(node_id),
                original_len=int(original_len),
                packing_factor=int(packing_factor),
                packed_ops=packed_ops,
                context_prefix=f"{context_prefix}_r{i}",
                timeout_s=float(timeout_s),
            )
        )
        if int(progress_every) > 0 and (i % int(progress_every) == 0):
            print(f"{context_prefix}: pre-unpacked row {i + 1}/{total}")
    return cache


def _get_or_unpack_cached_row(
    *,
    cache: Optional[dict],
    idx: int,
    packed_rows: np.ndarray,
    node_id: int,
    original_len: int,
    packing_factor: int,
    packed_ops: PackedMPCOps,
    context_prefix: str,
    timeout_s: float = 120.0,
) -> list:
    """
    Lazy packed-row unpack with memoization by sample index.
    """
    if cache is not None and int(idx) in cache:
        return cache[int(idx)]
    row = _packed_row_to_share_list(
        packed_rows[int(idx)],
        node_id=int(node_id),
        original_len=int(original_len),
        packing_factor=int(packing_factor),
        packed_ops=packed_ops,
        context_prefix=f"{context_prefix}_r{int(idx)}",
        timeout_s=float(timeout_s),
    )
    if cache is not None:
        cache[int(idx)] = row
    return row


def _get_or_unpack_cached_rows(
    *,
    cache: Optional[dict],
    indices: list,
    packed_rows: np.ndarray,
    node_id: int,
    original_len: int,
    packing_factor: int,
    packed_ops: PackedMPCOps,
    context_prefix: str,
    timeout_s: float = 120.0,
    metrics: Optional[dict] = None,
    metrics_key: str = "default",
) -> List[List[Share]]:
    """
    Batch version of lazy unpack: unpack only cache misses, in one amortized call.
    """
    idx_list = [int(i) for i in indices]
    out: List[Optional[List[Share]]] = [None] * len(idx_list)
    misses: List[int] = []
    miss_pos: List[int] = []
    for pos, idx in enumerate(idx_list):
        if cache is not None and idx in cache:
            out[pos] = cache[idx]
            if metrics is not None:
                m = metrics.setdefault(
                    metrics_key,
                    {"hits": 0, "misses": 0, "batch_calls": 0, "rows_unpacked": 0, "unpack_s": 0.0},
                )
                m["hits"] = int(m.get("hits", 0)) + 1
        else:
            misses.append(idx)
            miss_pos.append(pos)
    if misses:
        if metrics is not None:
            m = metrics.setdefault(
                metrics_key,
                {"hits": 0, "misses": 0, "batch_calls": 0, "rows_unpacked": 0, "unpack_s": 0.0},
            )
            m["misses"] = int(m.get("misses", 0)) + int(len(misses))
            m["batch_calls"] = int(m.get("batch_calls", 0)) + 1
        _t_unpack0 = time.time()
        miss_rows = [packed_rows[m] for m in misses]
        unpacked = packed_ops.unpack_packed_rows_to_lane_rows(
            packed_rows=miss_rows,
            original_len=int(original_len),
            packing_factor=int(packing_factor),
            context=f"{context_prefix}_batch",
            timeout=float(timeout_s),
        )
        if metrics is not None:
            m = metrics[metrics_key]
            m["rows_unpacked"] = int(m.get("rows_unpacked", 0)) + int(len(misses))
            m["unpack_s"] = float(m.get("unpack_s", 0.0)) + float(time.time() - _t_unpack0)
        for j, idx in enumerate(misses):
            row = unpacked[j]
            if cache is not None:
                cache[idx] = row
            out[miss_pos[j]] = row
    if any(r is None for r in out):
        raise ValueError("Batch lazy-unpack failed to populate all requested indices")
    return out  # type: ignore[return-value]


def _lazy_metrics_snapshot(metrics: Optional[dict], key: str) -> dict:
    m = (metrics or {}).get(key, {})
    return {
        "hits": int(m.get("hits", 0)),
        "misses": int(m.get("misses", 0)),
        "batch_calls": int(m.get("batch_calls", 0)),
        "rows_unpacked": int(m.get("rows_unpacked", 0)),
        "unpack_s": float(m.get("unpack_s", 0.0)),
    }


def _lazy_metrics_delta(metrics: Optional[dict], key: str, before: dict) -> dict:
    after = _lazy_metrics_snapshot(metrics, key)
    return {
        "hits": int(after["hits"] - int(before.get("hits", 0))),
        "misses": int(after["misses"] - int(before.get("misses", 0))),
        "batch_calls": int(after["batch_calls"] - int(before.get("batch_calls", 0))),
        "rows_unpacked": int(after["rows_unpacked"] - int(before.get("rows_unpacked", 0))),
        "unpack_s": float(after["unpack_s"] - float(before.get("unpack_s", 0.0))),
    }


def _wait_for_vector_from_sender(network, context: str, sender_id: int, timeout_s: float):
    start = time.time()
    while True:
        by_sender = network.channel.get_received_vector(context)
        if sender_id in by_sender:
            values = by_sender[sender_id]["values"]
            network.channel.clear_vector(context)
            return values
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for dataset context '{context}' from node {sender_id}")
        time.sleep(0.01)


def _to_uint64_array(values, expected_len: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.uint64)
    if int(arr.size) != int(expected_len):
        raise ValueError(f"Dataset share length mismatch: expected {expected_len}, got {int(arr.size)}")
    return arr


def prepare_distributed_dataset_shares(
    *,
    network,
    node_id: int,
    n_nodes: int,
    owner_node_id: int,
    t: int,
    field_size: int,
    shamir,
    pss,
    scale: int,
    mnist_samples: Optional[int],
    timeout_s: float,
    pss_packing_factor: int,
    use_pss_storage: bool,
    me: MembershipEpochScope,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    Owner node loads MNIST, secret-shares it, and distributes each node's shares.
    Non-owner nodes receive only their local share tensors.
    Returns:
        train_x_shares, train_y_shares, test_x_shares, test_y_shares, x_test_plain_or_none, y_test_plain_or_none, meta
    """
    meta_ctx = me.ctx("dataset/meta/v1")
    if int(node_id) == int(owner_node_id):
        (x_train, y_train), (x_test, y_test) = load_mnist_data(mnist_samples)
        n_train = int(len(x_train))
        n_test = int(len(x_test))
        feat_dim = int(x_train.shape[1])
        cls_dim = int(y_train.shape[1])
        pss_enabled = bool(use_pss_storage)
        pss_k = int(pss_packing_factor) if pss_enabled else 1
        feat_stored = (feat_dim + pss_k - 1) // pss_k if pss_enabled else feat_dim
        cls_stored = (cls_dim + pss_k - 1) // pss_k if pss_enabled else cls_dim

        local_train_x = np.zeros((n_train, feat_stored), dtype=np.uint64)
        local_train_y = np.zeros((n_train, cls_stored), dtype=np.uint64)
        local_test_x = np.zeros((n_test, feat_stored), dtype=np.uint64)
        local_test_y = np.zeros((n_test, cls_stored), dtype=np.uint64)

        meta_payload = [
            n_train,
            n_test,
            feat_dim,
            cls_dim,
            1 if pss_enabled else 0,
            pss_k,
            feat_stored,
            cls_stored,
        ]
        for target in range(1, n_nodes + 1):
            if target == node_id:
                continue
            network.channel.send_vector(target, meta_ctx, x=target, values=meta_payload)

        for split_name, x_src, y_src, x_dst, y_dst in (
            ("train", x_train, y_train, local_train_x, local_train_y),
            ("test", x_test, y_test, local_test_x, local_test_y),
        ):
            n_split = int(len(x_src))
            for idx in range(n_split):
                if pss_enabled:
                    x_per_node = _share_vector_for_all_nodes_pss(
                        x_src[idx], n_nodes, t, field_size, pss, scale, pss_k
                    )
                    y_per_node = _share_vector_for_all_nodes_pss(
                        y_src[idx], n_nodes, t, field_size, pss, scale, pss_k
                    )
                else:
                    x_per_node = _share_vector_for_all_nodes(
                        x_src[idx], n_nodes, t, field_size, shamir, scale
                    )
                    y_per_node = _share_vector_for_all_nodes(
                        y_src[idx], n_nodes, t, field_size, shamir, scale
                    )
                x_dst[idx, :] = np.asarray(x_per_node[node_id - 1], dtype=np.uint64)
                y_dst[idx, :] = np.asarray(y_per_node[node_id - 1], dtype=np.uint64)

                x_ctx = me.ctx(f"dataset/{split_name}/x/{idx}")
                y_ctx = me.ctx(f"dataset/{split_name}/y/{idx}")
                for target in range(1, n_nodes + 1):
                    if target == node_id:
                        continue
                    network.channel.send_vector(target, x_ctx, x=target, values=x_per_node[target - 1])
                    network.channel.send_vector(target, y_ctx, x=target, values=y_per_node[target - 1])

                if idx % 512 == 0:
                    print(f"Owner node {node_id}: shared {split_name} sample {idx + 1}/{n_split}")

        network.barrier(me.barrier_tag("dataset_distributed_v1"), timeout=timeout_s)
        meta = {
            "use_pss_storage": bool(pss_enabled),
            "packing_factor": int(pss_k),
            "feat_dim": int(feat_dim),
            "cls_dim": int(cls_dim),
            "feat_stored_len": int(feat_stored),
            "cls_stored_len": int(cls_stored),
            "membership_epoch": int(me.e),
        }
        return local_train_x, local_train_y, local_test_x, local_test_y, x_test, y_test, meta

    meta_vals = _wait_for_vector_from_sender(network, meta_ctx, int(owner_node_id), timeout_s)
    meta_arr = np.asarray(list(meta_vals), dtype=np.int64)
    if meta_arr.size not in (4, 8):
        raise ValueError(f"Invalid dataset meta from owner node {owner_node_id}: {meta_arr}")
    if meta_arr.size == 8:
        n_train, n_test, feat_dim, cls_dim, pss_flag, pss_k, feat_stored, cls_stored = [
            int(v) for v in meta_arr.tolist()
        ]
        use_pss_storage = bool(pss_flag)
    else:
        n_train, n_test, feat_dim, cls_dim = [int(v) for v in meta_arr.tolist()]
        use_pss_storage = False
        pss_k = 1
        feat_stored = feat_dim
        cls_stored = cls_dim

    local_train_x = np.zeros((n_train, feat_stored), dtype=np.uint64)
    local_train_y = np.zeros((n_train, cls_stored), dtype=np.uint64)
    local_test_x = np.zeros((n_test, feat_stored), dtype=np.uint64)
    local_test_y = np.zeros((n_test, cls_stored), dtype=np.uint64)

    for split_name, n_split, feat_len, cls_len, x_dst, y_dst in (
        ("train", n_train, feat_stored, cls_stored, local_train_x, local_train_y),
        ("test", n_test, feat_stored, cls_stored, local_test_x, local_test_y),
    ):
        for idx in range(n_split):
            x_ctx = me.ctx(f"dataset/{split_name}/x/{idx}")
            y_ctx = me.ctx(f"dataset/{split_name}/y/{idx}")
            x_vals = _wait_for_vector_from_sender(network, x_ctx, int(owner_node_id), timeout_s)
            y_vals = _wait_for_vector_from_sender(network, y_ctx, int(owner_node_id), timeout_s)
            x_dst[idx, :] = _to_uint64_array(x_vals, feat_len)
            y_dst[idx, :] = _to_uint64_array(y_vals, cls_len)
            if idx % 512 == 0:
                print(f"Node {node_id}: received {split_name} share sample {idx + 1}/{n_split}")

    network.barrier(me.barrier_tag("dataset_distributed_v1"), timeout=timeout_s)
    meta = {
        "use_pss_storage": bool(use_pss_storage),
        "packing_factor": int(pss_k),
        "feat_dim": int(feat_dim),
        "cls_dim": int(cls_dim),
        "feat_stored_len": int(feat_stored),
        "cls_stored_len": int(cls_stored),
        "membership_epoch": int(me.e),
    }
    return local_train_x, local_train_y, local_test_x, local_test_y, None, None, meta


def receive_dataset_shares_from_external_source(
    *,
    network,
    source_node_id: int,
    timeout_s: float,
    me: MembershipEpochScope,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Receive pre-shared dataset tensors from an external source (e.g. client node_id=0).
    No node loads raw MNIST in this path.
    """
    meta_ctx = me.ctx("dataset/meta/v1")
    meta_vals = _wait_for_vector_from_sender(network, meta_ctx, int(source_node_id), timeout_s)
    meta_arr = np.asarray(list(meta_vals), dtype=np.int64)
    if meta_arr.size not in (4, 8):
        raise ValueError(f"Invalid dataset meta from source node {source_node_id}: {meta_arr}")
    if meta_arr.size == 8:
        n_train, n_test, feat_dim, cls_dim, pss_flag, pss_k, feat_stored, cls_stored = [
            int(v) for v in meta_arr.tolist()
        ]
        use_pss_storage = bool(pss_flag)
    else:
        n_train, n_test, feat_dim, cls_dim = [int(v) for v in meta_arr.tolist()]
        use_pss_storage = False
        pss_k = 1
        feat_stored = feat_dim
        cls_stored = cls_dim

    local_train_x = np.zeros((n_train, feat_stored), dtype=np.uint64)
    local_train_y = np.zeros((n_train, cls_stored), dtype=np.uint64)
    local_test_x = np.zeros((n_test, feat_stored), dtype=np.uint64)
    local_test_y = np.zeros((n_test, cls_stored), dtype=np.uint64)

    for split_name, n_split, feat_len, cls_len, x_dst, y_dst in (
        ("train", n_train, feat_stored, cls_stored, local_train_x, local_train_y),
        ("test", n_test, feat_stored, cls_stored, local_test_x, local_test_y),
    ):
        for idx in range(n_split):
            x_ctx = me.ctx(f"dataset/{split_name}/x/{idx}")
            y_ctx = me.ctx(f"dataset/{split_name}/y/{idx}")
            x_vals = _wait_for_vector_from_sender(network, x_ctx, int(source_node_id), timeout_s)
            y_vals = _wait_for_vector_from_sender(network, y_ctx, int(source_node_id), timeout_s)
            x_dst[idx, :] = _to_uint64_array(x_vals, feat_len)
            y_dst[idx, :] = _to_uint64_array(y_vals, cls_len)
            if idx % 512 == 0:
                print(f"Node received {split_name} share sample {idx + 1}/{n_split} from source {source_node_id}")

    network.barrier(me.barrier_tag("dataset_distributed_v1"), timeout=timeout_s)
    meta = {
        "use_pss_storage": bool(use_pss_storage),
        "packing_factor": int(pss_k),
        "feat_dim": int(feat_dim),
        "cls_dim": int(cls_dim),
        "feat_stored_len": int(feat_stored),
        "cls_stored_len": int(cls_stored),
        "membership_epoch": int(me.e),
    }
    return local_train_x, local_train_y, local_test_x, local_test_y, meta


def _populate_dataset_kvs(
    cluster: KVSCluster,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    vD: int = 1,
) -> None:
    """Store dataset shares in KVS with version vD (data/train/{i}, data/test/{i})."""
    for i in range(len(train_x)):
        key = key_data_sample_split("train", i)
        kvs_put(cluster, key, (train_x[i].copy(), train_y[i].copy()), vD, quorum_size=1)
    for i in range(len(test_x)):
        key = key_data_sample_split("test", i)
        kvs_put(cluster, key, (test_x[i].copy(), test_y[i].copy()), vD, quorum_size=1)


def send_inference_shares_to_client(
    *,
    model,
    weights,
    test_x_shares: np.ndarray,
    node_id: int,
    client_node_id: int,
    network,
    client_connect_host: str,
    client_connect_port: int,
    n_samples: int,
    seed: int,
    context_prefix: str = "client_eval_final",
    eval_sync_tag: str = "client_eval_final_done",
    batched_receive: bool = False,
    dataset_meta: Optional[dict] = None,
    packed_ops: Optional[PackedMPCOps] = None,
    test_x_lane_cache: Optional[dict] = None,
    unpack_metrics: Optional[dict] = None,
):
    total = int(len(test_x_shares))
    if total <= 0 or int(n_samples) <= 0:
        return
    n_eval = min(int(n_samples), total)
    rng = np.random.default_rng(int(seed) + 777)
    eval_indices = rng.choice(total, size=n_eval, replace=False)

    if int(client_node_id) not in network.channel.connections:
        ok = network.channel.connect_to_node(
            int(client_node_id), str(client_connect_host), int(client_connect_port)
        )
        if not ok:
            raise RuntimeError(f"Unable to connect to client node {client_node_id} for final eval upload")

    # Send deterministic eval indices once from node 1 so client can map labels.
    if int(node_id) == 1:
        network.channel.send_vector(
            int(client_node_id),
            f"{context_prefix}/meta_indices",
            x=int(node_id),
            values=[int(v) for v in eval_indices.tolist()],
        )

    use_pss_storage = bool((dataset_meta or {}).get("use_pss_storage", False))
    feat_dim = int((dataset_meta or {}).get("feat_dim", int(test_x_shares.shape[1])))
    pss_k = int((dataset_meta or {}).get("packing_factor", 1))
    x_shares_cols = []
    if test_x_lane_cache is not None:
        x_shares_cols = _get_or_unpack_cached_rows(
            cache=test_x_lane_cache,
            indices=[int(i) for i in eval_indices.tolist()],
            packed_rows=test_x_shares,
            node_id=int(node_id),
            original_len=int(feat_dim),
            packing_factor=int(pss_k),
            packed_ops=packed_ops,
            context_prefix=f"{context_prefix}_lazy_test_x",
            timeout_s=120.0,
            metrics=unpack_metrics,
            metrics_key="client_eval_x",
        )
    else:
        for slot_idx, i in enumerate(eval_indices):
            row = test_x_shares[int(i)]
            if use_pss_storage:
                x_shares_cols.append(
                    _packed_row_to_share_list(
                        row,
                        node_id=int(node_id),
                        original_len=int(feat_dim),
                        packing_factor=int(pss_k),
                        packed_ops=packed_ops,
                        context_prefix=f"{context_prefix}_xslot{slot_idx}",
                        timeout_s=120.0,
                    )
                )
            else:
                x_shares_cols.append(_row_to_share_list(row, node_id))

    logits_cols, _ = model.forward_pass_batched(
        x_shares_cols, weights, node_id, context=context_prefix, open_relu=True, reconstruction_manager=None
    )

    if bool(batched_receive):
        flat = [int(s.y) for col in logits_cols for s in col]
        network.channel.send_vector(
            int(client_node_id),
            f"{context_prefix}/logits/batched",
            x=int(node_id),
            values=flat,
        )
    else:
        for slot_idx, col in enumerate(logits_cols):
            vec = [int(s.y) for s in col]
            network.channel.send_vector(
                int(client_node_id),
                f"{context_prefix}/logits/{slot_idx}",
                x=int(node_id),
                values=vec,
            )

    # Notify client that this node finished uploading final eval logits.
    network.channel.send_message(
        int(client_node_id),
        "sync",
        {"tag": str(eval_sync_tag)},
    )


def evaluate_model(
    model,
    weights,
    x_test,
    y_test,
    node_id,
    n_nodes,
    t,
    shamir,
    field_size,
    scale,
    reconstruction,
    n_test_samples=100,
    context_prefix="eval",
    fixed_indices=None,
    x_test_shared=None,
    dataset_meta: Optional[dict] = None,
    packed_ops: Optional[PackedMPCOps] = None,
    pss: Optional[PackedShamirSecretSharing] = None,
    pss_packing_factor: Optional[int] = None,
    x_test_lane_cache: Optional[dict] = None,
    unpack_metrics: Optional[dict] = None,
):
    print(f"Evaluating on {n_test_samples} test samples (batched)...")
    if x_test_shared is not None:
        total_samples = int(len(x_test_shared))
    elif x_test is not None:
        total_samples = int(len(x_test))
    else:
        raise ValueError("Either x_test or x_test_shared must be provided for evaluation")

    if fixed_indices is not None and len(fixed_indices) > 0:
        indices = np.asarray(fixed_indices, dtype=np.int64)[:n_test_samples]
    else:
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        indices = indices[:n_test_samples]
    
    x_shares_cols = []
    if x_test_shared is not None:
        use_pss_storage = bool((dataset_meta or {}).get("use_pss_storage", False))
        feat_dim = int((dataset_meta or {}).get("feat_dim", int(x_test_shared.shape[1])))
        pss_k = int((dataset_meta or {}).get("packing_factor", 1))
        if x_test_lane_cache is not None:
            x_shares_cols = _get_or_unpack_cached_rows(
                cache=x_test_lane_cache,
                indices=[int(i) for i in indices.tolist()],
                packed_rows=x_test_shared,
                node_id=int(node_id),
                original_len=int(feat_dim),
                packing_factor=int(pss_k),
                packed_ops=packed_ops,
                context_prefix=f"{context_prefix}_lazy_eval_x",
                timeout_s=120.0,
                metrics=unpack_metrics,
                metrics_key="eval_x",
            )
        else:
            for i in indices:
                row = x_test_shared[int(i)]
                if use_pss_storage:
                    x_shares_cols.append(
                        _packed_row_to_share_list(
                            row,
                            node_id=int(node_id),
                            original_len=int(feat_dim),
                            packing_factor=int(pss_k),
                            packed_ops=packed_ops,
                            context_prefix=f"{context_prefix}_x_{int(i)}",
                            timeout_s=120.0,
                        )
                    )
                else:
                    x_shares_cols.append(_row_to_share_list(row, node_id))
    else:
        use_pss_eval = bool(packed_ops is not None and pss is not None and int(n_nodes) > 1)
        for i in indices:
            if use_pss_eval:
                x_shares_cols.append(
                    _plaintext_vector_to_pss_lane_shares(
                        x_test[i],
                        node_id=int(node_id),
                        n_nodes=int(n_nodes),
                        t=int(t),
                        field_size=int(field_size),
                        scale=int(scale),
                        pss=pss,
                        packing_factor=int(pss_packing_factor or 1),
                        packed_ops=packed_ops,
                        context_prefix=f"{context_prefix}_plain_x_{int(i)}",
                        timeout_s=120.0,
                    )
                )
            else:
                x_shares_cols.append(image_to_shares(x_test[i], n_nodes, t, node_id, field_size, shamir, scale))
        
    # Forward Pass Batched
    if reconstruction is not None and int(n_nodes) > 1:
        reconstruction.network.barrier(f"{context_prefix}_fwd_ready", timeout=120.0)
    logits_cols, _ = model.forward_pass_batched(
        x_shares_cols,
        weights,
        node_id,
        context=context_prefix,
        open_relu=True,
        reconstruction_manager=reconstruction,
    )
    if reconstruction is not None and int(n_nodes) > 1:
        reconstruction.network.barrier(f"{context_prefix}_fwd_done", timeout=120.0)
    correct = 0
    loss_sum = 0.0
    max_abs_logit = 0.0
    mean_abs_logit_sum = 0.0
    mean_entropy_sum = 0.0
    if reconstruction:
        for i, col in enumerate(logits_cols):
            logits_val = []
            for j, share in enumerate(col):
                # Context must be globally unique per evaluation round to prevent
                # stale/mixed reconstructions across epochs.
                eval_ctx = f"{context_prefix}_logit_{i}_{j}"
                val = reconstruction.get_reconstructed_value([share], eval_ctx, use_cache=False)
                logits_val.append(val)
            if node_id == 1:
                logits_float = []
                for v in logits_val:
                    if v > field_size / 2:
                        v = v - field_size
                    logits_float.append(v / scale)
                if i == 0:
                    print(f"DEBUG EVAL LOGITS: {logits_float}")

                logits_np = np.asarray(logits_float, dtype=np.float64)
                abs_logits = np.abs(logits_np)
                max_abs_logit = max(max_abs_logit, float(np.max(abs_logits)))
                mean_abs_logit_sum += float(np.mean(abs_logits))
                logits_np = logits_np - np.max(logits_np)
                exp_l = np.exp(logits_np)
                probs = exp_l / np.maximum(np.sum(exp_l), 1e-12)
                entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
                mean_entropy_sum += entropy
                if y_test is None:
                    continue
                y_true = np.asarray(y_test[int(indices[i])], dtype=np.float64)
                loss_sum += float(-np.sum(y_true * np.log(np.maximum(probs, 1e-12))))

                pred = np.argmax(logits_float)
                true_label = np.argmax(y_test[indices[i]])
                if pred == true_label:
                    correct += 1
        if node_id == 1:
            denom = max(1, int(n_test_samples))
            diagnostics = {
                "max_abs_logit": float(max_abs_logit),
                "mean_abs_logit": float(mean_abs_logit_sum / denom),
                "mean_entropy": float(mean_entropy_sum / denom),
            }
            return correct / denom, loss_sum / denom, diagnostics
    return 0.0, 0.0, {"max_abs_logit": 0.0, "mean_abs_logit": 0.0, "mean_entropy": 0.0}


def _weights_to_flat_ys(weights: list) -> List[int]:
    """Flatten weight shares to list of y values (same order as _shares_to_flat_mod_p)."""
    vals: List[int] = []
    for layer in weights:
        if isinstance(layer, list) and layer and isinstance(layer[0], list):
            for row in layer:
                for s in row:
                    vals.append(int(s.y))
        else:
            for s in layer:
                vals.append(int(s.y))
    return vals


def _flat_ys_to_weights(flat_ys: List[int], node_id: int, shapes: List[Tuple]) -> list:
    """Reconstruct weight structure from flat y values and model weight shapes."""
    idx = 0
    out = []
    for shape in shapes:
        if len(shape) == 2:  # matrix: (rows, cols)
            rows, cols = shape
            layer = []
            for _ in range(rows):
                row = []
                for _ in range(cols):
                    y = flat_ys[idx]
                    idx += 1
                    row.append(Share(x=node_id, y=y, node_id=node_id))
                layer.append(row)
            out.append(layer)
        else:
            layer = []
            for _ in range(shape[0]):
                y = flat_ys[idx]
                idx += 1
                layer.append(Share(x=node_id, y=y, node_id=node_id))
            out.append(layer)
    return out


def _load_plain_weights_from_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(str(npz_path))
    w1 = np.asarray(data["w1"], dtype=np.float64)
    w2 = np.asarray(data["w2"], dtype=np.float64)
    b1 = np.asarray(data["b1"], dtype=np.float64)
    b2 = np.asarray(data["b2"], dtype=np.float64)
    return w1, w2, b1, b2


def _flatten_plain_weights_to_fixed_ints(
    *,
    w1: np.ndarray,
    w2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    scale: int,
    p: int,
    expected_shapes: List[Tuple],
) -> List[int]:
    expected_count = 0
    for shape in expected_shapes:
        if len(shape) == 2:
            expected_count += int(shape[0]) * int(shape[1])
        else:
            expected_count += int(shape[0])
    flat_float = np.concatenate([w1.reshape(-1), w2.reshape(-1), b1.reshape(-1), b2.reshape(-1)])
    if int(flat_float.size) != int(expected_count):
        raise ValueError(
            f"NPZ parameter count mismatch: expected {expected_count}, got {int(flat_float.size)} "
            f"(w1={w1.shape}, w2={w2.shape}, b1={b1.shape}, b2={b2.shape})"
        )
    flat_int = np.rint(flat_float * float(scale)).astype(np.int64, copy=False)
    return [int(v) % int(p) for v in flat_int.tolist()]


def _distribute_or_receive_weight_shares_from_npz(
    *,
    weights_npz_path: str,
    model,
    node_id: int,
    n_nodes: int,
    t: int,
    shamir,
    network,
    field_size: int,
    scale: int,
    me: MembershipEpochScope,
    dealer_node_id: int,
    timeout_s: float,
) -> list:
    ctx = me.ctx("init_weights_from_npz/v1")
    shapes = model.get_weight_shapes()
    dealer_node_id = int(dealer_node_id)
    if dealer_node_id < 1 or dealer_node_id > int(n_nodes):
        raise ValueError(f"Invalid dealer node id for NPZ weight distribution: {dealer_node_id}")

    if int(node_id) == dealer_node_id:
        w1, w2, b1, b2 = _load_plain_weights_from_npz(weights_npz_path)
        flat_mod_p = _flatten_plain_weights_to_fixed_ints(
            w1=w1,
            w2=w2,
            b1=b1,
            b2=b2,
            scale=int(scale),
            p=int(field_size),
            expected_shapes=shapes,
        )
        per_node: List[List[int]] = [[] for _ in range(int(n_nodes))]
        for secret in flat_mod_p:
            shares = shamir.share(int(secret), int(n_nodes), int(t))
            for s in shares:
                per_node[int(s.node_id) - 1].append(int(s.y) % int(field_size))
        for target in range(1, int(n_nodes) + 1):
            if int(target) == int(node_id):
                continue
            network.channel.send_vector(int(target), ctx, x=int(dealer_node_id), values=per_node[target - 1])
        local_ys = per_node[int(node_id) - 1]
        print(f"Node {node_id}: distributed secure weight shares from {weights_npz_path} ...")
    else:
        print(f"Node {node_id}: receiving secure weight shares from node {dealer_node_id} ...")
        local_ys = [int(v) for v in _wait_for_vector_from_sender(network, ctx, int(dealer_node_id), float(timeout_s))]
        print(f"Node {node_id}: weight shares received.")

    network.barrier(me.barrier_tag("init_weights_from_npz_done"), timeout=float(timeout_s))
    return _flat_ys_to_weights(local_ys, int(node_id), shapes)


def _receive_weight_shares_from_client(
    *,
    model,
    node_id: int,
    source_node_id: int,
    network,
    me: MembershipEpochScope,
    timeout_s: float,
) -> list:
    shapes = model.get_weight_shapes()
    ctx = me.ctx("init_weights_from_npz/v1")
    print(f"Node {node_id}: receiving secure weight shares from client node {source_node_id} ...")
    local_ys = [int(v) for v in _wait_for_vector_from_sender(network, ctx, int(source_node_id), float(timeout_s))]
    print(f"Node {node_id}: client weight shares received.")
    network.barrier(me.barrier_tag("init_weights_from_npz_done"), timeout=float(timeout_s))
    return _flat_ys_to_weights(local_ys, int(node_id), shapes)


def _dpss_refresh_weights(
    weights: list,
    *,
    node_id: int,
    n_nodes: int,
    t: int,
    field_size: int,
    shamir,
    network,
    model,
    context_prefix: str,
    seed: int,
) -> list:
    """
    Herzberg proactive refresh of all weight shares (distributed DPSS).
    Requires network and multi-node. Returns updated weights.
    """
    flat_ys = _weights_to_flat_ys(weights)
    committee = list(range(1, n_nodes + 1))
    barrier_fn = lambda: network.barrier(f"{context_prefix}_barrier", timeout=300.0)
    rng = random.Random(seed)
    new_ys = distributed_proactive_refresh_herzberg_vectorized(
        channel=network.channel,
        shamir=shamir,
        my_node_id=node_id,
        p=field_size,
        t=t,
        my_share_ys=flat_ys,
        committee_node_ids=committee,
        context_prefix=context_prefix,
        rng=rng,
        chunk_size=50000,
        barrier_fn=barrier_fn,
    )
    return _flat_ys_to_weights(new_ys, node_id, model.get_weight_shapes())
def _shares_to_flat_mod_p(shares_like, p: int) -> np.ndarray:
    vals = []
    if isinstance(shares_like, list) and shares_like and isinstance(shares_like[0], list):
        # Matrix layer
        for row in shares_like:
            for s in row:
                vals.append(int(s.y) % p)
    else:
        # Vector layer (bias)
        for s in shares_like:
            vals.append(int(s.y) % p)
    return np.asarray(vals, dtype=np.uint64)


def _mod_p_to_float(arr_u64: np.ndarray, p: int, scale: int) -> np.ndarray:
    signed = arr_u64.astype(np.int64, copy=False)
    signed = np.where(signed > (p // 2), signed - p, signed)
    return signed.astype(np.float64) / float(scale)

def _prepare_training_batch_inputs(
    *,
    batch_idx,
    args,
    epoch: int,
    start_idx: int,
    me: MembershipEpochScope,
    use_distributed_dataset: bool,
    dataset_meta: dict,
    train_x_shares,
    train_y_shares,
    use_kvs_dataset: bool,
    local_kvs,
    packed_ops,
    train_x_lane_cache,
    train_y_lane_cache,
    lazy_unpack_metrics,
    x_train,
    y_train,
    use_pss_storage: bool,
    pss,
    pss_packing_factor: int,
    field_size: int,
    scale: int,
    shamir,
) -> Tuple[list, list, Optional[dict]]:
    x_shares_cols = []
    y_shares_cols = []
    packed_batch_payload = None
    if use_distributed_dataset:
        feat_dim = int(dataset_meta.get("feat_dim", int(train_x_shares.shape[1])))
        cls_dim = int(dataset_meta.get("cls_dim", int(train_y_shares.shape[1])))
        use_pss_batch = bool(dataset_meta.get("use_pss_storage", False))
        pss_k = int(dataset_meta.get("packing_factor", 1))
        if use_kvs_dataset and local_kvs is not None:
            keys = [key_data_sample_split("train", int(i)) for i in batch_idx]
            results = kvs_get_batch(local_kvs, keys, min_version=1)
            x_rows = [r.value[0] for r in results if r is not None]
            y_rows = [r.value[1] for r in results if r is not None]
            if len(x_rows) != len(batch_idx):
                raise RuntimeError(
                    f"KVS batch retrieval failed: got {len(x_rows)}/{len(batch_idx)} samples"
                )
            packed_batch_payload = {
                "x_rows": x_rows,
                "y_rows": y_rows,
                "feat_dim": int(feat_dim),
                "cls_dim": int(cls_dim),
                "pss_k": int(pss_k),
            }
        elif bool(args.packed_end2end) and use_pss_batch and packed_ops is not None:
            packed_batch_payload = {
                "x_rows": [train_x_shares[int(i)] for i in batch_idx],
                "y_rows": [train_y_shares[int(i)] for i in batch_idx],
                "feat_dim": int(feat_dim),
                "cls_dim": int(cls_dim),
                "pss_k": int(pss_k),
            }
        elif train_x_lane_cache is not None and train_y_lane_cache is not None:
            idx_list = [int(i) for i in batch_idx.tolist()]
            x_shares_cols = _get_or_unpack_cached_rows(
                cache=train_x_lane_cache,
                indices=idx_list,
                packed_rows=train_x_shares,
                node_id=int(args.node_id),
                original_len=int(feat_dim),
                packing_factor=int(pss_k),
                packed_ops=packed_ops,
                context_prefix=me.ctx(f"lazy_train_x_e{epoch}_b{start_idx}"),
                timeout_s=120.0,
                metrics=lazy_unpack_metrics,
                metrics_key="train_x",
            )
            y_shares_cols = _get_or_unpack_cached_rows(
                cache=train_y_lane_cache,
                indices=idx_list,
                packed_rows=train_y_shares,
                node_id=int(args.node_id),
                original_len=int(cls_dim),
                packing_factor=int(pss_k),
                packed_ops=packed_ops,
                context_prefix=me.ctx(f"lazy_train_y_e{epoch}_b{start_idx}"),
                timeout_s=120.0,
                metrics=lazy_unpack_metrics,
                metrics_key="train_y",
            )
        else:
            for pos, i in enumerate(batch_idx):
                if use_pss_batch:
                    x_shares_cols.append(
                        _packed_row_to_share_list(
                            train_x_shares[int(i)],
                            node_id=int(args.node_id),
                            original_len=int(feat_dim),
                            packing_factor=int(pss_k),
                            packed_ops=packed_ops,
                            context_prefix=me.ctx(f"train_e{epoch}_b{start_idx}_s{pos}_x"),
                            timeout_s=120.0,
                        )
                    )
                    y_shares_cols.append(
                        _packed_row_to_share_list(
                            train_y_shares[int(i)],
                            node_id=int(args.node_id),
                            original_len=int(cls_dim),
                            packing_factor=int(pss_k),
                            packed_ops=packed_ops,
                            context_prefix=me.ctx(f"train_e{epoch}_b{start_idx}_s{pos}_y"),
                            timeout_s=120.0,
                        )
                    )
                else:
                    x_shares_cols.append(_row_to_share_list(train_x_shares[int(i)], args.node_id))
                    y_shares_cols.append(_row_to_share_list(train_y_shares[int(i)], args.node_id))
    else:
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]

        batch_seed = args.seed + epoch * 100000 + start_idx
        random.seed(batch_seed)
        np.random.seed(batch_seed)

        use_pss_batch = bool(use_pss_storage and packed_ops is not None)
        if use_pss_batch:
            # Batched unpack: one amortized MPC round per packed chunk for the whole mini-batch.
            packed_rows_x: List[List[int]] = []
            packed_rows_y: List[List[int]] = []
            for i in range(len(x_batch)):
                x_per = _share_vector_for_all_nodes_pss(
                    x_batch[i],
                    int(args.n_nodes),
                    int(args.t),
                    int(field_size),
                    pss,
                    int(scale),
                    int(pss_packing_factor),
                )
                packed_rows_x.append([int(v) for v in x_per[int(args.node_id) - 1]])
                y_per = _share_vector_for_all_nodes_pss(
                    y_batch[i],
                    int(args.n_nodes),
                    int(args.t),
                    int(field_size),
                    pss,
                    int(scale),
                    int(pss_packing_factor),
                )
                packed_rows_y.append([int(v) for v in y_per[int(args.node_id) - 1]])
            feat_d = int(x_batch.shape[1])
            cls_d = int(y_batch.shape[1])
            x_shares_cols = packed_ops.unpack_packed_rows_to_lane_rows(
                packed_rows=packed_rows_x,
                original_len=feat_d,
                packing_factor=int(pss_packing_factor),
                context=me.ctx(f"train_plain_e{epoch}_b{start_idx}_x"),
                timeout=120.0,
            )
            y_shares_cols = packed_ops.unpack_packed_rows_to_lane_rows(
                packed_rows=packed_rows_y,
                original_len=cls_d,
                packing_factor=int(pss_packing_factor),
                context=me.ctx(f"train_plain_e{epoch}_b{start_idx}_y"),
                timeout=120.0,
            )
        else:
            for i in range(len(x_batch)):
                x_shares_cols.append(
                    image_to_shares(
                        x_batch[i], args.n_nodes, args.t, args.node_id, field_size, shamir, scale
                    )
                )
                y_shares_cols.append(
                    label_to_shares(
                        y_batch[i], args.n_nodes, args.t, args.node_id, field_size, shamir, scale
                    )
                )

    return x_shares_cols, y_shares_cols, packed_batch_payload
def export_reconstructed_model(
    *,
    weights: list,
    node_id: int,
    opener_node_id: int,
    n_nodes: int,
    field_size: int,
    scale_factor: int,
    reconstruction,
    export_path: str,
    timeout_s: float = 180.0,
    me: Optional[MembershipEpochScope] = None,
) -> None:
    """
    Reconstruct final model weights on opener node and save to NPZ.
    Non-opener nodes only participate by broadcasting their local vectors.
    """
    if not export_path:
        return
    if reconstruction is None:
        if node_id == 1:
            print("[WARN] Model export requested but reconstruction manager is unavailable.")
        return

    net = reconstruction.network
    p = int(field_size)
    opener = int(opener_node_id)

    # Layer descriptors and shapes
    layer_specs = [
        ("w1", weights[0], "matrix", (len(weights[0]), len(weights[0][0]) if weights[0] else 0)),
        ("w2", weights[1], "matrix", (len(weights[1]), len(weights[1][0]) if weights[1] else 0)),
        ("b1", weights[2], "vector", (len(weights[2]),)),
        ("b2", weights[3], "vector", (len(weights[3]),)),
    ]

    reconstructed = {}
    _me = me if me is not None else MembershipEpochScope(0)
    base_ctx = _me.ctx("final_model_export")
    x0 = int(weights[0][0][0].x) if weights and weights[0] and weights[0][0] else int(node_id)

    for lname, layer, ltype, shape in layer_specs:
        local_vals = _shares_to_flat_mod_p(layer, p)
        ctx = f"{base_ctx}_{lname}"
        net.broadcast_vector(ctx, x=x0, values=local_vals)

        if int(node_id) == int(opener):
            opened = reconstruction.reconstruct_opened_vector_values(
                context=ctx,
                values_local=local_vals,
                x=x0,
                timeout=float(timeout_s),
            )
            arr_f = _mod_p_to_float(opened, p, int(scale_factor))
            if ltype == "matrix":
                arr_f = arr_f.reshape(shape[0], shape[1])
            else:
                arr_f = arr_f.reshape(shape[0])
            reconstructed[lname] = arr_f

    # Ensure all nodes finished vector broadcast/reconstruct before shutdown.
    try:
        net.barrier(_me.barrier_tag("final_model_export_done"), timeout=float(timeout_s))
    except Exception:
        pass

    if int(node_id) != int(opener):
        return

    out_path = Path(export_path)
    if out_path.suffix.lower() != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        w1=reconstructed["w1"],
        w2=reconstructed["w2"],
        b1=reconstructed["b1"],
        b2=reconstructed["b2"],
    )
    meta_path = out_path.with_suffix(".json")
    meta = {
        "field_size": int(field_size),
        "scale_factor": int(scale_factor),
        "n_nodes": int(n_nodes),
        "exported_by_node": int(node_id),
        "weights_file": str(out_path),
        "format": "npz",
        "membership_epoch": int(_me.e),
        "arrays": {"w1": list(reconstructed["w1"].shape), "w2": list(reconstructed["w2"].shape), "b1": list(reconstructed["b1"].shape), "b2": list(reconstructed["b2"].shape)},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Reconstructed model exported: {out_path}")
    print(f"Model metadata exported: {meta_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-id', type=int, required=True)
    parser.add_argument('--n-nodes', type=int, default=3)
    parser.add_argument('--t', type=int, default=1)
    parser.add_argument('--base-port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument(
        '--topology',
        type=str,
        default='',
        help='YAML path: explicit hosts/ports for MPC parties and optional client endpoint (see ml_training.topology).',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Gradient accumulation steps via effective batch grouping (default: 1). Effective batch = batch_size * accum_steps.')
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--field-size', type=int, default=2**32 - 5,
                        help='Finite field modulus (default: 2^32-5). For larger headroom, try 2^61-1.')
    parser.add_argument('--mnist-samples', type=int, default=None)
    parser.add_argument('--enable-network', action='store_true')
    parser.add_argument('--loss-mode', choices=['mse', 'softmax'], default='softmax',
                        help='Output gradient mode for batched secure training')
    parser.add_argument('--scale-factor', type=int, default=2**20,
                        help='Fixed-point scale factor (default: 2^20 = 1048576)')
    parser.add_argument('--softmax-temperature', type=float, default=1.0,
                        help='Temperature for secure softmax (default: 1.0)')
    parser.add_argument('--grad-clip', type=float, default=2.0,
                        help='Gradient clip bound in model fixed-point units (default: 2.0)')
    parser.add_argument('--logit-clip', type=float, default=8.0,
                        help='Logit clip bound before exp approximation (default: 8.0)')
    parser.add_argument('--exp-approx', choices=['taylor5', 'pade22'], default='pade22',
                        help='Secure exp approximation used by softmax (default: pade22)')
    parser.add_argument('--softmax-grad-mode', choices=['secure_approx', 'opened_exact'], default='secure_approx',
                        help='Gradient path for softmax CE: secure_approx (default) or opened_exact')
    parser.add_argument('--debug-numerics', action='store_true',
                        help='Enable extra numeric probes (logits/dz2/update stats) on first batch each epoch')
    parser.add_argument('--debug-division', action='store_true',
                        help='Enable secure-division debug summaries for first few division calls')
    parser.add_argument('--packed-forward-pilot', action='store_true',
                        help='Pilot forward dense kernel mode (larger batched multiply chunk for A/B testing).')
    parser.add_argument('--packed-forward-native', action='store_true',
                        help='Experimental true packed-native dense1 forward kernel (very expensive; research mode).')
    parser.add_argument('--packed-end2end', action='store_true',
                        help='Experimental packed batch API path (Phase-3 scaffold).')
    parser.add_argument('--explode-logit-threshold', type=float, default=10.0,
                        help='Warn if reconstructed |logit| exceeds this value (default: 10.0)')
    parser.add_argument('--loss-growth-threshold', type=float, default=5.0,
                        help='Instability threshold for loss growth vs previous epoch (default: 5.0)')
    parser.add_argument('--grad-norm-threshold', type=float, default=5.0,
                        help='Instability threshold for estimated gradient norm (default: 5.0)')
    parser.add_argument('--no-abort-on-instability', action='store_true',
                        help='Do not abort training when instability thresholds are violated')
    parser.add_argument('--dpss-refresh-interval', type=int, default=0,
                        help='Herzberg proactive refresh every N epochs (0=disabled). Requires --enable-network and n_nodes>1.')
    parser.add_argument('--export-reconstructed-model', type=str, default='',
                        help='Path to save reconstructed final model (.npz). Reconstruction/export is performed on opener node.')
    parser.add_argument('--export-timeout', type=float, default=180.0,
                        help='Timeout (seconds) for final model export reconstruction (default: 180)')
    parser.add_argument(
        '--init-weights-from-npz',
        type=str,
        default='',
        help='Initialize secure weight shares from reconstructed model NPZ (dealer distributes shares to all nodes).',
    )
    parser.add_argument(
        '--weights-dealer-node',
        type=int,
        default=0,
        help='Dealer node id for --init-weights-from-npz (default: n_nodes).',
    )
    parser.add_argument(
        '--receive-weights-shares-from-client',
        action='store_true',
        help='Receive model weight shares from external client sender instead of node dealer.',
    )
    parser.add_argument(
        '--weights-source-node-id',
        type=int,
        default=0,
        help='External sender node_id for --receive-weights-shares-from-client (default: 0).',
    )
    parser.add_argument('--distribute-dataset-shares', action='store_true',
                        help='Owner node secret-shares MNIST and distributes only per-node shares over MPC network.')
    parser.add_argument('--dataset-owner-node', type=int, default=1,
                        help='Node ID that loads raw MNIST and distributes shares when --distribute-dataset-shares is enabled.')
    parser.add_argument('--dataset-distribution-timeout', type=float, default=900.0,
                        help='Timeout (seconds) for dataset share distribution and reception.')
    parser.add_argument('--receive-dataset-shares-from-client', action='store_true',
                        help='Do not load MNIST on nodes; receive pre-shared dataset from external client sender.')
    parser.add_argument('--dataset-source-node-id', type=int, default=0,
                        help='Sender node_id used by external dataset distributor (default: 0).')
    parser.add_argument('--client-eval-after-training', action='store_true',
                        help='After training, send output-share logits to client so client can reconstruct accuracy.')
    parser.add_argument('--client-eval-samples', type=int, default=100,
                        help='Number of test samples to use for client-side final accuracy reconstruction.')
    parser.add_argument(
        '--client-eval-batched-receive',
        action='store_true',
        help='Send all eval logit shares in one vector for single client receive/reconstruct.',
    )
    parser.add_argument(
        '--membership-epoch',
        type=int,
        default=0,
        help='Membership epoch e: prefix MPC vector contexts and barrier/SYNC tags with m{e}_ (default: 0). '
        'Must match across all nodes and the client distributor.',
    )
    parser.add_argument('--enable-failure-detection', action='store_true',
                        help='Enable heartbeat-based failure detection; use dynamic n_active for packing safety.')
    parser.add_argument(
        '--enable-dropout-reshare-recovery',
        action='store_true',
        help='On node failure: pause, barrier among survivors, Lagrange redistribute weight shares, '
        're-check packing safety, then resume if safe. Requires --enable-failure-detection.',
    )
    parser.add_argument(
        '--enable-join-recovery',
        action='store_true',
        help='On node rejoin: pause, barrier among all active, Lagrange redistribute weight shares '
        'to new committee, then resume. Requires --enable-failure-detection.',
    )
    parser.add_argument('--use-kvs-dataset', action='store_true',
                        help='Store dataset shares in local KVS (vD) and retrieve mini-batches from KVS.')
    parser.add_argument('--use-weight-versioning', action='store_true',
                        help='Store weight shares to local KVS with v_theta after each epoch.')
    args = parser.parse_args()

    node_topology = None
    _top_path = str(getattr(args, "topology", "")).strip()
    if _top_path:
        node_topology = load_node_topology(_top_path, int(args.node_id))
        if int(args.n_nodes) != int(node_topology.n_nodes):
            print(
                f"[WARN] --n-nodes {args.n_nodes} does not match topology (n={node_topology.n_nodes}); "
                "using topology."
            )
        args.n_nodes = int(node_topology.n_nodes)

    if bool(args.enable_dropout_reshare_recovery):
        if not bool(args.enable_failure_detection):
            parser.error("--enable-dropout-reshare-recovery requires --enable-failure-detection")
        if not bool(args.enable_network) or int(args.n_nodes) < 2:
            parser.error("--enable-dropout-reshare-recovery requires --enable-network and --n-nodes >= 2")
    if bool(args.enable_join_recovery):
        if not bool(args.enable_failure_detection):
            parser.error("--enable-join-recovery requires --enable-failure-detection")
        if not bool(args.enable_network) or int(args.n_nodes) < 2:
            parser.error("--enable-join-recovery requires --enable-network and --n-nodes >= 2")

    me = (
        MutableMembershipEpochScope(int(args.membership_epoch))
        if bool(args.enable_join_recovery)
        else MembershipEpochScope(int(args.membership_epoch))
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    FIELD_SIZE = int(args.field_size)
    if FIELD_SIZE <= 3:
        raise ValueError("--field-size must be > 3")
    SCALE = int(args.scale_factor)
    if SCALE <= 0:
        raise ValueError("--scale-factor must be positive")
    if int(args.accum_steps) <= 0:
        raise ValueError("--accum-steps must be >= 1")
    if bool(args.packed_forward_native) and not bool(args.packed_end2end):
        raise ValueError("--packed-forward-native requires --packed-end2end")

    from ml_training.benchmark_stats import log_benchmark

    log_benchmark(
        role=f"node{int(args.node_id)}",
        phase="cold_start",
        wall_sec=float(time.time() - _PROCESS_T0),
    )
    _node_run_t0 = time.time()
    _dataset_prep_sec = 0.0

    print(f"Node {args.node_id} starting. Dataset: MNIST. Model: BATCHED MLP (784-128-10)")
    print(f"Membership epoch e={me.e} (MPC/barrier prefix m{me.e}_)")
    print(f"Fixed-point scale: {SCALE}, softmax temperature: {args.softmax_temperature}, grad clip: {args.grad_clip}, logit clip: {args.logit_clip}, grad mode: {args.softmax_grad_mode}")
    print(f"Batch config: batch_size={args.batch_size}, accum_steps={int(args.accum_steps)}, effective_batch={int(args.batch_size)*int(args.accum_steps)}")
    if bool(args.packed_forward_pilot):
        print("Forward kernel mode: packed-forward pilot enabled")
    if bool(args.packed_forward_native):
        print("Forward kernel mode: packed-native dense1 pilot enabled")
    if bool(args.packed_end2end):
        print("Packed end-to-end mode: experimental API enabled")

    shamir = ShamirSecretSharing(FIELD_SIZE)
    pss = PackedShamirSecretSharing(FIELD_SIZE)
    raw_packing = int(pss.max_packing_factor(int(args.n_nodes), int(args.t)))
    if raw_packing <= 0:
        raise ValueError(
            f"No valid PSS packing factor for n_nodes={args.n_nodes}, t={args.t}. "
            "Need n_nodes > t."
        )
    n_active = int(args.n_nodes)  # TODO: wire failure detector when available
    pss_packing_factor = min(raw_packing, get_max_safe_packing_factor(int(args.t), n_active))
    if pss_packing_factor < raw_packing and int(args.node_id) == 1:
        print(
            f"[Packing safety] Capped packing factor {raw_packing} -> {pss_packing_factor} "
            f"(2*(t+s-1) < n_active={n_active})"
        )
    triple_gen = BeaverTripleGenerator(FIELD_SIZE)
    default_pool = 8000 if os.environ.get("SENTRA_IN_OCCLUM") == "1" else 50000
    pool_size = int(os.environ.get("SENTRA_BEAVER_POOL_SIZE", str(default_pool)))
    triple_pool = BeaverTriplePool(triple_gen, initial_size=pool_size)
    
    network = None
    reconstruction = None
    failure_detector = None
    use_client_distributed_dataset = bool(args.receive_dataset_shares_from_client and args.enable_network and args.n_nodes > 1)
    if args.enable_network and args.n_nodes > 1:
        if node_topology is not None:
            node_configs = dict(node_topology.node_configs)
            listen_bind_port = int(node_topology.listen_port)
        else:
            node_configs = {i: {'host': args.host, 'port': args.base_port + i} for i in range(1, args.n_nodes + 1)}
            listen_bind_port = int(args.base_port + args.node_id)
        network = create_mpc_network(
            args.node_id, node_configs, port=listen_bind_port,
            membership_epoch=int(me.e),
        )
        reconstruction = create_reconstruction_manager(network, args.t, FIELD_SIZE)
        print("Waiting for network barrier...")
        network.barrier(me.barrier_tag("startup"), 600)
        print("Network ready.")
        try:
            record_membership_epoch_local_kvs(args.node_id, me.e)
        except Exception as exc:
            if int(args.node_id) == 1:
                print(f"[WARN] Could not record membership epoch to local KVS: {exc}")
        
        if args.node_id == args.n_nodes:
            print(f"Node {args.node_id}: Initializing Dealer Service...")
            dealer = BeaverTripleDealerService(
                network=network,
                dealer_node_id=args.n_nodes,
                n_nodes=args.n_nodes,
                t=args.t,
                field_size=FIELD_SIZE
            )
            dealer.register()
            print(f"Node {args.node_id}: Dealer Service Registered.")
        failure_detector = None
        if args.enable_failure_detection:
            failure_detector = NodeFailureDetector(
                network, heartbeat_interval=2.0, failure_timeout=6.0
            )
            failure_detector.start_monitoring()
            print(f"Node {args.node_id}: Failure detection enabled (n_active from heartbeat).")

    state_machine = None
    join_recovered_nodes_holder: List[int] = []  # Populated by recovery callback; read in training loop
    if failure_detector is not None:
        def _log_state(msg: str) -> None:
            print(f"Node {args.node_id}: {msg}")
        state_machine = TrainingStateMachine(log_fn=_log_state)
        failure_detector.register_failure_callback(
            lambda _: state_machine.request_pause("node_failure")
        )
        def _on_recovery(recovered: List[int]) -> None:
            join_recovered_nodes_holder.clear()
            join_recovered_nodes_holder.extend(recovered)
            state_machine.request_pause("node_rejoin")

        failure_detector.register_recovery_callback(_on_recovery)
        if network is not None and getattr(network, "channel", None) is not None:
            network.channel.on_send_failure = failure_detector.mark_peer_failed_immediate

    packed_ops = None
    if network is not None:
        packed_ops = PackedMPCOps(
            network=network,
            n_nodes=int(args.n_nodes),
            t=int(args.t),
            field_size=int(FIELD_SIZE),
        )

    use_owner_distributed_dataset = bool(args.distribute_dataset_shares and args.enable_network and args.n_nodes > 1)
    if use_owner_distributed_dataset and use_client_distributed_dataset:
        raise ValueError("Choose only one of --distribute-dataset-shares or --receive-dataset-shares-from-client")

    use_distributed_dataset = bool(use_owner_distributed_dataset or use_client_distributed_dataset)
    use_pss_storage = bool(network is not None and args.n_nodes > 1)
    use_kvs_dataset = bool(use_distributed_dataset and args.use_kvs_dataset)
    use_weight_versioning = bool(args.use_weight_versioning)
    dataset_meta = {
        "use_pss_storage": bool(use_pss_storage),
        "packing_factor": int(pss_packing_factor),
        "feat_dim": 784,
        "cls_dim": 10,
        "feat_stored_len": 784 if not use_pss_storage else (784 + pss_packing_factor - 1) // pss_packing_factor,
        "cls_stored_len": 10 if not use_pss_storage else (10 + pss_packing_factor - 1) // pss_packing_factor,
        "membership_epoch": int(me.e),
    }
    print(
        f"PSS mode: {'enabled' if use_pss_storage else 'disabled'}; "
        f"packing factor={pss_packing_factor}"
    )
    if use_owner_distributed_dataset:
        _t_dataset0 = time.time()
        print(
            f"Node {args.node_id}: distributed dataset mode enabled; owner node is {args.dataset_owner_node}. "
            "Non-owner nodes will not load raw MNIST."
        )
        train_x_shares, train_y_shares, test_x_shares, test_y_shares, x_test_plain, y_test_plain, dataset_meta = (
            prepare_distributed_dataset_shares(
                network=network,
                node_id=args.node_id,
                n_nodes=args.n_nodes,
                owner_node_id=int(args.dataset_owner_node),
                t=args.t,
                field_size=FIELD_SIZE,
                shamir=shamir,
                pss=pss,
                scale=SCALE,
                mnist_samples=args.mnist_samples,
                timeout_s=float(args.dataset_distribution_timeout),
                pss_packing_factor=int(pss_packing_factor),
                use_pss_storage=bool(use_pss_storage),
                me=me,
            )
        )
        print(f"Node {args.node_id}: dataset shares ready (train={len(train_x_shares)}, test={len(test_x_shares)}).")
        _dataset_prep_sec = float(time.time() - _t_dataset0)
        print(f"Dataset Share Prep Time: {_dataset_prep_sec:.6f}s")
    elif use_client_distributed_dataset:
        _t_dataset0 = time.time()
        print(
            f"Node {args.node_id}: waiting for dataset shares from external sender node_id={args.dataset_source_node_id}. "
            "This node will not load raw MNIST."
        )
        train_x_shares, train_y_shares, test_x_shares, test_y_shares, dataset_meta = receive_dataset_shares_from_external_source(
            network=network,
            source_node_id=int(args.dataset_source_node_id),
            timeout_s=float(args.dataset_distribution_timeout),
            me=me,
        )
        x_test_plain = None
        y_test_plain = None
        print(f"Node {args.node_id}: dataset shares received (train={len(train_x_shares)}, test={len(test_x_shares)}).")
        _dataset_prep_sec = float(time.time() - _t_dataset0)
        print(f"Dataset Share Prep Time: {_dataset_prep_sec:.6f}s")
    else:
        (x_train, y_train), (x_test, y_test) = load_mnist_data(args.mnist_samples)
        train_x_shares = None
        train_y_shares = None
        test_x_shares = None
        test_y_shares = None
        x_test_plain = x_test
        y_test_plain = y_test
        dataset_meta["feat_dim"] = int(x_train.shape[1])
        dataset_meta["cls_dim"] = int(y_train.shape[1])
        if use_pss_storage:
            dataset_meta["feat_stored_len"] = int((int(x_train.shape[1]) + pss_packing_factor - 1) // pss_packing_factor)
            dataset_meta["cls_stored_len"] = int((int(y_train.shape[1]) + pss_packing_factor - 1) // pss_packing_factor)
        print(f"Loaded {len(x_train)} training samples")

    local_kvs = None
    if use_kvs_dataset or use_weight_versioning:
        local_kvs = KVSCluster([int(args.node_id)])
        if use_kvs_dataset:
            print(f"Node {args.node_id}: Populating KVS with dataset (vD=1)...")
            _populate_dataset_kvs(
                local_kvs, train_x_shares, train_y_shares, test_x_shares, test_y_shares, vD=1
            )
            print(f"Node {args.node_id}: KVS dataset populated.")
        if use_weight_versioning:
            print(f"Node {args.node_id}: Weight versioning enabled (v_theta persisted after each epoch).")

    multiplier = SecureMultiplier(triple_pool, args.n_nodes, args.t, FIELD_SIZE, 
                                  reconstruction_manager=reconstruction,
                                  prss_seed=args.seed,
                                  triple_dealer_id=args.n_nodes if args.enable_network else None,
                                  privacy_mode=True)
    
    comparator = SecureComparator(multiplier, field_size=FIELD_SIZE)
    divider = SecureDivider(multiplier, FIELD_SIZE, SCALE, debug=bool(args.debug_division))
    model = BatchedSecureMNISTMLP(
        args.n_nodes,
        args.t,
        multiplier,
        FIELD_SIZE,
        comparator,
        divider,
        scale_factor=SCALE,
        grad_clip=float(args.grad_clip),
        use_packed_forward_pilot=bool(args.packed_forward_pilot),
    )
    # 5. Initialize/Distribute weights
    np.random.seed(args.seed) # Ensure identical initialization across all nodes
    random.seed(args.seed) # Essential because ShamirSecretSharing uses Python's native random module 
    weights = model.initialize_weights(node_id=args.node_id)
    if bool(getattr(args, "init_weights_from_npz", "")):
        if network is None or int(args.n_nodes) < 2:
            raise ValueError("--init-weights-from-npz requires --enable-network and --n-nodes >= 2")
        npz_path = str(args.init_weights_from_npz)
        if not os.path.isabs(npz_path):
            npz_path = os.path.abspath(npz_path)
        if bool(getattr(args, "receive_weights_shares_from_client", False)):
            print(
                f"Node {args.node_id}: model-weight init mode = client-shared "
                f"(source node_id={int(getattr(args, 'weights_source_node_id', 0))})"
            )
            weights = _receive_weight_shares_from_client(
                model=model,
                node_id=int(args.node_id),
                source_node_id=int(getattr(args, "weights_source_node_id", 0)),
                network=network,
                me=me,
                timeout_s=float(max(float(args.dataset_distribution_timeout), float(args.export_timeout), 300.0)),
            )
        else:
            dealer_node_id = int(args.weights_dealer_node) if int(args.weights_dealer_node) > 0 else int(args.n_nodes)
            print(f"Node {args.node_id}: model-weight init mode = dealer-node (dealer={dealer_node_id})")
            weights = _distribute_or_receive_weight_shares_from_npz(
                weights_npz_path=npz_path,
                model=model,
                node_id=int(args.node_id),
                n_nodes=int(args.n_nodes),
                t=int(args.t),
                shamir=shamir,
                network=network,
                field_size=int(FIELD_SIZE),
                scale=int(SCALE),
                me=me,
                dealer_node_id=dealer_node_id,
                timeout_s=float(max(float(args.dataset_distribution_timeout), float(args.export_timeout), 300.0)),
            )
    softmax = SecureSoftmax(
        multiplier,
        divider,
        FIELD_SIZE,
        SCALE,
        temperature=float(args.softmax_temperature),
        logit_clip=float(args.logit_clip),
        exp_approx=str(args.exp_approx),
        softmax_grad_mode=str(args.softmax_grad_mode),
    )

    if use_distributed_dataset:
        train_len = int(train_x_shares.shape[0])
        test_len = int(test_x_shares.shape[0])
    else:
        train_len = int(len(x_train))
        test_len = int(len(x_test))
    planned_eval_n = min(100, test_len)
    train_x_lane_cache = None
    train_y_lane_cache = None
    test_x_lane_cache = None
    lazy_unpack_metrics = {}
    if (
        use_distributed_dataset
        and bool(dataset_meta.get("use_pss_storage", False))
        and packed_ops is not None
    ):
        feat_dim = int(dataset_meta.get("feat_dim", int(train_x_shares.shape[1])))
        cls_dim = int(dataset_meta.get("cls_dim", int(train_y_shares.shape[1])))
        pss_k = int(dataset_meta.get("packing_factor", 1))
        # Lazy unpack cache: unpack packed rows only when accessed, then reuse.
        train_x_lane_cache = {}
        train_y_lane_cache = {}
        test_x_lane_cache = {}
        print("Lazy PSS unpack cache enabled (train/test).")
        if _env_truthy("SENTRA_PREWARM_TRAIN_UNPACK"):
            _prewarm_train_pss_unpack_cache(
                train_len=train_len,
                test_len=test_len,
                args=args,
                network=network,
                me=me,
                train_x_shares=train_x_shares,
                train_y_shares=train_y_shares,
                test_x_shares=test_x_shares,
                dataset_meta=dataset_meta,
                packed_ops=packed_ops,
                train_x_lane_cache=train_x_lane_cache,
                train_y_lane_cache=train_y_lane_cache,
                test_x_lane_cache=test_x_lane_cache,
                lazy_unpack_metrics=lazy_unpack_metrics,
                timeout_s=float(args.dataset_distribution_timeout),
            )

    rng_eval = np.random.default_rng(args.seed + 777)
    eval_n = planned_eval_n
    fixed_eval_indices = rng_eval.choice(test_len, size=eval_n, replace=False)
    fixed_pre_indices = fixed_eval_indices[:1]

    # Pre-train eval with packed test shares requires every node to run evaluate_model (lazy unpack
    # / MPC). Non-owners have y_test_plain is None but still must enter the same round as the owner.
    if y_test_plain is not None or test_x_shares is not None:
        print("\nStarting PRE-TRAIN Evaluation Check...")
        if reconstruction is not None:
            reconstruction.clear_cache()
        acc_pre, loss_pre, diag_pre = evaluate_model(model, weights, x_test_plain, y_test_plain, args.node_id, 
                                                     args.n_nodes, args.t, shamir, FIELD_SIZE, SCALE, 
                                                     reconstruction, n_test_samples=1, context_prefix=me.ctx("eval_pre"),
                                                     fixed_indices=fixed_pre_indices,
                                                     x_test_shared=test_x_shares,
                                                     dataset_meta=dataset_meta,
                                                     packed_ops=packed_ops,
                                                     pss=pss,
                                                     pss_packing_factor=pss_packing_factor,
                                                    x_test_lane_cache=test_x_lane_cache,
                                                    unpack_metrics=lazy_unpack_metrics)
        if int(args.node_id) == 1:
            print(f"Pre-Train Test Accuracy: {acc_pre*100:.2f}%")
            print(f"Pre-Train Test Loss: {loss_pre:.4f}")
            print(
                f"Pre-Train Diagnostics: mean|logit|={diag_pre['mean_abs_logit']:.4f}, "
                f"max|logit|={diag_pre['max_abs_logit']:.4f}"
            )

    print("\nStarting BATCHED Training...")
    _t_train0 = time.time()
    _resource_sampler_stop = None
    try:
        from ml_training.runtime_metrics import metrics_enabled, start_resource_sampler
        import threading

        if metrics_enabled():
            _resource_sampler_stop = threading.Event()
            start_resource_sampler(stop_event=_resource_sampler_stop)
    except Exception:
        _resource_sampler_stop = None
    prev_epoch_loss = None
    instability_detected = False
    training_aborted = False
    dropout_recovery_seq = [0]
    join_recovery_seq = [0]

    for epoch in range(args.num_epochs):
        epoch_train_x_before = _lazy_metrics_snapshot(lazy_unpack_metrics, "train_x")
        epoch_train_y_before = _lazy_metrics_snapshot(lazy_unpack_metrics, "train_y")
        epoch_eval_x_before = _lazy_metrics_snapshot(lazy_unpack_metrics, "eval_x")
        epoch_grad_norm_estimate = None
        epoch_diag = {}
        epoch_stage_sums = {
            "t_forward_s": 0.0,
            "t_output_grad_s": 0.0,
            "t_dw2_s": 0.0,
            "t_da1_dz1_s": 0.0,
            "t_dw1_s": 0.0,
            "t_update_s": 0.0,
            "t_train_batch_total_s": 0.0,
        }
        epoch_stage_samples = 0
        n_batches = 0
        indices = np.arange(train_len)
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(indices)
        
        eff_batch_span = int(args.batch_size) * int(args.accum_steps)
        for start_idx in range(0, train_len, eff_batch_span):
            if training_aborted:
                break
            batch_done = False
            while not batch_done:
                if args.node_id == 1:
                    time.sleep(0.001)

                if state_machine is not None and state_machine.is_paused():
                    weights, n_active, recovered_and_resume, aborted = handle_paused_training_recovery(
                        weights=weights,
                        state_machine=state_machine,
                        args=args,
                        join_recovered_nodes_holder=join_recovered_nodes_holder,
                        network=network,
                        failure_detector=failure_detector,
                        shamir=shamir,
                        me=me,
                        field_size=FIELD_SIZE,
                        pss_packing_factor=pss_packing_factor,
                        model=model,
                        join_recovery_seq=join_recovery_seq,
                        dropout_recovery_seq=dropout_recovery_seq,
                    )
                    if recovered_and_resume:
                        continue
                    if aborted:
                        training_aborted = True
                        break

                if training_aborted:
                    break

                n_active = int(failure_detector.get_n_active()) if failure_detector is not None else int(args.n_nodes)
                # Packing safety guard (step 5): 2*(t+s-1) < n_active
                if use_pss_storage and not check_packing_safety(args.t, pss_packing_factor, n_active):
                    raise PackingSafetyError(args.t, pss_packing_factor, n_active)

                batch_idx = indices[start_idx : start_idx + eff_batch_span]
                # Wall-clock for the whole batch iteration (share prep / unpack + train). Previously only
                # model.train_batch* was timed, which excluded e.g. PSS lane sharing from plaintext rows.
                start_time = time.time()
                try:
                    x_shares_cols, y_shares_cols, packed_batch_payload = _prepare_training_batch_inputs(
                        batch_idx=batch_idx,
                        args=args,
                        epoch=epoch,
                        start_idx=start_idx,
                        me=me,
                        use_distributed_dataset=use_distributed_dataset,
                        dataset_meta=dataset_meta,
                        train_x_shares=train_x_shares,
                        train_y_shares=train_y_shares,
                        use_kvs_dataset=use_kvs_dataset,
                        local_kvs=local_kvs,
                        packed_ops=packed_ops,
                        train_x_lane_cache=train_x_lane_cache,
                        train_y_lane_cache=train_y_lane_cache,
                        lazy_unpack_metrics=lazy_unpack_metrics,
                        x_train=(x_train if not use_distributed_dataset else None),
                        y_train=(y_train if not use_distributed_dataset else None),
                        use_pss_storage=use_pss_storage,
                        pss=pss,
                        pss_packing_factor=pss_packing_factor,
                        field_size=FIELD_SIZE,
                        scale=SCALE,
                        shamir=shamir,
                    )

                    print(f"Epoch {epoch+1} Batch {n_batches+1} ({len(batch_idx)} samples)...", end='\r')

                    lr = args.learning_rate * (0.95 ** epoch)
                    weights, batch_diag = execute_training_batch(
                        weights=weights,
                        model=model,
                        packed_batch_payload=packed_batch_payload,
                        batch_idx=batch_idx,
                        args=args,
                        me=me,
                        epoch=epoch,
                        start_idx=start_idx,
                        train_x_lane_cache=train_x_lane_cache,
                        train_y_lane_cache=train_y_lane_cache,
                        train_x_shares=train_x_shares,
                        train_y_shares=train_y_shares,
                        packed_ops=packed_ops,
                        lazy_unpack_metrics=lazy_unpack_metrics,
                        softmax=softmax,
                        lr=lr,
                        reconstruction=reconstruction,
                        x_shares_cols=x_shares_cols,
                        y_shares_cols=y_shares_cols,
                        get_or_unpack_cached_rows_fn=_get_or_unpack_cached_rows,
                    )
                except Exception as exc:
                    if (
                        failure_detector is not None
                        and state_machine is not None
                        and state_machine.is_paused()
                    ):
                        print(
                            f"Node {args.node_id}: MPC batch interrupted for failure handling "
                            f"({type(exc).__name__}: {exc})"
                        )
                        continue
                    raise

                end_time = time.time()
                if "grad_norm_estimate" in batch_diag and epoch_grad_norm_estimate is None:
                    epoch_grad_norm_estimate = float(batch_diag["grad_norm_estimate"])
                for k, v in batch_diag.items():
                    if k not in epoch_diag:
                        epoch_diag[k] = v
                if "t_train_batch_total_s" in batch_diag:
                    for sk in epoch_stage_sums.keys():
                        epoch_stage_sums[sk] += float(batch_diag.get(sk, 0.0))
                    epoch_stage_samples += 1
                if "relu_backward_mode" in batch_diag and "relu_backward_mode" not in epoch_diag:
                    epoch_diag["relu_backward_mode"] = str(batch_diag.get("relu_backward_mode"))
                if "packed_end2end_mode" in batch_diag and "packed_end2end_mode" not in epoch_diag:
                    epoch_diag["packed_end2end_mode"] = str(batch_diag.get("packed_end2end_mode"))

                # Stage probes (integration tests / numerics): computed in mnist_mlp_batched on first _b0 batch only.
                if (
                    bool(args.debug_numerics)
                    and int(args.node_id) == 1
                    and batch_diag.get("logit_probe") is not None
                ):
                    _lp = batch_diag["logit_probe"]
                    _dz = batch_diag["dz2_probe"]
                    print(f"Epoch {epoch + 1} Probe logits[0][:10]: {_lp}")
                    print(f"Epoch {epoch + 1} Probe dz2[0][:10]: {_dz}")
                    print(
                        f"Epoch {epoch + 1} Probe probs_est stats: "
                        f"sum={batch_diag['probs_est_sum']}, min={batch_diag['probs_est_min']}, "
                        f"max={batch_diag['probs_est_max']}, target_sum={batch_diag['target_sum']}"
                    )

                wall_s = end_time - start_time
                train_internal = batch_diag.get("t_train_batch_total_s")
                if train_internal is not None:
                    print(
                        f"Epoch {epoch+1} Batch {n_batches+1} completed in {wall_s:.2f}s "
                        f"(MPC train step internal: {float(train_internal):.2f}s)",
                        end="\n",
                    )
                else:
                    print(f"Epoch {epoch+1} Batch {n_batches+1} completed in {wall_s:.2f}s", end="\n")

                try:
                    from ml_training.runtime_metrics import inc_training_iteration

                    inc_training_iteration()
                except Exception:
                    pass

                if os.environ.get("SENTRA_STRICT_TRAIN_BARRIER", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    network.barrier(
                        me.ctx(f"train_batch_e{epoch}_b{start_idx}_done"),
                        timeout=600.0,
                    )

                n_batches += 1
                batch_done = True

            if training_aborted:
                break

        if training_aborted:
            break

        if training_aborted:
            break

        print(f"Epoch {epoch+1} Complete.")

        if use_weight_versioning and local_kvs is not None:
            v_theta = epoch + 1
            _t_ver0 = time.time()
            put_weights_versioned(local_kvs, weights, v_theta, quorum_size=1)
            try:
                from ml_training.runtime_metrics import add_versioning_sec

                add_versioning_sec(time.time() - _t_ver0)
            except Exception:
                pass
            if args.node_id == 1:
                print(f"Epoch {epoch+1}: Weights persisted to KVS (v_theta={v_theta}).")

        if (
            args.dpss_refresh_interval > 0
            and network is not None
            and args.n_nodes > 1
            and (epoch + 1) % args.dpss_refresh_interval == 0
        ):
            _t_dpss = time.time()
            print(f"Epoch {epoch+1}: Running DPSS Herzberg proactive refresh...")
            weights = _dpss_refresh_weights(
                weights,
                node_id=args.node_id,
                n_nodes=args.n_nodes,
                t=args.t,
                field_size=FIELD_SIZE,
                shamir=shamir,
                network=network,
                model=model,
                context_prefix=me.ctx(f"dpss_refresh_e{epoch+1}"),
                seed=args.seed + 1000 * (epoch + 1),
            )
            print(f"Epoch {epoch+1}: DPSS refresh completed in {time.time() - _t_dpss:.2f}s")
        if args.node_id == 1 and bool(dataset_meta.get("use_pss_storage", False)):
            dtx = _lazy_metrics_delta(lazy_unpack_metrics, "train_x", epoch_train_x_before)
            dty = _lazy_metrics_delta(lazy_unpack_metrics, "train_y", epoch_train_y_before)
            print(
                "Lazy PSS unpack stats (epoch "
                f"{epoch + 1}): "
                f"train_x[h={dtx['hits']},m={dtx['misses']},rows={dtx['rows_unpacked']},"
                f"batches={dtx['batch_calls']},s={dtx['unpack_s']:.3f}] "
                f"train_y[h={dty['hits']},m={dty['misses']},rows={dty['rows_unpacked']},"
                f"batches={dty['batch_calls']},s={dty['unpack_s']:.3f}]"
            )
        if args.node_id == 1 and epoch_stage_samples > 0:
            if "forward_kernel_mode" in epoch_diag:
                print(
                    f"Epoch {epoch+1} Forward Kernel: "
                    f"{epoch_diag.get('forward_kernel_mode')} "
                    f"(chunk={epoch_diag.get('forward_kernel_chunk', 'n/a')})"
                )
            if "relu_backward_mode" in epoch_diag:
                print(f"Epoch {epoch+1} ReLU Backward: {epoch_diag.get('relu_backward_mode')}")
            if "packed_end2end_mode" in epoch_diag:
                print(f"Epoch {epoch+1} Packed API: {epoch_diag.get('packed_end2end_mode')}")
            if "t_train_batch_total_s" in epoch_diag:
                print(
                    "Epoch "
                    f"{epoch+1} Batch Stage Timing (sampled): "
                    f"fwd={epoch_diag.get('t_forward_s', 0.0):.3f}s, "
                    f"dz2={epoch_diag.get('t_output_grad_s', 0.0):.3f}s, "
                    f"dw2={epoch_diag.get('t_dw2_s', 0.0):.3f}s, "
                    f"da1_dz1={epoch_diag.get('t_da1_dz1_s', 0.0):.3f}s, "
                    f"dw1={epoch_diag.get('t_dw1_s', 0.0):.3f}s, "
                    f"upd={epoch_diag.get('t_update_s', 0.0):.3f}s, "
                    f"total={epoch_diag.get('t_train_batch_total_s', 0.0):.3f}s"
                )
            n_s = float(epoch_stage_samples)
            avg = {k: float(v) / n_s for k, v in epoch_stage_sums.items()}
            total = max(1e-9, float(avg["t_train_batch_total_s"]))
            print(
                "Epoch "
                f"{epoch+1} Batch Stage Timing (avg over {epoch_stage_samples} batches): "
                f"fwd={avg['t_forward_s']:.3f}s ({100.0*avg['t_forward_s']/total:.1f}%), "
                f"dz2={avg['t_output_grad_s']:.3f}s ({100.0*avg['t_output_grad_s']/total:.1f}%), "
                f"dw2={avg['t_dw2_s']:.3f}s ({100.0*avg['t_dw2_s']/total:.1f}%), "
                f"da1_dz1={avg['t_da1_dz1_s']:.3f}s ({100.0*avg['t_da1_dz1_s']/total:.1f}%), "
                f"dw1={avg['t_dw1_s']:.3f}s ({100.0*avg['t_dw1_s']/total:.1f}%), "
                f"upd={avg['t_update_s']:.3f}s ({100.0*avg['t_update_s']/total:.1f}%), "
                f"total={avg['t_train_batch_total_s']:.3f}s"
            )

        prev_epoch_loss, instability_detected, should_abort = run_epoch_eval_and_stability_checks(
            epoch=epoch,
            args=args,
            model=model,
            weights=weights,
            x_test_plain=x_test_plain,
            y_test_plain=y_test_plain,
            shamir=shamir,
            field_size=FIELD_SIZE,
            scale=SCALE,
            reconstruction=reconstruction,
            eval_n=eval_n,
            me=me,
            fixed_eval_indices=fixed_eval_indices,
            test_x_shares=test_x_shares,
            dataset_meta=dataset_meta,
            packed_ops=packed_ops,
            pss=pss,
            pss_packing_factor=pss_packing_factor,
            test_x_lane_cache=test_x_lane_cache,
            lazy_unpack_metrics=lazy_unpack_metrics,
            epoch_grad_norm_estimate=epoch_grad_norm_estimate,
            epoch_diag=epoch_diag,
            prev_epoch_loss=prev_epoch_loss,
            instability_detected=instability_detected,
            epoch_eval_x_before=epoch_eval_x_before,
            evaluate_fn=evaluate_model,
            lazy_metrics_delta_fn=_lazy_metrics_delta,
        )
        if should_abort:
            training_aborted = True
            break
    # Emit explicit prover timing summary for post-run parsers/exporters.
    _training_sec = float(time.time() - _t_train0)
    print(f"Training Time: {_training_sec:.6f}s")
    log_benchmark(
        role=f"node{int(args.node_id)}",
        phase="training",
        wall_sec=_training_sec,
    )
    prover_total = 0.0
    try:
        stats = multiplier.prover_time_snapshot(reset=False)
        prover_total = float(stats.get("total_sec", 0.0))
        opener_id = int(multiplier._opened_fp_opener()) if hasattr(multiplier, "_opened_fp_opener") else 1
        print(f"Prover Node: {opener_id}")
        print(f"Prover Time: {prover_total:.6f}s")
        by_cat = stats.get("by_category", {})
        if isinstance(by_cat, dict) and by_cat:
            cat_parts = [f"{k}={float(v):.6f}s" for k, v in sorted(by_cat.items())]
            print("Prover Time Breakdown: " + ", ".join(cat_parts))
    except Exception:
        pass

    # Optional final model export (reconstructed on opener node).
    if args.export_reconstructed_model:
        try:
            export_reconstructed_model(
                weights=weights,
                node_id=args.node_id,
                opener_node_id=int(multiplier._opened_fp_opener()) if hasattr(multiplier, "_opened_fp_opener") else 1,
                n_nodes=args.n_nodes,
                field_size=FIELD_SIZE,
                scale_factor=SCALE,
                reconstruction=reconstruction,
                export_path=args.export_reconstructed_model,
                timeout_s=float(args.export_timeout),
                me=me,
            )
        except Exception as exc:
            print(f"[WARN] Failed to export reconstructed model: {exc}")

    if (
        bool(args.client_eval_after_training)
        and bool(use_client_distributed_dataset)
        and network is not None
        and test_x_shares is not None
    ):
        _t_eval_upload0 = time.time()
        try:
            if (
                node_topology is not None
                and node_topology.client_host is not None
                and node_topology.client_port is not None
            ):
                _c_host = str(node_topology.client_host)
                _c_port = int(node_topology.client_port)
            else:
                _c_host = str(args.host)
                _c_port = int(args.base_port) + int(args.dataset_source_node_id)
            send_inference_shares_to_client(
                model=model,
                weights=weights,
                test_x_shares=test_x_shares,
                node_id=args.node_id,
                client_node_id=int(args.dataset_source_node_id),
                network=network,
                client_connect_host=_c_host,
                client_connect_port=_c_port,
                n_samples=int(args.client_eval_samples),
                seed=int(args.seed),
                context_prefix=me.ctx("client_eval_final"),
                eval_sync_tag=me.barrier_tag("client_eval_final_done"),
                batched_receive=bool(getattr(args, "client_eval_batched_receive", False)),
                dataset_meta=dataset_meta,
                packed_ops=packed_ops,
                test_x_lane_cache=test_x_lane_cache,
                unpack_metrics=lazy_unpack_metrics,
            )
        except Exception as exc:
            print(f"[WARN] Failed to send client eval shares: {exc}")
        print(f"Client Eval Upload Time: {time.time() - _t_eval_upload0:.6f}s")

    if args.node_id == 1 and bool(dataset_meta.get("use_pss_storage", False)):
        ttx = _lazy_metrics_snapshot(lazy_unpack_metrics, "train_x")
        tty = _lazy_metrics_snapshot(lazy_unpack_metrics, "train_y")
        tev = _lazy_metrics_snapshot(lazy_unpack_metrics, "eval_x")
        tcl = _lazy_metrics_snapshot(lazy_unpack_metrics, "client_eval_x")
        print(
            "Lazy PSS unpack totals: "
            f"train_x[h={ttx['hits']},m={ttx['misses']},rows={ttx['rows_unpacked']},batches={ttx['batch_calls']},s={ttx['unpack_s']:.3f}] "
            f"train_y[h={tty['hits']},m={tty['misses']},rows={tty['rows_unpacked']},batches={tty['batch_calls']},s={tty['unpack_s']:.3f}] "
            f"eval_x[h={tev['hits']},m={tev['misses']},rows={tev['rows_unpacked']},batches={tev['batch_calls']},s={tev['unpack_s']:.3f}] "
            f"client_x[h={tcl['hits']},m={tcl['misses']},rows={tcl['rows_unpacked']},batches={tcl['batch_calls']},s={tcl['unpack_s']:.3f}]"
        )

    _extra: dict = {}
    if _dataset_prep_sec > 0:
        _extra["dataset_prep_sec"] = f"{_dataset_prep_sec:.6f}"
    if prover_total > 0:
        _extra["prover_sec"] = f"{prover_total:.6f}"
    try:
        if _resource_sampler_stop is not None:
            _resource_sampler_stop.set()
        from ml_training.runtime_metrics import emit_summary

        emit_summary(role=f"node{int(args.node_id)}")
    except Exception:
        pass
    log_benchmark(
        role=f"node{int(args.node_id)}",
        phase="dataset",
        wall_sec=_dataset_prep_sec if _dataset_prep_sec > 0 else None,
    )
    log_benchmark(
        role=f"node{int(args.node_id)}",
        phase="total",
        wall_sec=float(time.time() - _node_run_t0),
        extra=_extra or None,
    )

if __name__ == '__main__':
    main()

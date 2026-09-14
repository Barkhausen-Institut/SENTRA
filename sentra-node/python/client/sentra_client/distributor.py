"""
Client-side dataset distributor for secure batched MNIST training.

Runnable as ``python -m sentra_client`` or ``client_distributor.py`` (shim).
Topology: optional ``--topology`` YAML (see ``ml_training.topology``).
"""

from __future__ import annotations

import argparse
import random
import time

_PROCESS_T0 = time.time()

import numpy as np
from tensorflow import keras

from ml_training.membership_epoch import MembershipEpochScope
from ml_training.packing_safety import get_max_safe_packing_factor
from ml_training.secret_sharing import PackedShamirSecretSharing, ShamirSecretSharing, Share
from ml_training.secure_comm import create_mpc_network
from ml_training.topology import load_client_topology
from ml_training.util import loadMNISTDataset


def load_mnist_data(train_samples=None, test_samples=None):
    """Load MNIST on the client only (local mnist.npz; never keras download)."""
    from ml_training.util import resolve_mnist_npz_path

    mnist_path = resolve_mnist_npz_path()
    print(f"Client: loading MNIST from local file {mnist_path} (no network download)", flush=True)
    (x_train, y_train), (x_test, y_test) = loadMNISTDataset()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    if train_samples is not None:
        x_train = x_train[: int(train_samples)]
        y_train = y_train[: int(train_samples)]
    if test_samples is not None:
        x_test = x_test[: int(test_samples)]
        y_test = y_test[: int(test_samples)]
    # Benchmark subset = first N rows (see benchmarking/shared/mnist_benchmark_subset.py)

    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train_oh), (x_test, y_test_oh)


def load_raw_mnist_image_784(path: str) -> np.ndarray:
    """Load a 28x28 grayscale image as MNIST-space vector [0,1]."""
    import tensorflow as tf

    raw = tf.io.read_file(str(path))
    img = tf.image.decode_image(raw, channels=1, expand_animations=False)
    h = int(img.shape[0])
    w = int(img.shape[1])
    if h != 28 or w != 28:
        raise ValueError(f"--infer-raw-mnist requires a 28x28 image, got {h}x{w}")
    arr = tf.cast(img, tf.float32).numpy().reshape(28 * 28) / 255.0
    return np.asarray(arr, dtype=np.float32)


def load_plain_weights_from_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(str(npz_path))
    w1 = np.asarray(data["w1"], dtype=np.float64)
    w2 = np.asarray(data["w2"], dtype=np.float64)
    b1 = np.asarray(data["b1"], dtype=np.float64)
    b2 = np.asarray(data["b2"], dtype=np.float64)
    return w1, w2, b1, b2


def flatten_plain_weights_to_fixed_ints(
    *,
    w1: np.ndarray,
    w2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    scale: int,
    p: int,
) -> list[int]:
    # Batched MNIST MLP expected shapes: w1(128,784), w2(10,128), b1(128,), b2(10,)
    expected = [(128, 784), (10, 128), (128,), (10,)]
    expected_count = 128 * 784 + 10 * 128 + 128 + 10
    got_shapes = [tuple(w1.shape), tuple(w2.shape), tuple(b1.shape), tuple(b2.shape)]
    if got_shapes != expected:
        raise ValueError(
            f"NPZ weight shape mismatch for secure inference: expected {expected}, got {got_shapes}"
        )
    flat_float = np.concatenate([w1.reshape(-1), w2.reshape(-1), b1.reshape(-1), b2.reshape(-1)])
    if int(flat_float.size) != int(expected_count):
        raise ValueError(f"NPZ parameter count mismatch: expected {expected_count}, got {int(flat_float.size)}")
    flat_int = np.rint(flat_float * float(scale)).astype(np.int64, copy=False)
    return [int(v) % int(p) for v in flat_int.tolist()]


def share_vector_for_all_nodes_pss(values, n_nodes, t, field_size, pss, scale, packing_factor):
    secrets = [int(v * scale) % field_size for v in values]
    chunks = pss.share_vector(secrets, n_nodes, t, packing_factor=int(packing_factor))
    per_node = [[] for _ in range(n_nodes)]
    for chunk in chunks:
        for s in chunk:
            per_node[int(s.node_id) - 1].append(int(s.y))
    return per_node


def wait_for_vector_from_sender(network, context: str, sender_id: int, timeout_s: float):
    start = time.time()
    while True:
        by_sender = network.channel.get_received_vector(context)
        if sender_id in by_sender:
            values = by_sender[sender_id]["values"]
            network.channel.clear_vector(context)
            return values
        if float(timeout_s) > 0 and (time.time() - start > float(timeout_s)):
            raise TimeoutError(f"Timed out waiting for context '{context}' from node {sender_id}")
        time.sleep(0.01)


def mod_p_to_signed(v: int, p: int) -> int:
    v = int(v) % int(p)
    if v > (p // 2):
        return v - p
    return v


def _barrier_timeout(timeout_s: float) -> float:
    if float(timeout_s) > 0:
        return float(timeout_s)
    return 1e9


def wait_for_party_vectors(network, context: str, min_parties: int, timeout_s: float) -> dict:
    start = time.time()
    while True:
        by_sender = network.channel.get_received_vector(context)
        if len(by_sender) >= int(min_parties):
            return by_sender
        if float(timeout_s) > 0 and (time.time() - start > float(timeout_s)):
            raise TimeoutError(
                f"Timed out waiting for context '{context}' from {min_parties} parties (have {len(by_sender)})"
            )
        time.sleep(0.01)


def party_vectors_to_share_vectors(by_sender: dict) -> list:
    share_vectors = []
    for sender_id in sorted(by_sender.keys()):
        vals = [int(v) for v in by_sender[sender_id]["values"]]
        x_coord = int(by_sender[sender_id]["x"])
        share_vectors.append((x_coord, vals))
    return share_vectors


def reconstruct_logits_from_parties(share_vectors, n_classes: int, shamir, p: int) -> list:
    logits = []
    for j in range(int(n_classes)):
        shares_j = [Share(x=x, y=vals[j], node_id=x) for (x, vals) in share_vectors]
        rec = shamir.reconstruct(shares_j)
        logits.append(float(mod_p_to_signed(rec, p)))
    return logits


def reconstruct_batched_logits_from_parties(
    share_vectors, n_samples: int, n_classes: int, shamir, p: int
) -> list[list[float]]:
    all_logits: list[list[float]] = []
    for slot in range(int(n_samples)):
        logits = []
        base = slot * int(n_classes)
        for j in range(int(n_classes)):
            idx = base + j
            shares_j = [Share(x=x, y=vals[idx], node_id=x) for (x, vals) in share_vectors]
            rec = shamir.reconstruct(shares_j)
            logits.append(float(mod_p_to_signed(rec, p)))
        all_logits.append(logits)
    return all_logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribute secret-shared MNIST from client to training nodes")
    parser.add_argument(
        "--topology",
        type=str,
        default="",
        help="YAML path: explicit hosts/ports for each training party (see ml_training.topology).",
    )
    parser.add_argument(
        "--client-node-id",
        type=int,
        default=0,
        help="Sender node_id used in transport headers (default: 0). Ignored when --topology is set.",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=3,
        help="Number of training parties. Ignored when --topology is set.",
    )
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--mnist-samples", type=int, default=None)
    parser.add_argument(
        "--client-test-samples",
        type=int,
        default=-1,
        help="Number of test samples to distribute as shares. -1 means auto/default.",
    )
    parser.add_argument("--field-size", type=int, default=2**32 - 5)
    parser.add_argument("--scale-factor", type=int, default=2**20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init-weights-from-npz",
        type=str,
        default="",
        help="Path to reconstructed model NPZ used for client-side secure weight sharing.",
    )
    parser.add_argument(
        "--share-model-weights",
        action="store_true",
        help="Inference-only: client secret-shares NPZ model weights to nodes.",
    )
    parser.add_argument(
        "--infer-image",
        type=str,
        default="",
        help="Single image inference input; sends n_train=0 and one test sample to secure nodes.",
    )
    parser.add_argument(
        "--infer-raw-mnist",
        action="store_true",
        help="Treat --infer-image as already-MNIST 28x28 grayscale input.",
    )
    parser.add_argument("--barrier-timeout", type=float, default=900.0)
    parser.add_argument(
        "--collect-client-eval",
        action="store_true",
        help="Collect final inference-output shares from nodes and compute accuracy on client.",
    )
    parser.add_argument("--client-eval-samples", type=int, default=100)
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=0.0,
        help="Timeout in seconds for client-side eval waits (0 = no timeout).",
    )
    parser.add_argument(
        "--client-eval-batched-receive",
        action="store_true",
        help="Receive all eval logit shares in one vector (single wait + batched reconstruct).",
    )
    parser.add_argument(
        "--membership-epoch",
        type=int,
        default=0,
        help="Must match training nodes' --membership-epoch (MPC/barrier prefix m{e}_, default: 0).",
    )
    args = parser.parse_args()

    from ml_training.benchmark_stats import log_benchmark

    log_benchmark(
        role="client",
        phase="cold_start",
        wall_sec=float(time.time() - _PROCESS_T0),
    )
    _client_run_t0 = time.time()

    me = MembershipEpochScope(int(args.membership_epoch))

    random.seed(args.seed)
    np.random.seed(args.seed)

    field_size = int(args.field_size)
    scale = int(args.scale_factor)
    if field_size <= 3:
        raise ValueError("--field-size must be > 3")
    if scale <= 0:
        raise ValueError("--scale-factor must be positive")

    infer_path = str(getattr(args, "infer_image", "")).strip()
    if infer_path:
        if bool(getattr(args, "infer_raw_mnist", False)):
            x_vec = load_raw_mnist_image_784(infer_path)
        else:
            import tensorflow as tf

            raw = tf.io.read_file(str(infer_path))
            img = tf.image.decode_image(raw, channels=1, expand_animations=False)
            img = tf.image.resize(img, [28, 28], method="bilinear", antialias=True)
            x_vec = tf.cast(img, tf.float32).numpy().reshape(28 * 28) / 255.0
        x_train = np.zeros((0, 784), dtype=np.float32)
        y_train = np.zeros((0, 10), dtype=np.float32)
        x_test = np.asarray([x_vec], dtype=np.float32)
        # Placeholder label only for shape compatibility in infer-image mode.
        y_test = keras.utils.to_categorical(np.asarray([0], dtype=np.int64), 10).astype(np.float32)
    else:
        train_samples = int(args.mnist_samples) if args.mnist_samples is not None else None
        if int(args.client_test_samples) >= 0:
            test_samples = int(args.client_test_samples)
        elif args.collect_client_eval:
            test_samples = int(args.client_eval_samples)
        else:
            test_samples = train_samples
        (x_train, y_train), (x_test, y_test) = load_mnist_data(train_samples, test_samples)
    n_train = int(len(x_train))
    n_test = int(len(x_test))
    feat_dim = 784
    cls_dim = 10

    if str(args.topology).strip():
        ct = load_client_topology(str(args.topology).strip())
        client_node_id = int(ct.party_id)
        n_nodes = int(ct.n_nodes)
        node_configs = ct.node_configs
        listen_port = int(ct.listen_port)
    else:
        client_node_id = int(args.client_node_id)
        n_nodes = int(args.n_nodes)
        node_configs = {i: {"host": args.host, "port": args.base_port + i} for i in range(1, n_nodes + 1)}
        listen_port = int(args.base_port + client_node_id)

    print(f"Client {client_node_id}: loaded MNIST train={n_train}, test={n_test}")
    print(f"Client {client_node_id}: membership epoch e={me.e} (MPC/barrier prefix m{me.e}_)")

    network = create_mpc_network(
        node_id=client_node_id,
        node_configs=node_configs,
        port=listen_port,
        membership_epoch=int(me.e),
    )
    shamir = ShamirSecretSharing(field_size)
    pss = PackedShamirSecretSharing(field_size)
    raw_packing = int(pss.max_packing_factor(int(n_nodes), int(args.t)))
    if raw_packing <= 0:
        raise ValueError(f"No valid PSS packing factor for n_nodes={n_nodes}, t={args.t}. Need n_nodes > t.")
    n_active = int(n_nodes)
    pss_packing_factor = min(raw_packing, get_max_safe_packing_factor(int(args.t), n_active))
    if pss_packing_factor < raw_packing:
        print(
            f"Client {client_node_id}: [Packing safety] Capped packing factor "
            f"{raw_packing} -> {pss_packing_factor} (2*(t+s-1) < n_active={n_active})"
        )
    print(f"Client {client_node_id}: PSS mode enabled; packing factor={pss_packing_factor}")

    if bool(getattr(args, "share_model_weights", False)):
        npz_path = str(getattr(args, "init_weights_from_npz", "")).strip()
        if not npz_path:
            raise ValueError("--share-model-weights requires --init-weights-from-npz")
        print(f"Client {client_node_id}: model-weight init mode = client-shared (source NPZ: {npz_path})")
        w1, w2, b1, b2 = load_plain_weights_from_npz(npz_path)
        flat_mod_p = flatten_plain_weights_to_fixed_ints(
            w1=w1,
            w2=w2,
            b1=b1,
            b2=b2,
            scale=int(scale),
            p=int(field_size),
        )
        shamir_dist = ShamirSecretSharing(int(field_size))
        per_node = [[] for _ in range(int(n_nodes))]
        for secret in flat_mod_p:
            shares = shamir_dist.share(int(secret), int(n_nodes), int(args.t))
            for s in shares:
                per_node[int(s.node_id) - 1].append(int(s.y) % int(field_size))
        w_ctx = me.ctx("init_weights_from_npz/v1")
        for target in range(1, int(n_nodes) + 1):
            network.channel.send_vector(int(target), w_ctx, x=int(client_node_id), values=per_node[target - 1])
        print(f"Client {client_node_id}: secure model-weight shares distributed from NPZ.")

    meta_ctx = me.ctx("dataset/meta/v1")
    _t_dist0 = time.time()
    feat_stored = (int(feat_dim) + int(pss_packing_factor) - 1) // int(pss_packing_factor)
    cls_stored = (int(cls_dim) + int(pss_packing_factor) - 1) // int(pss_packing_factor)
    meta_payload = [n_train, n_test, feat_dim, cls_dim, 1, int(pss_packing_factor), int(feat_stored), int(cls_stored)]
    for target in range(1, n_nodes + 1):
        network.channel.send_vector(target, meta_ctx, x=target, values=meta_payload)

    for split_name, x_src, y_src in (("train", x_train, y_train), ("test", x_test, y_test)):
        n_split = int(len(x_src))
        for idx in range(n_split):
            x_per_node = share_vector_for_all_nodes_pss(
                x_src[idx], n_nodes, args.t, field_size, pss, scale, int(pss_packing_factor)
            )
            y_per_node = share_vector_for_all_nodes_pss(
                y_src[idx], n_nodes, args.t, field_size, pss, scale, int(pss_packing_factor)
            )

            x_ctx = me.ctx(f"dataset/{split_name}/x/{idx}")
            y_ctx = me.ctx(f"dataset/{split_name}/y/{idx}")
            for target in range(1, n_nodes + 1):
                network.channel.send_vector(target, x_ctx, x=target, values=x_per_node[target - 1])
                network.channel.send_vector(target, y_ctx, x=target, values=y_per_node[target - 1])

            if idx % 512 == 0:
                print(f"Client {client_node_id}: distributed {split_name} sample {idx + 1}/{n_split}")

    print("Client distribution complete: dataset shares sent to all nodes.")
    _dist_sec = float(time.time() - _t_dist0)
    print(f"Client Distribution Time: {_dist_sec:.6f}s")
    log_benchmark(
        role="client",
        phase="distribution",
        wall_sec=_dist_sec,
        extra={"train_samples": n_train, "test_samples": n_test},
    )

    if args.collect_client_eval:
        _t_eval0 = time.time()
        n_eval = min(int(args.client_eval_samples), n_test)
        eval_indices_vals = wait_for_vector_from_sender(
            network, me.ctx("client_eval_final/meta_indices"), sender_id=1, timeout_s=float(args.eval_timeout)
        )
        eval_indices = np.asarray(list(eval_indices_vals), dtype=np.int64)
        if eval_indices.size > n_eval:
            eval_indices = eval_indices[:n_eval]
        n_eval = int(eval_indices.size)

        correct = 0
        p = int(field_size)
        n_classes = 10
        batched_receive = bool(getattr(args, "client_eval_batched_receive", False))
        _t_share_wait0 = time.time()

        if batched_receive:
            ctx = me.ctx("client_eval_final/logits/batched")
            by_sender = wait_for_party_vectors(
                network, ctx, int(args.t) + 1, float(args.eval_timeout)
            )
            network.channel.clear_vector(ctx)
            _share_wait_sec = float(time.time() - _t_share_wait0)
            _t_reconstruct0 = time.time()
            share_vectors = party_vectors_to_share_vectors(by_sender)
            all_logits = reconstruct_batched_logits_from_parties(
                share_vectors, n_eval, n_classes, shamir, p
            )
            _reconstruct_sec = float(time.time() - _t_reconstruct0)
            for slot, logits in enumerate(all_logits):
                logits_np = np.asarray(logits, dtype=np.float64)
                pred = int(np.argmax(logits_np))
                if infer_path:
                    print(f"Client predicted digit: {pred}")
                    print(f"Client logits (10 classes): {logits_np.tolist()}")
                else:
                    true_label = int(np.argmax(np.asarray(y_test[int(eval_indices[slot])], dtype=np.float64)))
                    if pred == true_label:
                        correct += 1
        else:
            for slot in range(n_eval):
                ctx = me.ctx(f"client_eval_final/logits/{slot}")
                by_sender = wait_for_party_vectors(
                    network, ctx, int(args.t) + 1, float(args.eval_timeout)
                )
                network.channel.clear_vector(ctx)

                share_vectors = party_vectors_to_share_vectors(by_sender)
                logits = reconstruct_logits_from_parties(share_vectors, n_classes, shamir, p)

                pred = int(np.argmax(np.asarray(logits, dtype=np.float64)))
                logits_np = np.asarray(logits, dtype=np.float64)
                if infer_path:
                    print(f"Client predicted digit: {pred}")
                    print(f"Client logits (10 classes): {logits_np.tolist()}")
                else:
                    true_label = int(np.argmax(np.asarray(y_test[int(eval_indices[slot])], dtype=np.float64)))
                    if pred == true_label:
                        correct += 1
            _share_wait_sec = float(time.time() - _t_share_wait0)
            _reconstruct_sec = 0.0

        if infer_path:
            print("Client Final Accuracy: N/A (infer-image mode; label is a dummy placeholder)")
        else:
            acc = float(correct) / float(max(1, n_eval))
            print(f"Client Final Accuracy ({n_eval} samples): {acc*100:.2f}%")
        _eval_sec = float(time.time() - _t_eval0)
        print(f"Client Eval Time: {_eval_sec:.6f}s")
        if batched_receive:
            print(f"Client Eval Share Wait Time: {_share_wait_sec:.6f}s")
            print(f"Client Eval Reconstruct Time: {_reconstruct_sec:.6f}s")
        log_benchmark(
            role="client",
            phase="eval",
            wall_sec=_eval_sec,
            extra={
                "eval_samples": n_eval,
                "batched_receive": int(batched_receive),
                "share_wait_sec": f"{_share_wait_sec:.6f}",
                "reconstruct_sec": f"{_reconstruct_sec:.6f}",
            },
        )

        network.barrier(me.barrier_tag("client_eval_final_done"), timeout=_barrier_timeout(float(args.eval_timeout)))

    log_benchmark(
        role="client",
        phase="total",
        wall_sec=float(time.time() - _client_run_t0),
    )
    time.sleep(0.5)
    network.stop()

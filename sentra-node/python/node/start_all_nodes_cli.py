import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start all SENTRA nodes")
    parser.add_argument("--n-nodes", type=int, default=5, help="Number of nodes to start (default: 5)")
    parser.add_argument("--base-port", type=int, default=8000, help="Base port number (default: 8000)")
    parser.add_argument("--host", type=str, default="localhost", help="Host for all nodes (default: localhost)")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size passed to each node (default: 8)")
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation grouping for batched runner; effective batch = batch-size * accum-steps",
    )
    parser.add_argument("--num-epochs", type=int, default=1, help="Epoch count passed to each node (default: 1)")
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate passed to each node (default: 0.01)"
    )
    parser.add_argument("--t", type=int, default=1, help="Privacy threshold (default: 1)")
    parser.add_argument("--dataset", choices=["mnist"], default="mnist", help="Dataset mode for each node (default: mnist)")
    parser.add_argument("--mnist-samples", type=int, default=128, help="MNIST sample count per node when --dataset mnist (default: 128)")
    parser.add_argument(
        "--post-metrics-barrier-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait at post-metrics barrier before shutdown (default: 180)",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Global random seed passed to all nodes (default: 2026)")
    parser.add_argument("--batched", action="store_true", dest="batched", help="Deprecated; batched mode is always enabled.")
    parser.set_defaults(batched=True)
    parser.add_argument("--loss-mode", choices=["mse", "softmax"], default="softmax", help="Output gradient mode for batched MNIST secure training (default: softmax)")
    parser.add_argument("--scale-factor", type=int, default=2**20, help="Fixed-point scale for batched secure MNIST (default: 2^20 = 1048576)")
    parser.add_argument("--field-size", type=int, default=2**32 - 5, help="Finite field modulus for batched secure MNIST (default: 2^32-5)")
    parser.add_argument("--softmax-temperature", type=float, default=1.0, help="Softmax temperature for batched secure MNIST (default: 1.0)")
    parser.add_argument("--grad-clip", type=float, default=2.0, help="Gradient clip bound for batched secure MNIST (default: 2.0)")
    parser.add_argument("--logit-clip", type=float, default=8.0, help="Logit clip bound before secure exp approximation (default: 8.0)")
    parser.add_argument("--exp-approx", choices=["taylor5", "pade22"], default="pade22", help="Secure exp approximation used in softmax (default: pade22)")
    parser.add_argument("--softmax-grad-mode", choices=["secure_approx", "opened_exact"], default="secure_approx", help="Softmax CE gradient path in batched mode (default: secure_approx)")
    parser.add_argument("--explode-logit-threshold", type=float, default=10.0, help="Logit explosion threshold for batched secure MNIST (default: 10.0)")
    parser.add_argument("--loss-growth-threshold", type=float, default=5.0, help="Loss growth threshold for instability checks (default: 5.0)")
    parser.add_argument("--grad-norm-threshold", type=float, default=5.0, help="Estimated gradient norm threshold for instability checks (default: 5.0)")
    parser.add_argument("--no-abort-on-instability", action="store_true", help="Do not abort batched secure training when instability is detected")
    parser.add_argument("--debug-numerics", action="store_true", help="Enable numeric probes (updates/logits/dz2) in batched secure mode")
    parser.add_argument("--debug-division", action="store_true", help="Enable secure division debug summaries in batched secure mode")
    parser.add_argument("--packed-forward-pilot", action="store_true", help="Enable packed forward kernel pilot mode in batched secure runner")
    parser.add_argument("--packed-forward-native", action="store_true", help="Enable experimental packed-native dense1 forward kernel in batched secure runner")
    parser.add_argument("--packed-end2end", action="store_true", help="Enable experimental packed API path in batched secure runner")
    parser.add_argument("--dpss-refresh-interval", type=int, default=0, help="Herzberg proactive refresh every N epochs in batched mode (0=disabled). Requires --batched --enable-network.")
    parser.add_argument("--membership-epoch", type=int, default=0, help="Membership epoch e for batched nodes + client distributor: MPC/barrier prefix m{e}_ (default: 0).")
    parser.add_argument("--enable-failure-detection", action="store_true", help="Enable heartbeat-based failure detection; use dynamic n_active for packing safety.")
    parser.add_argument("--enable-dropout-reshare-recovery", action="store_true", help="Forward to batched runner: on failure, Lagrange reshare weights among survivors and resume if safe.")
    parser.add_argument("--enable-join-recovery", action="store_true", help="Forward to batched runner: on node rejoin, Lagrange reshare weights to new committee and resume.")
    parser.add_argument("--export-reconstructed-model", type=str, default="", help="Export reconstructed final model to this .npz path (opener node writes file)")
    parser.add_argument("--export-timeout", type=float, default=180.0, help="Timeout in seconds for final model reconstruction/export (default: 180)")
    parser.add_argument("--distribute-dataset-shares", action="store_true", help="Owner node shares MNIST to peers (local simulation mode).")
    parser.add_argument("--dataset-owner-node", type=int, default=1, help="Owner node id for --distribute-dataset-shares (default: 1).")
    parser.add_argument("--receive-dataset-shares-from-client", action="store_true", help="Nodes receive only pre-shared dataset from external client distributor.")
    parser.add_argument("--dataset-source-node-id", type=int, default=0, help="External sender node_id for client distributor mode (default: 0).")
    parser.add_argument("--dataset-distribution-timeout", type=float, default=900.0, help="Timeout for dataset share distribution/reception barrier (seconds).")
    parser.add_argument(
        "--start-client-distributor",
        action="store_true",
        help="In headless mode, auto-start the client module (python -m sentra_client) after launching nodes.",
    )
    parser.add_argument("--client-eval-after-training", action="store_true", help="After training, nodes send final inference shares to client; client reconstructs accuracy.")
    parser.add_argument("--client-eval-samples", type=int, default=100, help="Sample count for client-side reconstructed final accuracy.")
    parser.add_argument("--client-test-samples", type=int, default=-1, help="Number of test shares to distribute from client (-1: auto; with client eval uses client-eval-samples).")
    parser.add_argument("--client-eval-timeout", type=float, default=0.0, help="Timeout for client-side evaluation share collection and barrier in seconds (0 = no timeout).")
    parser.add_argument(
        "--client-eval-batched-receive",
        action="store_true",
        help="Send/receive all eval logit shares in one vector (single client wait + batched Shamir reconstruct).",
    )
    parser.add_argument(
        "--init-weights-from-npz",
        type=str,
        default="",
        help="Initialize secure node weights from reconstructed model (.npz).",
    )
    parser.add_argument(
        "--infer-image",
        type=str,
        default="",
        help="Run inference on a single image via secure multi-party execution (requires --init-weights-from-npz).",
    )
    parser.add_argument(
        "--infer-raw-mnist",
        action="store_true",
        help="Treat --infer-image input as 28x28 MNIST PNG (no drawn-digit preprocessing).",
    )
    parser.add_argument(
        "--client-shares-model-weights",
        action="store_true",
        help="Inference-only: client secret-shares NPZ model weights to nodes (no node-side plaintext model load).",
    )
    parser.add_argument("--use-kvs-dataset", action="store_true", help="Store dataset shares in local KVS and retrieve mini-batches from KVS. Requires distributed dataset.")
    parser.add_argument("--use-weight-versioning", action="store_true", help="Store weight shares to local KVS with v_theta after each epoch.")
    parser.add_argument("--headless", action="store_true", help="Run all nodes in this terminal and wait for completion (writes per-node logs)")
    parser.add_argument("--record-results-xlsx", type=str, default="", help="Append run parameters + parsed final metrics to an Excel file (implies --headless)")
    return parser


def parse_and_validate_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.batched:
        args.dataset = "mnist"

    if args.distribute_dataset_shares and args.receive_dataset_shares_from_client:
        raise ValueError("Use only one dataset-sharing mode: owner-node or external client distributor")
    if args.start_client_distributor and not args.receive_dataset_shares_from_client:
        raise ValueError("--start-client-distributor requires --receive-dataset-shares-from-client")
    if args.client_eval_after_training and not args.receive_dataset_shares_from_client:
        raise ValueError("--client-eval-after-training requires --receive-dataset-shares-from-client")
    if getattr(args, "enable_dropout_reshare_recovery", False):
        if not bool(args.enable_failure_detection):
            raise ValueError("--enable-dropout-reshare-recovery requires --enable-failure-detection")
    if getattr(args, "enable_join_recovery", False):
        if not bool(args.enable_failure_detection):
            raise ValueError("--enable-join-recovery requires --enable-failure-detection")
    if getattr(args, "use_kvs_dataset", False):
        if not (bool(args.distribute_dataset_shares) or bool(args.receive_dataset_shares_from_client)):
            raise ValueError("--use-kvs-dataset requires --distribute-dataset-shares or --receive-dataset-shares-from-client")
    if bool(args.infer_image):
        if not bool(args.init_weights_from_npz):
            raise ValueError("--infer-image requires --init-weights-from-npz")
        if not bool(args.receive_dataset_shares_from_client):
            raise ValueError("--infer-image requires --receive-dataset-shares-from-client")
        if not bool(args.client_eval_after_training):
            raise ValueError("--infer-image requires --client-eval-after-training")
        if int(args.n_nodes) < 2:
            raise ValueError("--infer-image requires --n-nodes >= 2")
        # Inference-only mode: client sends one test sample, nodes skip training.
        args.num_epochs = 0
        args.mnist_samples = 0
        args.client_test_samples = 1
        args.client_eval_samples = 1
        # Default to stronger confidentiality for inference mode.
        args.client_shares_model_weights = True
    if bool(args.client_shares_model_weights) and not bool(args.infer_image):
        raise ValueError("--client-shares-model-weights is supported only with --infer-image")
    if (
        bool(args.packed_forward_native)
        and bool(args.start_client_distributor)
        and bool(args.client_eval_after_training)
        and float(args.client_eval_timeout) > 0.0
    ):
        print(
            "Packed native forward is enabled; overriding --client-eval-timeout to 0 "
            "(no timeout) to avoid premature client disconnect during long runs."
        )
        args.client_eval_timeout = 0.0

    return args

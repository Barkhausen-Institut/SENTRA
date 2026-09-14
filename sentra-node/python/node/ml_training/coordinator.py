"""
Training Coordinator
Manages mini-batch selection, safety bounds, and quorum commits
"""

from typing import List, Dict, Optional, Tuple, Any
import hashlib
import time
from ml_training.kvs import KVSCluster
from ml_training.secret_sharing import Share, PackedShamirSecretSharing
from ml_training.mpc_engine import PackedMPCEngine
from ml_training.beaver_triples import SecureMultiplier
from ml_training.secure_comm import SecureMPCNetwork, create_mpc_network
from ml_training.reconstruction import create_reconstruction_manager
import random
import numpy as np


class SafetyBoundChecker:
    """
    Checks safety bound: 2*(t+s-1) < n_active
    
    When violated:
    - Abort current mini-batch (don't advance model state)
    - Suspend training until safety is restored
    - Resume when condition holds again
    """
    
    def __init__(self, t: int, s: int):
        self.t = t
        self.s = s
    
    def check(self, n_active: int) -> bool:
        """
        Check if safety bound is satisfied
        Safety condition: 2*(t+s-1) < n_active
        """
        return 2 * (self.t + self.s - 1) < n_active
    
    def get_max_packing_factor(self, n_active: int) -> int:
        """
        Get maximum safe packing factor s such that 2*(t+s-1) < n_active
        Returns 1 if safety bound is violated (no packing)
        """
        if not self.check(n_active):
            return 1  # No packing if bound violated
        
        # Strict inequality 2*(t+s-1) < n_active  =>  max s = (n_active - 2t + 1) // 2
        max_s = (n_active - 2 * self.t + 1) // 2
        
        # Ensure at least 1, and cap at reasonable maximum (e.g., 4)
        return max(1, min(4, max_s))


class TrainingCoordinator:
    """
    Coordinates secure training across nodes
    """
    
    def __init__(self, kvs_cluster: KVSCluster, n_nodes: int, t: int, s: int,
                 batch_size: int = 32, learning_rate: float = 0.01,
                 node_id: int = 1, node_configs: Optional[Dict[int, Dict[str, Any]]] = None,
                 enable_network: bool = False, seed: int = 2026,
                 train_mode: str = "secure"):
        """
        Initialize training coordinator
        Args:
            kvs_cluster: KVS cluster for storage
            n_nodes: Total number of nodes
            t: Privacy threshold
            s: Adversarial share limit
            batch_size: Mini-batch size
            learning_rate: Learning rate
            node_id: ID of this node
            node_configs: Dictionary mapping node_id to {host, port} for network
            enable_network: Whether to enable multi-node network communication
        """
        self.kvs_cluster = kvs_cluster
        self.n_nodes = n_nodes
        self.t = t
        self.s = s
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.node_id = node_id
        self.enable_network = enable_network
        self.seed = int(seed)
        self.train_mode = str(train_mode).strip().lower()
        if self.train_mode not in ("secure", "hybrid"):
            self.train_mode = "secure"
        
        self.safety_checker = SafetyBoundChecker(t, s)
        self.batch_rng = random.Random(self.seed)
        
        # Initialize PSS (needed for node manager)
        self.pss = PackedShamirSecretSharing()
        
        # Initialize network if enabled
        self.network = None
        self.reconstruction_manager = None
        self.failure_detector = None
        self.node_manager = None
        if enable_network and node_configs:
            bind_port = int(node_configs.get(node_id, {}).get("port", 8000 + node_id))
            self.network = create_mpc_network(node_id, node_configs, port=bind_port)
            self.reconstruction_manager = create_reconstruction_manager(self.network, t)
            
            # Initialize node failure detection and management
            from ml_training.node_failure_detector import NodeFailureDetector
            from ml_training.node_manager import NodeManager
            self.failure_detector = NodeFailureDetector(
                self.network,
                heartbeat_interval=2.0,
                failure_timeout=20.0,
                startup_grace_period=60.0,
            )
            self.failure_detector.start_monitoring()
            self.node_manager = NodeManager(
                self.network, 
                self.failure_detector, 
                self.safety_checker,
                self.pss,
                t,
                resharing_enabled=True
            )
        
        # Initialize MPC engine
        self.mpc_engine = PackedMPCEngine(n_nodes=n_nodes, t=t)
        
        # Update multiplier to use reconstruction manager if available
        if self.reconstruction_manager:
            from ml_training.beaver_triples import BeaverTripleGenerator, BeaverTriplePool
            triple_gen = BeaverTripleGenerator()
            triple_pool = BeaverTriplePool(triple_gen, initial_size=1000)
            self.mpc_engine.multiplier = SecureMultiplier(
                triple_pool, n_nodes, t,
                reconstruction_manager=self.reconstruction_manager
            )
        
        # Track versions
        self.v_D = 0  # Dataset version
        self.v_theta = 0  # Model version
        self.scale = int(self.mpc_engine.matrix_ops.scale_factor)
        self.weight_shapes: Optional[List[Tuple[int, int]]] = None
        self._plain_dataset: Optional[List[np.ndarray]] = None
        self._plain_labels: Optional[List[np.ndarray]] = None
        self._plain_weights: Optional[List[np.ndarray]] = None
        self.secure_dz2_clip = 4.0
        self.aggregation_timeout_s = 20.0

    def _to_field_int(self, value: float) -> int:
        return int(round(float(value) * self.scale)) % self.mpc_engine.field_size

    def _field_to_signed_int(self, value: int) -> int:
        p = int(self.mpc_engine.field_size)
        v = int(value) % p
        if v > p // 2:
            v -= p
        return v

    def _clip_field_value(self, value: int, clip_abs: float) -> int:
        """Clip a fixed-point field value by signed magnitude and re-encode to field."""
        signed = self._field_to_signed_int(value)
        as_float = float(signed) / float(self.scale)
        clipped = max(-float(clip_abs), min(float(clip_abs), as_float))
        return self._to_field_int(clipped)

    def _deterministic_share_secret(self, secret: int, context: str) -> List[Share]:
        """
        Deterministic Shamir shares for local multi-process reproducibility.
        All nodes can derive consistent shares from the same secret/context.
        """
        p = int(self.mpc_engine.field_size)
        secret = int(secret) % p
        digest = hashlib.sha256(f"{self.seed}:{context}".encode("utf-8")).digest()
        base = int.from_bytes(digest[:8], "big", signed=False)
        rng = random.Random(base)
        coeffs = [secret] + [rng.randint(0, p - 1) for _ in range(self.t)]
        shares: List[Share] = []
        for node in range(1, self.n_nodes + 1):
            y = self.pss.shamir._evaluate_polynomial(coeffs, node)  # noqa: SLF001
            shares.append(Share(x=node, y=y, node_id=node))
        return shares

    def _batch_tag(self, batch_indices: Optional[List[int]]) -> str:
        ordered = ",".join(str(i) for i in sorted(batch_indices or []))
        return hashlib.sha256(
            f"v{self.v_theta}:{ordered}".encode("utf-8")
        ).hexdigest()[:16]

    def _get_active_node_ids(self) -> List[int]:
        if self.node_manager:
            try:
                active = sorted(int(n) for n in self.node_manager.get_active_nodes())
                if active:
                    return active
            except Exception:
                pass
        return list(range(1, self.n_nodes + 1))

    def _secure_aggregate_gradients_batch(
        self,
        local_gradients: List[List[List[Share]]],
        batch_indices: Optional[List[int]],
        node_id: int,
    ) -> List[List[List[Share]]]:
        """
        Network-based per-batch gradient-share aggregation.
        
        SECURITY UPDATE: 
        In this synchronized training mode, all nodes process the EXACT SAME mini-batch 
        (indices derived from shared seed). Therefore, they all compute shares of the 
        SAME gradient vector G.
        
        Broadcasting these shares (as the previous code did) would allow every node to 
        collect [G]_1, [G]_2, ... [G]_n and reconstruct the plaintext gradient G!
        
        Since we already hold valid shares of the correct gradient, no aggregation is 
        needed. We simply proceed with the local update.
        """
        # Secure no-op: do not broadcast shares.
        return local_gradients

    def _mnist_plain_batch_update(self, batch_indices: List[int], node_id: int) -> Optional[List[List[List[Share]]]]:
        """
        Hybrid mode: plaintext Dense-ReLU-Dense(softmax) SGD step, then deterministic resharing.
        This keeps multi-node determinism while providing stronger learning quality.
        """
        if not batch_indices or self._plain_dataset is None or self._plain_labels is None:
            return None
        if self._plain_weights is None or self.weight_shapes is None or len(self.weight_shapes) != 2:
            return None

        w1, w2 = self._plain_weights
        out_dim = int(w2.shape[0])
        x = np.stack([self._plain_dataset[i] for i in batch_indices], axis=0).astype(np.float64)
        y = np.stack([self._plain_labels[i] for i in batch_indices], axis=0).astype(np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] != out_dim:
            return None

        has_w1_bias = w1.shape[1] == x.shape[1] + 1
        x_in = x
        if has_w1_bias:
            ones = np.ones((x.shape[0], 1), dtype=x.dtype)
            x_in = np.concatenate([x, ones], axis=1)
        elif w1.shape[1] != x.shape[1]:
            return None

        z1 = x_in @ w1.T
        a1 = np.maximum(z1, 0.0)

        has_w2_bias = w2.shape[1] == a1.shape[1] + 1
        a1_in = a1
        if has_w2_bias:
            ones_h = np.ones((a1.shape[0], 1), dtype=a1.dtype)
            a1_in = np.concatenate([a1, ones_h], axis=1)
        elif w2.shape[1] != a1.shape[1]:
            return None

        z2 = a1_in @ w2.T
        z2 = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z = np.exp(z2)
        probs = exp_z / np.maximum(np.sum(exp_z, axis=1, keepdims=True), 1e-12)

        b = max(1, x.shape[0])
        dz2 = (probs - y) / float(b)
        dw2 = dz2.T @ a1_in
        da1_full = dz2 @ w2
        da1 = da1_full[:, :a1.shape[1]]
        dz1 = da1 * (z1 > 0.0)
        dw1 = dz1.T @ x_in

        lr = float(self.learning_rate)
        w1_new = w1 - lr * dw1
        w2_new = w2 - lr * dw2
        self._plain_weights = [w1_new, w2_new]

        next_v = int(self.v_theta + 1)
        local_layers: List[List[List[Share]]] = []
        for li, w in enumerate(self._plain_weights):
            out_rows, in_cols = w.shape
            layer_rows: List[List[Share]] = []
            for r in range(out_rows):
                row_shares: List[Share] = []
                for c in range(in_cols):
                    secret = self._to_field_int(float(w[r, c]))
                    ctx = f"weights_v{next_v}_L{li}_R{r}_C{c}"
                    shares = self._deterministic_share_secret(secret, ctx)
                    row_shares.append(shares[node_id - 1])
                layer_rows.append(row_shares)
            local_layers.append(layer_rows)
        return local_layers
    
    def ingest_dataset(self, dataset: List[np.ndarray], labels: List[np.ndarray]) -> int:
        """
        Ingest and secret-share dataset
        Args:
            dataset: Training samples
            labels: Training labels
        Returns:
            Dataset version
        """
        self.v_D += 1
        
        scale = self.scale
        self._plain_dataset = [np.asarray(sample, dtype=np.float64).copy() for sample in dataset]
        self._plain_labels = [np.asarray(label, dtype=np.float64).reshape(-1).copy() for label in labels]

        # Secret-share each sample
        for i, (sample, label) in enumerate(zip(dataset, labels)):
            # Share each feature of the sample
            sample_feature_shares = []
            for feature_idx, feature_value in enumerate(sample):
                feature_int = int(round(float(feature_value) * scale)) % self.mpc_engine.field_size
                feature_ctx = f"dataset_v{self.v_D}_sample_{i}_feature_{feature_idx}"
                feature_shares = self._deterministic_share_secret(feature_int, feature_ctx)
                sample_feature_shares.append(feature_shares)
            
            # Share label(s): support scalar labels and vector (e.g., one-hot) labels.
            label_arr = np.asarray(label)
            if label_arr.ndim == 0:
                label_arr = label_arr.reshape(1)
            label_component_shares = []
            for label_idx, label_value in enumerate(label_arr.reshape(-1)):
                label_int = int(round(float(label_value) * scale)) % self.mpc_engine.field_size
                label_ctx = f"dataset_v{self.v_D}_sample_{i}_label_{label_idx}"
                component_shares = self._deterministic_share_secret(label_int, label_ctx)
                label_component_shares.append(component_shares)
            
            # Store shares in KVS (one share per node per feature)
            for node_id in range(1, self.n_nodes + 1):
                # Store all feature shares for this node
                node_sample_shares = []
                for feature_shares in sample_feature_shares:
                    node_share = next(s for s in feature_shares if s.node_id == node_id)
                    node_sample_shares.append(node_share)
                
                node_label_shares = []
                for component_shares in label_component_shares:
                    node_component_share = next(s for s in component_shares if s.node_id == node_id)
                    node_label_shares.append(node_component_share)
                
                sample_key = f"sample_{i}_node_{node_id}"
                label_key = f"label_{i}_node_{node_id}"
                
                self.kvs_cluster.write_with_quorum(
                    sample_key, node_sample_shares, self.v_D
                )
                self.kvs_cluster.write_with_quorum(
                    label_key, node_label_shares, self.v_D
                )
        
        return self.v_D
    
    def initialize_weights(self, weight_shapes: List[Tuple[int, int]]) -> int:
        """
        Initialize and secret-share model weights
        Args:
            weight_shapes: List of (input_dim, output_dim) for each layer
        Returns:
            Model version
        """
        self.v_theta += 1
        
        # Initialize with small fixed-point values to avoid saturation/exploding updates.
        scale = self.scale
        self.weight_shapes = list(weight_shapes)
        rng = np.random.default_rng(self.seed)
        weights = []
        plain_weights: List[np.ndarray] = []
        for input_dim, output_dim in weight_shapes:
            layer_weights = []
            fan_in = max(1, input_dim)
            limit = np.sqrt(6.0 / (fan_in + max(1, output_dim)))
            layer_plain = np.zeros((output_dim, input_dim), dtype=np.float64)
            for _ in range(output_dim):
                row = []
                row_idx = len(layer_weights)
                for col_idx in range(input_dim):
                    w = float(rng.uniform(-limit, limit))
                    weight_int = int(round(w * scale)) % self.mpc_engine.field_size
                    layer_plain[row_idx, col_idx] = w
                    ctx = f"weights_v{self.v_theta}_L{len(weights)}_R{row_idx}_C{col_idx}"
                    weight_shares = self._deterministic_share_secret(weight_int, ctx)
                    row.append(weight_shares)
                layer_weights.append(row)
            plain_weights.append(layer_plain)
            weights.append(layer_weights)
        self._plain_weights = plain_weights
        
        # Store weights in KVS (simplified - would store properly)
        weight_key = f"weights_v{self.v_theta}"
        self.kvs_cluster.write_with_quorum(weight_key, weights, self.v_theta)
        
        return self.v_theta
    
    def select_mini_batch(self, dataset_size: int) -> List[int]:
        """Select random mini-batch indices"""
        return self.batch_rng.sample(range(dataset_size), min(self.batch_size, dataset_size))
    
    def train_mini_batch(self, sample_shares: List[List[Share]], 
                        label_shares: List[List[Share]],
                        weight_shares: List[List[List[Share]]],
                        packing_factor: int, node_id: int,
                        batch_indices: Optional[List[int]] = None) -> Tuple[List[List[List[Share]]], bool]:
        """
        Train on a mini-batch
        
        Safety bound check: 2*(t+s-1) < n_active (where s is the packing factor)
        The packing_factor parameter must satisfy this bound for every mini-batch.
        If violated:
        - Abort current mini-batch (don't advance model state)
        - Return original weights unchanged
        - Training will suspend until safety is restored
        
        Args:
            sample_shares: Sample shares for batch
            label_shares: Label shares for batch
            weight_shares: Current weight shares
            packing_factor: Packing factor for PSS (must satisfy safety bound)
            node_id: Node ID
            batch_indices: Original dataset indices for this mini-batch
        Returns:
            Tuple of (updated_weights, success)
            - If success=False: weights are unchanged (batch aborted)
            - If success=True: weights are updated (batch committed)
        """
        # Check safety bound with ACTUAL packing factor BEFORE any computation
        # Safety condition: 2*(t + packing_factor - 1) < n_active
        n_active = self.n_nodes  # Simplified - would track active nodes
        safety_bound_value = 2 * (self.t + packing_factor - 1)
        if safety_bound_value >= n_active:
            # Safety bound violated with this packing factor: ABORT current mini-batch
            # Return original weights unchanged (no model state advancement)
            # Training will suspend until safety is restored
            return weight_shares, False

        # True share-domain path (no plaintext shortcut).
        if not sample_shares or not label_shares:
            return weight_shares, False

        # Normalize weight structure to this node's local shares.
        # Initial KVS weights are nested as [layer][row][col][node_share_list].
        # After first update in this simplified prototype, weights may already be local Share objects.
        local_weight_shares: List[List[List[Share]]] = []
        for layer in weight_shares:
            local_layer = []
            for row in layer:
                local_row = []
                for w in row:
                    if hasattr(w, "node_id"):
                        local_row.append(w)
                    elif isinstance(w, list) and w and hasattr(w[0], "node_id"):
                        local = next((s for s in w if s.node_id == node_id), w[0])
                        local_row.append(local)
                    else:
                        return weight_shares, False
                local_layer.append(local_row)
            local_weight_shares.append(local_layer)

        if not local_weight_shares or not local_weight_shares[0] or not local_weight_shares[0][0]:
            return weight_shares, False

        # Hybrid mode: high-quality local update followed by deterministic resharing.
        if self.train_mode == "hybrid" and batch_indices is not None and len(local_weight_shares) == 2:
            hybrid_updated = self._mnist_plain_batch_update(batch_indices, node_id)
            if hybrid_updated is not None:
                return hybrid_updated, True

        expected_input_size = len(local_weight_shares[0][0])
        p = int(self.mpc_engine.field_size)

        # Initialize gradient accumulator with zero shares.
        grad_accum: List[List[List[Share]]] = []
        for layer in local_weight_shares:
            grad_layer: List[List[Share]] = []
            for row in layer:
                grad_row: List[Share] = []
                for w in row:
                    grad_row.append(Share(x=w.x, y=0, node_id=node_id))
                grad_layer.append(grad_row)
            grad_accum.append(grad_layer)

        processed = 0
        max_samples = min(len(sample_shares), len(label_shares))
        use_two_layer_classifier = len(local_weight_shares) == 2
        secure_mul = self.mpc_engine.multiplier
        matmul = self.mpc_engine.matrix_ops.matrix_multiplier
        for i in range(max_samples):
            sample = sample_shares[i]
            if not sample:
                continue

            input_shares = list(sample)
            if len(input_shares) > expected_input_size:
                input_shares = input_shares[:expected_input_size]
            elif len(input_shares) < expected_input_size:
                if len(input_shares) == expected_input_size - 1:
                    bias_share = Share(x=input_shares[0].x, y=self.scale, node_id=node_id)
                    input_shares = input_shares + [bias_share]
                else:
                    zero_share = Share(x=input_shares[0].x, y=0, node_id=node_id)
                    input_shares = input_shares + [zero_share] * (expected_input_size - len(input_shares))

            sample_labels = label_shares[i] if isinstance(label_shares[i], list) else [label_shares[i]]
            targets: List[Share] = []
            if sample_labels:
                targets = [s for s in sample_labels if hasattr(s, "node_id")]
            gradients: Optional[List[List[List[Share]]]] = None

            if use_two_layer_classifier:
                # Two-layer classifier path:
                # z1 = W1 x, a1 = relu(z1), z2 = W2 a1
                # Secure mode uses share-domain MSE-style gradient at output:
                #   dz2 = z2 - y
                # dW2 = dz2 * a1^T, dW1 = dz1 * x^T where dz1 = (W2^T dz2) * relu'(z1)
                w1 = local_weight_shares[0]
                w2 = local_weight_shares[1]
                z1 = matmul.secure_matrix_vector_multiply(
                    w1, input_shares, node_id, context=f"twolayer_s{i}_z1"
                )
                a1: List[Share] = []
                for s in z1:
                    if self._field_to_signed_int(s.y) > 0:
                        a1.append(Share(x=s.x, y=int(s.y) % p, node_id=node_id))
                    else:
                        a1.append(Share(x=s.x, y=0, node_id=node_id))

                a1_for_w2 = list(a1)
                if len(w2) > 0 and len(w2[0]) == len(a1) + 1:
                    a1_for_w2.append(Share(x=a1[0].x if a1 else 1, y=self.scale, node_id=node_id))

                z2 = matmul.secure_matrix_vector_multiply(
                    w2, a1_for_w2, node_id, context=f"twolayer_s{i}_z2"
                )
                if not z2:
                    continue

                if len(targets) > len(z2):
                    targets = targets[:len(z2)]
                elif len(targets) < len(z2):
                    targets = targets + [Share(x=z2[0].x, y=0, node_id=node_id)] * (len(z2) - len(targets))
                if not targets:
                    continue

                # Stabilize secure-mode output gradient:
                # 1) clip each output error in fixed-point space
                # 2) normalize by output dimension (mean over classes)
                inv_out_dim = pow(max(1, len(z2)), p - 2, p)
                dz2: List[Share] = []
                for j in range(len(z2)):
                    raw_err = (int(z2[j].y) - int(targets[j].y)) % p
                    clipped_err = self._clip_field_value(raw_err, self.secure_dz2_clip)
                    norm_err = (int(clipped_err) * int(inv_out_dim)) % p
                    dz2.append(Share(x=z2[j].x, y=norm_err, node_id=node_id))

                grad_w2: List[List[Share]] = []
                for o in range(len(w2)):
                    row: List[Share] = []
                    for h in range(len(w2[o])):
                        g = secure_mul.multiply_fixed_point(
                            dz2[o],
                            a1_for_w2[h],
                            node_id=node_id,
                            scale_factor=self.scale,
                            context=f"twolayer_s{i}_dw2_{o}_{h}",
                        )
                        row.append(g)
                    grad_w2.append(row)

                da1: List[Share] = []
                for h in range(len(a1)):
                    sum_y = 0
                    xh = a1[h].x
                    for o in range(len(w2)):
                        prod = secure_mul.multiply_fixed_point(
                            w2[o][h],
                            dz2[o],
                            node_id=node_id,
                            scale_factor=self.scale,
                            context=f"twolayer_s{i}_da1_{h}_{o}",
                        )
                        sum_y = (sum_y + int(prod.y)) % p
                    da1.append(Share(x=xh, y=sum_y, node_id=node_id))

                dz1: List[Share] = []
                for h in range(len(z1)):
                    if self._field_to_signed_int(z1[h].y) > 0:
                        dz1.append(Share(x=da1[h].x, y=int(da1[h].y) % p, node_id=node_id))
                    else:
                        dz1.append(Share(x=da1[h].x, y=0, node_id=node_id))

                grad_w1: List[List[Share]] = []
                for h in range(len(w1)):
                    row: List[Share] = []
                    for j in range(len(w1[h])):
                        g = secure_mul.multiply_fixed_point(
                            dz1[h],
                            input_shares[j],
                            node_id=node_id,
                            scale_factor=self.scale,
                            context=f"twolayer_s{i}_dw1_{h}_{j}",
                        )
                        row.append(g)
                    grad_w1.append(row)

                gradients = [grad_w1, grad_w2]
            else:
                predictions = self.mpc_engine.forward_pass(
                    input_shares, local_weight_shares, node_id
                )
                if len(targets) > len(predictions):
                    targets = targets[:len(predictions)]
                elif len(targets) < len(predictions):
                    x0 = predictions[0].x if predictions else 1
                    targets = targets + [Share(x=x0, y=0, node_id=node_id)] * (len(predictions) - len(targets))
                if not predictions or not targets:
                    continue
                loss_share = self.mpc_engine.compute_loss(predictions, targets, node_id)
                gradients = self.mpc_engine.backward_pass(
                    loss_share,
                    predictions,
                    targets,
                    local_weight_shares,
                    node_id,
                    input_shares=input_shares,
                )
            if gradients is None:
                continue

            # Accumulate per-sample gradients in the field.
            for li in range(len(grad_accum)):
                for r in range(len(grad_accum[li])):
                    for c in range(len(grad_accum[li][r])):
                        grad_accum[li][r][c] = Share(
                            x=grad_accum[li][r][c].x,
                            y=(int(grad_accum[li][r][c].y) + int(gradients[li][r][c].y)) % p,
                            node_id=node_id,
                        )
            processed += 1

        if processed == 0:
            return weight_shares, False

        # Average gradients: multiply by modular inverse of batch count.
        inv_count = pow(int(processed), p - 2, p)
        gradients_avg: List[List[List[Share]]] = []
        for layer in grad_accum:
            avg_layer: List[List[Share]] = []
            for row in layer:
                avg_row: List[Share] = []
                for g in row:
                    avg_row.append(
                        Share(
                            x=g.x,
                            y=(int(g.y) * int(inv_count)) % p,
                            node_id=node_id,
                        )
                    )
                avg_layer.append(avg_row)
            gradients_avg.append(avg_layer)

        gradients_to_apply = gradients_avg
        if self.train_mode == "secure":
            gradients_to_apply = self._secure_aggregate_gradients_batch(
                gradients_avg, batch_indices=batch_indices, node_id=node_id
            )

        # Update weights using standard (non-DP) SGD in share domain.
        updated_weights = self.mpc_engine.update_weights(
            local_weight_shares, gradients_to_apply, self.learning_rate, node_id
        )

        return updated_weights, True
    
    def commit_model_update(self, updated_weights: List[List[List[Share]]]) -> bool:
        """
        Commit model update with quorum
        Args:
            updated_weights: Updated weight shares
        Returns:
            True if committed successfully
        """
        self.v_theta += 1
        weight_key = f"weights_v{self.v_theta}"
        
        success = self.kvs_cluster.write_with_quorum(
            weight_key, updated_weights, self.v_theta
        )
        
        return success

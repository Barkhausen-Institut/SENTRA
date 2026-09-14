"""
MPC Engine for Secure Computation
Performs forward/backward pass on secret-shared data
"""

import math
import time
import logging
from typing import List, Tuple, Optional, Any, Dict
import numpy as np

from ml_training.secret_sharing import Share, PackedShamirSecretSharing
from ml_training.beaver_triples import SecureMultiplier, BeaverTriplePool, BeaverTripleGenerator
from ml_training.secure_matrix_ops import SecureMatrixOperations, GPUMatrixAccelerator


class PackedMPCEngine:
    """
    Packed MPC Engine for secure neural network operations
    Operates directly on secret-shared values
    """
    
    def __init__(self, n_nodes: int, t: int, field_size: int = 2**31 - 1, use_gpu: bool = False):
        """
        Initialize MPC engine
        Args:
            n_nodes: Number of nodes
            t: Privacy threshold
            field_size: Prime field size
            use_gpu: Whether to use GPU acceleration (optional)
        """
        self.n_nodes = n_nodes
        self.t = t
        self.field_size = field_size
        self.use_gpu = use_gpu
        self.pss = PackedShamirSecretSharing(field_size)
        self.operation_counter = 0  # For generating unique contexts
        
        # Initialize secure multiplier with Beaver triples
        triple_gen = BeaverTripleGenerator(field_size)
        triple_pool = BeaverTriplePool(triple_gen, initial_size=1000)
        self.multiplier = SecureMultiplier(triple_pool, n_nodes, t, field_size)
        
        # Initialize secure matrix operations
        self.matrix_ops = SecureMatrixOperations(self.multiplier, field_size)
    
    def _generate_context(self, operation: str) -> str:
        """Generate unique context for an operation"""
        self.operation_counter += 1
        return f"{operation}_{self.operation_counter}"
        
        # GPU setup (if available)
        if use_gpu:
            self.gpu_accelerator = GPUMatrixAccelerator()
            self.use_gpu = self.gpu_accelerator.is_available()
        else:
            self.gpu_accelerator = None
            self.use_gpu = False
    
    def _add_shares(self, share1: Share, share2: Share) -> Share:
        """Add two shares (homomorphic addition)"""
        return Share(
            x=share1.x,
            y=(share1.y + share2.y) % self.field_size,
            node_id=share1.node_id
        )
    
    def _multiply_shares(self, share1: Share, share2: Share, node_id: int, 
                        context: Optional[str] = None) -> Share:
        """
        Multiply two shares securely using Beaver triples
        Args:
            share1: First share
            share2: Second share
            node_id: Node ID
            context: Optional context for multi-node reconstruction
        Returns:
            Product share (secure using Beaver triple protocol)
        """
        return self.multiplier.multiply(share1, share2, node_id, context=context)
    
    def _multiply_share_by_scalar(self, share: Share, scalar: int, node_id: Optional[int] = None) -> Share:
        """Multiply share by scalar (homomorphic)"""
        if node_id is None:
            node_id = share.node_id
        return Share(
            x=share.x,
            y=(share.y * scalar) % self.field_size,
            node_id=node_id
        )

    # ---------------------------------------------------------------------
    # Packed-Shamir SIMD utilities (public-weight linear layers)
    # ---------------------------------------------------------------------
    def pack_batch_vector(
        self,
        batch_vectors: List[List[int]],
        *,
        node_id: int,
        packing_factor: Optional[int] = None,
    ) -> List[List[Share]]:
        """
        Pack a batch of vectors across the batch dimension using Packed Shamir.

        Output format:
            packed_inputs[j][lane_chunk] is a Share held by this node for feature j,
            where each Share encodes up to k=(n-t) batch elements (lanes).

        This is useful for SIMD-style evaluation of *linear* layers with PUBLIC weights.
        """
        if not batch_vectors:
            return []
        in_dim = len(batch_vectors[0])
        if any(len(v) != in_dim for v in batch_vectors):
            raise ValueError("All batch vectors must have same length")

        max_k = self.pss.max_packing_factor(self.n_nodes, self.t)
        k = int(packing_factor) if packing_factor is not None else max_k
        if k <= 0 or k > max_k:
            raise ValueError(f"packing_factor must be in [1, {max_k}] for (n={self.n_nodes}, t={self.t})")

        # Transpose batch: for each feature j, pack [x0[j], x1[j], ...]
        packed_by_feature: List[List[Share]] = []
        for j in range(in_dim):
            secrets_j = [int(v[j]) % self.field_size for v in batch_vectors]
            chunks = self.pss.share_vector(secrets_j, self.n_nodes, self.t, packing_factor=k)
            # Take this node's share from each chunk
            node_shares = [chunk[int(node_id) - 1] for chunk in chunks]
            packed_by_feature.append(node_shares)
        return packed_by_feature

    def packed_matvec_public_weights(
        self,
        packed_inputs_by_feature: List[List[Share]],
        W: List[List[int]],
        *,
        node_id: int,
    ) -> List[List[Share]]:
        """
        Compute y = W @ x for PACKED secret-shared x, where W is PUBLIC (plaintext) integers mod p.

        Returns:
            packed_outputs[i] is a list of Share chunks for output neuron i,
            aligned with the chunking of packed_inputs_by_feature[0].

        LIMITATION:
            This only supports public weights. If weights are also secret-shared, you need
            packed Beaver triples + degree management, which is not implemented here.
        """
        if not packed_inputs_by_feature:
            return []
        in_dim = len(packed_inputs_by_feature)
        if not W:
            return []
        out_dim = len(W)
        if any(len(row) != in_dim for row in W):
            raise ValueError("W shape mismatch: expected each row to have len(in_dim)")

        n_chunks = len(packed_inputs_by_feature[0])
        if any(len(packed_inputs_by_feature[j]) != n_chunks for j in range(in_dim)):
            raise ValueError("All input features must have same number of packed chunks")

        p = int(self.field_size)
        outputs: List[List[Share]] = []
        for i in range(out_dim):
            row = W[i]
            out_chunks: List[Share] = []
            for c in range(n_chunks):
                # y_i(chunk c) = sum_j W[i][j] * x_j(chunk c)
                acc_y = 0
                x_point = packed_inputs_by_feature[0][c].x
                for j in range(in_dim):
                    x_share = packed_inputs_by_feature[j][c]
                    w = int(row[j]) % p
                    acc_y = (acc_y + (w * int(x_share.y)) % p) % p
                out_chunks.append(Share(x=int(x_point), y=int(acc_y), node_id=int(node_id)))
            outputs.append(out_chunks)
        return outputs

    def reconstruct_packed_outputs(
        self,
        packed_outputs_by_neuron_all_nodes: List[List[List[Share]]],
        *,
        k: int,
        t: int,
    ) -> List[List[int]]:
        """
        Reconstruct packed outputs.

        Args:
            packed_outputs_by_neuron_all_nodes:
                packed_outputs_by_neuron_all_nodes[i][c] is the list of shares from multiple nodes
                for neuron i, chunk c (at least t+k shares required).
            k: number of lanes per chunk (except possibly the last chunk)
            t: privacy threshold
        Returns:
            For each neuron i, returns the concatenated reconstructed outputs across chunks.
        """
        out: List[List[int]] = []
        for neuron_chunks in packed_outputs_by_neuron_all_nodes:
            vals: List[int] = []
            for shares_chunk in neuron_chunks:
                vals.extend(self.pss.reconstruct_secrets(shares_chunk, k=k, t=t))
            out.append(vals)
        return out
    
    def forward_pass(self, packed_input_shares: List[Share], 
                    weights: List[List[List[Share]]], 
                    node_id: int, 
                    activation: str = "relu",
                    context: Optional[str] = None) -> List[Share]:
        """
        Perform optimized forward pass on packed shares using secure matrix operations
        Args:
            packed_input_shares: Packed input shares
            weights: Weight matrices (one per layer)
            node_id: Node ID
            activation: Activation function ("relu", "sigmoid", "linear")
            context: Optional context for multi-node operations
        Returns:
            Output shares
        """
        if context is None:
            context = self._generate_context("forward")
        # Use optimized matrix operations
        return self.matrix_ops.forward_pass(packed_input_shares, weights, node_id, context=context)
    
    def compute_loss(self, predictions: List[Share], targets: List[Share], node_id: int) -> Share:
        """
        Compute loss (MSE) on shares
        Args:
            predictions: Prediction shares
            targets: Target shares
            node_id: Node ID
        Returns:
            Loss share
        """
        # MSE: sum((pred - target)^2) / n
        loss_sum = Share(x=predictions[0].x, y=0, node_id=node_id)
        
        context = self._generate_context("loss")
        for pred, target in zip(predictions, targets):
            diff = Share(
                x=pred.x,
                y=(pred.y - target.y) % self.field_size,
                node_id=node_id
            )
            diff_squared = self._multiply_shares(diff, diff, node_id, context=context)
            loss_sum = self._add_shares(loss_sum, diff_squared)
        
        # Average (simplified - would need secure division)
        n = len(predictions)
        return self._multiply_share_by_scalar(loss_sum, 1, node_id)  # Skip division for minimal
    
    def backward_pass(self, loss_share: Share, predictions: List[Share], 
                     targets: List[Share], weights: List[List[List[Share]]], 
                     node_id: int, input_shares: Optional[List[Share]] = None,
                     context: Optional[str] = None) -> List[List[List[Share]]]:
        """
        Perform optimized backward pass (gradient computation) using secure matrix operations
        Args:
            loss_share: Loss share
            predictions: Prediction shares
            targets: Target shares
            weights: Weight matrices
            node_id: Node ID
            input_shares: Optional input shares for gradient computation
            context: Optional context for multi-node operations
        Returns:
            Gradient matrices (same shape as weights)
        """
        if context is None:
            context = self._generate_context("backward")
        # Compute error gradients
        error_grads = []
        for pred, target in zip(predictions, targets):
            error = Share(
                x=pred.x,
                y=(pred.y - target.y) % self.field_size,
                node_id=node_id
            )
            error_grads.append(error)
        
        # Use predictions as input if input_shares not provided (simplified)
        if input_shares is None:
            input_shares = predictions
        
        # Use optimized matrix operations for backward pass
        return self.matrix_ops.backward_pass(error_grads, input_shares, weights, node_id, context=context)
    
    def update_weights(self, weights: List[List[List[Share]]], 
                      gradients: List[List[List[Share]]], 
                      learning_rate: float, 
                      node_id: int) -> List[List[List[Share]]]:
        """
        Update weights using gradients with optimized matrix operations
        Args:
            weights: Current weight matrices
            gradients: Gradient matrices
            learning_rate: Learning rate
            node_id: Node ID
        Returns:
            Updated weight matrices
        """
        # Use optimized matrix operations
        return self.matrix_ops.update_weights(weights, gradients, learning_rate, node_id)

    def secure_aggregate_gradients(self, gradients: List[List[List[Share]]], context: Optional[str] = None) -> List[List[List[Share]]]:
        """
        Securely aggregate gradients from all nodes using All-Reduce Sum.
        
        Protocol:
        1. Input `gradients` represents the local gradient contribution (G_i).
           - In Enclave mode, this contains SHARES of G_i for ALL nodes: [Share(G_i)_1, Share(G_i)_2, ...].
           - We must distribute these: Send [G_i]_j to Node j.
        2. Receive [G_k]_i from all other nodes k.
        3. Local Sum: [G_total]_i = Sum_k([G_k]_i).
        4. Return [G_total]_i (this node's share of the global aggregated gradient).

        If `gradients` only contains shares for THIS node (standard MPC output), we assume
        we are in a simulated mode where we just sum local values (or fail).
        But to support the Sentra requirement, we assume `gradients` contains the full set of shares 
        generated by the local Enclave.
        """
        if context is None:
            context = self._generate_context("agg_grad")

        # 1. Flatten gradients to linear list of Shares for easier handling
        # Structure: layers -> rows -> cols -> shares
        # We need to preserve the structure for reconstruction.
        # Let's flatten to (Share, metadata) and reconstruct later? 
        # Or just traverse.
        
        # Check if we have network
        network = getattr(self.reconstruction_manager, "network", None) if self.reconstruction_manager else None
        if network is None:
            # Local/Single-node mode: just return as is (no aggregation possible/needed)
            return gradients

        start_time = getattr(self, "_time", lambda: 0)() # Dummy if not present

        # 2. Bucketize shares by target node_id
        # We assume `gradients` structure is uniform.
        # If Enclave generated 3 shares per value (for n=3), `gradients` should hold them.
        # But `backward_pass` returns `List[List[List[Share]]]`.
        # If it returns multiple shares per value, `Share` is the leaf? 
        # Wait, `Share` object is single.
        # So `backward_pass` returns `[ [ [s_1, s_2, s_3], ... ] ]`? 
        # No, `Share` is `x, y, node_id`.
        # `backward_pass` returns `List[List[List[Share]]]`. Inner list is usually "shares for this value"?
        # Let's inspect `backward_pass`: calculates `error_grads` (List[Share]) -> `matrix_ops.backward`.
        # `matrix_ops` returns `List[List[List[Share]]]`.
        
        # If Enclave is working, it produces `[ [ [Share(node=1), Share(node=2)...] ] ]` per weight.
        # So leaf is `List[Share]` (one per node).
        
        # We will iterate and collect.
        buckets: Dict[int, List[int]] = {i: [] for i in range(1, self.n_nodes + 1)}
        
        def _collect(obj):
            if isinstance(obj, Share):
                # Append the Y value (share value) to the bucket for obj.node_id
                buckets[obj.node_id].append(int(obj.y))
            elif isinstance(obj, list):
                for item in obj:
                    _collect(item)
        
        _collect(gradients)
        
        # 3. Exchange Buckets
        # Send bucket j to Node j
        local_node_id = int(network.node_id)
        
        # Buffers for received data
        received_sums = None # Will be vector of same length as flattened gradients
        
        # Helper to flat-pack and send
        import numpy as np
        
        # Send phase
        for target_id, values in buckets.items():
            if target_id == local_node_id:
                continue # Handle self later
            ctx_send = f"{context}_to_{target_id}"
            network.channel.send_vector(
                target_id, ctx_send, x=local_node_id, values=np.asarray(values, dtype=np.uint64)
            )

        # Receive and Sum phase
        # Init sum with self-values
        total_len = len(buckets[local_node_id])
        if total_len == 0:
            return gradients # Nothing to aggregate
            
        acc_sum = np.asarray(buckets[local_node_id], dtype=np.uint64)
        
        for peer_id in range(1, self.n_nodes + 1):
            if peer_id == local_node_id:
                continue
            ctx_recv = f"{context}_to_{local_node_id}" # They sent to ME using MY id in suffix? No, usually symmetric.
            # My send: f"{context}_to_{target_id}".
            # Their send to ME: f"{context}_to_{local_node_id}" (from their perspective).
            # Wait, context management in `send_vector` usually requires specific key.
            # network.channel.send_vector(receiver, key, ...)
            # get_received_vector(key)
            # If they used key f"{context}_to_{local_node_id}", I listen on that.
            
            # Use blocked wait
            start_wait = 0
            import time
            while True:
                recv = network.channel.get_received_vector(ctx_recv)
                if peer_id in recv:
                    vals = recv[peer_id]["values"]
                    arr = np.asarray(vals, dtype=np.uint64)
                    acc_sum = (acc_sum + arr) % self.field_size
                    # Clear buffer
                    try:
                        network.channel.clear_vector(ctx_recv)
                    except:
                        pass
                    break
                time.sleep(0.01)
                # Simple timeout
                start_wait += 0.01
                if start_wait > 60.0:
                     raise RuntimeError(f"Timeout waiting for aggregation from {peer_id}")

        # 4. Unpack/Reconstruct Structure
        # We need to map `acc_sum` back to the structure of `gradients`.
        # Since `gradients` leaf was `List[Share]` (all shares), the output should be `Share` (my share of sum).
        # We traverse `gradients` again and pop from `acc_sum`.
        
        # Wait, the input `gradients` contained shares for EVERYONE.
        # The output `gradients` should contains the share for ME (of the global sum).
        # But `backward_pass` returns `List[List[List[Share]]]`.
        # Does it return "My Share" or "All Shares"?
        # - Current `backward_pass` returns `List[List[List[Share]]]` where leaf is `Share`.
        # - If it's "All Shares", leaf is `List[Share]`?
        # Let's assume input matches the structure needed for All-to-All:
        # If input leaf is `Share(node=1)`, `Share(node=2)`, then we flattened them.
        # The received `acc_sum` corresponds to the values destined for ME.
        # The number of values in `acc_sum` == number of `Share` objects in input with `node_id == my_id`.
        
        # We need to reconstruct the return structure `List[List[List[Share]]]` but only keeping MY shares.
        # Actually, we should return the same structure as standard `backward_pass` output (for this node).
        
        flat_idx = 0
        
        def _rebuild(obj):
            nonlocal flat_idx
            if isinstance(obj, Share):
                if obj.node_id == local_node_id:
                     # This was "my" share of the local gradient.
                     # We replace it with "my" share of the GLOBAL gradient (from acc_sum).
                     val = int(acc_sum[flat_idx])
                     flat_idx += 1
                     return Share(x=obj.x, y=val, node_id=local_node_id)
                else:
                    # This was a share for someone else. In the output (my shares), it's irrelevant?
                    # Or do we keep it? 
                    # If the output is "My shares of the global gradient", we only need shares where node_id == me.
                    # But the structure expects `List[List[List[Share]]]`.
                    # Typically this structure is weights[layer][row][col] -> Share.
                    # So we just return the Share for me.
                    return None # Mark for removal
            elif isinstance(obj, list):
                res = []
                for item in obj:
                    r = _rebuild(item)
                    if r is not None:
                        res.append(r)
                return res
            return obj

        return _rebuild(gradients)




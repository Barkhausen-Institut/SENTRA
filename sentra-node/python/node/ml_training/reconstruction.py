"""
Secure Share Reconstruction Protocol
Implements threshold cryptography for reconstructing secrets from shares
"""

from typing import List, Optional, Dict, Tuple, Sequence, Union
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_comm import SecureMPCNetwork, SecureChannel
import time
import numpy as np
from array import array


class SecureReconstruction:
    """
    Secure reconstruction protocol for MPC
    Coordinates share collection and reconstruction across nodes
    """
    
    def __init__(self, network: SecureMPCNetwork, t: int, field_size: int):
        """
        Initialize secure reconstruction
        Args:
            network: SecureMPCNetwork instance
            t: Privacy threshold (need t+1 shares to reconstruct)
        """
        self.network = network
        self.t = t
        self.field_size = field_size
        self.sss = ShamirSecretSharing(field_size)
    
    def reconstruct_value(self, local_shares: List[Share], context: str,
                         required_nodes: Optional[List[int]] = None,
                         timeout: float = 5.0) -> int:
        """
        Reconstruct a secret value from shares across nodes
        
        Args:
            local_shares: Shares held by this node
            context: Unique context identifier
            required_nodes: Nodes to request shares from
            timeout: Timeout for reconstruction
        Returns:
            Reconstructed secret value
        """
        # In this codebase, each node typically holds ONE share for a given context.
        # For multi-node, we use a proactive broadcast-and-collect pattern:
        # - broadcast our local share for `context` to all peers
        # - wait until we have t+1 unique-node shares (including our own)

        if not local_shares:
            raise ValueError("local_shares is empty")

        # If caller already provided >= t+1 shares, reconstruct immediately.
        if len(local_shares) >= self.t + 1:
            return self.sss.reconstruct(local_shares[: self.t + 1])

        # Broadcast the (single) local share for this context.
        # Use the SAME `context` key so peers can store/retrieve consistently.
        self.network.broadcast_share(local_shares[0], context)

        start_time = time.time()
        shares_by_node: Dict[int, Share] = {local_shares[0].node_id: local_shares[0]}

        while time.time() - start_time < timeout:
            received = self.network.get_received_shares(context)
            for share in received:
                shares_by_node[share.node_id] = share

            if len(shares_by_node) >= self.t + 1:
                break

            time.sleep(0.05)

        if len(shares_by_node) < self.t + 1:
            raise RuntimeError(
                f"Insufficient shares for reconstruction: got {len(shares_by_node)}, need {self.t + 1}"
            )

        secret = self.sss.reconstruct(list(shares_by_node.values())[: self.t + 1])
        # Prevent unbounded growth of per-context buffers
        try:
            self.network.channel.clear_context(context)
        except Exception:
            pass
        return secret
    
    def reconstruct_for_multiplication(self, d_shares: List[Share], e_shares: List[Share],
                                      context: str, required_nodes: Optional[List[int]] = None) -> tuple:
        """
        Reconstruct d and e values for Beaver triple multiplication
        
        Args:
            d_shares: Shares of d = x - a
            e_shares: Shares of e = y - b
            context: Context identifier
            required_nodes: Nodes to request shares from
        Returns:
            Tuple of (d_reconstructed, e_reconstructed)
        """
        # Reconstruct d
        d_context = f"{context}_d"
        d_recon = self.reconstruct_value(d_shares, d_context, required_nodes)
        
        # Reconstruct e
        e_context = f"{context}_e"
        e_recon = self.reconstruct_value(e_shares, e_context, required_nodes)
        
        return d_recon, e_recon
    
    def batch_reconstruct(self, share_groups: List[List[Share]], context_prefix: str,
                         required_nodes: Optional[List[int]] = None) -> List[int]:
        """
        Reconstruct multiple secrets efficiently
        
        Args:
            share_groups: List of share lists (one per secret)
            context_prefix: Prefix for context identifiers
            required_nodes: Nodes to request shares from
        Returns:
            List of reconstructed secrets
        """
        reconstructed = []
        for i, shares in enumerate(share_groups):
            context = f"{context_prefix}_{i}"
            secret = self.reconstruct_value(shares, context, required_nodes)
            reconstructed.append(secret)
        return reconstructed


class MPCReconstructionManager:
    """
    Manages reconstruction operations for MPC
    Coordinates with network and handles caching
    """
    
    def __init__(self, network: SecureMPCNetwork, t: int, field_size: int):
        """
        Initialize reconstruction manager
        Args:
            network: SecureMPCNetwork instance
            t: Privacy threshold
        """
        self.network = network
        self.t = t
        self.field_size = field_size
        self.reconstructor = SecureReconstruction(network, t, field_size)
        self.reconstruction_cache: Dict[str, int] = {}

    def _required_peer_ids(self) -> List[int]:
        """Deterministic Shamir peer set (lowest node ids); do not use arbitrary recv.keys()."""
        return sorted(int(nid) for nid in self.network.node_configs.keys())[: self.t + 1]

    def get_reconstructed_values_batch(
        self,
        local_shares: List[Share],
        contexts: List[str],
        timeout: float = 120.0,
        use_cache: bool = False,
    ) -> List[int]:
        """
        Batch reconstruct many secrets (same t/field) with a single share-broadcast step.

        This is the core primitive we use to amortize Beaver openings:
        - each node broadcasts many (context, share) items
        - each node waits until it has t+1 unique shares per context
        - reconstruct all secrets locally
        """
        if len(local_shares) != len(contexts):
            raise ValueError("local_shares and contexts length mismatch")
        if not local_shares:
            return []

        # Optional cache fast-path
        if use_cache:
            cached = []
            missing_local_shares: List[Share] = []
            missing_contexts: List[str] = []
            missing_indices: List[int] = []
            for i, ctx in enumerate(contexts):
                if ctx in self.reconstruction_cache:
                    cached.append(self.reconstruction_cache[ctx])
                else:
                    cached.append(None)
                    missing_local_shares.append(local_shares[i])
                    missing_contexts.append(ctx)
                    missing_indices.append(i)
        else:
            cached = None
            missing_local_shares = local_shares
            missing_contexts = contexts
            missing_indices = list(range(len(contexts)))

        # Broadcast missing shares in a batch (network may choose to send individually).
        if missing_local_shares:
            self.network.broadcast_shares_batch(missing_local_shares, missing_contexts)

        # Collect shares for each missing context
        start_time = time.time()
        shares_by_context: Dict[str, Dict[int, Share]] = {}
        for share, ctx in zip(missing_local_shares, missing_contexts):
            shares_by_context[ctx] = {share.node_id: share}

        pending = set(missing_contexts)
        while pending and (time.time() - start_time < timeout):
            # Poll each pending context (simple but correct; could be optimized)
            done_now = []
            for ctx in pending:
                received = self.network.get_received_shares(ctx)
                by_node = shares_by_context[ctx]
                for s in received:
                    by_node[s.node_id] = s
                if len(by_node) >= self.t + 1:
                    done_now.append(ctx)
            for ctx in done_now:
                pending.remove(ctx)
            if pending:
                time.sleep(0.05)

        if pending:
            raise RuntimeError(f"Batch reconstruction timed out; pending={len(pending)} contexts")

        reconstructed_missing: Dict[str, int] = {}
        for ctx in missing_contexts:
            shares = list(shares_by_context[ctx].values())[: self.t + 1]
            reconstructed_missing[ctx] = self.reconstructor.sss.reconstruct(shares)
            if use_cache:
                self.reconstruction_cache[ctx] = reconstructed_missing[ctx]

        # Clear per-context buffers to avoid memory blow-up during training
        for ctx in missing_contexts:
            try:
                self.network.channel.clear_context(ctx)
            except Exception:
                pass

        if use_cache:
            out: List[int] = []
            for i, ctx in enumerate(contexts):
                if cached[i] is not None:
                    out.append(cached[i])
                else:
                    out.append(reconstructed_missing[ctx])
            return out

        return [reconstructed_missing[ctx] for ctx in contexts]

    def reconstruct_for_multiplication_batch(
        self,
        d_local_shares: List[Share],
        e_local_shares: List[Share],
        context_prefix: str,
        timeout: float = 120.0,
    ) -> Tuple[List[int], List[int]]:
        """
        Batch reconstruct d and e for many Beaver multiplications.
        """
        if len(d_local_shares) != len(e_local_shares):
            raise ValueError("d_local_shares and e_local_shares length mismatch")
        if not d_local_shares:
            return [], []

        # SIMD-style vector opening:
        # Instead of 2048 separate contexts, open two vectors (d and e) under two contexts.
        d_ctx = f"{context_prefix}_d_vec"
        e_ctx = f"{context_prefix}_e_vec"

        d_vals_local = [s.y for s in d_local_shares]
        e_vals_local = [s.y for s in e_local_shares]
        x = d_local_shares[0].x
        return self.reconstruct_for_multiplication_batch_values(
            d_vals_local=d_vals_local,
            e_vals_local=e_vals_local,
            x=x,
            context_prefix=context_prefix,
            timeout=timeout,
        )

    def reconstruct_for_multiplication_batch_values(
        self,
        d_vals_local: Union[Sequence[int], np.ndarray],
        e_vals_local: Union[Sequence[int], np.ndarray],
        x: int,
        context_prefix: str,
        timeout: float = 120.0,
    ) -> Tuple[List[int], List[int]]:
        """
        Same as reconstruct_for_multiplication_batch, but avoids allocating Share objects.
        Used by array-based conv/matmul kernels.
        """
        if len(d_vals_local) != len(e_vals_local):
            raise ValueError("d_vals_local and e_vals_local length mismatch")
        # Works for lists, arrays, numpy arrays
        if len(d_vals_local) == 0:
            return [], []

        d_ctx = f"{context_prefix}_d_vec"
        e_ctx = f"{context_prefix}_e_vec"
        required_peers = self._required_peer_ids()

        # Broadcast d and e in one message per peer (one round-trip instead of two)
        self.network.broadcast_vector_pair(
            d_ctx, e_ctx, x=x, values_d=d_vals_local, values_e=e_vals_local
        )

        start = time.time()
        # Helper: Lagrange coefficients at x=0 for the chosen x-points
        def _lagrange_coeffs_at_zero(xs: List[int], p: int) -> List[int]:
            coeffs: List[int] = []
            for i, x_i in enumerate(xs):
                num = 1
                den = 1
                for j, x_j in enumerate(xs):
                    if i == j:
                        continue
                    num = (num * (-x_j)) % p
                    den = (den * (x_i - x_j)) % p
                # p is prime in our setup; use Fermat inverse for speed
                inv_den = pow(den % p, p - 2, p)
                coeffs.append((num * inv_den) % p)
            return coeffs

        while time.time() - start < timeout:
            d_recv = self.network.channel.get_received_vector(d_ctx)
            e_recv = self.network.channel.get_received_vector(e_ctx)

            # Add our own (so we don't depend on loopback)
            d_recv[self.network.node_id] = {"x": x, "values": d_vals_local}
            e_recv[self.network.node_id] = {"x": x, "values": e_vals_local}

            if all(nid in d_recv for nid in required_peers) and all(nid in e_recv for nid in required_peers):
                # Ensure all vectors have correct length
                ok = True
                chosen_nodes = list(required_peers)
                chosen_nodes_e = list(required_peers)
                for nid in chosen_nodes:
                    v = d_recv[nid]
                    if len(v["values"]) != len(d_vals_local):
                        ok = False
                for nid in chosen_nodes_e:
                    v = e_recv[nid]
                    if len(v["values"]) != len(e_vals_local):
                        ok = False
                if ok:
                    # Vectorized reconstruction using precomputed Lagrange coeffs
                    p = int(self.field_size)

                    xs_d = [int(d_recv[nid]["x"]) for nid in chosen_nodes]
                    xs_e = [int(e_recv[nid]["x"]) for nid in chosen_nodes_e]
                    lambdas_d = _lagrange_coeffs_at_zero(xs_d, p)
                    lambdas_e = _lagrange_coeffs_at_zero(xs_e, p)

                    # Vectorized reconstruction: sum_i lambda_i * y_i  (mod p)
                    L = len(d_vals_local)
                    # uint64 multiply overflows for large fields (e.g. 2^61-1).
                    # Use object-int modular arithmetic in that regime.
                    use_object_mod = p > 0xFFFFFFFF
                    if use_object_mod:
                        d_out = np.zeros((L,), dtype=object)
                        e_out = np.zeros((L,), dtype=object)
                    else:
                        d_out = np.zeros((L,), dtype=np.uint64)
                        e_out = np.zeros((L,), dtype=np.uint64)

                    for lam, nid in zip(lambdas_d, chosen_nodes):
                        vec = d_recv[nid]["values"]
                        if use_object_mod:
                            v = np.asarray(vec, dtype=object)
                            d_out = (d_out + (v * int(lam)) % p) % p
                        else:
                            d_out = (d_out + (np.asarray(vec, dtype=np.uint64) * np.uint64(lam)) % np.uint64(p)) % np.uint64(p)

                    for lam, nid in zip(lambdas_e, chosen_nodes_e):
                        vec = e_recv[nid]["values"]
                        if use_object_mod:
                            v = np.asarray(vec, dtype=object)
                            e_out = (e_out + (v * int(lam)) % p) % p
                        else:
                            e_out = (e_out + (np.asarray(vec, dtype=np.uint64) * np.uint64(lam)) % np.uint64(p)) % np.uint64(p)

                    # Cleanup buffers
                    try:
                        self.network.channel.clear_vector(d_ctx)
                        self.network.channel.clear_vector(e_ctx)
                    except Exception:
                        pass

                    # Return numpy arrays (indexable like lists) to avoid Python list materialization
                    if use_object_mod:
                        return np.asarray(d_out, dtype=np.uint64), np.asarray(e_out, dtype=np.uint64)
                    return d_out, e_out

            time.sleep(0.02)

        # Provide more context for debugging desynchronization issues (e.g. one node ahead).
        try:
            d_recv = self.network.channel.get_received_vector(d_ctx)
            e_recv = self.network.channel.get_received_vector(e_ctx)
            d_keys = sorted(list(d_recv.keys()))
            e_keys = sorted(list(e_recv.keys()))
        except Exception:
            d_keys = []
            e_keys = []
        raise RuntimeError(
            f"Batch reconstruction timed out; pending={len(d_vals_local)} elements; "
            f"received d from nodes={d_keys}, e from nodes={e_keys}"
        )

    def reconstruct_opened_vector_values(
        self,
        *,
        context: str,
        values_local: Union[Sequence[int], np.ndarray],
        x: int,
        timeout: float = 120.0,
    ) -> np.ndarray:
        """
        Open/reconstruct a single vector of field elements (values) from node shares.

        Expected usage:
        - all nodes call network.broadcast_vector(context, x=..., values=local_vector)
        - one designated node calls this to collect >= t+1 vectors and reconstruct the secret vector

        Returns:
            numpy uint64 array of reconstructed field values (mod p)
        """
        if len(values_local) == 0:
            return np.zeros((0,), dtype=np.uint64)

        required_peers = self._required_peer_ids()

        start = time.time()

        # Helper: Lagrange coefficients at x=0 for the chosen x-points
        def _lagrange_coeffs_at_zero(xs: List[int], p: int) -> List[int]:
            coeffs: List[int] = []
            for i, x_i in enumerate(xs):
                num = 1
                den = 1
                for j, x_j in enumerate(xs):
                    if i == j:
                        continue
                    num = (num * (-x_j)) % p
                    den = (den * (x_i - x_j)) % p
                inv_den = pow(den % p, p - 2, p)
                coeffs.append((num * inv_den) % p)
            return coeffs

        while time.time() - start < timeout:
            recv = self.network.channel.get_received_vector(context)
            # Add our own vector (so we don't depend on loopback)
            recv[self.network.node_id] = {"x": x, "values": values_local}

            if all(nid in recv for nid in required_peers):
                chosen_nodes = list(required_peers)
                # Validate lengths
                ok = True
                for nid in chosen_nodes:
                    if len(recv[nid]["values"]) != len(values_local):
                        ok = False
                        break
                if not ok:
                    time.sleep(0.02)
                    continue

                p = int(self.field_size)
                xs = [int(recv[nid]["x"]) for nid in chosen_nodes]
                lambdas = _lagrange_coeffs_at_zero(xs, p)

                L = len(values_local)
                use_object_mod = p > 0xFFFFFFFF
                if use_object_mod:
                    out = np.zeros((L,), dtype=object)
                else:
                    out = np.zeros((L,), dtype=np.uint64)
                    p_u64 = np.uint64(p)
                for lam, nid in zip(lambdas, chosen_nodes):
                    vec = recv[nid]["values"]
                    # vec can be numpy array, array('I'), or python list
                    if isinstance(vec, np.ndarray):
                        vec_u64 = np.asarray(vec, dtype=np.uint64)
                    elif isinstance(vec, array):
                        # array('I') / array('Q') support buffer interface
                        if vec.typecode == "Q":
                            vec_u64 = np.asarray(np.frombuffer(vec, dtype=np.uint64), dtype=np.uint64)
                        else:
                            vec_u64 = np.asarray(np.frombuffer(vec, dtype=np.uint32), dtype=np.uint64)
                    else:
                        vec_u64 = np.asarray(vec, dtype=np.uint64)
                    if use_object_mod:
                        v = np.asarray(vec_u64, dtype=object)
                        out = (out + (v * int(lam)) % p) % p
                    else:
                        out = (out + (vec_u64 * np.uint64(lam)) % p_u64) % p_u64

                # Cleanup buffers
                try:
                    self.network.channel.clear_vector(context)
                except Exception:
                    pass
                if use_object_mod:
                    return np.asarray(out, dtype=np.uint64)
                return out

            time.sleep(0.02)

        raise RuntimeError(f"Vector reconstruction timed out for context={context!r}")

    def reconstruct_for_multiplication(
        self,
        d_shares: List[Share],
        e_shares: List[Share],
        context: str,
        required_nodes: Optional[List[int]] = None,
        timeout: float = 120.0,
    ) -> Tuple[int, int]:
        """
        Compatibility wrapper for scalar Beaver multiplication.
        SecureMultiplier._multiply_with_reconstruction expects this API.
        """
        # Delegate to SecureReconstruction logic (uses contexts: f"{context}_d" and f"{context}_e")
        return self.reconstructor.reconstruct_for_multiplication(
            d_shares, e_shares, context, required_nodes=required_nodes, timeout=timeout
        )
    
    def get_reconstructed_value(self, shares: List[Share], context: str,
                               use_cache: bool = True) -> int:
        """
        Get reconstructed value, using cache if available
        
        Args:
            shares: Shares to reconstruct
            context: Context identifier
            use_cache: Whether to use cached values
        Returns:
            Reconstructed value
        """
        if use_cache and context in self.reconstruction_cache:
            return self.reconstruction_cache[context]
        
        value = self.reconstructor.reconstruct_value(shares, context)
        
        if use_cache:
            self.reconstruction_cache[context] = value
        
        return value
    
    def clear_cache(self):
        """Clear reconstruction cache"""
        self.reconstruction_cache.clear()


def create_reconstruction_manager(network: SecureMPCNetwork, t: int, field_size: int = 2**31 - 1) -> MPCReconstructionManager:
    """
    Factory function to create reconstruction manager
    
    Args:
        network: SecureMPCNetwork instance
        t: Privacy threshold
    Returns:
        Configured MPCReconstructionManager instance
    """
    return MPCReconstructionManager(network, t, field_size)

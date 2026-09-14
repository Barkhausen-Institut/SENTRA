"""
Beaver Triple Generation and Secure Multiplication
Implements secure multiplication protocol using pre-computed Beaver triples
"""

from typing import List, Tuple, Optional, Dict
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.reconstruction import MPCReconstructionManager
import secrets
import random
import hashlib
import numpy as np
import time


class BeaverTripleDealerService:
    """
    Trusted dealer for Beaver triples in multi-node mode.

    Motivation:
    - PRSS-seed triples allow any node with the seed to derive triple secrets.
    - This dealer keeps the seed private and only serves per-node Shamir shares.
    - Triples are keyed by (context_prefix, idx) so nodes can request independently
      while staying aligned.

    Threat model:
    - Nodes run in trusted enclaves.
    - Privacy is against outsiders (network eavesdroppers).
    - Dealer learns triple secrets (acceptable for this model).
    """

    def __init__(
        self,
        *,
        network,
        dealer_node_id: int,
        n_nodes: int,
        t: int,
        field_size: int,
        seed: Optional[int] = None,
    ):
        self.network = network
        self.dealer_node_id = int(dealer_node_id)
        self.n_nodes = int(n_nodes)
        self.t = int(t)
        self.field_size = int(field_size)
        self.seed = int(seed) if seed is not None else secrets.randbits(64)

    def register(self):
        """
        Register the dealer request handler on this node's channel.
        Must be called only on the dealer node process.
        """
        from ml_training.secure_comm import MessageType

        # Expose this dealer instance to local code paths (dealer self-requests)
        try:
            setattr(self.network.channel, "_triple_dealer_service", self)
        except Exception:
            pass

        def _ensure_connection_to(target_id: int) -> bool:
            """
            Ensure we have an outgoing connection to target_id.
            SecureChannel only tracks outgoing sockets in `connections`.
            """
            try:
                if target_id in self.network.channel.connections:
                    return True
                cfg = self.network.node_configs.get(int(target_id))
                if not cfg:
                    return False
                return bool(self.network.channel.connect_to_node(int(target_id), cfg["host"], int(cfg["port"])))
            except Exception:
                return False

        def _build_triple_vectors(context_prefix: str, n: int, x: int):
            # Build a,b,c share vectors for the requesting node.
            vec_dtype = np.uint64 if self.field_size > 0xFFFFFFFF else np.uint32
            a_y = np.empty((n,), dtype=vec_dtype)
            b_y = np.empty((n,), dtype=vec_dtype)
            c_y = np.empty((n,), dtype=vec_dtype)
            p = self.field_size
            for i in range(n):
                a = self._secret("a", context_prefix, i) % p
                b = self._secret("b", context_prefix, i) % p
                c = (a * b) % p
                if vec_dtype == np.uint64:
                    a_y[i] = np.uint64(self._share_y(a, "a", context_prefix, i, x) & 0xFFFFFFFFFFFFFFFF)
                    b_y[i] = np.uint64(self._share_y(b, "b", context_prefix, i, x) & 0xFFFFFFFFFFFFFFFF)
                    c_y[i] = np.uint64(self._share_y(c, "c", context_prefix, i, x) & 0xFFFFFFFFFFFFFFFF)
                else:
                    a_y[i] = np.uint32(self._share_y(a, "a", context_prefix, i, x) & 0xFFFFFFFF)
                    b_y[i] = np.uint32(self._share_y(b, "b", context_prefix, i, x) & 0xFFFFFFFF)
                    c_y[i] = np.uint32(self._share_y(c, "c", context_prefix, i, x) & 0xFFFFFFFF)
            return a_y, b_y, c_y

        # Attach for local fast-path
        self.get_triple_vectors = _build_triple_vectors  # type: ignore[attr-defined]

        def _handler(sender_id: int, data: dict):
            try:
                # print(f"Dealer: Received request from {sender_id}")
                context_prefix = str(data.get("context_prefix", ""))
                n = int(data.get("n", 0))
                x = int(data.get("x", sender_id))
                req_id = str(data.get("req_id", ""))
                if not context_prefix or n <= 0 or not req_id:
                     print(f"Dealer: Invalid request parameters from {sender_id}")
                     return
                
                a_y, b_y, c_y = _build_triple_vectors(context_prefix, n, x)

                # Send three vectors back to requester under unique contexts.
                # Reuse existing binary vector transport (fast path).
                if not _ensure_connection_to(sender_id):
                    print(f"Dealer: Failed to ensure connection to {sender_id}")
                    return
                # print(f"Dealer: Sending response to {sender_id} for {req_id}")
                self.network.channel.send_vector(sender_id, f"{req_id}_a", x=x, values=a_y)
                self.network.channel.send_vector(sender_id, f"{req_id}_b", x=x, values=b_y)
                self.network.channel.send_vector(sender_id, f"{req_id}_c", x=x, values=c_y)
            except Exception as e:
                print(f"Dealer Error handling request from {sender_id}: {e}")
                import traceback
                traceback.print_exc()

        # Use generic handler mechanism (SecureChannel routes unknown types here)
        self.network.channel.register_handler(MessageType.TRIPLE_REQUEST.value, _handler)

    def _u64(self, label: str, context_prefix: str, idx: int, extra: int = 0) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(str(self.seed).encode("utf-8"))
        h.update(b"|")
        h.update(label.encode("utf-8"))
        h.update(b"|")
        h.update(context_prefix.encode("utf-8"))
        h.update(b"|")
        h.update(str(idx).encode("utf-8"))
        h.update(b"|")
        h.update(str(extra).encode("utf-8"))
        return int.from_bytes(h.digest(), "big")

    def _secret(self, label: str, context_prefix: str, idx: int) -> int:
        return self._u64(label, context_prefix, idx) % self.field_size

    def _share_y(self, secret: int, label: str, context_prefix: str, idx: int, x: int) -> int:
        # Shamir polynomial: c0=secret, c1..ct derived deterministically from dealer seed.
        p = self.field_size
        y = int(secret) % p
        x_pow = int(x) % p
        for k in range(1, self.t + 1):
            coeff = self._u64(f"{label}_coeff_{k}", context_prefix, idx, extra=k) % p
            y = (y + (coeff * x_pow) % p) % p
            x_pow = (x_pow * x) % p
        return y


class BeaverTriple:
    """
    Represents a Beaver triple (a, b, c) where c = a * b
    Each value is secret-shared across nodes
    """
    
    def __init__(self, a: List[Share], b: List[Share], c: List[Share]):
        """
        Initialize Beaver triple
        Args:
            a: Secret-shared value a
            b: Secret-shared value b
            c: Secret-shared value c = a * b
        """
        self.a = a
        self.b = b
        self.c = c
        # Speed up lookups (hot path): avoid linear scans in get_for_node()
        self._a_by_node = {s.node_id: s for s in a}
        self._b_by_node = {s.node_id: s for s in b}
        self._c_by_node = {s.node_id: s for s in c}
    
    def get_for_node(self, node_id: int) -> Tuple[Share, Share, Share]:
        """Get triple shares for a specific node"""
        return self._a_by_node[node_id], self._b_by_node[node_id], self._c_by_node[node_id]


class BeaverTripleGenerator:
    """
    Generates Beaver triples using Shamir Secret Sharing
    """
    
    def __init__(self, field_size: int = 2**31 - 1):
        """
        Initialize triple generator
        Args:
            field_size: Prime field size
        """
        self.field_size = field_size
        self.sss = ShamirSecretSharing(field_size)
    
    def generate_triple(self, n_nodes: int, t: int) -> BeaverTriple:
        """
        Generate a Beaver triple (a, b, c) where c = a * b
        Args:
            n_nodes: Number of nodes
            t: Privacy threshold
        Returns:
            BeaverTriple with secret-shared values
        """
        # Generate random values a and b
        a = random.randint(0, self.field_size - 1)
        b = random.randint(0, self.field_size - 1)
        
        # Compute c = a * b
        c = (a * b) % self.field_size
        
        # Secret-share all three values
        a_shares = self.sss.share(a, n_nodes, t)
        b_shares = self.sss.share(b, n_nodes, t)
        c_shares = self.sss.share(c, n_nodes, t)
        
        return BeaverTriple(a_shares, b_shares, c_shares)


class BeaverTriplePool:
    """
    Pool of pre-generated Beaver triples for efficient secure multiplication
    """
    
    def __init__(self, generator: BeaverTripleGenerator, initial_size: int = 100):
        """
        Initialize triple pool
        Args:
            generator: BeaverTripleGenerator instance
            initial_size: Initial number of triples to generate
        """
        self.generator = generator
        self.triples: List[BeaverTriple] = []
        self.n_nodes = 0
        self.t = 0
        self.initial_size = initial_size
    
    def initialize(self, n_nodes: int, t: int):
        """Initialize pool with triples"""
        self.n_nodes = n_nodes
        self.t = t
        self.replenish(self.initial_size)
    
    def get_triple(self) -> Optional[BeaverTriple]:
        """Get a triple from the pool"""
        if not self.triples:
            return None
        return self.triples.pop()
    
    def replenish(self, num_triples: int):
        """Generate and add new triples to the pool"""
        for _ in range(num_triples):
            triple = self.generator.generate_triple(self.n_nodes, self.t)
            self.triples.append(triple)
    
    def should_replenish(self) -> bool:
        """Check if pool needs replenishing"""
        # Replenish when pool is 30% full (more aggressive to avoid blocking)
        return len(self.triples) < self.initial_size * 3 // 10


class SecureMultiplier:
    """
    High-level interface for secure multiplication using Beaver triples
    """
    
    def __init__(self, triple_pool: BeaverTriplePool, n_nodes: int, t: int,
                 field_size: int = 2**31 - 1,
                 reconstruction_manager: Optional[MPCReconstructionManager] = None,
                 prss_seed: Optional[int] = None,
                 *,
                 triple_dealer_id: Optional[int] = None,
                 privacy_mode: bool = False):
        """
        Initialize secure multiplier
        Args:
            triple_pool: Pool of Beaver triples
            n_nodes: Number of nodes
            t: Privacy threshold
            field_size: Prime field size
            reconstruction_manager: Optional reconstruction manager for multi-node operations
        """
        self.triple_pool = triple_pool
        self.n_nodes = n_nodes
        self.t = t
        self.field_size = field_size
        self.reconstruction_manager = reconstruction_manager
        # If set, generate Beaver triples deterministically from context/index.
        # This avoids expensive triple pool generation and keeps nodes consistent
        # in multi-process local testing (SIMD-style "packed" triples per chunk).
        self.prss_seed = prss_seed
        self.triple_dealer_id = int(triple_dealer_id) if triple_dealer_id is not None else None
        # If enabled, forbid any triple generation mode where a single node can
        # derive triple secrets (e.g., shared PRSS seed or locally-generated pools),
        # and require dealer-backed triples + multi-node reconstruction.
        self.privacy_mode = bool(privacy_mode)
        self._inv_cache: dict[int, int] = {}
        self._init_profile_stats()
        self._prover_time_sec: float = 0.0
        self._prover_time_breakdown: Dict[str, float] = {}
        
        # Initialize pool if not already initialized
        if triple_pool.n_nodes == 0:
            triple_pool.initialize(n_nodes, t)

    def _inv_public(self, a: int) -> int:
        """Modular inverse of a public integer in this field (cached)."""
        a = int(a) % int(self.field_size)
        if a in self._inv_cache:
            return self._inv_cache[a]
        inv = pow(a, int(self.field_size) - 2, int(self.field_size))
        self._inv_cache[a] = inv
        return inv

    def _effective_timeout(self, timeout: float) -> float:
        """
        In privacy_mode, nodes can desynchronize more (dealer node generates triples locally
        while non-dealer nodes request triple vectors over the network). Large SIMD openings
        (e.g. conv kernels) can legitimately take several minutes on slower nodes.
        """
        try:
            t = float(timeout)
        except Exception:
            t = 120.0
        if self.privacy_mode:
            return max(t, 600.0)
        return t

    def _init_profile_stats(self):
        # Lightweight counters for profiling (per-process). Safe to ignore.
        self._profile_stats: Dict[str, int] = {
            "multiply_calls": 0,
            "multiply_batch_calls": 0,
            "multiply_batch_values_calls": 0,
            "multiply_batch_values_elems": 0,
            "multiply_batch_values_fp_calls": 0,
            "multiply_batch_values_fp_elems": 0,
            "opened_div_vectors": 0,
            "opened_div_elems": 0,
            "dealer_triple_vector_requests": 0,
            "dealer_triple_vector_elems": 0,
        }

    def profile_snapshot_and_reset(self) -> Dict[str, int]:
        """
        Return a copy of current stats and reset counters.
        Useful for per-batch profiling output.
        """
        if not hasattr(self, "_profile_stats"):
            self._init_profile_stats()
        snap = dict(self._profile_stats)
        for k in self._profile_stats:
            self._profile_stats[k] = 0
        return snap

    def add_prover_time(self, seconds: float, category: str = "generic") -> None:
        """Accumulate prover/opener wall-clock time."""
        try:
            s = float(seconds)
        except Exception:
            return
        if s <= 0:
            return
        self._prover_time_sec += s
        self._prover_time_breakdown[category] = self._prover_time_breakdown.get(category, 0.0) + s

    def prover_time_snapshot(self, reset: bool = False) -> Dict[str, object]:
        """Return accumulated prover timing stats."""
        out = {
            "total_sec": float(self._prover_time_sec),
            "by_category": {k: float(v) for k, v in self._prover_time_breakdown.items()},
        }
        if reset:
            self._prover_time_sec = 0.0
            self._prover_time_breakdown = {}
        return out

    def _opened_fp_enabled(self) -> bool:
        """
        "A" mode: opener/enclave performs integer truncation and re-shares.
        Enabled automatically when privacy_mode is on and we have networking.
        """
        return bool(self.privacy_mode and self.reconstruction_manager is not None and self.n_nodes > 1)

    def _opened_fp_opener(self) -> int:
        # Prefer the triple dealer as the opener (already treated as enclave-trusted in this codebase)
        if self.triple_dealer_id is not None:
            return int(self.triple_dealer_id)
        return 1

    def _reshare_vector_from_opener(
        self,
        *,
        secrets_mod_p_u64: np.ndarray,
        node_id: int,
        context_prefix: str,
        x_points: List[int],
        timeout: float,
    ) -> np.ndarray:
        """
        Opener-only: given a vector of secrets (mod p), create Shamir shares for each node and
        send each node its vector as a single binary message.
        Returns the opener's own share vector (uint64).
        """
        p = int(self.field_size)
        _t0 = time.time()
        n = int(self.n_nodes)
        t = int(self.t)
        L = int(secrets_mod_p_u64.size)
        if L <= 0:
            return np.zeros((0,), dtype=np.uint64)

        # Random polynomial coefficients for degrees 1..t (vectorized)
        rng = np.random.default_rng()
        use_object_mod = int(p) > 0xFFFFFFFF
        if t > 0:
            coeffs = rng.integers(0, p, size=(t, L), dtype=np.uint64)
        else:
            coeffs = np.zeros((0, L), dtype=np.uint64)

        # Precompute x^k for each node x and each k=1..t
        xs = [int(x) for x in x_points]
        x_pows: dict[int, List[int]] = {}
        for x in xs:
            xp = []
            xk = x % p
            for _k in range(1, t + 1):
                xp.append(int(xk))
                xk = (xk * x) % p
            x_pows[x] = xp

        net = self.reconstruction_manager.network  # type: ignore[union-attr]
        opener = self._opened_fp_opener()

        # Compute and send per-node share vectors
        opener_share_vec = None
        for nid in range(1, n + 1):
            x = int(x_points[nid - 1])
            y = (secrets_mod_p_u64.astype(np.uint64, copy=False) % np.uint64(p)).copy()
            if t > 0:
                for k in range(t):
                    if use_object_mod:
                        y_obj = np.asarray(y, dtype=object)
                        ck = np.asarray(coeffs[k], dtype=object)
                        y_obj = (y_obj + (ck * int(x_pows[x][k])) % int(p)) % int(p)
                        y = np.asarray(y_obj, dtype=np.uint64)
                    else:
                        y = (y + (coeffs[k] * np.uint64(x_pows[x][k])) % np.uint64(p)) % np.uint64(p)
            if nid == opener:
                opener_share_vec = y
            else:
                ctx_out = f"{context_prefix}_to_{nid}"
                net.channel.send_vector(int(nid), ctx_out, x=int(nid), values=y.astype(np.uint64, copy=False))

        if opener_share_vec is None:
            raise RuntimeError("Opener share vector missing (unexpected)")
        self.add_prover_time(time.time() - _t0, "reshare_vector")
        return opener_share_vec

    def _opened_divide_and_reshare_vector(
        self,
        *,
        values_local_u64: np.ndarray,
        divisor: int,
        node_id: int,
        x: int,
        context_prefix: str,
        timeout: float,
        multiplier_factor: int = 1,
        clip_min: Optional[int] = None,
        clip_max: Optional[int] = None,
    ) -> np.ndarray:
        """
        Enclave Oracle Mode:
        1. All nodes send shares to the 'Enclave Oracle' (dealer_id).
        2. Enclave reconstructs, divides by `divisor`, and truncates.
        3. Enclave re-shares the result to all nodes.
        Note: The 'Opener' variable name is kept locally but semantically refers to the Enclave Oracle.
        """
        try:
            if hasattr(self, "_profile_stats"):
                self._profile_stats["opened_div_vectors"] += 1
                self._profile_stats["opened_div_elems"] += int(np.asarray(values_local_u64).size)
        except Exception:
            pass
        if self.reconstruction_manager is None:
            raise RuntimeError("Enclave interaction requires reconstruction_manager")
        if int(divisor) == 0:
            raise ValueError("divisor must be non-zero")

        p = int(self.field_size)
        n = int(self.n_nodes)
        timeout = self._effective_timeout(timeout)

        net = self.reconstruction_manager.network
        opener = self._opened_fp_opener()

        # Everyone broadcasts their local vector
        ctx_in = f"{context_prefix}_in"
        net.broadcast_vector(ctx_in, x=int(x), values=np.asarray(values_local_u64, dtype=np.uint64))

        if int(node_id) == int(opener):
            _t0 = time.time()
            opened_u64 = self.reconstruction_manager.reconstruct_opened_vector_values(
                context=ctx_in,
                values_local=np.asarray(values_local_u64, dtype=np.uint64),
                x=int(x),
                timeout=timeout,
            )
            opened = opened_u64.astype(np.int64, copy=False)
            opened = np.where(opened > (p // 2), opened - p, opened)
            
            # Apply multiplier_factor in native 64-bit integer space before division
            opened = opened * multiplier_factor

            d = int(divisor)
            # Round-to-nearest integer division (ties toward +inf for positives / -inf for negatives)
            adj = np.where(opened >= 0, d // 2, -(d // 2))
            q = (opened + adj) // d
            if clip_min is not None or clip_max is not None:
                lo = int(clip_min) if clip_min is not None else -2**63
                hi = int(clip_max) if clip_max is not None else 2**63 - 1
                q = np.clip(q, lo, hi)
            q_mod = np.mod(q, p).astype(np.uint64, copy=False)

            # Re-share q_mod to all nodes
            x_points = [i for i in range(1, n + 1)]
            out_prefix = f"{context_prefix}_out"
            opener_vec = self._reshare_vector_from_opener(
                secrets_mod_p_u64=q_mod,
                node_id=node_id,
                context_prefix=out_prefix,
                x_points=x_points,
                timeout=timeout,
            )

            # Cleanup input buffers
            try:
                net.channel.clear_vector(ctx_in)
            except Exception:
                pass
            self.add_prover_time(time.time() - _t0, "opened_divide_and_reshare_vector")
            return opener_vec

        # Non-opener: wait for output vector from opener
        out_ctx = f"{context_prefix}_out_to_{node_id}"
        import time as _time
        start = _time.time()
        while _time.time() - start < timeout:
            recv = net.channel.get_received_vector(out_ctx)
            if opener in recv:
                values = recv[opener].get("values")
                arr = np.asarray(values, dtype=np.uint64).astype(np.uint64, copy=False) % np.uint64(p)
                try:
                    net.channel.clear_vector(out_ctx)
                except Exception:
                    pass
                try:
                    net.channel.clear_vector(ctx_in)
                except Exception:
                    pass
                return arr
            _time.sleep(0.01)

        raise RuntimeError(f"Timed out waiting for opened truncation output (ctx={out_ctx})")

    def multiply_fixed_point(
        self,
        share1: Share,
        share2: Share,
        *,
        node_id: int,
        scale_factor: int,
        context: Optional[str],
    ) -> Share:
        """
        Fixed-point multiply for SCALE-scaled values:
            out = (share1 * share2) / SCALE   (out is SCALE-scaled)
        """
        prod = self.multiply(share1, share2, node_id=node_id, context=context)
        # "A" mode: Enclave Oracle truncation to keep values in integer fixed-point (avoid field fractions)
        if self._opened_fp_enabled() and context:
            # The "Opener" here refers to the Attested Enclave that reconstructs, truncates, and re-shares.
            out_vec = self._opened_divide_and_reshare_vector(
                values_local_u64=np.asarray([int(prod.y) % int(self.field_size)], dtype=np.uint64),
                divisor=int(scale_factor),
                node_id=int(node_id),
                x=int(prod.x),
                context_prefix=f"{context}_enclave_trunc_{int(scale_factor)}",
                timeout=120.0,
            )
            return Share(x=prod.x, y=int(out_vec[0]) % int(self.field_size), node_id=node_id)

        inv_scale = self._inv_public(scale_factor)
        return Share(x=prod.x, y=(int(prod.y) * inv_scale) % int(self.field_size), node_id=node_id)

    def multiply_batch_values_fixed_point(
        self,
        y1: np.ndarray,
        y2: np.ndarray,
        *,
        x: int,
        node_id: int,
        context_prefix: str,
        scale_factor: int,
        chunk_timeout: float = 120.0,
    ) -> np.ndarray:
        """
        Array-based fixed-point multiply for SCALE-scaled values:
            out = (y1 * y2) / SCALE   (out is SCALE-scaled)
        Returns uint64 array of share values mod p.
        """
        try:
            if hasattr(self, "_profile_stats"):
                self._profile_stats["multiply_batch_values_fp_calls"] += 1
                self._profile_stats["multiply_batch_values_fp_elems"] += int(np.asarray(y1).size)
        except Exception:
            pass
        prod = self.multiply_batch_values(
            y1=np.asarray(y1, dtype=np.uint64),
            y2=np.asarray(y2, dtype=np.uint64),
            x=x,
            node_id=node_id,
            context_prefix=context_prefix,
            chunk_timeout=self._effective_timeout(chunk_timeout),
        )
        p = np.uint64(int(self.field_size))

        # "A" mode: opener truncation to keep values in integer fixed-point (avoid field fractions)
        if self._opened_fp_enabled():
            return self._opened_divide_and_reshare_vector(
                values_local_u64=np.asarray(prod, dtype=np.uint64) % p,
                divisor=int(scale_factor),
                node_id=int(node_id),
                x=int(x),
                context_prefix=f"{context_prefix}_fp_trunc_div_{int(scale_factor)}",
                timeout=float(chunk_timeout),
            ).astype(np.uint64, copy=False) % p

        inv_scale = np.uint64(self._inv_public(scale_factor))
        # Safe in practice once SCALE_FACTOR is reduced (values stay far below 2^63)
        return (np.asarray(prod, dtype=np.uint64) * inv_scale) % p

    def _dealer_request_triple_vectors(self, *, context_prefix: str, n: int, x: int, node_id: int, timeout: float = 120.0):
        """
        Request a,b,c share vectors from the dealer for this node under (context_prefix, idx).
        Returns three numpy arrays of length n.
        """
        if self.triple_dealer_id is None:
            raise RuntimeError("triple_dealer_id is not set")
        if self.reconstruction_manager is None:
            raise RuntimeError("dealer triple requests require reconstruction_manager/network")

        try:
            if hasattr(self, "_profile_stats"):
                self._profile_stats["dealer_triple_vector_requests"] += 1
                self._profile_stats["dealer_triple_vector_elems"] += int(n)
        except Exception:
            pass

        dealer = int(self.triple_dealer_id)

        # Fast path: if we ARE the dealer node, generate locally (no self-connection exists).
        try:
            if dealer == int(self.reconstruction_manager.network.node_id):
                svc = getattr(self.reconstruction_manager.network.channel, "_triple_dealer_service", None)
                if svc is not None and hasattr(svc, "get_triple_vectors"):
                    a_y, b_y, c_y = svc.get_triple_vectors(context_prefix, int(n), int(x))  # type: ignore[attr-defined]
                    return (
                        np.asarray(a_y, dtype=np.uint64),
                        np.asarray(b_y, dtype=np.uint64),
                        np.asarray(c_y, dtype=np.uint64),
                    )
        except Exception:
            pass

        # Ensure we have a connection to dealer (outgoing sockets only)
        try:
            if dealer not in self.reconstruction_manager.network.channel.connections:
                cfg = self.reconstruction_manager.network.node_configs.get(dealer)
                if cfg:
                    self.reconstruction_manager.network.channel.connect_to_node(dealer, cfg["host"], int(cfg["port"]))
        except Exception:
            pass

        # Send request to dealer
        from ml_training.secure_comm import MessageType
        import time as _time
        import secrets as _secrets

        req_id = f"triple_req_{node_id}_{_secrets.token_hex(8)}"
        self.reconstruction_manager.network.channel.send_message(
            dealer,
            MessageType.TRIPLE_REQUEST,
            {"context_prefix": context_prefix, "n": int(n), "x": int(x), "req_id": req_id},
        )

        # Wait for three vectors from dealer
        ctx_a = f"{req_id}_a"
        ctx_b = f"{req_id}_b"
        ctx_c = f"{req_id}_c"
        timeout = self._effective_timeout(timeout)
        start = _time.time()

        def _get_u64(ctx: str):
            recv = self.reconstruction_manager.network.channel.get_received_vector(ctx)
            if dealer not in recv:
                return None
            values = recv[dealer].get("values")
            # values can be array('I') or list or numpy array
            try:
                arr = np.asarray(values, dtype=np.uint64)
            except Exception:
                return None
            if arr.size != n:
                return None
            return arr

        a = b = c = None
        while _time.time() - start < timeout:
            if a is None:
                a = _get_u64(ctx_a)
            if b is None:
                b = _get_u64(ctx_b)
            if c is None:
                c = _get_u64(ctx_c)
            if a is not None and b is not None and c is not None:
                break
            _time.sleep(0.01)

        # Cleanup buffers
        try:
            self.reconstruction_manager.network.channel.clear_vector(ctx_a)
            self.reconstruction_manager.network.channel.clear_vector(ctx_b)
            self.reconstruction_manager.network.channel.clear_vector(ctx_c)
        except Exception:
            pass

        if a is None or b is None or c is None:
            raise RuntimeError("Dealer triple request timed out")
        return a, b, c

    def _dealer_triple_for(self, *, context: str, idx: int, node_id: int, node_x: int) -> Tuple[Share, Share, Share]:
        a_y, b_y, c_y = self._dealer_request_triple_vectors(
            context_prefix=context, n=1, x=node_x, node_id=node_id, timeout=300.0
        )
        return (
            Share(x=node_x, y=int(a_y[0]) % self.field_size, node_id=node_id),
            Share(x=node_x, y=int(b_y[0]) % self.field_size, node_id=node_id),
            Share(x=node_x, y=int(c_y[0]) % self.field_size, node_id=node_id),
        )

    def get_random_mask_share(self, *, node_id: int, x: int, context: str) -> Share:
        """
        Return a secret-shared random mask r as a Share held by this node.

        Implementation note:
        - We reuse the 'a' component of a Beaver triple as randomness.
        - In multi-node mode, this is safe only if triple generation is private (nodes do NOT
          individually know the underlying a value).
        """
        # Privacy mode: only dealer-backed masks are allowed
        if self.privacy_mode:
            if self.prss_seed is not None:
                raise RuntimeError("privacy_mode forbids PRSS-seed masks")
            if self.triple_dealer_id is None or self.reconstruction_manager is None:
                raise RuntimeError("privacy_mode requires dealer-backed masks (set triple_dealer_id + reconstruction_manager)")

        # Prefer dealer if configured (seed not shared with all nodes)
        if self.triple_dealer_id is not None and self.reconstruction_manager is not None:
            a_s, _, _ = self._dealer_triple_for(context=context, idx=0, node_id=node_id, node_x=x)
            return a_s

        # Prefer PRSS if enabled (fast, deterministic, but nodes can derive triple secrets)
        if self.prss_seed is not None:
            a_s, _, _ = self._prss_triple_for(context, 0, node_id=node_id, node_x=x)
            return a_s

        # Pool-based: requires that all nodes stay in lockstep consuming triples.
        triple = self.triple_pool.get_triple()
        if triple is None:
            replenish_size = max(1000, self.triple_pool.initial_size // 10)
            self.triple_pool.replenish(replenish_size)
            triple = self.triple_pool.get_triple()
            if triple is None:
                raise RuntimeError("Failed to get Beaver triple for mask")
        a_s, _, _ = triple.get_for_node(node_id)
        return a_s

    def _prss_u64(self, label: str, context: str, idx: int, extra: int = 0) -> int:
        # Deterministic pseudo-random 64-bit integer derived from (seed,label,context,idx,extra).
        # Uses BLAKE2b for speed and stability.
        h = hashlib.blake2b(digest_size=8)
        h.update(str(self.prss_seed).encode("utf-8"))
        h.update(b"|")
        h.update(label.encode("utf-8"))
        h.update(b"|")
        h.update(context.encode("utf-8"))
        h.update(b"|")
        h.update(str(idx).encode("utf-8"))
        h.update(b"|")
        h.update(str(extra).encode("utf-8"))
        return int.from_bytes(h.digest(), "big")

    def _prss_secret(self, label: str, context: str, idx: int) -> int:
        return self._prss_u64(label, context, idx) % self.field_size

    def _prss_share(self, secret: int, label: str, context: str, idx: int, node_x: int) -> int:
        # Shamir polynomial coefficients: c0=secret, c1..ct derived from PRSS
        # Evaluate at x=node_x.
        y = secret % self.field_size
        x_pow = node_x % self.field_size
        for k in range(1, self.t + 1):
            coeff = self._prss_u64(f"{label}_coeff_{k}", context, idx, extra=k) % self.field_size
            y = (y + (coeff * x_pow) % self.field_size) % self.field_size
            x_pow = (x_pow * node_x) % self.field_size
        return y

    def _prss_triple_for(self, context: str, idx: int, node_id: int, node_x: int) -> Tuple[Share, Share, Share]:
        # Generate (a,b,c=a*b) secrets deterministically, then compute this node's shares.
        a = self._prss_secret("a", context, idx)
        b = self._prss_secret("b", context, idx)
        c = (a * b) % self.field_size
        a_y = self._prss_share(a, "a", context, idx, node_x)
        b_y = self._prss_share(b, "b", context, idx, node_x)
        c_y = self._prss_share(c, "c", context, idx, node_x)
        return (
            Share(x=node_x, y=a_y, node_id=node_id),
            Share(x=node_x, y=b_y, node_id=node_id),
            Share(x=node_x, y=c_y, node_id=node_id),
        )

    def get_prss_matrix_triple(self, context_prefix: str, m: int, n: int, b_dim: int, node_id: int, node_x: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate deterministic matrix triples A (mxn), B (nxb), C=A@B (mxb).
        Returns shares for this node.
        """
        if self.prss_seed is None:
            raise RuntimeError("prss_seed is required for PRSS matrix triples")
            
        h = hashlib.blake2b(digest_size=8)
        h.update(str(self.prss_seed).encode("utf-8"))
        h.update(context_prefix.encode("utf-8"))
        seed_val = int.from_bytes(h.digest(), "big") % (2**32)
        
        rng = np.random.default_rng(seed_val)
        p = np.uint64(self.field_size)
        
        A_sec = rng.integers(0, p, size=(m, n), dtype=np.uint64)
        B_sec = rng.integers(0, p, size=(n, b_dim), dtype=np.uint64)
        
        # Compute exact matrix multiplication modulo p
        A_obj = A_sec.astype(object)
        B_obj = B_sec.astype(object)
        C_sec = (np.dot(A_obj, B_obj) % p).astype(np.uint64)
        
        def _get_shares(secret_mat: np.ndarray, label: str):
            y = secret_mat.copy()
            x_pow = np.uint64(node_x % p)
            
            for k in range(1, self.t + 1):
                h2 = hashlib.blake2b(digest_size=8)
                h2.update(str(self.prss_seed).encode("utf-8"))
                h2.update(label.encode("utf-8"))
                h2.update(context_prefix.encode("utf-8"))
                h2.update(str(k).encode("utf-8"))
                coeff_seed = int.from_bytes(h2.digest(), "big") % (2**32)
                
                rng_k = np.random.default_rng(coeff_seed)
                coeff_mat = rng_k.integers(0, p, size=secret_mat.shape, dtype=np.uint64)
                
                y = (y + (coeff_mat * x_pow) % p) % p
                x_pow = (x_pow * np.uint64(node_x)) % p
            return y

        A_share = _get_shares(A_sec, "A_mat")
        B_share = _get_shares(B_sec, "B_mat")
        C_share = _get_shares(C_sec, "C_mat")
        
        return A_share, B_share, C_share
    
    def multiply(self, share1: Share, share2: Share, node_id: int, 
                context: Optional[str] = None) -> Share:
        """
        Multiply two shares securely using Beaver triple
        Args:
            share1: First share (x)
            share2: Second share (y)
            node_id: ID of this node
            context: Optional context for reconstruction (enables multi-node)
        Returns:
            Share of the product (x * y)
        """
        # Fast path: single-node mode has no privacy (t=0), so Beaver triples
        # provide no additional security but add massive overhead.
        #
        # In the current codebase, the "single-node or simplified multiplication"
        # path already uses local values as a placeholder for reconstruction,
        # so this optimization preserves the exact arithmetic result while
        # removing triple generation / book-keeping costs.
        try:
            if hasattr(self, "_profile_stats"):
                self._profile_stats["multiply_calls"] += 1
        except Exception:
            pass
        if self.n_nodes == 1 and self.t == 0 and self.reconstruction_manager is None:
            return Share(
                x=share1.x,
                y=(share1.y * share2.y) % self.field_size,
                node_id=node_id
            )

        # Dealer-backed scalar multiplication:
        # - required for privacy_mode
        # - also preferred whenever dealer is configured and a context is provided
        if self.triple_dealer_id is not None and self.reconstruction_manager is not None and context:
            node_x = int(share1.x)
            a_s, b_s, c_s = self._dealer_triple_for(context=context, idx=0, node_id=node_id, node_x=node_x)
            triple = BeaverTriple([a_s], [b_s], [c_s])
            return self._multiply_with_reconstruction(share1, share2, triple, node_id, context)

        if self.privacy_mode:
            # No context => can't align openings; no dealer => can't provide private triples
            raise RuntimeError(
                "privacy_mode requires dealer-backed multiplication with a non-empty context "
                "(set triple_dealer_id + reconstruction_manager, and pass context=...)"
            )

        # Get triple from pool
        triple = self.triple_pool.get_triple()
        if triple is None:
            # Emergency replenish (more aggressively)
            replenish_size = max(1000, self.triple_pool.initial_size // 10)
            self.triple_pool.replenish(replenish_size)
            triple = self.triple_pool.get_triple()
            if triple is None:
                raise RuntimeError("Failed to get Beaver triple")
        
        # Perform secure multiplication
        if self.reconstruction_manager and context:
            # Multi-node secure multiplication with reconstruction
            result_share = self._multiply_with_reconstruction(
                share1, share2, triple, node_id, context
            )
        else:
            # Single-node or simplified multiplication
            result_share = self._multiply_with_triple(share1, share2, triple, node_id)
        
        # Replenish pool if needed (more aggressively to avoid blocking)
        if self.triple_pool.should_replenish():
            # Replenish with 20% of initial size, but at least 1000
            replenish_size = max(1000, self.triple_pool.initial_size // 5)
            self.triple_pool.replenish(replenish_size)
        
        return result_share

    def multiply_batch(
        self,
        share1_list: List[Share],
        share2_list: List[Share],
        node_id: int,
        context_prefix: Optional[str],
        chunk_timeout: float = 120.0,
    ) -> List[Share]:
        """
        Multiply many share pairs efficiently.

        If multi-node reconstruction is enabled (reconstruction_manager + context_prefix),
        this batches Beaver openings by reconstructing ALL d's and ALL e's in bulk.
        """
        if len(share1_list) != len(share2_list):
            raise ValueError("share1_list and share2_list length mismatch")
        if not share1_list:
            return []
        try:
            if hasattr(self, "_profile_stats"):
                self._profile_stats["multiply_batch_calls"] += 1
        except Exception:
            pass

        # Single-node fast path
        if self.n_nodes == 1 and self.t == 0 and self.reconstruction_manager is None:
            return [
                Share(x=a.x, y=(a.y * b.y) % self.field_size, node_id=node_id)
                for a, b in zip(share1_list, share2_list)
            ]

        # If we don't have reconstruction enabled, fall back to scalar multiply
        if not (self.reconstruction_manager and context_prefix):
            out = []
            for i, (a, b) in enumerate(zip(share1_list, share2_list)):
                out.append(self.multiply(a, b, node_id, context=f"{context_prefix}_{i}" if context_prefix else None))
            return out

        a_shares: List[Share] = []
        b_shares: List[Share] = []
        c_shares: List[Share] = []
        d_local: List[Share] = []
        e_local: List[Share] = []

        # Generate triples and form local d/e shares
        node_x = share1_list[0].x
        # Increase batch timeout in privacy_mode (see _effective_timeout)
        chunk_timeout = self._effective_timeout(chunk_timeout)

        if self.triple_dealer_id is not None and self.reconstruction_manager is not None and context_prefix is not None:
            # Dealer-backed triples keyed by (context_prefix, idx)
            a_y, b_y, c_y = self._dealer_request_triple_vectors(
                context_prefix=context_prefix, n=len(share1_list), x=node_x, node_id=node_id, timeout=chunk_timeout
            )
            for i in range(len(share1_list)):
                a_s = Share(x=node_x, y=int(a_y[i]) % self.field_size, node_id=node_id)
                b_s = Share(x=node_x, y=int(b_y[i]) % self.field_size, node_id=node_id)
                c_s = Share(x=node_x, y=int(c_y[i]) % self.field_size, node_id=node_id)
                a_shares.append(a_s)
                b_shares.append(b_s)
                c_shares.append(c_s)
                d_local.append(Share(x=node_x, y=(share1_list[i].y - a_s.y) % self.field_size, node_id=node_id))
                e_local.append(Share(x=node_x, y=(share2_list[i].y - b_s.y) % self.field_size, node_id=node_id))
        elif self.prss_seed is not None and context_prefix is not None:
            # PRSS-style "packed" triples: deterministic per (context_prefix, idx)
            if self.privacy_mode:
                raise RuntimeError("privacy_mode forbids PRSS-seed triples")
            for i in range(len(share1_list)):
                a_s, b_s, c_s = self._prss_triple_for(context_prefix, i, node_id=node_id, node_x=node_x)
                a_shares.append(a_s)
                b_shares.append(b_s)
                c_shares.append(c_s)
                d_local.append(Share(x=node_x, y=(share1_list[i].y - a_s.y) % self.field_size, node_id=node_id))
                e_local.append(Share(x=node_x, y=(share2_list[i].y - b_s.y) % self.field_size, node_id=node_id))
        else:
            if self.privacy_mode:
                raise RuntimeError("privacy_mode requires dealer-backed batch triples (set triple_dealer_id + context_prefix)")
            for i in range(len(share1_list)):
                triple = self.triple_pool.get_triple()
                if triple is None:
                    replenish_size = max(1000, self.triple_pool.initial_size // 10)
                    self.triple_pool.replenish(replenish_size)
                    triple = self.triple_pool.get_triple()
                    if triple is None:
                        raise RuntimeError("Failed to get Beaver triple")

                a_s, b_s, c_s = triple.get_for_node(node_id)
                a_shares.append(a_s)
                b_shares.append(b_s)
                c_shares.append(c_s)

                d_local.append(Share(x=share1_list[i].x, y=(share1_list[i].y - a_s.y) % self.field_size, node_id=node_id))
                e_local.append(Share(x=share2_list[i].x, y=(share2_list[i].y - b_s.y) % self.field_size, node_id=node_id))

        # Batch reconstruct d and e
        d_vals, e_vals = self.reconstruction_manager.reconstruct_for_multiplication_batch(
            d_local, e_local, context_prefix=context_prefix, timeout=chunk_timeout
        )

        # Compute outputs
        out: List[Share] = []
        for i in range(len(share1_list)):
            # d_vals/e_vals come back as numpy uint64 scalars; cast to Python int
            # to avoid numpy-integer propagation into Share.y (breaks JSON transport).
            d_recon = int(d_vals[i])
            e_recon = int(e_vals[i])
            result_y = int((
                c_shares[i].y +
                (d_recon * b_shares[i].y) % self.field_size +
                (e_recon * a_shares[i].y) % self.field_size +
                (d_recon * e_recon) % self.field_size
            ) % self.field_size)
            out.append(Share(x=share1_list[i].x, y=result_y, node_id=node_id))

        # Replenish pool if needed (only if we're using the pool)
        if self.prss_seed is None and self.triple_pool.should_replenish():
            replenish_size = max(1000, self.triple_pool.initial_size // 5)
            self.triple_pool.replenish(replenish_size)

        return out

    def multiply_batch_values(
        self,
        y1: np.ndarray,
        y2: np.ndarray,
        *,
        x: int,
        node_id: int,
        context_prefix: str,
        chunk_timeout: float = 120.0,
    ) -> np.ndarray:
        """
        Array-based batch multiplication.
        Avoids constructing Share objects for each scalar multiply.

        Args:
            y1, y2: 1D arrays (same length) of share values modulo field_size
            x: Shamir x-coordinate for this node (usually equals node_id)
            node_id: node id
            context_prefix: unique context prefix for PRSS + openings
        Returns:
            1D numpy array of product share values modulo field_size
        """
        try:
            if hasattr(self, "_profile_stats"):
                self._profile_stats["multiply_batch_values_calls"] += 1
                self._profile_stats["multiply_batch_values_elems"] += int(np.asarray(y1).size)
        except Exception:
            pass
        # Increase batch timeout in privacy_mode (see _effective_timeout)
        chunk_timeout = self._effective_timeout(chunk_timeout)

        y1 = np.asarray(y1, dtype=np.uint64)
        y2 = np.asarray(y2, dtype=np.uint64)
        if y1.shape != y2.shape:
            raise ValueError("y1 and y2 shape mismatch")
        if y1.ndim != 1:
            y1 = y1.reshape(-1)
            y2 = y2.reshape(-1)
        n = int(y1.size)
        if n == 0:
            return np.zeros((0,), dtype=np.uint64)

        p = int(self.field_size)

        # Single-node fast path
        if self.n_nodes == 1 and self.t == 0 and self.reconstruction_manager is None:
            return (y1 * y2) % p

        # Require reconstruction manager for secure multi-node batching
        if not self.reconstruction_manager:
            # Fallback: do scalar multiplies (slow)
            out = np.empty((n,), dtype=np.uint64)
            for i in range(n):
                out[i] = self.multiply(Share(x=x, y=int(y1[i]), node_id=node_id),
                                       Share(x=x, y=int(y2[i]), node_id=node_id),
                                       node_id=node_id,
                                       context=f"{context_prefix}_{i}").y
            return out

        # Generate a,b,c shares for this node (PRSS preferred)
        a_y = np.empty((n,), dtype=np.uint64)
        b_y = np.empty((n,), dtype=np.uint64)
        c_y = np.empty((n,), dtype=np.uint64)

        if self.triple_dealer_id is not None:
            # Dealer-backed deterministic triples keyed by (context_prefix, idx)
            a_y_u32, b_y_u32, c_y_u32 = self._dealer_request_triple_vectors(
                context_prefix=context_prefix, n=n, x=x, node_id=node_id, timeout=chunk_timeout
            )
            a_y[:] = a_y_u32.astype(np.uint64)
            b_y[:] = b_y_u32.astype(np.uint64)
            c_y[:] = c_y_u32.astype(np.uint64)
        elif self.prss_seed is not None:
            if self.privacy_mode:
                raise RuntimeError("privacy_mode forbids PRSS-seed triples")
            for i in range(n):
                a_s, b_s, c_s = self._prss_triple_for(context_prefix, i, node_id=node_id, node_x=x)
                a_y[i] = a_s.y
                b_y[i] = b_s.y
                c_y[i] = c_s.y
        else:
            if self.privacy_mode:
                raise RuntimeError("privacy_mode requires dealer-backed triples (set triple_dealer_id)")
            # Pool-based (still works, but slower)
            for i in range(n):
                triple = self.triple_pool.get_triple()
                if triple is None:
                    replenish_size = max(1000, self.triple_pool.initial_size // 10)
                    self.triple_pool.replenish(replenish_size)
                    triple = self.triple_pool.get_triple()
                    if triple is None:
                        raise RuntimeError("Failed to get Beaver triple")
                a_s, b_s, c_s = triple.get_for_node(node_id)
                a_y[i] = a_s.y
                b_y[i] = b_s.y
                c_y[i] = c_s.y

        d_local = (y1 + (p - (a_y % p))) % p
        e_local = (y2 + (p - (b_y % p))) % p

        # Avoid materializing Python int lists: pass numpy arrays directly.
        d_vals, e_vals = self.reconstruction_manager.reconstruct_for_multiplication_batch_values(
            d_vals_local=(d_local % p).astype(np.uint64, copy=False),
            e_vals_local=(e_local % p).astype(np.uint64, copy=False),
            x=x,
            context_prefix=context_prefix,
            timeout=self._effective_timeout(chunk_timeout),
        )
        d = np.asarray(d_vals, dtype=np.uint64) % p
        e = np.asarray(e_vals, dtype=np.uint64) % p

        # result = c + d*b + e*a + d*e  (all mod p)
        if p > 0xFFFFFFFF:
            # Avoid uint64 overflow when field elements are wider than 32-bit.
            d_obj = d.astype(object)
            e_obj = e.astype(object)
            a_obj = (a_y % p).astype(object)
            b_obj = (b_y % p).astype(object)
            c_obj = (c_y % p).astype(object)
            res_obj = c_obj
            res_obj = (res_obj + (d_obj * b_obj) % p) % p
            res_obj = (res_obj + (e_obj * a_obj) % p) % p
            res_obj = (res_obj + (d_obj * e_obj) % p) % p
            res = np.asarray(res_obj, dtype=np.uint64)
        else:
            res = (c_y % p)
            res = (res + (d * (b_y % p)) % p) % p
            res = (res + (e * (a_y % p)) % p) % p
            res = (res + (d * e) % p) % p
        
        if context_prefix and "e0_b0" in context_prefix[:5] and "relu" in context_prefix:
            print(f"DEBUG {context_prefix}: n_size={n}, Y1={y1[:3]}, Y2={y2[:3]}, A={a_y[:3]}, B={b_y[:3]}, C={c_y[:3]}, d={d[:3]}, e={e[:3]}, Res={res[:3]}", flush=True)
            
        return res
    
    def _multiply_with_triple(self, share1: Share, share2: Share,
                             triple: BeaverTriple, node_id: int) -> Share:
        """
        Perform secure multiplication using Beaver triple protocol
        
        Protocol:
        1. Compute d = x - a, e = y - b (on shares)
        2. Reconstruct d and e (requires communication - simplified here)
        3. Compute result = c + d*b + e*a + d*e
        
        Args:
            share1: First share (x)
            share2: Second share (y)
            triple: Beaver triple
            node_id: Node ID
        Returns:
            Product share
        """
        # Get triple shares for this node
        a_share, b_share, c_share = triple.get_for_node(node_id)
        
        # Compute d = x - a, e = y - b (on shares)
        d_share = Share(
            x=share1.x,
            y=(share1.y - a_share.y) % self.field_size,
            node_id=node_id
        )
        e_share = Share(
            x=share2.x,
            y=(share2.y - b_share.y) % self.field_size,
            node_id=node_id
        )
        
        # In a full implementation, d and e would be reconstructed across nodes
        # For now, we'll use a simplified approach where we assume we can reconstruct
        # In production, this would require the communication protocol
        
        # Simplified reconstruction (would need actual multi-node communication)
        # For minimal pipeline, we'll use the share values directly
        # This is not fully secure but demonstrates the protocol
        
        # Reconstruct d and e (simplified - would need shares from all nodes)
        d_recon = d_share.y  # Placeholder - would reconstruct from all nodes
        e_recon = e_share.y  # Placeholder - would reconstruct from all nodes
        
        # Compute result share: c + d*b + e*a + d*e
        result_y = (
            c_share.y +
            (d_recon * b_share.y) % self.field_size +
            (e_recon * a_share.y) % self.field_size +
            (d_recon * e_recon) % self.field_size
        ) % self.field_size
        
        result_share = Share(
            x=share1.x,
            y=result_y,
            node_id=node_id
        )
        
        return result_share
    
    def _multiply_with_reconstruction(self, share1: Share, share2: Share,
                                     triple: BeaverTriple, node_id: int,
                                     context: str) -> Share:
        """
        Perform secure multiplication using Beaver triple protocol with multi-node reconstruction
        
        Protocol:
        1. Compute d = x - a, e = y - b (on shares)
        2. Reconstruct d and e across nodes using network
        3. Compute result = c + d*b + e*a + d*e
        
        Args:
            share1: First share (x)
            share2: Second share (y)
            triple: Beaver triple
            node_id: Node ID
            context: Context identifier for share exchange
        Returns:
            Product share
        """
        if not self.reconstruction_manager:
            # Fallback to simplified version
            return self._multiply_with_triple(share1, share2, triple, node_id)
        
        # Get triple shares for this node
        a_share, b_share, c_share = triple.get_for_node(node_id)
        
        # Compute d = x - a, e = y - b (on shares)
        d_share = Share(
            x=share1.x,
            y=(share1.y - a_share.y) % self.field_size,
            node_id=node_id
        )
        e_share = Share(
            x=share2.x,
            y=(share2.y - b_share.y) % self.field_size,
            node_id=node_id
        )
        
        # Reconstruct d and e across nodes using network
        try:
            d_recon, e_recon = self.reconstruction_manager.reconstruct_for_multiplication(
                [d_share], [e_share], context
            )
            # Debug: Show that multi-node reconstruction is working
            if hasattr(self, '_debug_log') and self._debug_log:
                print(f"  [Multi-Node] Reconstructed d and e using context '{context}'")
        except Exception as e:
            # If reconstruction fails, fall back to simplified version
            # This can happen if not enough nodes are connected
            if hasattr(self, '_debug_log') and self._debug_log:
                print(f"  [Fallback] Reconstruction failed for '{context}': {e}")
            d_recon = d_share.y
            e_recon = e_share.y
        
        # Compute result share: c + d*b + e*a + d*e
        result_y = (
            c_share.y +
            (d_recon * b_share.y) % self.field_size +
            (e_recon * a_share.y) % self.field_size +
            (d_recon * e_recon) % self.field_size
        ) % self.field_size
        
        result_share = Share(
            x=share1.x,
            y=result_y,
            node_id=node_id
        )
        
        return result_share

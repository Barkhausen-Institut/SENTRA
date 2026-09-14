"""
Secure Division Operations
Implements secure division and averaging on secret-shared values
"""

from typing import List, Optional
from ml_training.secret_sharing import Share
from ml_training.beaver_triples import SecureMultiplier
import numpy as np
import time


class SecureDivider:
    """
    Performs secure division operations on secret-shared values
    """
    
    def __init__(
        self,
        multiplier: SecureMultiplier,
        field_size: int = 2**31 - 1,
        scale_factor: int = 1000,
        *,
        debug: bool = False,
        debug_limit: int = 3,
    ):
        """
        Initialize secure divider
        Args:
            multiplier: SecureMultiplier instance
            field_size: Prime field size
            scale_factor: Scaling factor for integer representation
        """
        self.multiplier = multiplier
        self.field_size = field_size
        self.scale_factor = scale_factor
        self.debug = bool(debug)
        self._debug_limit = int(debug_limit)
        self._debug_count = 0
        # Precompute modular inverse of scale_factor (for fixed-point rescaling)
        # field_size is assumed prime in this codebase
        self._inv_scale_factor = pow(int(self.scale_factor) % self.field_size, self.field_size - 2, self.field_size)

    def _opened_enabled(self) -> bool:
        return bool(
            getattr(self.multiplier, "privacy_mode", False)
            and getattr(self.multiplier, "reconstruction_manager", None) is not None
            and int(getattr(self.multiplier, "n_nodes", 1)) > 1
        )

    def secure_inverse_share(self, denominator: Share, node_id: int, context: str = "inv") -> Share:
        """
        Securely compute the multiplicative inverse of a secret-shared value in the field.

        This uses the classic MPC trick:
        - sample secret-shared random r != 0
        - open u = denominator * r  (u reveals nothing about denominator if r is uniform and unknown)
        - inv(denominator) = r * inv(u)

        IMPORTANT:
        - Requires multi-node reconstruction (multiplier.reconstruction_manager)
        - Requires a source of secret-shared randomness. In this codebase, we reuse
          Beaver triples as that randomness source (multiplier must be configured consistently across nodes).
        """
        recon = getattr(self.multiplier, "reconstruction_manager", None)
        if recon is None:
            # Single-node fallback (share holds the full secret).
            if int(getattr(self.multiplier, "n_nodes", 1)) == 1 and int(getattr(self.multiplier, "t", 0)) == 0:
                p = int(self.field_size)
                denom = int(denominator.y) % p
                if denom == 0:
                    raise RuntimeError("secure_inverse_share failed: denominator was zero")
                inv = pow(denom, p - 2, p)
                return Share(x=denominator.x, y=inv, node_id=node_id)
            raise RuntimeError(
                "secure_inverse_share requires multi-node reconstruction_manager; "
                "run with --enable-network and a properly configured multiplier."
            )

        p = int(self.field_size)
        x = denominator.x

        # We may (extremely rarely) hit u == 0; retry a few times.
        for attempt in range(5):
            # Get a secret-shared random mask r (as a Share) without revealing it.
            # This consumes one triple internally (see SecureMultiplier.get_random_mask_share).
            r_share = self.multiplier.get_random_mask_share(node_id=node_id, x=x, context=f"{context}_r_{attempt}")

            # u = denominator * r  (still secret-shared)
            u_share = self.multiplier.multiply(denominator, r_share, node_id=node_id, context=f"{context}_u_{attempt}")

            # Open u (reconstruct across nodes). This reveals only a random field element.
            u_open = int(recon.get_reconstructed_value([u_share], context=f"{context}_open_u_{attempt}")) % p
            if u_open == 0:
                continue

            inv_u = pow(u_open, p - 2, p)

            # inv_den = r * inv(u)  (scalar multiply by public inv_u)
            inv_den_y = (int(r_share.y) * inv_u) % p
            return Share(x=x, y=inv_den_y, node_id=node_id)

        raise RuntimeError("secure_inverse_share failed: opened u was zero repeatedly (unexpected)")
    
    def secure_scalar_divide(self, share: Share, divisor: int, node_id: int, context: str = "scalar_div") -> Share:
        """
        Divide share by a public scalar
        Args:
            share: Share to divide
            divisor: Public divisor
            node_id: Node ID
        Returns:
            Share of quotient
        """
        if int(divisor) == 0:
            raise ValueError("divisor must be non-zero")

        # In privacy_mode we want integer fixed-point semantics (no field fractions),
        # so we use Enclave Oracle-based truncation instead of multiplying by modular inverse.
        if self._opened_enabled():
            # IMPORTANT: `context` must be deterministic and identical across nodes.
            # Callers should pass a unique context per division to avoid collisions.
            ctx = f"{context}_div{int(divisor)}"
            p = int(self.field_size)
            out_vec = self.multiplier._opened_divide_and_reshare_vector(  # type: ignore[attr-defined]
                values_local_u64=np.asarray([int(share.y) % p], dtype=np.uint64),
                divisor=int(divisor),
                node_id=int(node_id),
                x=int(share.x),
                context_prefix=ctx,
                timeout=120.0,
            )
            return Share(x=share.x, y=int(out_vec[0]) % p, node_id=node_id)

        # Legacy (field division) path
        inv = self._mod_inverse(divisor)
        return Share(x=share.x, y=(int(share.y) * int(inv)) % int(self.field_size), node_id=node_id)
        
    def secure_scalar_divide_batch(
        self,
        shares: List[Share],
        divisor: int,
        node_id: int,
        context: str = "scalar_div_batch",
        multiplier_factor: int = 1,
        clip_min: Optional[int] = None,
        clip_max: Optional[int] = None,
    ) -> List[Share]:
        """
        Divide a list of shares by a public scalar.
        Optionally pre-multiply the scalar securely within the enclave bounds.
        """
        if not shares:
            return []
        if int(divisor) == 0:
            raise ValueError("divisor must be non-zero")
            
        if self._opened_enabled():
            ctx = f"{context}_div{int(divisor)}"
            p = int(self.field_size)
            import numpy as np
            y_vals = np.array([s.y for s in shares], dtype=np.uint64)
            out_vec = self.multiplier._opened_divide_and_reshare_vector(  # type: ignore[attr-defined]
                values_local_u64=y_vals % np.uint64(p),
                divisor=int(divisor),
                node_id=int(node_id),
                x=int(shares[0].x),
                context_prefix=ctx,
                timeout=120.0,
                multiplier_factor=multiplier_factor,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            return [Share(x=shares[0].x, y=int(v) % p, node_id=node_id) for v in out_vec]

        # Legacy (field division) path
        inv = self._mod_inverse(divisor)
        return [Share(x=shares[0].x, y=(int(s.y) * int(inv)) % int(self.field_size), node_id=node_id) for s in shares]
    
    def secure_average(self, shares: List[Share], node_id: int) -> Share:
        """
        Compute average of shares
        Args:
            shares: List of shares
            node_id: Node ID
        Returns:
            Share of average
        """
        if not shares:
            return Share(x=0, y=0, node_id=node_id)
        
        # Sum all shares
        sum_share = Share(x=shares[0].x, y=0, node_id=node_id)
        for share in shares:
            sum_share = Share(
                x=sum_share.x,
                y=(sum_share.y + share.y) % self.field_size,
                node_id=node_id
            )
        
        # Divide by count
        n = len(shares)
        return self.secure_scalar_divide(sum_share, n, node_id)
    
    def secure_divide_shares(self, numerator: Share, denominator: Share, 
                            node_id: int, context: str = "div") -> Share:
        """
        Securely divide numerator by denominator (both are shares)
        
        For MPC, this requires computing the inverse of denominator securely.
        Currently uses approximation - in production, use full secure division protocol.
        
        Args:
            numerator: Share of numerator
            denominator: Share of denominator
            node_id: Node ID
            context: Context for secure operations
        Returns:
            Share of quotient
        """
        # Single-node fixed-point fallback:
        # numerator and denominator are SCALE-scaled integers (mod p).
        # We want output = round((numerator/denominator) * SCALE).
        if (
            getattr(self.multiplier, "reconstruction_manager", None) is None
            and int(getattr(self.multiplier, "n_nodes", 1)) == 1
            and int(getattr(self.multiplier, "t", 0)) == 0
        ):
            p = int(self.field_size)
            num_open = int(numerator.y) % p
            den_open = int(denominator.y) % p
            if num_open > (p // 2):
                num_open -= p
            if den_open > (p // 2):
                den_open -= p
            if den_open == 0:
                raise RuntimeError("secure_divide_shares failed: denominator was zero")
            S = int(self.scale_factor)
            numS = int(num_open) * S
            adj = (abs(den_open) // 2) * (1 if numS >= 0 else -1)
            q = (numS + adj) // int(den_open)
            return Share(x=numerator.x, y=int(q % p), node_id=node_id)

        # Enclave/opened fixed-point division (Option A):
        # numerator and denominator are SCALE-scaled integers (mod p).
        # We want output = round((numerator/denominator) * SCALE) as an integer, then re-share.
        if self._opened_enabled():
            recon = getattr(self.multiplier, "reconstruction_manager", None)
            if recon is None:
                raise RuntimeError("opened division requires reconstruction_manager")

            p = int(self.field_size)
            n = int(getattr(self.multiplier, "n_nodes", 1))
            opener = 1
            if getattr(self.multiplier, "triple_dealer_id", None) is not None:
                opener = int(getattr(self.multiplier, "triple_dealer_id"))

            net = recon.network
            timeout = float(getattr(self.multiplier, "_effective_timeout", lambda t: t)(120.0))

            num_ctx = f"{context}_num"
            den_ctx = f"{context}_den"
            # Everyone broadcasts local shares for numerator/denominator
            net.broadcast_vector(num_ctx, x=int(numerator.x), values=np.asarray([int(numerator.y) % p], dtype=np.uint64))
            net.broadcast_vector(den_ctx, x=int(denominator.x), values=np.asarray([int(denominator.y) % p], dtype=np.uint64))

            if int(node_id) == int(opener):
                _t0 = time.time()
                num_open_u64 = recon.reconstruct_opened_vector_values(
                    context=num_ctx, values_local=np.asarray([int(numerator.y) % p], dtype=np.uint64), x=int(numerator.x), timeout=timeout
                )
                den_open_u64 = recon.reconstruct_opened_vector_values(
                    context=den_ctx, values_local=np.asarray([int(denominator.y) % p], dtype=np.uint64), x=int(denominator.x), timeout=timeout
                )
                num_open = int(num_open_u64[0]) % p
                den_open = int(den_open_u64[0]) % p

                # Interpret as signed integers for fixed-point
                if num_open > (p // 2):
                    num_open -= p
                if den_open > (p // 2):
                    den_open -= p
                if den_open == 0:
                    # Avoid crash; treat as zero output (shouldn't happen in valid softmax/log inputs)
                    q = 0
                else:
                    # q = round(num * SCALE / den)
                    S = int(self.scale_factor)
                    numS = int(num_open) * S
                    # Round-to-nearest (sign-aware)
                    adj = (abs(den_open) // 2) * (1 if numS >= 0 else -1)
                    q = (numS + adj) // int(den_open)

                q_mod = int(q % p)
                out_prefix = f"{context}_out"
                opener_vec = self.multiplier._reshare_vector_from_opener(  # type: ignore[attr-defined]
                    secrets_mod_p_u64=np.asarray([q_mod], dtype=np.uint64),
                    node_id=int(node_id),
                    context_prefix=out_prefix,
                    x_points=[i for i in range(1, n + 1)],
                    timeout=timeout,
                )
                # Cleanup input buffers
                try:
                    net.channel.clear_vector(num_ctx)
                    net.channel.clear_vector(den_ctx)
                except Exception:
                    pass
                if hasattr(self.multiplier, "add_prover_time"):
                    self.multiplier.add_prover_time(time.time() - _t0, "secure_divide_shares_opened")
                return Share(x=numerator.x, y=int(opener_vec[0]) % p, node_id=node_id)

            # Non-opener: wait for reshared output
            out_ctx = f"{context}_out_to_{node_id}"
            import time as _time
            start = _time.time()
            while _time.time() - start < timeout:
                recv = net.channel.get_received_vector(out_ctx)
                if opener in recv:
                    values = recv[opener].get("values")
                    arr = np.asarray(values, dtype=np.uint64)
                    try:
                        net.channel.clear_vector(out_ctx)
                    except Exception:
                        pass
                    try:
                        net.channel.clear_vector(num_ctx)
                        net.channel.clear_vector(den_ctx)
                    except Exception:
                        pass
                    return Share(x=numerator.x, y=int(arr[0]) % p, node_id=node_id)
                _time.sleep(0.01)

            raise RuntimeError(f"Timed out waiting for opened division output (ctx={out_ctx})")

        # Fixed-point convention used across the codebase:
        # - numerator and denominator are SCALE_FACTOR-scaled field elements
        # - we want (numerator/denominator) scaled by SCALE_FACTOR
        #
        # In the field:
        #   inv_den = 1/denominator
        #   ratio_unscaled = numerator * inv_den
        #   ratio_scaled = ratio_unscaled * SCALE_FACTOR
        inv_den = self.secure_inverse_share(denominator, node_id=node_id, context=f"{context}_inv")
        ratio_unscaled = self.multiplier.multiply(numerator, inv_den, node_id=node_id, context=f"{context}_mul")
        ratio_scaled_y = (int(ratio_unscaled.y) * int(self.scale_factor)) % int(self.field_size)
        return Share(x=numerator.x, y=ratio_scaled_y, node_id=node_id)
        
    def secure_divide_shares_batch(
        self,
        numerators: List[Share],
        denominators: List[Share],
        node_id: int,
        context: str = "div_batch",
        *,
        enforce_probability_range: bool = False,
        probability_group_size: Optional[int] = None,
    ) -> List[Share]:
        if not numerators or not denominators:
            return []
        if len(numerators) != len(denominators):
            raise ValueError("Mismatched lengths")
            
        p = int(self.field_size)
        x0 = numerators[0].x

        # Single-node fixed-point fallback (batch).
        if (
            getattr(self.multiplier, "reconstruction_manager", None) is None
            and int(getattr(self.multiplier, "n_nodes", 1)) == 1
            and int(getattr(self.multiplier, "t", 0)) == 0
        ):
            out = []
            S = int(self.scale_factor)
            for n_sh, d_sh in zip(numerators, denominators):
                num_open = int(n_sh.y) % p
                den_open = int(d_sh.y) % p
                if num_open > (p // 2):
                    num_open -= p
                if den_open > (p // 2):
                    den_open -= p
                if den_open == 0:
                    raise RuntimeError("secure_divide_shares_batch failed: denominator was zero")
                numS = int(num_open) * S
                adj = (abs(den_open) // 2) * (1 if numS >= 0 else -1)
                q = (numS + adj) // int(den_open)
                out.append(int(q))

            if enforce_probability_range:
                q_arr = np.asarray(out, dtype=np.int64)
                q_arr = np.clip(q_arr, 0, S)
                g = int(probability_group_size) if probability_group_size else 0
                if g > 0 and (len(q_arr) % g == 0):
                    q2 = q_arr.reshape(-1, g)
                    sums = np.sum(q2, axis=1, keepdims=True)
                    sums[sums <= 0] = 1
                    q2 = np.round((q2.astype(np.float64) * float(S)) / sums.astype(np.float64)).astype(np.int64)
                    q2 = np.clip(q2, 0, S)
                    row_sum = np.sum(q2, axis=1)
                    residual = (S - row_sum).astype(np.int64)
                    argm = np.argmax(q2, axis=1)
                    for ridx in range(q2.shape[0]):
                        q2[ridx, argm[ridx]] = np.clip(q2[ridx, argm[ridx]] + residual[ridx], 0, S)
                    q_arr = q2.reshape(-1)
                out = q_arr.tolist()

            return [Share(x=x0, y=int(v) % p, node_id=node_id) for v in out]
        
        if self._opened_enabled():
            recon = getattr(self.multiplier, "reconstruction_manager", None)
            n = int(getattr(self.multiplier, "n_nodes", 1))
            opener = int(getattr(self.multiplier, "triple_dealer_id", 1)) if getattr(self.multiplier, "triple_dealer_id", None) is not None else 1
            net = recon.network
            timeout = float(getattr(self.multiplier, "_effective_timeout", lambda t: t)(120.0))
            
            num_ctx = f"{context}_num"
            den_ctx = f"{context}_den"
            num_vals = np.array([s.y for s in numerators], dtype=np.uint64) % p
            den_vals = np.array([s.y for s in denominators], dtype=np.uint64) % p
            
            net.broadcast_vector(num_ctx, x=int(x0), values=num_vals)
            net.broadcast_vector(den_ctx, x=int(x0), values=den_vals)
            
            if int(node_id) == int(opener):
                _t0 = time.time()
                num_open_u64 = recon.reconstruct_opened_vector_values(context=num_ctx, values_local=num_vals, x=int(x0), timeout=timeout)
                den_open_u64 = recon.reconstruct_opened_vector_values(context=den_ctx, values_local=den_vals, x=int(x0), timeout=timeout)
                
                num_open = num_open_u64 % p
                den_open = den_open_u64 % p

                # Signed fixed-point division.
                # For large fields, avoid int64 overflow by using Python-int object arrays.
                use_object_math = p > 0xFFFFFFFF
                if use_object_math:
                    num_signed = np.asarray(num_open, dtype=object)
                    den_signed = np.asarray(den_open, dtype=object)
                    half = p // 2
                    num_signed = np.where(num_open > half, num_signed - p, num_signed)
                    den_signed = np.where(den_open > half, den_signed - p, den_signed)
                else:
                    mask_n = num_open > (p // 2)
                    mask_d = den_open > (p // 2)
                    num_signed = num_open.astype(np.int64)
                    num_signed[mask_n] -= p
                    den_signed = den_open.astype(np.int64)
                    den_signed[mask_d] -= p
                
                # To avoid division by zero
                zero_den = int(np.sum(den_signed == 0))
                den_signed = np.where(den_signed == 0, 1, den_signed)

                S = int(self.scale_factor)
                if use_object_math:
                    numS = np.asarray(num_signed, dtype=object) * int(S)
                    signs = np.where(np.asarray(numS, dtype=object) >= 0, 1, -1)
                    adj = (np.abs(np.asarray(den_signed, dtype=object)) // 2) * signs
                    q = (np.asarray(numS, dtype=object) + np.asarray(adj, dtype=object)) // np.asarray(den_signed, dtype=object)
                else:
                    numS = num_signed * S
                    adj = (np.abs(den_signed) // 2) * np.where(numS >= 0, 1, -1)
                    q = (numS + adj) // den_signed

                if self.debug and self._debug_count < self._debug_limit and ("e0_b0" in str(context) or "eval_pre" in str(context)):
                    self._debug_count += 1
                    try:
                        print(
                            f"[DEBUG DIV] ctx={context} "
                            f"num_abs_max={float(np.max(np.abs(np.asarray(num_signed, dtype=np.float64)))):.4f} "
                            f"den_abs_min={float(np.min(np.abs(np.asarray(den_signed, dtype=np.float64)))):.4f} "
                            f"den_abs_max={float(np.max(np.abs(np.asarray(den_signed, dtype=np.float64)))):.4f} "
                            f"q_abs_max={float(np.max(np.abs(np.asarray(q, dtype=np.float64)))):.4f} "
                            f"zero_den={zero_den}",
                            flush=True,
                        )
                    except Exception:
                        pass
                
                # Optional projection for softmax probability divisions.
                # In fixed-point, valid probs are in [0, SCALE] and should sum to SCALE per sample.
                if enforce_probability_range:
                    S = int(self.scale_factor)
                    q = np.clip(q, 0, S)
                    g = int(probability_group_size) if probability_group_size else 0
                    if g > 0 and (len(q) % g == 0):
                        q2 = q.reshape(-1, g)
                        sums = np.sum(q2, axis=1, keepdims=True)
                        sums[sums <= 0] = 1
                        q2 = np.round((q2.astype(np.float64) * float(S)) / sums.astype(np.float64)).astype(np.int64)
                        q2 = np.clip(q2, 0, S)
                        # Correct residual so each row sums to exactly S
                        row_sum = np.sum(q2, axis=1)
                        residual = (S - row_sum).astype(np.int64)
                        # Add/subtract residual on argmax entry for each row
                        argm = np.argmax(q2, axis=1)
                        for ridx in range(q2.shape[0]):
                            q2[ridx, argm[ridx]] = np.clip(q2[ridx, argm[ridx]] + residual[ridx], 0, S)
                        q = q2.reshape(-1)

                q_mod = q % p
                out_prefix = f"{context}_out"
                opener_vec = self.multiplier._reshare_vector_from_opener(
                    secrets_mod_p_u64=np.asarray(q_mod, dtype=np.uint64),
                    node_id=int(node_id),
                    context_prefix=out_prefix,
                    x_points=[i for i in range(1, n+1)],
                    timeout=timeout
                )
                try:
                    net.channel.clear_vector(num_ctx)
                    net.channel.clear_vector(den_ctx)
                except:
                    pass
                if hasattr(self.multiplier, "add_prover_time"):
                    self.multiplier.add_prover_time(time.time() - _t0, "secure_divide_shares_batch_opened")
                return [Share(x=x0, y=int(v) % p, node_id=node_id) for v in opener_vec]
                
            out_ctx = f"{context}_out_to_{node_id}"
            import time as _time
            start = _time.time()
            while _time.time() - start < timeout:
                recv = net.channel.get_received_vector(out_ctx)
                if opener in recv:
                    arr = np.asarray(recv[opener].get("values"), dtype=np.uint64)
                    try:
                        net.channel.clear_vector(out_ctx)
                        net.channel.clear_vector(num_ctx)
                        net.channel.clear_vector(den_ctx)
                    except:
                        pass
                    return [Share(x=x0, y=int(v) % p, node_id=node_id) for v in arr]
                _time.sleep(0.01)
            raise RuntimeError(f"Timed out waiting for batched opened division output (ctx={out_ctx})")

        # Fallback to slow loop
        return [self.secure_divide_shares(n_sh, d_sh, node_id, f"{context}_{i}") for i, (n_sh, d_sh) in enumerate(zip(numerators, denominators))]
    
    def _mod_inverse(self, a: int) -> int:
        """Compute modular inverse using extended Euclidean algorithm"""
        if a < 0:
            a = a % self.field_size
        
        old_r, r = a, self.field_size
        old_s, s = 1, 0
        
        while r != 0:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
        
        if old_r != 1:
            raise ValueError(f"{a} has no modular inverse modulo {self.field_size}")
        
        return old_s % self.field_size


class SecureAverager:
    """
    Performs secure averaging operations for gradient aggregation
    """
    
    def __init__(self, divider: SecureDivider):
        """
        Initialize secure averager
        Args:
            divider: SecureDivider instance
        """
        self.divider = divider
    
    def average_gradient_batch(self, gradient_batch: List[List[Share]], 
                               batch_size: int, node_id: int) -> List[Share]:
        """
        Average gradients across a batch
        Args:
            gradient_batch: List of gradient lists (one per sample)
            batch_size: Batch size
            node_id: Node ID
        Returns:
            Averaged gradient shares
        """
        if not gradient_batch:
            return []
        
        # Sum gradients across batch
        num_grads = len(gradient_batch[0])
        summed_grads = []
        
        for i in range(num_grads):
            sum_share = Share(x=gradient_batch[0][i].x, y=0, node_id=node_id)
            for grad_list in gradient_batch:
                sum_share = Share(
                    x=sum_share.x,
                    y=(sum_share.y + grad_list[i].y) % self.divider.field_size,
                    node_id=node_id
                )
            summed_grads.append(sum_share)
        
        # Average by dividing by batch size
        averaged_grads = []
        for sum_share in summed_grads:
            avg_share = self.divider.secure_scalar_divide(sum_share, batch_size, node_id)
            averaged_grads.append(avg_share)
        
        return averaged_grads

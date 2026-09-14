"""
Secure Softmax and Cross-Entropy Loss Operations
Implements secure softmax and cross-entropy loss for classification
Uses numerically stable approximations suitable for MPC
"""

from typing import List, Tuple
from ml_training.secret_sharing import Share
from ml_training.beaver_triples import SecureMultiplier
from ml_training.secure_division import SecureDivider
import numpy as np
import time


class SecureSoftmax:
    """
    Performs secure softmax operations on secret-shared logits
    Uses polynomial approximation for exp(x) in secure computation
    """
    
    def __init__(self, multiplier: SecureMultiplier, divider: SecureDivider, 
                 field_size: int = 2**32 - 5, scale_factor: int = 1000,
                 temperature: float = 2.0,
                 logit_clip: float = 8.0,
                 exp_approx: str = "pade22",
                 softmax_grad_mode: str = "secure_approx"):
        """
        Initialize secure softmax
        Args:
            multiplier: SecureMultiplier instance
            divider: SecureDivider instance
            field_size: Prime field size
            scale_factor: Scaling factor for integer representation
            temperature: Temperature scaling factor (default 2.0 for softer probabilities)
                         Higher temperature = softer probabilities (less confident)
                         Lower temperature = sharper probabilities (more confident)
        """
        self.multiplier = multiplier
        self.divider = divider
        self.field_size = field_size
        self.SCALE_FACTOR = scale_factor
        self.temperature = temperature
        self.logit_clip = float(logit_clip)
        self.exp_approx = str(exp_approx).strip().lower()
        if self.exp_approx not in {"taylor5", "pade22"}:
            raise ValueError("exp_approx must be one of: taylor5, pade22")
        self.softmax_grad_mode = str(softmax_grad_mode).strip().lower()
        if self.softmax_grad_mode not in {"secure_approx", "opened_exact"}:
            raise ValueError("softmax_grad_mode must be one of: secure_approx, opened_exact")

        # Precompute inverses for fixed-point polynomial coefficients (public scalars)
        p = int(self.field_size)
        self._inv_scale = pow(int(self.SCALE_FACTOR) % p, p - 2, p)
        self._inv2 = pow(2, p - 2, p)
        self._inv3 = pow(3, p - 2, p)
        self._inv5 = pow(5, p - 2, p)
        self._inv7 = pow(7, p - 2, p)

    def _opened_enabled(self) -> bool:
        return bool(
            getattr(self.multiplier, "privacy_mode", False)
            and getattr(self.multiplier, "reconstruction_manager", None) is not None
            and int(getattr(self.multiplier, "n_nodes", 1)) > 1
        )

    def _opened_exact_softmax_grad_batch(
        self,
        logits_cols: List[List[Share]],
        target_cols: List[List[Share]],
        *,
        node_id: int,
        context: str,
    ) -> List[List[Share]]:
        """
        Optional debug/benchmark path:
        - Open logits/targets at the opener
        - Compute exact float softmax gradient dz = (softmax - target) / T
        - Re-share dz back to all nodes
        """
        if not logits_cols:
            return []
        if len(logits_cols) != len(target_cols):
            raise ValueError("Mismatched batch sizes for logits and targets")

        if not self._opened_enabled():
            raise RuntimeError("opened_exact softmax gradient requires opened/privacy mode with reconstruction manager")

        recon = getattr(self.multiplier, "reconstruction_manager", None)
        if recon is None:
            raise RuntimeError("opened_exact softmax gradient requires reconstruction_manager")

        p = int(self.field_size)
        n = int(getattr(self.multiplier, "n_nodes", 1))
        opener = int(self.multiplier._opened_fp_opener())  # type: ignore[attr-defined]
        timeout = float(getattr(self.multiplier, "_effective_timeout", lambda t: t)(120.0))
        net = recon.network

        batch_size = len(logits_cols)
        num_classes = len(logits_cols[0])
        for c in logits_cols:
            if len(c) != num_classes:
                raise ValueError("Inconsistent class dimension in logits")
        for c in target_cols:
            if len(c) != num_classes:
                raise ValueError("Inconsistent class dimension in targets")

        x0 = int(logits_cols[0][0].x)

        flat_logits = np.asarray([int(s.y) % p for col in logits_cols for s in col], dtype=np.uint64)
        flat_targets = np.asarray([int(s.y) % p for col in target_cols for s in col], dtype=np.uint64)

        ctx_l = f"{context}_opened_exact_logits"
        ctx_t = f"{context}_opened_exact_targets"
        net.broadcast_vector(ctx_l, x=x0, values=flat_logits)
        net.broadcast_vector(ctx_t, x=x0, values=flat_targets)

        if int(node_id) == opener:
            _t0 = time.time()
            logits_open_u64 = recon.reconstruct_opened_vector_values(
                context=ctx_l,
                values_local=flat_logits,
                x=x0,
                timeout=timeout,
            )
            targets_open_u64 = recon.reconstruct_opened_vector_values(
                context=ctx_t,
                values_local=flat_targets,
                x=x0,
                timeout=timeout,
            )

            logits_open = logits_open_u64.astype(np.int64, copy=False)
            logits_open = np.where(logits_open > (p // 2), logits_open - p, logits_open)
            targets_open = targets_open_u64.astype(np.int64, copy=False)
            targets_open = np.where(targets_open > (p // 2), targets_open - p, targets_open)

            scale = float(self.SCALE_FACTOR)
            logits_f = logits_open.astype(np.float64) / scale
            targets_f = targets_open.astype(np.float64) / scale
            logits_m = logits_f.reshape(batch_size, num_classes)
            targets_m = targets_f.reshape(batch_size, num_classes)

            shifted = logits_m - np.max(logits_m, axis=1, keepdims=True)
            expv = np.exp(shifted)
            probs = expv / np.maximum(np.sum(expv, axis=1, keepdims=True), 1e-12)
            dz = probs - targets_m
            if abs(float(self.temperature) - 1.0) > 1e-9:
                dz = dz / float(self.temperature)

            dz_i = np.rint(dz * scale).astype(np.int64, copy=False).reshape(-1)
            dz_mod = np.mod(dz_i, p).astype(np.uint64, copy=False)

            out_prefix = f"{context}_opened_exact_out"
            opener_vec = self.multiplier._reshare_vector_from_opener(  # type: ignore[attr-defined]
                secrets_mod_p_u64=dz_mod,
                node_id=int(node_id),
                context_prefix=out_prefix,
                x_points=[i for i in range(1, n + 1)],
                timeout=timeout,
            )
            try:
                net.channel.clear_vector(ctx_l)
                net.channel.clear_vector(ctx_t)
            except Exception:
                pass

            flat_out = [Share(x=x0, y=int(v) % p, node_id=node_id) for v in opener_vec]
            out_cols: List[List[Share]] = []
            for i in range(batch_size):
                out_cols.append(flat_out[i * num_classes : (i + 1) * num_classes])
            if hasattr(self.multiplier, "add_prover_time"):
                self.multiplier.add_prover_time(time.time() - _t0, "softmax_opened_exact_grad")
            return out_cols

        out_ctx = f"{context}_opened_exact_out_to_{node_id}"
        import time as _time
        start = _time.time()
        while _time.time() - start < timeout:
            recv = net.channel.get_received_vector(out_ctx)
            if opener in recv:
                arr = np.asarray(recv[opener].get("values"), dtype=np.uint64)
                try:
                    net.channel.clear_vector(out_ctx)
                    net.channel.clear_vector(ctx_l)
                    net.channel.clear_vector(ctx_t)
                except Exception:
                    pass
                flat_out = [Share(x=x0, y=int(v) % p, node_id=node_id) for v in arr]
                out_cols: List[List[Share]] = []
                for i in range(batch_size):
                    out_cols.append(flat_out[i * num_classes : (i + 1) * num_classes])
                return out_cols
            _time.sleep(0.01)
        raise RuntimeError(f"Timed out waiting for opened exact softmax gradient output (ctx={out_ctx})")

    def _opened_clip_batch(
        self,
        shares: List[Share],
        *,
        node_id: int,
        context: str,
        clip_min: int,
        clip_max: int,
    ) -> List[Share]:
        if not shares:
            return []

        # Fallback path for non-opened mode (legacy behavior).
        if not self._opened_enabled():
            p = int(self.field_size)
            out: List[Share] = []
            for s in shares:
                v = int(s.y) % p
                if v > p // 2:
                    v -= p
                if v < int(clip_min):
                    v = int(clip_min)
                elif v > int(clip_max):
                    v = int(clip_max)
                out.append(Share(x=s.x, y=v % p, node_id=node_id))
            return out

        recon = getattr(self.multiplier, "reconstruction_manager", None)
        if recon is None:
            raise RuntimeError("opened clipping requires reconstruction_manager")

        p = int(self.field_size)
        n = int(getattr(self.multiplier, "n_nodes", 1))
        opener = int(self.multiplier._opened_fp_opener())  # type: ignore[attr-defined]
        timeout = float(getattr(self.multiplier, "_effective_timeout", lambda t: t)(120.0))
        net = recon.network

        x0 = int(shares[0].x)
        ctx_in = f"{context}_clip_in"
        vals_local = np.asarray([int(s.y) % p for s in shares], dtype=np.uint64)
        net.broadcast_vector(ctx_in, x=x0, values=vals_local)

        if int(node_id) == opener:
            _t0 = time.time()
            opened_u64 = recon.reconstruct_opened_vector_values(
                context=ctx_in,
                values_local=vals_local,
                x=x0,
                timeout=timeout,
            )
            opened = opened_u64.astype(np.int64, copy=False)
            opened = np.where(opened > (p // 2), opened - p, opened)
            clipped = np.clip(opened, int(clip_min), int(clip_max)).astype(np.int64, copy=False)
            clipped_mod = np.mod(clipped, p).astype(np.uint64, copy=False)

            out_prefix = f"{context}_clip_out"
            opener_vec = self.multiplier._reshare_vector_from_opener(  # type: ignore[attr-defined]
                secrets_mod_p_u64=clipped_mod,
                node_id=int(node_id),
                context_prefix=out_prefix,
                x_points=[i for i in range(1, n + 1)],
                timeout=timeout,
            )
            try:
                net.channel.clear_vector(ctx_in)
            except Exception:
                pass
            if hasattr(self.multiplier, "add_prover_time"):
                self.multiplier.add_prover_time(time.time() - _t0, "softmax_opened_clip")
            return [Share(x=x0, y=int(v) % p, node_id=node_id) for v in opener_vec]

        # Non-opener waits for reshare output.
        out_ctx = f"{context}_clip_out_to_{node_id}"
        import time as _time
        start = _time.time()
        while _time.time() - start < timeout:
            recv = net.channel.get_received_vector(out_ctx)
            if opener in recv:
                arr = np.asarray(recv[opener].get("values"), dtype=np.uint64)
                try:
                    net.channel.clear_vector(out_ctx)
                    net.channel.clear_vector(ctx_in)
                except Exception:
                    pass
                return [Share(x=x0, y=int(v) % p, node_id=node_id) for v in arr]
            _time.sleep(0.01)
        raise RuntimeError(f"Timed out waiting for opened clipping output (ctx={out_ctx})")

    def _opened_groupwise_max_as_shares(
        self,
        shares: List[Share],
        *,
        node_id: int,
        context: str,
        group_size: int,
    ) -> List[Share]:
        """
        Compute group-wise maxima in opened mode and return them as secret shares
        aligned with the flattened input layout (same length as shares).
        """
        if not shares:
            return []
        g = int(group_size)
        if g <= 0 or (len(shares) % g) != 0:
            raise ValueError("group_size must divide number of shares")

        if not self._opened_enabled():
            p = int(self.field_size)
            out: List[Share] = []
            for i in range(0, len(shares), g):
                vals = []
                for s in shares[i:i+g]:
                    v = int(s.y) % p
                    if v > p // 2:
                        v -= p
                    vals.append(v)
                m = max(vals)
                out.extend([Share(x=shares[0].x, y=m % p, node_id=node_id) for _ in range(g)])
            return out

        recon = getattr(self.multiplier, "reconstruction_manager", None)
        if recon is None:
            raise RuntimeError("opened max requires reconstruction_manager")

        p = int(self.field_size)
        n = int(getattr(self.multiplier, "n_nodes", 1))
        opener = int(self.multiplier._opened_fp_opener())  # type: ignore[attr-defined]
        timeout = float(getattr(self.multiplier, "_effective_timeout", lambda t: t)(120.0))
        net = recon.network

        x0 = int(shares[0].x)
        ctx_in = f"{context}_max_in"
        vals_local = np.asarray([int(s.y) % p for s in shares], dtype=np.uint64)
        net.broadcast_vector(ctx_in, x=x0, values=vals_local)

        if int(node_id) == opener:
            _t0 = time.time()
            opened_u64 = recon.reconstruct_opened_vector_values(
                context=ctx_in,
                values_local=vals_local,
                x=x0,
                timeout=timeout,
            )
            opened = opened_u64.astype(np.int64, copy=False)
            opened = np.where(opened > (p // 2), opened - p, opened)
            rows = opened.reshape(-1, g)
            max_per_row = np.max(rows, axis=1)
            max_flat = np.repeat(max_per_row, g)
            max_mod = np.mod(max_flat, p).astype(np.uint64, copy=False)

            out_prefix = f"{context}_max_out"
            opener_vec = self.multiplier._reshare_vector_from_opener(  # type: ignore[attr-defined]
                secrets_mod_p_u64=max_mod,
                node_id=int(node_id),
                context_prefix=out_prefix,
                x_points=[i for i in range(1, n + 1)],
                timeout=timeout,
            )
            try:
                net.channel.clear_vector(ctx_in)
            except Exception:
                pass
            if hasattr(self.multiplier, "add_prover_time"):
                self.multiplier.add_prover_time(time.time() - _t0, "softmax_opened_group_max")
            return [Share(x=x0, y=int(v) % p, node_id=node_id) for v in opener_vec]

        out_ctx = f"{context}_max_out_to_{node_id}"
        import time as _time
        start = _time.time()
        while _time.time() - start < timeout:
            recv = net.channel.get_received_vector(out_ctx)
            if opener in recv:
                arr = np.asarray(recv[opener].get("values"), dtype=np.uint64)
                try:
                    net.channel.clear_vector(out_ctx)
                    net.channel.clear_vector(ctx_in)
                except Exception:
                    pass
                return [Share(x=x0, y=int(v) % p, node_id=node_id) for v in arr]
            _time.sleep(0.01)
        raise RuntimeError(f"Timed out waiting for opened max output (ctx={out_ctx})")

    def _clip_share(self, s: Share) -> Share:
        p = int(self.field_size)
        v = int(s.y) % p
        if v > p // 2:
            v -= p
        clip_abs = int(round(self.logit_clip * float(self.SCALE_FACTOR)))
        if v > clip_abs:
            v = clip_abs
        elif v < -clip_abs:
            v = -clip_abs
        return Share(x=s.x, y=v % p, node_id=s.node_id)

    def _fp_mul(self, a: Share, b: Share, node_id: int, context: str) -> Share:
        """
        Fixed-point multiply of two SCALE_FACTOR-scaled shares:
            (a*b)/SCALE_FACTOR  (result is SCALE_FACTOR-scaled)
        """
        # Use multiplier's fixed-point multiply so "A" mode truncation applies consistently.
        return self.multiplier.multiply_fixed_point(
            a,
            b,
            node_id=node_id,
            scale_factor=int(self.SCALE_FACTOR),
            context=context,
        )

    def _fp_add(self, a: Share, b: Share, node_id: int) -> Share:
        return Share(x=a.x, y=(int(a.y) + int(b.y)) % int(self.field_size), node_id=node_id)

    def _fp_sub(self, a: Share, b: Share, node_id: int) -> Share:
        return Share(x=a.x, y=(int(a.y) - int(b.y)) % int(self.field_size), node_id=node_id)
        
    def _fp_mul_batch(self, A: List[Share], B: List[Share], node_id: int, context: str) -> List[Share]:
        import numpy as np
        y1 = np.array([a.y for a in A], dtype=np.uint64)
        y2 = np.array([b.y for b in B], dtype=np.uint64)
        prod = self.multiplier.multiply_batch_values_fixed_point(
            y1, y2, x=A[0].x, node_id=node_id, context_prefix=context, scale_factor=int(self.SCALE_FACTOR)
        )
        return [Share(x=A[0].x, y=int(v), node_id=node_id) for v in prod]
        
    def _fp_add_batch(self, A: List[Share], B: List[Share], node_id: int) -> List[Share]:
        p = int(self.field_size)
        return [Share(x=a.x, y=(int(a.y) + int(b.y)) % p, node_id=node_id) for a, b in zip(A, B)]
        
    def _fp_sub_batch(self, A: List[Share], B: List[Share], node_id: int) -> List[Share]:
        p = int(self.field_size)
        return [Share(x=a.x, y=(int(a.y) - int(b.y)) % p, node_id=node_id) for a, b in zip(A, B)]

    def _exp_poly(self, x: Share, node_id: int, context: str) -> Share:
        """
        MPC-safe exp approximation (fixed-point) using a 5th order Taylor polynomial:
          exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120

        This avoids reading Share values as plaintext. Accuracy depends on logits being
        reasonably small in magnitude (which must be enforced by the numeric pipeline).
        """
        p = int(self.field_size)
        one = Share(x=x.x, y=int(self.SCALE_FACTOR) % p, node_id=node_id)

        x2 = self._fp_mul(x, x, node_id, context=f"{context}_x2")
        x3 = self._fp_mul(x2, x, node_id, context=f"{context}_x3")
        x4 = self._fp_mul(x3, x, node_id, context=f"{context}_x4")
        x5 = self._fp_mul(x4, x, node_id, context=f"{context}_x5")

        # IMPORTANT: do NOT use modular inverses for division (creates field fractions).
        # Use truncating scalar division to preserve integer fixed-point semantics.
        t2 = self.divider.secure_scalar_divide(x2, 2, node_id, context=f"{context}_div2")
        t3 = self.divider.secure_scalar_divide(x3, 6, node_id, context=f"{context}_div6")
        t4 = self.divider.secure_scalar_divide(x4, 24, node_id, context=f"{context}_div24")
        t5 = self.divider.secure_scalar_divide(x5, 120, node_id, context=f"{context}_div120")

        out = self._fp_add(one, x, node_id)
        out = self._fp_add(out, t2, node_id)
        out = self._fp_add(out, t3, node_id)
        out = self._fp_add(out, t4, node_id)
        out = self._fp_add(out, t5, node_id)
        return out
        
    def _exp_poly_batch(self, x_list: List[Share], node_id: int, context: str) -> List[Share]:
        """
        MPC-safe exp approximation for a batch of shares.
        """
        if not x_list:
            return []
            
        p = int(self.field_size)
        one_list = [Share(x=x_list[0].x, y=int(self.SCALE_FACTOR) % p, node_id=node_id) for _ in range(len(x_list))]
        
        x2 = self._fp_mul_batch(x_list, x_list, node_id, f"{context}_x2")
        x3 = self._fp_mul_batch(x2, x_list, node_id, f"{context}_x3")
        x4 = self._fp_mul_batch(x3, x_list, node_id, f"{context}_x4")
        x5 = self._fp_mul_batch(x4, x_list, node_id, f"{context}_x5")
        
        t2 = self.divider.secure_scalar_divide_batch(x2, 2, node_id, f"{context}_div2")
        t3 = self.divider.secure_scalar_divide_batch(x3, 6, node_id, f"{context}_div6")
        t4 = self.divider.secure_scalar_divide_batch(x4, 24, node_id, f"{context}_div24")
        t5 = self.divider.secure_scalar_divide_batch(x5, 120, node_id, f"{context}_div120")
        
        out = self._fp_add_batch(one_list, x_list, node_id)
        out = self._fp_add_batch(out, t2, node_id)
        out = self._fp_add_batch(out, t3, node_id)
        out = self._fp_add_batch(out, t4, node_id)
        out = self._fp_add_batch(out, t5, node_id)
        return out

    def _exp_pade22_batch(self, x_list: List[Share], node_id: int, context: str) -> List[Share]:
        """
        exp(x) approximation via [2/2] Padé:
          exp(x) ~= (1 + x/2 + x^2/12) / (1 - x/2 + x^2/12)
        Better behaved than low-order Taylor for wider negative ranges and
        remains positive when denominator stays positive.
        """
        if not x_list:
            return []
        p = int(self.field_size)
        one = [Share(x=x_list[0].x, y=int(self.SCALE_FACTOR) % p, node_id=node_id) for _ in range(len(x_list))]
        x2 = self._fp_mul_batch(x_list, x_list, node_id, f"{context}_x2")
        x_half = self.divider.secure_scalar_divide_batch(x_list, 2, node_id, f"{context}_xdiv2")
        x2_12 = self.divider.secure_scalar_divide_batch(x2, 12, node_id, f"{context}_x2div12")

        num = self._fp_add_batch(one, x_half, node_id)
        num = self._fp_add_batch(num, x2_12, node_id)

        den = self._fp_sub_batch(one, x_half, node_id)
        den = self._fp_add_batch(den, x2_12, node_id)

        return self.divider.secure_divide_shares_batch(num, den, node_id, f"{context}_pade_div")

    def _atanh_series(self, y: Share, node_id: int, context: str, terms: int = 5) -> Share:
        """
        Approximate atanh(y) via series:
          atanh(y) = y + y^3/3 + y^5/5 + ...  (|y|<1)
        returns SCALE_FACTOR-scaled share.
        """
        p = int(self.field_size)
        out = Share(x=y.x, y=int(y.y) % p, node_id=node_id)
        y2 = self._fp_mul(y, y, node_id, context=f"{context}_y2")
        ypow = Share(x=y.x, y=int(y.y) % p, node_id=node_id)  # y^(2k+1)
        denom = 1
        for k in range(1, terms):
            # y^(2k+1) = y^(2k-1) * y^2
            ypow = self._fp_mul(ypow, y2, node_id, context=f"{context}_ypow_{k}")
            denom = 2 * k + 1
            term = self.divider.secure_scalar_divide(ypow, denom, node_id, context=f"{context}_div_{denom}")
            out = self._fp_add(out, term, node_id)
        return out

    def _log_fp(self, x: Share, node_id: int, context: str) -> Share:
        """
        Approximate ln(x) for x in (0,1] (fixed-point scaled) using:
          ln(x) = 2 * atanh((x-1)/(x+1))

        Note: This is approximate and requires x not too close to 0 for good accuracy.
        """
        p = int(self.field_size)
        one = Share(x=x.x, y=int(self.SCALE_FACTOR) % p, node_id=node_id)
        num = self._fp_sub(x, one, node_id)
        den = self._fp_add(x, one, node_id)
        y = self.divider.secure_divide_shares(num, den, node_id=node_id, context=f"{context}_div")
        at = self._atanh_series(y, node_id, context=f"{context}_atanh", terms=6)
        two = 2 % p
        return Share(x=x.x, y=(int(at.y) * two) % p, node_id=node_id)
    
    def softmax(self, logits: List[Share], node_id: int, context: str = "softmax") -> List[Share]:
        """
        Single-sample softmax routed through the batched path to ensure
        consistent probability projection and normalization behavior.
        """
        if len(logits) == 0:
            return []
        cols = self.softmax_batch([logits], node_id=node_id, context=context)
        return cols[0] if cols else []
        
    def softmax_batch(self, logits_cols: List[List[Share]], node_id: int, context: str = "softmax_batch") -> List[List[Share]]:
        """
        Batched Softmax computation.
        Args:
            logits_cols: List of batch columns (each column is a list of logits for that sample)
        Returns:
            probs_cols: List of batch columns
        """
        if not logits_cols:
            return []
            
        p = int(self.field_size)
        batch_size = len(logits_cols)
        num_classes = len(logits_cols[0])
        
        # 1. Flatten into 1D arrays for batch execution
        flat_logits = []
        for col in logits_cols:
            flat_logits.extend(col)
            
        # 2. Temperature scaling (optional, default T=1)
        if self.temperature != 1.0:
            temp_int = int(round(float(self.temperature)))
            flat_logits = self.divider.secure_scalar_divide_batch(flat_logits, temp_int, node_id, f"{context}_temp")
            
        # 3. Stabilization shift.
        # In opened mode, use per-sample max subtraction (numerically strongest).
        # Fallback to mean subtraction in non-opened mode.
        normalized_logits = []
        if self._opened_enabled():
            max_flat = self._opened_groupwise_max_as_shares(
                flat_logits,
                node_id=node_id,
                context=f"{context}_maxshift",
                group_size=num_classes,
            )
            for l, m in zip(flat_logits, max_flat):
                normalized_logits.append(Share(x=l.x, y=(int(l.y) - int(m.y)) % p, node_id=node_id))
        else:
            col_means = []
            for i in range(batch_size):
                sum_val = Share(x=logits_cols[i][0].x, y=0, node_id=node_id)
                for j in range(num_classes):
                    y_val = flat_logits[i * num_classes + j].y
                    sum_val = Share(x=sum_val.x, y=(int(sum_val.y) + int(y_val)) % p, node_id=node_id)
                col_means.append(sum_val)
            col_means = self.divider.secure_scalar_divide_batch(col_means, num_classes, node_id, f"{context}_mean")
            for i in range(batch_size):
                m = col_means[i].y
                for j in range(num_classes):
                    l = flat_logits[i * num_classes + j]
                    normalized_logits.append(Share(x=l.x, y=(int(l.y) - int(m)) % p, node_id=node_id))
        clip_abs = int(round(self.logit_clip * float(self.SCALE_FACTOR)))
        normalized_logits = self._opened_clip_batch(
            normalized_logits,
            node_id=node_id,
            context=f"{context}_clip_logits",
            clip_min=-clip_abs,
            clip_max=clip_abs,
        )
                
        # 4. Exponentiation approximation
        if self.exp_approx == "pade22":
            exp_logits_flat = self._exp_pade22_batch(normalized_logits, node_id, f"{context}_exp")
        else:
            exp_logits_flat = self._exp_poly_batch(normalized_logits, node_id, f"{context}_exp")
        # Force exp approximation outputs into valid non-negative range.
        # This prevents negative Taylor artifacts from corrupting probability gradients.
        exp_upper = int(round(np.exp(self.logit_clip) * float(self.SCALE_FACTOR)))
        if exp_upper < 1:
            exp_upper = 1
        exp_logits_flat = self._opened_clip_batch(
            exp_logits_flat,
            node_id=node_id,
            context=f"{context}_clip_exp",
            clip_min=1,
            clip_max=exp_upper,
        )
        
        # 5. Compute sumexp per column
        sumexp_cols = []
        for i in range(batch_size):
            sumexp = Share(x=exp_logits_flat[0].x, y=0, node_id=node_id)
            for j in range(num_classes):
                e_val = exp_logits_flat[i * num_classes + j].y
                sumexp = Share(x=sumexp.x, y=(int(sumexp.y) + int(e_val)) % p, node_id=node_id)
            sumexp_cols.append(sumexp)
            
        # 6. Divide to get probabilities
        # We expand sumexp_cols to match the flattened array so we can do ONE secure_divide_shares_batch
        denoms_flat = []
        for i in range(batch_size):
            for _ in range(num_classes):
                denoms_flat.append(sumexp_cols[i])
                
        probs_flat = self.divider.secure_divide_shares_batch(
            exp_logits_flat,
            denoms_flat,
            node_id,
            f"{context}_div",
            enforce_probability_range=True,
            probability_group_size=num_classes,
        )
        
        # 7. Reshape back
        probs_cols = []
        for i in range(batch_size):
            probs_cols.append(probs_flat[i * num_classes : (i+1) * num_classes])
            
        return probs_cols
    
    def cross_entropy_loss(self, logits: List[Share], target: List[Share], 
                          node_id: int, context: str = "ce_loss") -> Share:
        """
        Compute cross-entropy loss: -log(softmax(logits)[true_class])
        Uses log-softmax trick for numerical stability
        
        Args:
            logits: List of shares representing logits [num_classes]
            target: List of shares representing one-hot target [num_classes]
            node_id: Node ID
            context: Context for secure operations
        Returns:
            Share of cross-entropy loss (scaled)
        """
        # MPC-safe CE:  L = - sum_i (y_i * log(p_i))
        # - y is one-hot encoded but secret-shared; we must NOT "find true class" from shares.
        # - probabilities p_i are secret-shared outputs of softmax.
        p_mod = int(self.field_size)
        probs = self.softmax(logits, node_id, f"{context}_softmax")
        if not probs:
            return Share(x=logits[0].x if logits else 0, y=0, node_id=node_id)

        acc = Share(x=probs[0].x, y=0, node_id=node_id)
        for i, (p_i, y_i) in enumerate(zip(probs, target)):
            logp = self._log_fp(p_i, node_id=node_id, context=f"{context}_log_{i}")
            # y_i is SCALE-scaled (0 or SCALE). fp-mul yields y_i/scale * logp = indicator * logp.
            term = self._fp_mul(y_i, logp, node_id=node_id, context=f"{context}_mask_{i}")
            acc = Share(x=acc.x, y=(int(acc.y) + int(term.y)) % p_mod, node_id=node_id)

        # loss = -acc
        return Share(x=acc.x, y=(-int(acc.y)) % p_mod, node_id=node_id)
    
    def cross_entropy_loss_gradient(self, logits: List[Share], target: List[Share],
                                   node_id: int, context: str = "ce_grad") -> List[Share]:
        """
        Compute gradient of cross-entropy loss w.r.t. logits
        Gradient = softmax(logits) - target (one-hot)
        
        This is the key advantage: gradient is simple!
        
        Args:
            logits: List of shares representing logits [num_classes]
            target: List of shares representing one-hot target [num_classes]
            node_id: Node ID
            context: Context for secure operations
        Returns:
            List of shares representing gradients [num_classes]
        """
        # MPC-safe gradient:
        # - Let p = softmax(logits / T).
        # - Then dL/dlogits = (p - y) / T.
        # - p and y are SCALE-scaled, so (p-y) is SCALE-scaled and we apply a public 1/T factor.
        p_mod = int(self.field_size)
        probs = self.softmax(logits, node_id, f"{context}_softmax")
        grads: List[Share] = []
        for i, (p_i, y_i) in enumerate(zip(probs, target)):
            grads.append(Share(x=p_i.x, y=(int(p_i.y) - int(y_i.y)) % p_mod, node_id=node_id))
        if self.temperature != 1.0 and grads:
            temp_int = int(round(float(self.temperature)))
            if abs(float(self.temperature) - float(temp_int)) > 1e-9:
                raise ValueError("Non-integer temperature is not supported in fixed-point privacy_mode")
            grads = [
                self.divider.secure_scalar_divide(g, temp_int, node_id, context=f"{context}_tempgrad_{i}")
                for i, g in enumerate(grads)
            ]
        return grads
        
    def cross_entropy_loss_gradient_batch(self, logits_cols: List[List[Share]], target_cols: List[List[Share]], node_id: int, context: str = "ce_grad_batch") -> List[List[Share]]:
        """
        Compute gradient of cross-entropy loss w.r.t. logits for a batch
        """
        if self.softmax_grad_mode == "opened_exact":
            return self._opened_exact_softmax_grad_batch(
                logits_cols,
                target_cols,
                node_id=node_id,
                context=f"{context}_opened_exact",
            )

        p_mod = int(self.field_size)
        probs_cols = self.softmax_batch(logits_cols, node_id, f"{context}_softmax")
        
        batch_size = len(logits_cols)
        if batch_size == 0:
            return []
        num_classes = len(logits_cols[0])
            
        grads_cols = []
        for i in range(batch_size):
            grads = []
            for j in range(num_classes):
                p_i = probs_cols[i][j]
                y_i = target_cols[i][j]
                grads.append(Share(x=p_i.x, y=(int(p_i.y) - int(y_i.y)) % p_mod, node_id=node_id))
            grads_cols.append(grads)
            
        if self.temperature != 1.0 and grads_cols:
            temp_int = int(round(float(self.temperature)))
            if abs(float(self.temperature) - float(temp_int)) > 1e-9:
                raise ValueError("Non-integer temperature is not supported in fixed-point privacy_mode")
            # Flatten to divide
            flat_grads = []
            for col in grads_cols:
                flat_grads.extend(col)
                
            flat_grads = self.divider.secure_scalar_divide_batch(flat_grads, temp_int, node_id, f"{context}_tempgrad")
            
            grads_cols = []
            for i in range(batch_size):
                grads_cols.append(flat_grads[i * num_classes : (i+1) * num_classes])
                
        return grads_cols

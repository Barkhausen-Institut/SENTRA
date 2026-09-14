"""
Secret Sharing Implementation
Shamir Secret Sharing and Packed Shamir Secret Sharing (PSS)

Notes on "Packed Shamir":
- This implements *true* packed Shamir secret sharing: multiple secrets are embedded
  as evaluations of a single polynomial at distinct (public) x-positions.
- A packed polynomial that hides k secrets with privacy threshold t has degree d=t+k-1
  and therefore needs at least (t+k) shares to reconstruct the packed secrets.
- With n parties, the maximum packing factor is k <= n - t.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class Share:
    """Represents a secret share"""
    x: int  # Evaluation point
    y: int  # Share value
    node_id: int  # Node that holds this share


class ShamirSecretSharing:
    """
    Shamir Secret Sharing implementation
    Splits a secret into n shares, requires t+1 shares to reconstruct
    """
    
    def __init__(self, field_size: int = 2**31 - 1):
        """
        Initialize Shamir Secret Sharing
        Args:
            field_size: Prime field size (default: 2^31 - 1)
        """
        self.field_size = field_size
    
    def share(self, secret: int, n: int, t: int) -> List[Share]:
        """
        Split secret into n shares with threshold t
        Args:
            secret: Secret value to share
            n: Number of shares
            t: Privacy threshold (need t+1 shares to reconstruct)
        Returns:
            List of n shares
        """
        if t >= n:
            raise ValueError("Threshold t must be less than n")
        
        # Ensure secret is in field
        secret = secret % self.field_size
        
        # Generate random polynomial coefficients
        # f(x) = secret + a1*x + a2*x^2 + ... + at*x^t
        coefficients = [secret]
        for _ in range(t):
            coefficients.append(random.randint(0, self.field_size - 1))
        
        # Generate shares at points 1, 2, ..., n
        shares = []
        for i in range(1, n + 1):
            y = self._evaluate_polynomial(coefficients, i)
            shares.append(Share(x=i, y=y, node_id=i))
        
        return shares
    
    def reconstruct(self, shares: List[Share]) -> int:
        """
        Reconstruct secret from shares using Lagrange interpolation
        Args:
            shares: List of at least t+1 shares
        Returns:
            Reconstructed secret
        """
        if len(shares) < 2:
            raise ValueError("Need at least 2 shares to reconstruct")
        
        # Use Lagrange interpolation
        secret = 0
        for i, share_i in enumerate(shares):
            numerator = 1
            denominator = 1
            
            for j, share_j in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-share_j.x)) % self.field_size
                    denominator = (denominator * (share_i.x - share_j.x)) % self.field_size
            
            # Compute Lagrange basis polynomial value
            lagrange_basis = (numerator * self._mod_inverse(denominator)) % self.field_size
            secret = (secret + share_i.y * lagrange_basis) % self.field_size
        
        return secret
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x"""
        result = 0
        power = 1
        for coeff in coefficients:
            result = (result + coeff * power) % self.field_size
            power = (power * x) % self.field_size
        return result
    
    def _mod_inverse(self, a: int) -> int:
        """Compute modular inverse using extended Euclidean algorithm"""
        if a < 0:
            a = a % self.field_size
        
        # Extended Euclidean algorithm
        old_r, r = a, self.field_size
        old_s, s = 1, 0
        
        while r != 0:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
        
        if old_r != 1:
            raise ValueError(f"{a} has no modular inverse modulo {self.field_size}")
        
        return old_s % self.field_size


class PackedShamirSecretSharing:
    """
    Packed Shamir Secret Sharing (PSS)
    Packs multiple secrets into a single polynomial for SIMD-style parallelism
    """
    
    def __init__(self, field_size: int = 2**31 - 1):
        """
        Initialize Packed Shamir Secret Sharing
        Args:
            field_size: Prime field size
        """
        self.field_size = field_size
        self.shamir = ShamirSecretSharing(field_size)

    def max_packing_factor(self, n: int, t: int) -> int:
        """
        Maximum number of secrets that can be packed into one polynomial for (n,t).
        For degree d=t+k-1, we need at least d+1=t+k points => k <= n - t.
        """
        if t < 0 or n <= 0:
            raise ValueError("Invalid n/t")
        return max(0, int(n) - int(t))

    def _inv(self, a: int) -> int:
        """Field inverse (field_size is assumed prime)."""
        p = int(self.field_size)
        return pow(int(a) % p, p - 2, p)

    def _eval_poly(self, coeffs: List[int], x: int) -> int:
        """Evaluate polynomial with coeffs[0] + coeffs[1] x + ... at x (mod p)."""
        p = int(self.field_size)
        x = int(x) % p
        out = 0
        power = 1
        for c in coeffs:
            out = (out + (int(c) % p) * power) % p
            power = (power * x) % p
        return out

    def _lagrange_value_at(self, x0: int, xs: List[int], ys: List[int]) -> int:
        """Interpolate f(x0) from points (xs[i], ys[i]) in the field."""
        p = int(self.field_size)
        x0 = int(x0) % p
        xs_mod = [int(x) % p for x in xs]
        ys_mod = [int(y) % p for y in ys]
        if len(xs_mod) != len(ys_mod):
            raise ValueError("xs/ys length mismatch")
        if len(xs_mod) < 1:
            raise ValueError("Need at least one point")

        out = 0
        m = len(xs_mod)
        for i in range(m):
            xi = xs_mod[i]
            num = 1
            den = 1
            for j in range(m):
                if i == j:
                    continue
                xj = xs_mod[j]
                num = (num * (x0 - xj)) % p
                den = (den * (xi - xj)) % p
            out = (out + ys_mod[i] * (num * self._inv(den) % p)) % p
        return out

    def share_secrets(
        self,
        secrets: List[int],
        n: int,
        t: int,
        *,
        secret_xs: Optional[List[int]] = None,
    ) -> List[Share]:
        """
        Create n packed shares that encode k=len(secrets) secrets with privacy threshold t.

        We keep party evaluation points as x=1..n (so existing networking assumptions hold),
        and place secrets at distinct public positions secret_xs. By default we use:
            secret_xs = [-1, -2, ..., -k]
        which avoids colliding with party x=1..n.
        """
        p = int(self.field_size)
        if n <= 0:
            raise ValueError("n must be positive")
        if t < 0 or t >= n:
            raise ValueError("Threshold t must satisfy 0 <= t < n")
        if secrets is None:
            raise ValueError("secrets must not be None")
        k = int(len(secrets))
        if k <= 0:
            return []

        if k > self.max_packing_factor(n, t):
            raise ValueError(f"packing_factor k={k} exceeds maximum n-t={n - t} for (n={n}, t={t})")

        # Secret x-positions (public, distinct, not overlapping party points 1..n)
        if secret_xs is None:
            secret_xs = [-(i + 1) for i in range(k)]
        if len(secret_xs) != k:
            raise ValueError("secret_xs length must equal number of secrets")

        # Normalize and validate x-points
        party_xs = [i for i in range(1, n + 1)]
        # Ensure all x are distinct mod p
        all_xs = [(int(x) % p) for x in (party_xs + list(secret_xs))]
        if len(set(all_xs)) != len(all_xs):
            raise ValueError("secret_xs collide with party x points modulo field")

        # Interpolation helper for the k secrets at secret_xs (degree < k)
        secret_xs_mod = [int(x) % p for x in secret_xs]
        secrets_mod = [int(s) % p for s in secrets]

        # Construct masking polynomial r(x) = P(x) * q(x)
        # where P(x) = ∏(x - secret_xs[j]) (degree k)
        # and q(x) is random degree (t-1). Then r(secret_xs[j])=0.
        if t == 0:
            q_coeffs: List[int] = [0]
        else:
            q_coeffs = [random.randint(0, p - 1) for _ in range(t)]  # degree t-1

        def _P(x: int) -> int:
            x = int(x) % p
            out = 1
            for sx in secret_xs_mod:
                out = (out * (x - sx)) % p
            return out

        shares: List[Share] = []
        for node_id, x in enumerate(party_xs, start=1):
            # g(x): interpolate the secrets-only polynomial at x
            gx = self._lagrange_value_at(x, secret_xs_mod, secrets_mod)
            rx = (_P(x) * self._eval_poly(q_coeffs, x)) % p
            y = (gx + rx) % p
            shares.append(Share(x=int(x), y=int(y), node_id=int(node_id)))

        return shares

    def reconstruct_secrets(
        self,
        shares: List[Share],
        k: int,
        t: int,
        *,
        secret_xs: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Reconstruct k packed secrets from party shares.

        Requires at least (t+k) shares, since the packed polynomial degree is d=t+k-1.
        """
        p = int(self.field_size)
        k = int(k)
        t = int(t)
        if k <= 0:
            return []
        if t < 0:
            raise ValueError("t must be non-negative")
        need = t + k
        if shares is None or len(shares) < need:
            raise ValueError(f"Need at least {need} shares to reconstruct packed secrets (got {0 if shares is None else len(shares)})")

        if secret_xs is None:
            secret_xs = [-(i + 1) for i in range(k)]
        if len(secret_xs) != k:
            raise ValueError("secret_xs length mismatch")

        chosen = shares[:need]
        xs = [int(s.x) % p for s in chosen]
        ys = [int(s.y) % p for s in chosen]

        out: List[int] = []
        for sx in secret_xs:
            out.append(int(self._lagrange_value_at(sx, xs, ys)) % p)
        return out

    def share_vector(
        self,
        secrets: List[int],
        n: int,
        t: int,
        *,
        packing_factor: Optional[int] = None,
    ) -> List[List[Share]]:
        """
        Convenience: pack a long list of secrets into multiple packed polynomials.
        Returns a list of "chunks", each chunk is a list of n shares (one per node).
        """
        if secrets is None:
            return []
        if not secrets:
            return []
        max_k = self.max_packing_factor(n, t)
        if max_k <= 0:
            raise ValueError("No packing possible for given (n,t)")
        k = int(packing_factor) if packing_factor is not None else max_k
        if k <= 0 or k > max_k:
            raise ValueError(f"packing_factor must be in [1, {max_k}]")

        chunks: List[List[Share]] = []
        for start in range(0, len(secrets), k):
            group = secrets[start:start + k]
            chunks.append(self.share_secrets(group, n, t))
        return chunks

    def reconstruct_vector(
        self,
        chunks: List[List[Share]],
        *,
        t: int,
        packing_factor: Optional[int] = None,
    ) -> List[int]:
        """
        Reconstruct a vector that was shared with share_vector(...).

        Args:
            chunks: list of chunks produced by share_vector; each chunk is List[Share] of length n
            t: privacy threshold used during sharing
            packing_factor: if provided, use this k for all full chunks; the last chunk may be shorter
        Returns:
            Flat list of reconstructed secrets (concatenated across chunks).
        """
        if not chunks:
            return []
        out: List[int] = []
        for chunk in chunks:
            # Infer k from packing_factor or from maximum allowed by (n,t), but cap to chunk length.
            n = len(chunk)
            if n <= 0:
                continue
            max_k = self.max_packing_factor(n, t)
            k = int(packing_factor) if packing_factor is not None else max_k
            k = max(1, min(k, max_k))
            # If this chunk was generated from a shorter tail group, we can only reconstruct up to that length.
            # The share_vector(...) API uses share_secrets(group, ...) where group length is the true k for that chunk.
            # We can't infer it reliably from shares alone, so we detect it by trying from k downwards.
            # (k is small: <= n-t.)
            recovered = None
            for kk in range(k, 0, -1):
                need = int(t) + int(kk)
                if len(chunk) < need:
                    continue
                try:
                    recovered = self.reconstruct_secrets(chunk[:need], k=kk, t=t)
                    break
                except Exception:
                    continue
            if recovered is None:
                raise ValueError("Failed to reconstruct packed chunk")
            out.extend(recovered)
        return out
    
    # Legacy API (incorrect in previous versions):
    # "pack_share/unpack_share" are not meaningful for true packed Shamir because packing
    # must happen at secret-sharing time (one polynomial), not by combining already-shared secrets.
    def pack_share(self, shares: List[Share], packing_factor: int) -> List[Share]:
        raise NotImplementedError(
            "pack_share() was a placeholder and is not a valid packed-Shamir primitive. "
            "Use share_secrets(...) / share_vector(...) instead."
        )

    def unpack_share(self, packed_shares: List[Share], packing_factor: int) -> List[Share]:
        raise NotImplementedError(
            "unpack_share() was a placeholder and is not a valid packed-Shamir primitive. "
            "Use reconstruct_secrets(...) instead."
        )



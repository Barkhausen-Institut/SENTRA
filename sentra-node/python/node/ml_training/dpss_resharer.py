"""
DPSS-style share update and committee redistribution (Shamir, synchronous, semi-honest).

Scope (what this module implements)
------------------------------------
1) **Redistribution without reconstructing the secret at any single party** (when run in a
   distributed fashion): each old holder i sends only ``λ_i(x'_j) * y_i`` to each new
   holder j; j sums to obtain a share of the *same* underlying polynomial value ``f(x'_j)``.
   In local tests we simulate the combined effect in one place (equivalent output).

2) **Proactive refresh (Herzberg-style)**: same committee, same secret ``f(0)``, fresh
   degree-``t`` polynomial. Each party i samples ``R_i`` with ``deg(R_i) ≤ t`` and
   ``R_i(0)=0``; new share j is ``y_j + Σ_i R_i(j)``. No party ever computes ``f(0)``.

What this is *not* (documented limitation)
------------------------------------------
`Long Live The Honey Badger` (Yurek et al., USENIX Security 2023) is **asynchronous,
Byzantine-tolerant** DPSS with strong liveness/fairness properties. Implementing that
protocol in full (RBC, vote structures, batch amortization, high-threshold path) is a
large standalone project. This module provides **standard synchronous semi-honest**
primitives that satisfy the core *cryptographic* goal: **no scalar reconstruction of the
secret** during refresh or fixed-threshold Shamir redistribution.

See: https://www.usenix.org/system/files/sec23fall-prepub-356-yurek.pdf
"""

from __future__ import annotations

from typing import List, Sequence
import random

from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_comm import SecureMPCNetwork


def _mod_inv(a: int, p: int) -> int:
    a = int(a) % int(p)
    if a == 0:
        raise ValueError("no inverse of 0")
    return pow(a, int(p) - 2, int(p))


def lagrange_interpolate_at(
    xs: Sequence[int],
    ys: Sequence[int],
    x_target: int,
    p: int,
) -> int:
    """
    Return f(x_target) where f is the unique polynomial of degree < len(xs)
    interpolating (xs[i], ys[i]) in F_p.
    """
    p = int(p)
    if len(xs) != len(ys) or len(xs) == 0:
        raise ValueError("xs and ys must be same non-empty length")
    m = len(xs)
    x_t = int(x_target) % p
    out = 0
    for i in range(m):
        xi = int(xs[i]) % p
        yi = int(ys[i]) % p
        num = 1
        den = 1
        for j in range(m):
            if i == j:
                continue
            xj = int(xs[j]) % p
            num = (num * (x_t - xj)) % p
            den = (den * (xi - xj)) % p
        lam = (num * _mod_inv(den, p)) % p
        out = (out + yi * lam) % p
    return out


def _pick_lagrange_subset(shares: List[Share], t: int) -> List[Share]:
    """Pick t+1 shares with distinct evaluation points (deterministic by x)."""
    if len(shares) < t + 1:
        raise ValueError(f"Need at least {t + 1} shares for Lagrange step, got {len(shares)}")
    by_x: dict = {}
    for s in shares:
        by_x[int(s.x)] = s
    if len(by_x) < t + 1:
        raise ValueError("Not enough distinct x-coordinates among shares")
    sorted_x = sorted(by_x.keys())
    chosen_x = sorted_x[: t + 1]
    return [by_x[x] for x in chosen_x]


class DPSSResharer:
    """
    Synchronous semi-honest Shamir redistribution / proactive refresh.

    ``reshare_shares`` defaults to non-reconstructing methods; use
    ``reshare_shares_via_reconstruct`` only for debugging or compatibility.
    """

    def __init__(
        self,
        network: SecureMPCNetwork,
        shamir: ShamirSecretSharing,
        t: int,
        old_node_ids: List[int],
        new_node_ids: List[int],
        *,
        rng: random.Random | None = None,
    ):
        self.network = network
        self.shamir = shamir
        self.t = int(t)
        self.old_node_ids = list(old_node_ids)
        self.new_node_ids = list(new_node_ids)
        self.field_size = int(shamir.field_size)
        self._rng = rng or random.Random()

    def reshare_shares(self, old_shares: List[Share]) -> List[Share]:
        """
        Redistribute or refresh shares without calling ``ShamirSecretSharing.reconstruct``.

        - If ``old_node_ids`` and ``new_node_ids`` are the same set (same committee size),
          applies **proactive refresh** (new random polynomial, same secret).
        - Otherwise uses **Lagrange redistribution** at new evaluation points 1..|new|.
        """
        active = [s for s in old_shares if s.node_id in self.old_node_ids]
        if len(active) < self.t + 1:
            raise ValueError(
                f"Need at least {self.t + 1} shares from old committee, got {len(active)}"
            )

        old_set = set(self.old_node_ids)
        new_set = set(self.new_node_ids)
        same_committee = old_set == new_set and len(self.old_node_ids) == len(self.new_node_ids)

        if same_committee:
            return self._proactive_refresh_herzberg(active)

        new_sorted = sorted(self.new_node_ids)
        return self._redistribute_lagrange_non_reconstructing(active, new_sorted)

    def reshare_shares_via_reconstruct(self, old_shares: List[Share]) -> List[Share]:
        """
        Legacy path: reconstruct from t+1 shares and re-share (NOT secure DPSS).

        Kept for regression tests and debugging only.
        """
        if len(old_shares) < self.t + 1:
            raise ValueError(f"Need at least {self.t + 1} shares, got {len(old_shares)}")

        active_old_shares = [s for s in old_shares if s.node_id in self.old_node_ids]

        if len(active_old_shares) < self.t + 1:
            raise ValueError(f"Need at least {self.t + 1} shares from active old nodes")

        shares_for_reconstruction = active_old_shares[: self.t + 1]
        secret = self.shamir.reconstruct(shares_for_reconstruction)
        n_new = len(self.new_node_ids)
        new_shares = self.shamir.share(secret, n_new, self.t)

        reshared_shares = []
        for i, new_share in enumerate(new_shares):
            reshared_shares.append(
                Share(x=new_share.x, y=new_share.y, node_id=self.new_node_ids[i])
            )
        return reshared_shares

    def _redistribute_lagrange_non_reconstructing(
        self,
        active: List[Share],
        new_node_ids_sorted: List[int],
    ) -> List[Share]:
        """
        New share for j-th new member: f(x=j+1) = Σ_{i∈S} λ_i(x) y_i  (|S|=t+1).

        Distributed form: holder of y_i sends λ_i(x'_j)·y_i to party j; no party sums to f(0).
        """
        subset = _pick_lagrange_subset(active, self.t)
        xs = [int(s.x) for s in subset]
        ys = [int(s.y) % self.field_size for s in subset]
        p = self.field_size

        out: List[Share] = []
        for nid in new_node_ids_sorted:
            x_new = int(nid)
            y_new = lagrange_interpolate_at(xs, ys, x_new, p)
            out.append(Share(x=x_new, y=y_new % p, node_id=x_new))
        return out

    def _proactive_refresh_herzberg(self, active: List[Share]) -> List[Share]:
        """
        Herzberg proactive refresh: same secret f(0), new polynomial
        F(x) = f(x) + Σ_k R_k(x) with deg R_k ≤ t and R_k(0) = 0.

        Simulates each party k ∈ old committee sampling R_k and sending R_k(x_pt) to every
        evaluation point x_pt in the committee. No party computes f(0).
        """
        p = self.field_size
        t = self.t
        by_nid: dict[int, Share] = {}
        for s in active:
            nid = int(s.node_id)
            if nid in self.old_node_ids:
                by_nid[nid] = s

        committee = sorted(self.old_node_ids)
        if len(by_nid) < len(committee):
            raise ValueError(
                "Proactive refresh needs one share per committee member in old_node_ids"
            )

        xs = sorted({int(by_nid[nid].x) % p for nid in committee})
        if len(xs) < t + 1:
            raise ValueError("Need at least t+1 distinct evaluation points for refresh")

        new_y_by_x: dict[int, int] = {
            int(by_nid[nid].x) % p: int(by_nid[nid].y) % p for nid in committee
        }

        for _k in committee:
            coeffs = [0]
            for _ in range(t):
                coeffs.append(self._rng.randrange(0, p))
            for x_pt in xs:
                delta = int(self.shamir._evaluate_polynomial(coeffs, x_pt)) % p
                new_y_by_x[x_pt] = (new_y_by_x[x_pt] + delta) % p

        out: List[Share] = []
        for nid in sorted(self.new_node_ids):
            s0 = by_nid[int(nid)]
            x_pt = int(s0.x) % p
            out.append(Share(x=x_pt, y=new_y_by_x[x_pt] % p, node_id=int(nid)))
        return out

    def reshare_weight_shares(self, weight_shares_list: List[List[List[Share]]]) -> List[List[List[Share]]]:
        """
        Reshare all weight shares (non-reconstructing ``reshare_shares`` per scalar).
        """
        reshared_weights: List[List[List[Share]]] = []

        for layer_idx, layer_shares in enumerate(weight_shares_list):
            reshared_layer: List[List[Share]] = []

            for row_idx, row_shares in enumerate(layer_shares):
                reshared_row: List[Share] = []

                for col_idx, shares_for_weight in enumerate(row_shares):
                    try:
                        new_shares = self.reshare_shares(shares_for_weight)
                        reshared_row.append(new_shares)
                    except Exception as e:
                        print(
                            f"Warning: Failed to reshare weight at layer={layer_idx}, "
                            f"row={row_idx}, col={col_idx}: {e}"
                        )
                        reshared_row.append(shares_for_weight)

                reshared_layer.append(reshared_row)

            reshared_weights.append(reshared_layer)

        return reshared_weights

    def can_reshare(self, available_shares: int) -> bool:
        return int(available_shares) >= self.t + 1

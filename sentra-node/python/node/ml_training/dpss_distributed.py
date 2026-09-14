"""
Distributed (message-passing) DPSS building blocks — Option B.

Each party only uses its own share plus vectors received over ``SecureChannel``:
no single party holds all shares.  Pair with ``SecureMPCNetwork.barrier`` in real
runs so all sends complete before receives (tests often run send-all then recv-all).

Conventions
-----------
- Evaluation point for party ``i`` is ``x = i`` (same as default ``ShamirSecretSharing.share``).
- ``subset_old_node_ids`` must be exactly the Lagrange set ``S`` (|S| = t+1), sorted,
  and identical at every party.
"""

from __future__ import annotations

import random
from typing import Any, Callable, List, Optional, Protocol, Sequence, Literal

from ml_training.secret_sharing import ShamirSecretSharing


class VectorChannel(Protocol):
    """Minimal channel surface used by distributed DPSS rounds."""

    def send_vector(self, target_node_id: int, context: str, x: int, values: Sequence[int]) -> None: ...

    def get_received_vector(self, context: str) -> dict[int, dict[str, Any]]: ...

    def clear_vector(self, context: str) -> None: ...


def _mod_inv(a: int, p: int) -> int:
    a = int(a) % int(p)
    if a == 0:
        raise ValueError("no inverse of 0")
    return pow(a, int(p) - 2, int(p))


def lagrange_lambda_basis(x_i: int, xs: Sequence[int], x_target: int, p: int) -> int:
    """
    Lagrange basis coefficient ℓ_i(x_target) for the set ``xs`` (distinct mod p),
    where share i sits at evaluation point ``x_i ∈ xs``.
    """
    p = int(p)
    xi = int(x_i) % p
    xt = int(x_target) % p
    num = 1
    den = 1
    for xj in xs:
        xj = int(xj) % p
        if xj == xi:
            continue
        num = (num * (xt - xj)) % p
        den = (den * (xi - xj)) % p
    return (num * _mod_inv(den, p)) % p


def distributed_redistribute_lagrange(
    channel: VectorChannel,
    *,
    my_node_id: int,
    p: int,
    t: int,
    my_share_y: Optional[int],
    subset_old_node_ids: Sequence[int],
    new_committee_ids: Sequence[int],
    context_prefix: str,
    phase: Literal["send", "recv", "both"] = "both",
) -> Optional[int]:
    """
    One-round Lagrange redistribution to new evaluation points = new node IDs.

    **Send phase** (for each ``i ∈ S`` that is this node): for each ``j`` in
    ``new_committee_ids``, send ``λ_i(j) · y_i`` to ``j`` using context
    ``{context_prefix}/to_{j}`` and ``x=i`` (sender id).

    **Receive phase** (if ``my_node_id ∈ new_committee_ids``): sum contributions from
    all senders in ``S``.

    Returns the new share value at ``x = my_node_id`` for new committee members;
    ``None`` if this node is not in ``new_committee_ids``.
    """
    p = int(p)
    t = int(t)
    sid = int(my_node_id)
    subset = [int(x) for x in subset_old_node_ids]
    new_ids = [int(x) for x in new_committee_ids]
    if len(subset) != t + 1:
        raise ValueError(f"subset_old_node_ids must have length t+1={t+1}, got {len(subset)}")
    if len(set(subset)) != len(subset):
        raise ValueError("subset_old_node_ids must be distinct")

    if phase not in ("send", "recv", "both"):
        raise ValueError(f"phase must be send|recv|both, got {phase!r}")

    xs = sorted(subset)

    # Send (only parties in S who hold y_i)
    if phase in ("send", "both") and sid in subset and my_share_y is not None:
        yi = int(my_share_y) % p
        xi = sid  # x-coordinate = node id
        for j in new_ids:
            lam = lagrange_lambda_basis(xi, xs, j, p)
            val = (lam * yi) % p
            channel.send_vector(j, f"{context_prefix}/to_{j}", x=sid, values=[val])

    if phase == "send" or sid not in new_ids:
        return None

    ctx = f"{context_prefix}/to_{sid}"
    recv = channel.get_received_vector(ctx)
    acc = 0
    for old_id in subset:
        payload = recv.get(int(old_id))
        if payload is None:
            raise RuntimeError(
                f"Missing contribution from old holder {old_id} at node {sid} (context {ctx})"
            )
        vals = payload.get("values") or []
        if not vals:
            raise RuntimeError(f"Empty values from sender {old_id} at node {sid}")
        acc = (acc + int(vals[0])) % p
    channel.clear_vector(ctx)
    return acc


def distributed_proactive_refresh_herzberg_send(
    channel: VectorChannel,
    shamir: ShamirSecretSharing,
    *,
    my_node_id: int,
    p: int,
    t: int,
    committee_node_ids: Sequence[int],
    context_prefix: str,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """
    Sample ``R_k`` with ``R_k(0)=0``, ``deg ≤ t``, and send ``R_k(j)`` to each ``j ≠ k``.

    Returns the coefficient list so the caller can pass it to
    ``distributed_proactive_refresh_herzberg_recv`` on the same machine.
    """
    p = int(p)
    t = int(t)
    sid = int(my_node_id)
    committee = sorted(int(x) for x in committee_node_ids)
    if sid not in committee:
        raise ValueError(f"my_node_id {sid} must be in committee {committee}")
    rng = rng or random.Random()

    coeffs = [0]
    for _ in range(t):
        coeffs.append(rng.randrange(0, p))

    for j in committee:
        if j == sid:
            continue
        delta = int(shamir._evaluate_polynomial(coeffs, j)) % p
        channel.send_vector(j, f"{context_prefix}/refresh/to_{j}", x=sid, values=[delta])

    return coeffs


def distributed_proactive_refresh_herzberg_recv(
    channel: VectorChannel,
    shamir: ShamirSecretSharing,
    *,
    my_node_id: int,
    p: int,
    t: int,
    my_share_y: int,
    coeffs: Sequence[int],
    committee_node_ids: Sequence[int],
    context_prefix: str,
) -> int:
    """
    Finish Herzberg refresh after ``distributed_proactive_refresh_herzberg_send`` using the
    **same** ``coeffs`` returned from that call for this node.
    """
    p = int(p)
    t = int(t)
    sid = int(my_node_id)
    committee = sorted(int(x) for x in committee_node_ids)
    if sid not in committee:
        raise ValueError(f"my_node_id {sid} must be in committee {committee}")
    cl = [int(c) % p for c in coeffs]
    if len(cl) != t + 1 or cl[0] != 0:
        raise ValueError("coeffs must have length t+1 with coeffs[0]==0")

    ctx = f"{context_prefix}/refresh/to_{sid}"
    recv = channel.get_received_vector(ctx)
    acc = (int(my_share_y) + int(shamir._evaluate_polynomial(cl, sid))) % p
    for sender_id, payload in recv.items():
        vals = payload.get("values") or []
        if not vals:
            raise RuntimeError(f"Empty refresh payload from {sender_id} at node {sid}")
        acc = (acc + int(vals[0])) % p

    expected_senders = {k for k in committee if k != sid}
    if set(recv.keys()) != expected_senders:
        raise RuntimeError(
            f"Node {sid}: expected senders {sorted(expected_senders)}, got {sorted(recv.keys())}"
        )
    channel.clear_vector(ctx)
    return acc


def distributed_proactive_refresh_herzberg_vectorized(
    channel: VectorChannel,
    shamir: ShamirSecretSharing,
    *,
    my_node_id: int,
    p: int,
    t: int,
    my_share_ys: Sequence[int],
    committee_node_ids: Sequence[int],
    context_prefix: str,
    rng: Optional[random.Random] = None,
    chunk_size: int = 50000,
    barrier_fn: Optional[Callable[[], None]] = None,
    phase: Literal["send", "recv", "both"] = "both",
    send_state: Optional[Any] = None,
) -> Any:
    """
    Herzberg refresh for many secrets (batched messages).

    For each secret s, party k sends R_{k,s}(j) to j. Uses ``chunk_size`` to split
    into sub-rounds. If ``barrier_fn`` is provided (e.g. network.barrier), it is
    called after all sends and before any recv.

    When ``phase="send"``, returns state for recv. When ``phase="recv"``, pass
    that state as ``send_state`` and returns new shares. When ``phase="both"``,
    returns new shares (for real network with barrier).
    """
    p = int(p)
    t = int(t)
    sid = int(my_node_id)
    committee = sorted(int(x) for x in committee_node_ids)
    if sid not in committee:
        raise ValueError(f"my_node_id {sid} must be in committee {committee}")
    rng = rng or random.Random()
    ys = [int(y) % p for y in my_share_ys]
    M = len(ys)
    if M == 0:
        return [] if phase != "send" else []

    if phase == "recv":
        if send_state is None:
            raise ValueError("send_state required when phase='recv'")
        all_coeffs = send_state
    else:
        all_coeffs = []
        offset = 0
        chunk_idx = 0
        while offset < M:
            end = min(offset + chunk_size, M)
            chunk_len = end - offset
            prefix = f"{context_prefix}_c{chunk_idx}"
            coeffs_list: List[List[int]] = []
            for _ in range(chunk_len):
                coeffs = [0]
                for _ in range(t):
                    coeffs.append(rng.randrange(0, p))
                coeffs_list.append(coeffs)
            all_coeffs.append((offset, end, prefix, coeffs_list, ys[offset:end]))
            if phase in ("send", "both"):
                for j in committee:
                    if j == sid:
                        continue
                    deltas = [
                        int(shamir._evaluate_polynomial(coeffs_list[s], j)) % p
                        for s in range(chunk_len)
                    ]
                    channel.send_vector(j, f"{prefix}/refresh/to_{j}", x=sid, values=deltas)
            offset = end
            chunk_idx += 1

        if phase == "send":
            return all_coeffs

    if phase in ("recv", "both") and barrier_fn is not None:
        barrier_fn()

    out: List[int] = list(ys)
    for offset, end, prefix, coeffs_list, chunk_ys in all_coeffs:
        chunk_len = end - offset
        ctx = f"{prefix}/refresh/to_{sid}"
        recv = channel.get_received_vector(ctx)
        for s in range(chunk_len):
            acc = (chunk_ys[s] + int(shamir._evaluate_polynomial(coeffs_list[s], sid))) % p
            for sender_id, payload in recv.items():
                vals = payload.get("values") or []
                if s >= len(vals):
                    raise RuntimeError(
                        f"Node {sid}: secret {offset + s}: short values from {sender_id}"
                    )
                acc = (acc + int(vals[s])) % p
            out[offset + s] = acc
        expected_senders = {k for k in committee if k != sid}
        if set(recv.keys()) != expected_senders:
            raise RuntimeError(
                f"Node {sid}: expected senders {sorted(expected_senders)}, "
                f"got {sorted(recv.keys())}"
            )
        channel.clear_vector(ctx)
    return out


def distributed_proactive_refresh_herzberg(
    channel: VectorChannel,
    shamir: ShamirSecretSharing,
    *,
    my_node_id: int,
    p: int,
    t: int,
    my_share_y: int,
    committee_node_ids: Sequence[int],
    context_prefix: str,
    rng: Optional[random.Random] = None,
) -> int:
    """
    Convenience: send + recv in one call (fine when messages are already delivered,
    e.g. after a barrier). Prefer split send/recv when driving an in-order simulator.
    """
    coeffs = distributed_proactive_refresh_herzberg_send(
        channel,
        shamir,
        my_node_id=my_node_id,
        p=p,
        t=t,
        committee_node_ids=committee_node_ids,
        context_prefix=context_prefix,
        rng=rng,
    )
    return distributed_proactive_refresh_herzberg_recv(
        channel,
        shamir,
        my_node_id=my_node_id,
        p=p,
        t=t,
        my_share_y=my_share_y,
        coeffs=coeffs,
        committee_node_ids=committee_node_ids,
        context_prefix=context_prefix,
    )

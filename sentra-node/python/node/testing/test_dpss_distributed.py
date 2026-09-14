"""
In-process simulation of distributed DPSS rounds (no TCP).

Uses a shared mailbox backend compatible with ``VectorChannel``.
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from threading import Lock
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_training.dpss_distributed import (
    distributed_proactive_refresh_herzberg_recv,
    distributed_proactive_refresh_herzberg_send,
    distributed_proactive_refresh_herzberg_vectorized,
    distributed_redistribute_lagrange,
    lagrange_lambda_basis,
)
from ml_training.dpss_resharer import lagrange_interpolate_at
from ml_training.secret_sharing import ShamirSecretSharing, Share


class SimVectorBackend:
    """(receiver_id, context) -> sender_id -> payload."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._mail: DefaultDict[Tuple[int, str], Dict[int, Dict[str, Any]]] = defaultdict(dict)

    def send(self, from_id: int, to_id: int, context: str, x: int, values: Sequence[int]) -> None:
        with self._lock:
            self._mail[(int(to_id), context)][int(from_id)] = {
                "x": int(x),
                "values": [int(v) for v in values],
            }

    def snapshot_get(self, recv_id: int, context: str) -> Dict[int, Dict[str, Any]]:
        with self._lock:
            return dict(self._mail.get((int(recv_id), context), {}))

    def clear(self, recv_id: int, context: str) -> None:
        with self._lock:
            self._mail.pop((int(recv_id), context), None)


class SimVectorChannel:
    def __init__(self, node_id: int, backend: SimVectorBackend) -> None:
        self.node_id = int(node_id)
        self.backend = backend

    def send_vector(self, target_node_id: int, context: str, x: int, values: Sequence[int]) -> None:
        self.backend.send(self.node_id, target_node_id, context, x, values)

    def get_received_vector(self, context: str) -> Dict[int, Dict[str, Any]]:
        return self.backend.snapshot_get(self.node_id, context)

    def clear_vector(self, context: str) -> None:
        self.backend.clear(self.node_id, context)


def _run_redistribute(
    shamir: ShamirSecretSharing,
    shares: List[Share],
    t: int,
    subset: List[int],
    new_committee: List[int],
    prefix: str,
) -> Dict[int, int]:
    p = shamir.field_size
    backend = SimVectorBackend()
    by_nid = {int(s.node_id): int(s.y) % p for s in shares}
    senders = sorted(by_nid.keys())
    receivers = sorted(set(by_nid.keys()) | set(new_committee))

    for nid in senders:
        ch = SimVectorChannel(nid, backend)
        distributed_redistribute_lagrange(
            ch,
            my_node_id=nid,
            p=p,
            t=t,
            my_share_y=by_nid[nid] if nid in subset else None,
            subset_old_node_ids=subset,
            new_committee_ids=new_committee,
            context_prefix=prefix,
            phase="send",
        )

    out: Dict[int, int] = {}
    for nid in receivers:
        ch = SimVectorChannel(nid, backend)
        y_new = distributed_redistribute_lagrange(
            ch,
            my_node_id=nid,
            p=p,
            t=t,
            my_share_y=by_nid[nid] if nid in subset else None,
            subset_old_node_ids=subset,
            new_committee_ids=new_committee,
            context_prefix=prefix,
            phase="recv",
        )
        if nid in new_committee:
            assert y_new is not None
            out[nid] = y_new % p
    return out


def test_lagrange_lambda_matches_interpolation() -> None:
    shamir = ShamirSecretSharing(field_size=127)
    p = shamir.field_size
    xs = [1, 2, 3]
    ys = [11, 22, 33]
    target = 7
    acc = 0
    for xi, yi in zip(xs, ys):
        acc = (acc + yi * lagrange_lambda_basis(xi, xs, target, p)) % p
    assert acc == lagrange_interpolate_at(xs, ys, target, p)


def test_distributed_redistribute_same_secret_new_committee() -> None:
    secret = 42
    n_old, t = 5, 2
    shamir = ShamirSecretSharing(field_size=1009)
    shares = shamir.share(secret, n_old, t)
    subset = [1, 2, 3]
    new_committee = [10, 11, 12]

    new_y_by_nid = _run_redistribute(
        shamir, shares, t, subset, new_committee, prefix="sim_redist"
    )
    new_shares = [Share(x=nid, y=new_y_by_nid[nid], node_id=nid) for nid in sorted(new_committee)]
    assert shamir.reconstruct(new_shares[: t + 1]) == secret % shamir.field_size


def test_distributed_redistribute_holder_not_in_subset_but_in_new_committee() -> None:
    """Party 5 receives Lagrange sum though it does not hold a share in S."""
    secret = 7
    n_old, t = 5, 2
    shamir = ShamirSecretSharing(field_size=1009)
    shares = shamir.share(secret, n_old, t)
    subset = [1, 2, 3]
    new_committee = [4, 5, 10]
    p = shamir.field_size
    by_nid = {int(s.node_id): int(s.y) % p for s in shares}
    backend = SimVectorBackend()

    for nid in sorted(by_nid.keys()):
        ch = SimVectorChannel(nid, backend)
        distributed_redistribute_lagrange(
            ch,
            my_node_id=nid,
            p=p,
            t=t,
            my_share_y=by_nid[nid] if nid in subset else None,
            subset_old_node_ids=subset,
            new_committee_ids=new_committee,
            context_prefix="edge",
            phase="send",
        )

    ch5 = SimVectorChannel(5, backend)
    y5 = distributed_redistribute_lagrange(
        ch5,
        my_node_id=5,
        p=p,
        t=t,
        my_share_y=None,
        subset_old_node_ids=subset,
        new_committee_ids=new_committee,
        context_prefix="edge",
        phase="recv",
    )
    assert y5 is not None
    xs = subset
    ys = [by_nid[i] for i in xs]
    assert y5 % p == lagrange_interpolate_at(xs, ys, 5, p)


def test_distributed_herzberg_preserves_secret() -> None:
    secret = 99
    n, t = 4, 2
    shamir = ShamirSecretSharing(field_size=1013)
    shares = shamir.share(secret, n, t)
    committee = [1, 2, 3, 4]
    p = shamir.field_size
    backend = SimVectorBackend()
    by_nid = {int(s.node_id): int(s.y) % p for s in shares}

    coeffs_by_nid: Dict[int, List[int]] = {}
    for nid in committee:
        ch = SimVectorChannel(nid, backend)
        coeffs_by_nid[nid] = distributed_proactive_refresh_herzberg_send(
            ch,
            shamir,
            my_node_id=nid,
            p=p,
            t=t,
            committee_node_ids=committee,
            context_prefix="hrz",
            rng=random.Random(nid),
        )

    for nid in committee:
        ch = SimVectorChannel(nid, backend)
        by_nid[nid] = distributed_proactive_refresh_herzberg_recv(
            ch,
            shamir,
            my_node_id=nid,
            p=p,
            t=t,
            my_share_y=by_nid[nid],
            coeffs=coeffs_by_nid[nid],
            committee_node_ids=committee,
            context_prefix="hrz",
        )

    refreshed = [Share(x=nid, y=by_nid[nid], node_id=nid) for nid in committee]
    assert shamir.reconstruct(refreshed[: t + 1]) == secret % p


def test_herzberg_vectorized_preserves_secrets() -> None:
    """Vectorized Herzberg refresh preserves all secrets (simulated with phase split)."""
    secrets = [11, 22, 33]
    n, t = 3, 1
    shamir = ShamirSecretSharing(field_size=127)
    p = shamir.field_size
    committee = [1, 2, 3]
    backend = SimVectorBackend()
    shares_by_secret: List[Dict[int, int]] = []
    for sec in secrets:
        shs = shamir.share(sec, n, t)
        shares_by_secret.append({int(s.node_id): int(s.y) % p for s in shs})

    state_by_nid: Dict[int, Any] = {}
    for nid in committee:
        ch = SimVectorChannel(nid, backend)
        ys = [shares_by_secret[s][nid] for s in range(len(secrets))]
        state_by_nid[nid] = distributed_proactive_refresh_herzberg_vectorized(
            ch,
            shamir,
            my_node_id=nid,
            p=p,
            t=t,
            my_share_ys=ys,
            committee_node_ids=committee,
            context_prefix="vec",
            rng=random.Random(nid),
            chunk_size=2,
            phase="send",
        )

    for nid in committee:
        ch = SimVectorChannel(nid, backend)
        ys = [shares_by_secret[s][nid] for s in range(len(secrets))]
        new_ys = distributed_proactive_refresh_herzberg_vectorized(
            ch,
            shamir,
            my_node_id=nid,
            p=p,
            t=t,
            my_share_ys=ys,
            committee_node_ids=committee,
            context_prefix="vec",
            send_state=state_by_nid[nid],
            phase="recv",
        )
        for s in range(len(secrets)):
            shares_by_secret[s][nid] = new_ys[s]

    for s, sec in enumerate(secrets):
        refreshed = [Share(x=nid, y=shares_by_secret[s][nid], node_id=nid) for nid in committee]
        assert shamir.reconstruct(refreshed[: t + 1]) == sec % p


def test_herzberg_recv_rejects_bad_coeffs() -> None:
    n, t = 3, 1
    shamir = ShamirSecretSharing(field_size=127)
    shares = shamir.share(3, n, t)
    committee = [1, 2, 3]
    p = shamir.field_size
    backend = SimVectorBackend()
    by_nid = {int(s.node_id): int(s.y) % p for s in shares}

    for nid in committee:
        ch = SimVectorChannel(nid, backend)
        distributed_proactive_refresh_herzberg_send(
            ch,
            shamir,
            my_node_id=nid,
            p=p,
            t=t,
            committee_node_ids=committee,
            context_prefix="bad",
            rng=random.Random(42),
        )

    with pytest.raises(ValueError, match="coeffs must"):
        ch = SimVectorChannel(1, backend)
        distributed_proactive_refresh_herzberg_recv(
            ch,
            shamir,
            my_node_id=1,
            p=p,
            t=t,
            my_share_y=by_nid[1],
            coeffs=[0],
            committee_node_ids=committee,
            context_prefix="bad",
        )

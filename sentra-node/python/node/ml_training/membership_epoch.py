"""
SENTRA membership epoch ``e`` (protocol step 6: epoch activation).

All parties must agree on the same integer ``e`` for a run. MPC vector contexts
and barrier tags are prefixed so stale traffic from a previous epoch cannot mix
with the current run (same TCP ports, new session).

KVS: use ``SENTRA_MEMBERSHIP_EPOCH_KVS_KEY`` with a monotonic version when a
distributed KVS is available. For single-process tests, ``record_membership_epoch_local_kvs``
writes to an in-memory cluster for this node only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ml_training.kvs import KVSCluster

# Protocol-facing key (for versioned put/get when KVS is wired end-to-end).
SENTRA_MEMBERSHIP_EPOCH_KVS_KEY = "sentra/membership_epoch"


@dataclass(frozen=True)
class MembershipEpochScope:
    """
    Immutable membership epoch ``e``. Use ``ctx()`` for vector / MPC context
    strings and ``barrier_tag()`` for ``SecureMPCNetwork.barrier`` and SYNC tags.
    """

    e: int

    def __post_init__(self) -> None:
        if int(self.e) < 0:
            raise ValueError("membership epoch e must be >= 0")

    def barrier_tag(self, name: str) -> str:
        """Barrier or SYNC tag, e.g. ``m3_startup``."""
        return f"m{int(self.e)}_{str(name)}"

    def ctx(self, name: str) -> str:
        """MPC / vector context string, e.g. ``m3_dataset/meta/v1``."""
        return f"m{int(self.e)}_{str(name)}"


class MutableMembershipEpochScope:
    """
    Mutable membership epoch ``e`` for join recovery (protocol step 6: bump e).
    Use ``bump()`` after join reshare; updates ctx/barrier_tag. Compatible API
    with MembershipEpochScope for ctx/barrier_tag/e.
    """

    def __init__(self, e: int):
        self._e = int(e)
        if self._e < 0:
            raise ValueError("membership epoch e must be >= 0")

    @property
    def e(self) -> int:
        return self._e

    def bump(self) -> int:
        """Increment e (call after join recovery). Returns new e."""
        self._e += 1
        return self._e

    def barrier_tag(self, name: str) -> str:
        return f"m{self._e}_{str(name)}"

    def ctx(self, name: str) -> str:
        return f"m{self._e}_{str(name)}"


def record_membership_epoch_local_kvs(node_id: int, e: int) -> "KVSCluster":
    """
    Record ``e`` on an in-process single-node KVS cluster (this process only).

    When a replicated KVS exists, write the same key with quorum instead.
    """
    from ml_training.kvs import KVSCluster

    cluster = KVSCluster([int(node_id)])
    cluster.write_with_quorum(SENTRA_MEMBERSHIP_EPOCH_KVS_KEY, int(e), version=1)
    return cluster


def read_membership_epoch_local_kvs(cluster: "KVSCluster", min_version: int = 1) -> Optional[int]:
    """Read epoch from local KVS cluster if present."""
    val = cluster.read_with_min_version(SENTRA_MEMBERSHIP_EPOCH_KVS_KEY, min_version=min_version)
    if val is None:
        return None
    return int(val.data)

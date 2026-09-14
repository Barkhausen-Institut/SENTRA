"""
SENTRA protocol-facing KVS API (steps 7-8, 11).

Key schemas:
- data/{i}/vD   : dataset sample i shares, version vD
- weights/layer/{name}/vθ : weight shares, version vθ

API: put(key, value, version), get(key, min_version) -> (value, version)
"""

from typing import Optional, Tuple, Any, List
from dataclasses import dataclass

from ml_training.kvs import KVSCluster, KVSValue


def key_data_sample(i: int) -> str:
    """Key for dataset sample i (version tag vD in caller)."""
    return f"data/{i}"


def key_data_sample_split(split: str, i: int) -> str:
    """Key for dataset sample i in split (train/test). Version vD in caller."""
    return f"data/{split}/{i}"


def key_weight_layer(layer_name: str) -> str:
    """Key prefix for weight layer (append /vθ when storing)."""
    return f"weights/layer/{layer_name}"


@dataclass
class GetResult:
    """Result of get(key, min_version): value and version."""

    value: Any
    version: int

    def as_tuple(self) -> Tuple[Any, int]:
        return (self.value, self.version)


def put(
    cluster: KVSCluster,
    key: str,
    value: Any,
    version: int,
    quorum_size: Optional[int] = None,
) -> bool:
    """
    Versioned put. Rejects stale: only accepts version > current.
    Returns True if quorum write succeeded.
    """
    return cluster.write_with_quorum(key, value, version, quorum_size=quorum_size)


def get(
    cluster: KVSCluster,
    key: str,
    min_version: int = 0,
) -> Optional[GetResult]:
    """
    Version-aware get. Returns (value, version) or None if not found or stale.
    """
    v = cluster.read_with_min_version(key, min_version=min_version)
    if v is None:
        return None
    return GetResult(value=v.data, version=v.version)


def get_batch(
    cluster: KVSCluster,
    keys: List[str],
    min_version: int = 0,
) -> List[Optional[GetResult]]:
    """
    Batched get for mini-batch retrieval.
    Returns list of GetResult or None per key.
    """
    return [get(cluster, k, min_version) for k in keys]

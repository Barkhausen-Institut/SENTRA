"""
Weight versioning vθ for SENTRA (protocol step 11).

After each optimizer step (or epoch): vθ += 1, put updated share blobs with
rollback metadata (keep vθ-1 until commit barrier).
"""

from typing import Optional, Any, List
from ml_training.kvs import KVSCluster
from ml_training.sentra_kvs import put, get, GetResult

WEIGHT_KEY_PREFIX = "weights/vθ"


def weight_key(v_theta: int) -> str:
    """Key for weight shares at version vθ."""
    return f"{WEIGHT_KEY_PREFIX}/{v_theta}"


def put_weights(
    cluster: KVSCluster,
    weights: Any,
    v_theta: int,
    quorum_size: Optional[int] = None,
) -> bool:
    """
    Store weight shares at version vθ.
    Use CAS for commit: put with expected v_theta-1, new v_theta.
    """
    key = weight_key(v_theta)
    return put(cluster, key, weights, v_theta, quorum_size=quorum_size)


def get_weights(
    cluster: KVSCluster,
    min_version: int = 1,
) -> Optional[GetResult]:
    """
    Get weight shares with version >= min_version.
    Returns (weights, version) or None.
    """
    key = f"{WEIGHT_KEY_PREFIX}/current"
    return get(cluster, key, min_version=min_version)


def put_weights_versioned(
    cluster: KVSCluster,
    weights: Any,
    v_theta: int,
    use_cas: bool = False,
    expected_version: Optional[int] = None,
    quorum_size: Optional[int] = None,
) -> bool:
    """
    Store weights at v_theta. If use_cas, requires expected_version = v_theta - 1.
    """
    key = f"{WEIGHT_KEY_PREFIX}/current"
    if use_cas and expected_version is not None:
        return cluster.write_with_quorum_cas(
            key, weights, expected_version, v_theta, quorum_size=quorum_size
        )
    return put(cluster, key, weights, v_theta, quorum_size=quorum_size)

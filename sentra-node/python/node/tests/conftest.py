"""
Pytest configuration and fixtures
"""

import socket
from typing import Any, Dict

import numpy as np
import pytest
from ml_training.beaver_triples import BeaverTripleGenerator, BeaverTriplePool, SecureMultiplier
from ml_training.secret_sharing import PackedShamirSecretSharing, ShamirSecretSharing


def ephemeral_local_node_configs(count: int) -> Dict[int, Dict[str, Any]]:
    """Unique loopback ports for this call (avoids EADDRINUSE across tests / leftover binds)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        base = int(s.getsockname()[1])
    return {i: {"host": "127.0.0.1", "port": base + i} for i in range(1, int(count) + 1)}


@pytest.fixture
def local_node_configs(n_nodes):
    return ephemeral_local_node_configs(n_nodes)


@pytest.fixture
def local_node_configs_pair():
    return ephemeral_local_node_configs(2)


@pytest.fixture
def field_size():
    """Prime field size for testing"""
    return 2**31 - 1


@pytest.fixture
def n_nodes():
    """Number of nodes for testing"""
    return 5


@pytest.fixture
def threshold():
    """Privacy threshold"""
    return 1


@pytest.fixture
def shamir_sharing(field_size):
    """Shamir secret sharing instance"""
    return ShamirSecretSharing(field_size=field_size)


@pytest.fixture
def packed_sharing(field_size):
    """Packed Shamir secret sharing instance"""
    return PackedShamirSecretSharing(field_size=field_size)


@pytest.fixture
def sample_shares(n_nodes, threshold, shamir_sharing):
    """Generate sample shares for testing"""
    secret = 42
    shares = shamir_sharing.share(secret, n_nodes, threshold)
    return shares, secret


@pytest.fixture
def beaver_generator(field_size):
    """Beaver triple generator"""
    return BeaverTripleGenerator(field_size=field_size)


@pytest.fixture
def triple_pool(beaver_generator):
    """Beaver triple pool"""
    return BeaverTriplePool(beaver_generator, initial_size=10)


@pytest.fixture
def secure_multiplier(triple_pool, n_nodes, threshold, field_size):
    """Secure multiplier instance"""
    triple_pool.initialize(n_nodes, threshold)
    return SecureMultiplier(triple_pool, n_nodes, threshold, field_size)


@pytest.fixture
def sample_matrix():
    """Sample matrix for testing"""
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def sample_vector():
    """Sample vector for testing"""
    return np.array([1, 2, 3])

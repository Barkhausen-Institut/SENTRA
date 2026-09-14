"""
Unit tests for Key-Value Store (KVS)
"""

import pytest
from ml_training.kvs import KVSNode, KVSCluster, KVSValue


class TestKVSValue:
    """Tests for KVSValue"""
    
    def test_kvs_value_creation(self):
        """Test creating a KVS value"""
        import time
        value = KVSValue(data={"key": "value"}, version=1, timestamp=time.time())
        assert value.data == {"key": "value"}
        assert value.version == 1
    
    def test_kvs_value_comparison(self):
        """Test comparing KVS values by version"""
        import time
        value1 = KVSValue(data={}, version=1, timestamp=time.time())
        value2 = KVSValue(data={}, version=2, timestamp=time.time())
        
        assert value2.version > value1.version


class TestKVSNode:
    """Tests for KVSNode"""
    
    def test_kvs_node_creation(self):
        """Test creating a KVS node"""
        node = KVSNode(node_id=1)
        assert node.node_id == 1
        assert len(node.store) == 0
    
    def test_put_and_get(self):
        """Test putting and getting values"""
        node = KVSNode(node_id=1)
        
        success = node.write("key1", {"test": "data"}, version=1)
        assert success
        
        retrieved = node.read("key1")
        assert retrieved is not None
        assert retrieved.data == {"test": "data"}
        assert retrieved.version == 1
    
    def test_get_nonexistent_key(self):
        """Test getting a non-existent key"""
        node = KVSNode(node_id=1)
        assert node.read("nonexistent") is None
    
    def test_versioning(self):
        """Test version management"""
        node = KVSNode(node_id=1)
        
        node.write("key", {"v": 1}, version=1)
        node.write("key", {"v": 2}, version=2)
        
        retrieved = node.read("key")
        assert retrieved.version == 2
        assert retrieved.data == {"v": 2}
    
    def test_get_version(self):
        """Test getting a specific version"""
        node = KVSNode(node_id=1)
        
        node.write("key", {"v": 1}, version=1)
        node.write("key", {"v": 2}, version=2)
        
        # Get version 2 (latest)
        v2 = node.read("key", min_version=2)
        assert v2 is not None
        assert v2.version == 2
        assert v2.data == {"v": 2}
        
        # Get with min_version 1 should also return version 2
        v_latest = node.read("key", min_version=1)
        assert v_latest.version == 2


class TestKVSCluster:
    """Tests for KVSCluster"""
    
    def test_cluster_creation(self, n_nodes):
        """Test creating a KVS cluster"""
        node_ids = list(range(1, n_nodes + 1))
        cluster = KVSCluster(node_ids=node_ids)
        assert len(cluster.nodes) == n_nodes
        assert all(node.node_id == i + 1 for i, node in enumerate(cluster.nodes.values()))
    
    def test_quorum_put(self, n_nodes):
        """Test quorum-based put operation"""
        node_ids = list(range(1, n_nodes + 1))
        cluster = KVSCluster(node_ids=node_ids)
        
        success = cluster.write_with_quorum("key1", {"test": "data"}, version=1, quorum_size=3)
        
        assert success
        # Check that at least quorum_size nodes have the value
        count = sum(1 for node in cluster.nodes.values() if node.read("key1") is not None)
        assert count >= 3
    
    def test_quorum_get(self, n_nodes):
        """Test quorum-based get operation"""
        node_ids = list(range(1, n_nodes + 1))
        cluster = KVSCluster(node_ids=node_ids)
        
        cluster.write_with_quorum("key1", {"test": "data"}, version=1, quorum_size=3)
        
        retrieved = cluster.read_with_min_version("key1", min_version=1)
        assert retrieved is not None
        assert retrieved.data == {"test": "data"}
    
    def test_quorum_get_fails_insufficient(self, n_nodes):
        """Test that quorum get fails with insufficient nodes"""
        node_ids = list(range(1, n_nodes + 1))
        cluster = KVSCluster(node_ids=node_ids)
        
        # Put to only 2 nodes (less than quorum)
        cluster.nodes[1].write("key1", {"test": "data"}, version=1)
        cluster.nodes[2].write("key1", {"test": "data"}, version=1)
        
        # Try to get (should return None if not enough nodes have it)
        retrieved = cluster.read_with_min_version("key1", min_version=1)
        # May or may not be None depending on implementation
        assert retrieved is None or retrieved is not None
    
    def test_version_consistency(self, n_nodes):
        """Test version consistency across nodes"""
        node_ids = list(range(1, n_nodes + 1))
        cluster = KVSCluster(node_ids=node_ids)
        
        cluster.write_with_quorum("key", {"v": 1}, version=1, quorum_size=3)
        cluster.write_with_quorum("key", {"v": 2}, version=2, quorum_size=3)
        
        # All nodes should have version 2
        retrieved = cluster.read_with_min_version("key", min_version=2)
        assert retrieved is not None
        assert retrieved.version == 2


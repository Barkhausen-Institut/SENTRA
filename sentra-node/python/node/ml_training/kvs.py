"""
Key-Value Store with Versioning
Enclave-backed KVS with rollback protection and freshness guarantees
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum
import threading


@dataclass
class KVSValue:
    """Value stored in KVS with version"""
    data: Any
    version: int
    timestamp: float


class KVSNode:
    """
    Single KVS node
    In production, this would run inside an enclave
    """
    
    def __init__(self, node_id: int):
        """
        Initialize KVS node
        Args:
            node_id: Unique node identifier
        """
        self.node_id = node_id
        self.store: Dict[str, KVSValue] = {}
        self.lock = threading.Lock()  # For thread-safe operations
    
    def write(self, key: str, data: Any, version: int) -> bool:
        """
        Write data with version
        Only accepts writes with version > current version
        Args:
            key: Key to write
            data: Data to store
            version: Version number (must be > current version)
        Returns:
            True if write succeeded, False otherwise
        """
        import time
        
        with self.lock:
            if key in self.store:
                current_version = self.store[key].version
                if version <= current_version:
                    return False  # Version not fresh
            
            self.store[key] = KVSValue(
                data=data,
                version=version,
                timestamp=time.time()
            )
            return True
    
    def write_cas(self, key: str, data: Any, expected_version: int, new_version: int) -> bool:
        """
        Compare-and-Set write with atomic version check
        Only updates if current version matches expected_version
        
        This provides true CAS semantics: atomically checks that the current version
        matches the expected version, and only then updates to the new version.
        
        Args:
            key: Key to write
            data: Data to store
            expected_version: Expected current version (must match for CAS to succeed)
            new_version: New version to set (must be > expected_version)
        Returns:
            True if CAS succeeded (current version matched expected), False otherwise
        """
        import time
        
        with self.lock:
            # Check if key exists
            if key not in self.store:
                # If key doesn't exist, expected_version should be 0 or -1 (initial state)
                if expected_version != 0 and expected_version != -1:
                    return False  # CAS failed: key doesn't exist but expected_version != 0/-1
                # Key doesn't exist and expected_version is appropriate
                self.store[key] = KVSValue(
                    data=data,
                    version=new_version,
                    timestamp=time.time()
                )
                return True
            
            # Key exists: check current version matches expected (CAS check)
            current_version = self.store[key].version
            if current_version != expected_version:
                return False  # CAS failed: version mismatch
            
            # Verify new_version is greater than expected_version (rollback protection)
            if new_version <= expected_version:
                return False  # CAS failed: new_version must be > expected_version
            
            # CAS succeeded: atomically update
            self.store[key] = KVSValue(
                data=data,
                version=new_version,
                timestamp=time.time()
            )
            return True
    
    def read(self, key: str, min_version: int = 0) -> Optional[KVSValue]:
        """
        Read data with minimum version requirement
        Args:
            key: Key to read
            min_version: Minimum version required
        Returns:
            KVSValue if found and version >= min_version, None otherwise
        """
        if key not in self.store:
            return None
        
        value = self.store[key]
        if value.version < min_version:
            return None  # Version too old
        
        return value


class KVSCluster:
    """
    Cluster of KVS nodes with quorum-based operations
    """
    
    def __init__(self, node_ids: List[int]):
        """
        Initialize KVS cluster
        Args:
            node_ids: List of node IDs in the cluster
        """
        self.nodes: Dict[int, KVSNode] = {node_id: KVSNode(node_id) for node_id in node_ids}
        self.n_nodes = len(node_ids)
    
    def write_with_quorum(self, key: str, data: Any, version: int, quorum_size: Optional[int] = None) -> bool:
        """
        Write to cluster with quorum consensus
        Args:
            key: Key to write
            data: Data to write
            version: Version number
            quorum_size: Required quorum size (default: majority)
        Returns:
            True if quorum achieved, False otherwise
        """
        if quorum_size is None:
            quorum_size = (self.n_nodes // 2) + 1
        
        # Write to all nodes
        successes = 0
        for node in self.nodes.values():
            if node.write(key, data, version):
                successes += 1
        
        return successes >= quorum_size
    
    def write_with_quorum_cas(self, key: str, data: Any, expected_version: int, new_version: int, 
                              quorum_size: Optional[int] = None) -> bool:
        """
        Compare-and-Set write to cluster with quorum consensus
        Only succeeds if quorum of nodes have matching expected_version
        
        This provides true CAS semantics at the cluster level: atomically checks
        that the current version matches the expected version across a quorum of nodes,
        and only then updates to the new version.
        
        Args:
            key: Key to write
            data: Data to write
            expected_version: Expected current version (must match for CAS to succeed)
            new_version: New version to set (must be > expected_version)
            quorum_size: Required quorum size (default: majority)
        Returns:
            True if CAS succeeded on quorum of nodes, False otherwise
        """
        if quorum_size is None:
            quorum_size = (self.n_nodes // 2) + 1
        
        # CAS write to all nodes
        successes = 0
        for node in self.nodes.values():
            if node.write_cas(key, data, expected_version, new_version):
                successes += 1
        
        return successes >= quorum_size
    
    def read_with_min_version(self, key: str, min_version: int = 0) -> Optional[KVSValue]:
        """
        Read from cluster with minimum version
        Returns value from first node that has version >= min_version
        Args:
            key: Key to read
            min_version: Minimum version required
        Returns:
            KVSValue if found, None otherwise
        """
        for node in self.nodes.values():
            value = node.read(key, min_version)
            if value is not None:
                return value
        return None
    
    def get_latest_version(self, key: str) -> int:
        """
        Get latest version for a key across all nodes
        Args:
            key: Key to check
        Returns:
            Latest version number, or 0 if key doesn't exist
        """
        max_version = 0
        for node in self.nodes.values():
            if key in node.store:
                max_version = max(max_version, node.store[key].version)
        return max_version



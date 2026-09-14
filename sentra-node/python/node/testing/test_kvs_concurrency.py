
import sys
import threading
import time
import random
import unittest
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Mock dependencies BEFORE importing sentra_training_node
sys.modules["coordination"] = MagicMock()
sys.modules["kvstore"] = MagicMock()
sys.modules["mpc"] = MagicMock()

from ml_training.sentra_training_node import (
    SentraTrainingNode, 
    NodeIdentity, 
    QuorumConfig, 
    RetryConfig, 
    TrainingConfig,
    KVSOperationError,
    VersionedKVSClient
)

# Mock KVS Storage
class MockKVStore:
    def __init__(self):
        self.store = {}
        self.lock = threading.Lock()

    def get(self, key, min_version=0, **kwargs):
        with self.lock:
            if key not in self.store:
                # Simulate "not found" or empty
                return MagicMock(value=None, version=0, quorum_ok=True)
            val, ver = self.store[key]
            # In a real KVS, min_version might block or return stale, 
            # here we just return what we have (Sentra client handles staleness checks)
            return MagicMock(value=val, version=ver, quorum_ok=True)

    def compare_and_set(self, key, old_version, new_value, new_version, **kwargs):
        with self.lock:
            current = self.store.get(key, (None, 0))
            curr_val, curr_ver = current
            
            # If key doesn't exist, we expect old_version=0
            if curr_ver != old_version:
                return MagicMock(success=False, quorum_ok=True)
            
            self.store[key] = (new_value, new_version)
            return MagicMock(success=True, quorum_ok=True)

# Test Case
class TestKVSConcurrency(unittest.TestCase):
    def setUp(self):
        self.mock_store = MockKVStore()
        
        # Common configs
        self.identity = NodeIdentity("enclave1", "token", "jwt", "creds")
        self.quorum = QuorumConfig(1, 1)
        self.retry = RetryConfig(max_attempts=3, base_delay_s=0.01)
        self.t_config = TrainingConfig(
            node_id=1, n_nodes=3, t=1, s=1, packing_factor=1,
            model_key="model", model_version_key="model_version",
            dataset_prefix="data", minibatch_size=10, max_steps=10
        )

    def test_cas_commit_model_atomicity(self):
        # Create a node
        node = SentraTrainingNode(
            kvs_endpoint="mock",
            identity=self.identity,
            quorum=self.quorum,
            retry=self.retry,
            config=self.t_config
        )
        
        # Patch the internal client to use our MockKVStore
        node.kvs._client = self.mock_store
        
        # Initialize model in store (version 0)
        self.mock_store.store["model"] = ("initial_params", 0)
        node._model_version = 0
        node._model_params = "initial_params"
        
        # Define a worker that tries to update the model
        successes = []
        errors = []
        
        def worker(worker_id):
            # Each worker tries to commit a new model based on version 0
            # Expected behavior: Only one succeeds, others fail with KVSOperationError
            # because they all start with local _model_version=0.
            try:
                # We manually invoke _cas_commit_model
                # But _cas_commit_model reads self._model_version.
                # All threads share 'node', so they share self._model_version.
                # However, _cas_commit_model does:
                #   old_version = self._model_version
                #   ...
                #   compare_and_set(..., old_version, ...)
                
                # To simulate "Concurrent Nodes", we should probably use separate Node instances
                # connected to the SAME store.
                new_ver = node._cas_commit_model(f"params_{worker_id}")
                successes.append((worker_id, new_ver))
            except KVSOperationError:
                errors.append(worker_id)
            except Exception as e:
                print(f"Worker {worker_id} failed unexpected: {e}")

        # BUT: Testing threads sharing the SAME node object is not quite right for "distributed" test.
        # Correct test: 2 separate Node objects sharing the MockStore.
        
        node1 = SentraTrainingNode(kvs_endpoint="mock", identity=self.identity, quorum=self.quorum, retry=self.retry, config=self.t_config)
        node1.kvs._client = self.mock_store
        node1._model_version = 0 
        
        node2 = SentraTrainingNode(kvs_endpoint="mock", identity=self.identity, quorum=self.quorum, retry=self.retry, config=self.t_config)
        node2.kvs._client = self.mock_store
        node2._model_version = 0

        # Worker function for distinct nodes
        def run_node_update(n, uid):
            try:
                n._cas_commit_model(f"params_{uid}")
                successes.append(uid)
            except KVSOperationError:
                errors.append(uid)

        t1 = threading.Thread(target=run_node_update, args=(node1, 1))
        t2 = threading.Thread(target=run_node_update, args=(node2, 2))
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        print(f"Successes: {successes}, Errors: {errors}")
        
        # Assertions
        self.assertEqual(len(successes), 1, "Exactly one update should succeed")
        self.assertEqual(len(errors), 1, "Exactly one update should fail due to conflict")
        
        # Verify store state
        val, ver = self.mock_store.store["model"]
        self.assertEqual(ver, 1)
        self.assertTrue(val.startswith("params_"))

    def test_version_mirror_update(self):
        # Verify that model_version_key is also updated
        node = SentraTrainingNode(kvs_endpoint="mock", identity=self.identity, quorum=self.quorum, retry=self.retry, config=self.t_config)
        node.kvs._client = self.mock_store
        
        # Initialize both model and model_version keys
        node.kvs._client.store["model"] = ("params", 10)
        node.kvs._client.store["model_version"] = (10, 10)  # Value 10, Version 10
        node._model_version = 10
        
        node._cas_commit_model("new_params")
        
        # Check mirror key
        val, ver = self.mock_store.store["model_version"]
        self.assertEqual(val, 11)
        self.assertEqual(ver, 11)  # logic sets new_value=new_version and new_version=new_version

if __name__ == "__main__":
    unittest.main()

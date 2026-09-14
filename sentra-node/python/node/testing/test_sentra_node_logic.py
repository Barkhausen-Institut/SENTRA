
import sys
import unittest
from unittest.mock import MagicMock, patch
import time
import os
import pytest

# Ensure the project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This test suite targets the optional `SentraTrainingNode` runtime path which depends on
# external service adapters (`coordination`, `kvstore`, `mpc`) and is not part of the
# `start_all_nodes.py` MNIST MLP workflow. Keep it opt-in so CI/local runs don't fail.
if os.getenv("SENTRA_RUN_SENTRA_NODE_LOGIC", "1") != "1":
    pytest.skip(
        "Disabled via SENTRA_RUN_SENTRA_NODE_LOGIC=0.",
        allow_module_level=True,
    )

# Mock external dependencies BEFORE importing sentra_training_node
sys.modules["kvstore"] = MagicMock()
sys.modules["mpc"] = MagicMock()
sys.modules["coordination"] = MagicMock()

import coordination
import kvstore
import mpc
import ml_training.sentra_training_node as sentra_node

from ml_training.sentra_training_node import (
    SentraTrainingNode,
    TrainingConfig,
    NodeIdentity,
    QuorumConfig,
    RetryConfig,
    VersionedValue,
    MiniBatchShares,
    KVSOperationError,
    StaleVersionError,
    QuorumError
)

class TestSentraTrainingNode(unittest.TestCase):
    def setUp(self):
        self.identity = NodeIdentity(
            enclave_id="test_enclave",
            epoch_token="test_token",
            epoch_jwt="test_jwt",
            mtls_credentials="test_creds"
        )
        self.quorum = QuorumConfig(read_quorum=2, write_quorum=2)
        self.retry = RetryConfig(max_attempts=2, base_delay_s=0.01)
        self.config = TrainingConfig(
            node_id=1,
            n_nodes=3,
            t=1,
            s=1,
            packing_factor=1,
            model_key="model",
            model_version_key="model_version",
            dataset_prefix="dataset",
            minibatch_size=8,
            max_steps=5,
            membership_poll_interval_s=0.01
        )
        
        # Reset mocks
        kvstore.Client = MagicMock()
        mpc.PackedEngine = MagicMock()
        coordination.notify = MagicMock()
        coordination.get_events = MagicMock(return_value=[])

        # Ensure the module under test sees the same mocked adapters.
        sentra_node.kvstore = kvstore
        sentra_node.mpc = mpc
        sentra_node.coordination = coordination

        # Setup default mock behavior
        self.mock_kvs_client = MagicMock()
        kvstore.Client.return_value = self.mock_kvs_client

        # Default recursive mock for engine
        self.mock_engine = MagicMock()
        mpc.PackedEngine.return_value = self.mock_engine

    def test_initialization(self):
        node = SentraTrainingNode(
            kvs_endpoint="https://kvs",
            identity=self.identity,
            quorum=self.quorum,
            retry=self.retry,
            config=self.config
        )
        self.assertIsNotNone(node)
        kvstore.Client.assert_called_with("https://kvs", "test_creds")

    def test_safety_bound_logic(self):
        node = SentraTrainingNode(
            kvs_endpoint="https://kvs",
            identity=self.identity,
            quorum=self.quorum,
            retry=self.retry,
            config=self.config
        )
        # bound: 2 * (t + s - 1) < n_active
        # t=1, s=1 => 2 * (1) = 2.
        # n_active=3 => 2 < 3 (True)
        self.assertTrue(node._safety_bound_holds(1, 1, 3))
        
        # n_active=2 => 2 < 2 (False)
        self.assertFalse(node._safety_bound_holds(1, 1, 2))
        
        # t=1, s=2 => 2*(2) = 4
        # n_active=4 => 4 < 4 (False)
        self.assertFalse(node._safety_bound_holds(1, 2, 4))
        # n_active=5 => 4 < 5 (True)
        self.assertTrue(node._safety_bound_holds(1, 2, 5))

    def test_kvs_get_retry_success(self):
        node = SentraTrainingNode(
            kvs_endpoint="https://kvs",
            identity=self.identity,
            quorum=self.quorum,
            retry=self.retry,
            config=self.config
        )
        
        # Fail once, then succeed
        self.mock_kvs_client.get.side_effect = [
            ConnectionError("fail"),
            {"version": 1, "value": "data", "quorum_ok": True}
        ]
        
        result = node.kvs.get("key", min_version=0)
        self.assertEqual(result.value, "data")
        self.assertEqual(result.version, 1)
        self.assertEqual(self.mock_kvs_client.get.call_count, 2)

    def test_kvs_get_stale_error(self):
        node = SentraTrainingNode(
            kvs_endpoint="https://kvs",
            identity=self.identity,
            quorum=self.quorum,
            retry=self.retry,
            config=self.config
        )
        
        self.mock_kvs_client.get.return_value = {"version": 0, "value": "old", "quorum_ok": True}
        
        with self.assertRaises(KVSOperationError):
            try:
                node.kvs.get("key", min_version=1)
            except KVSOperationError as e:
                # Check if cause was StaleVersionError
                self.assertIsInstance(e.__cause__, StaleVersionError)
                raise e

    def test_training_loop_flow(self):
        node = SentraTrainingNode(
            kvs_endpoint="https://kvs",
            identity=self.identity,
            quorum=self.quorum,
            retry=self.retry,
            config=self.config
        )
        
        # Setup mocks for training loop
        # 1. load_model
        self.mock_kvs_client.get.side_effect = lambda key, **kwargs: {
            "model": {"version": 0, "value": "initial_model", "quorum_ok": True},
            "model_version": {"version": 0, "value": 0, "quorum_ok": True},
            "dataset/batch/0/inputs": {"version": 1, "value": ["in1"], "quorum_ok": True},
            "dataset/batch/0/targets": {"version": 1, "value": ["tgt1"], "quorum_ok": True},
             # For subsequent checks in loop if any
        }.get(key, {"version": 0, "value": None, "quorum_ok": True})
        
        # 2. compare_and_set success
        self.mock_kvs_client.compare_and_set.return_value = {"success": True, "quorum_ok": True}
        
        # 3. Engine returns
        self.mock_engine.forward.return_value = "predictions"
        self.mock_engine.backward.return_value = "gradients"
        self.mock_engine.secure_aggregate_gradients.return_value = "agg_grads"
        self.mock_engine.apply_gradients.return_value = "new_model"
        
        # Run only 1 step
        node.config = TrainingConfig(
            node_id=1, n_nodes=3, t=1, s=1, packing_factor=1,
            model_key="model", model_version_key="model_version",
            dataset_prefix="dataset", minibatch_size=8,
            max_steps=1, membership_poll_interval_s=0.01
        )
        
        node.train()
        
        # Verifications
        # KVS
        self.mock_kvs_client.get.assert_any_call(
            key="model", min_version=0, read_quorum=2, epoch_token="test_token", jwt="test_jwt"
        )
        self.mock_kvs_client.compare_and_set.assert_called()
        
        # Engine
        # SentraTrainingNode calls via getattr, so we check if the mock methods were assigned
        # The code does: fn = getattr(self.engine, name, None); fn(...)
        # Since self.engine is a MagicMock, getattr returns a MagicMock, which is callable.
        # We need to verify that 'pack_shares' etc. were accessed.
        
        # Ideally we should see calls to the returned mocks of getattr
        # But for MagicMock, we can check if methods were called.
        # The node class tries multiple names, e.g. ["pack_shares", "pack_inputs", ...]
        # mocking getattr is tricky.
        
        # However, the code:
        # packed_inputs = self._call_engine(["pack_shares", ...], ...)
        # will try self.engine.pack_shares.
        # It's a MagicMock, so it exists. It will be called.
        
        # Let's verify specific calls that likely happened first in the list
        # _pack_batch: "pack_shares"
        # _forward_backward: "forward", "backward"
        # _secure_aggregate: "secure_aggregate_gradients"
        # _update_model: "apply_gradients"
        
        self.assertTrue(self.mock_engine.pack_shares.called or self.mock_engine.pack_inputs.called)
        self.assertTrue(self.mock_engine.forward.called or self.mock_engine.forward_pass.called)
        self.assertTrue(self.mock_engine.backward.called or self.mock_engine.compute_gradients.called)
        self.assertTrue(self.mock_engine.secure_aggregate_gradients.called)
        self.assertTrue(self.mock_engine.apply_gradients.called)

    def test_resharing_trigger(self):
        node = SentraTrainingNode(
            kvs_endpoint="https://kvs",
            identity=self.identity,
            quorum=self.quorum,
            retry=self.retry,
            config=self.config
        )
        
        # Mock membership to simulate breach
        # t=1, s=1 -> bound 2. Need > 2 nodes.
        # Simulate only 2 active nodes.
        with patch.object(node.membership, 'active_count', return_value=2):
             # Also assume training loop throws exception or we break it, 
             # otherwise it loops 'wait_until_allowed' potentially forever if we don't handle logic.
             # Actually, code does:
             # if not safety_bound: request_reshare; wait_until_allowed; continue
             
             # We need to make wait_until_allowed blocking or raise an exception to exit the test
             # or better, rely on side_effect to change state.
             
             # Let's just test the private method logic or the membership interaction directly
             # calling `train` is risky if it loops.
             
             # Instead, let's call `_request_resharing` manually and see effects
             node._request_resharing("test_reason")
             
             coordination.notify.assert_called()
             self.assertTrue(node.membership._reshare_in_progress.is_set())
             
             # Also check if engine reshare called
             self.assertTrue(self.mock_engine.dpss_reshare.called or self.mock_engine.reshare.called)

    def test_evaluate(self):
        node = SentraTrainingNode(
            kvs_endpoint="https://kvs",
            identity=self.identity,
            quorum=self.quorum,
            retry=self.retry,
            config=self.config
        )
        
        # Mock load_model
        self.mock_kvs_client.get.return_value = {"version": 5, "value": "model_params", "quorum_ok": True}
        self.mock_engine.forward.return_value = "encrypted_pred"
        
        preds = node.evaluate(test_dataset_shares=["share1", "share2"])
        
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0], "encrypted_pred")
        coordination.notify.assert_called_with({
            "event": "evaluation_completed",
            "node_id": 1,
            "model_version": 5,
            "num_samples": 2,
            "timestamp": unittest.mock.ANY
        })

if __name__ == '__main__':
    unittest.main()

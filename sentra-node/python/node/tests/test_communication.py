"""
Unit tests for Communication Protocol and Reconstruction
"""

import pytest
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_comm import (
    SecureChannel,
    SecureMPCNetwork,
    MessageType,
    create_mpc_network
)
from ml_training.reconstruction import (
    SecureReconstruction,
    MPCReconstructionManager,
    create_reconstruction_manager
)


class TestSecureChannel:
    """Tests for SecureChannel"""
    
    def test_channel_creation(self):
        """Test creating a secure channel"""
        # Note: Actual network channels require sockets
        # This tests the structure
        channel = SecureChannel(
            node_id=1,
            port=8001,
            use_tls=False
        )
        assert channel.node_id == 1
        assert channel.port == 8001
    
    def test_message_types(self):
        """Test message type enumeration"""
        assert MessageType.SHARE_EXCHANGE is not None
        assert MessageType.RECONSTRUCTION_REQUEST is not None
        assert MessageType.RECONSTRUCTION_RESPONSE is not None
        assert MessageType.HEARTBEAT is not None


class TestSecureMPCNetwork:
    """Tests for SecureMPCNetwork"""
    
    def test_network_creation(self, local_node_configs_pair):
        """Test creating an MPC network"""
        node_configs = local_node_configs_pair
        p1 = int(node_configs[1]["port"])
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs, port=p1)
        try:
            assert network.node_id == 1
            assert len(network.node_configs) == 2
        finally:
            network.stop()
    
    def test_create_network_helper(self, local_node_configs_pair):
        """Test network creation helper function"""
        node_configs = local_node_configs_pair
        p1 = int(node_configs[1]["port"])
        network = create_mpc_network(1, node_configs, port=p1)
        try:
            assert network is not None
            assert network.node_id == 1
        finally:
            network.stop()


class TestSecureReconstruction:
    """Tests for SecureReconstruction"""
    
    def test_reconstruction_creation(self, local_node_configs, threshold, field_size):
        """Test creating reconstruction manager"""
        node_configs = local_node_configs
        p1 = int(node_configs[1]["port"])
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs, port=p1)
        try:
            recon = SecureReconstruction(network, threshold, field_size)
            assert recon.t == threshold
        finally:
            network.stop()
    
    def test_reconstruct_from_shares(self, local_node_configs, n_nodes, threshold, field_size):
        """Test reconstructing secret from shares"""
        node_configs = local_node_configs
        p1 = int(node_configs[1]["port"])
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs, port=p1)
        try:
            recon = SecureReconstruction(network, threshold, field_size)
            shamir = ShamirSecretSharing(field_size=field_size)
            
            secret = 42
            shares = shamir.share(secret, n_nodes, threshold)
            
            reconstructed = recon.reconstruct_value(shares, context="test")
            assert reconstructed == secret
        finally:
            network.stop()
    
    def test_reconstruct_insufficient_shares(self, local_node_configs, n_nodes, threshold, field_size):
        """Test that reconstruction fails with insufficient shares"""
        node_configs = local_node_configs
        p1 = int(node_configs[1]["port"])
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs, port=p1)
        try:
            recon = SecureReconstruction(network, threshold, field_size)
            shamir = ShamirSecretSharing(field_size=field_size)
            
            secret = 42
            shares = shamir.share(secret, n_nodes, threshold)
            
            if threshold >= 1:
                with pytest.raises(RuntimeError, match="Insufficient shares"):
                    recon.reconstruct_value([shares[0]], context="test", timeout=0.1)
        finally:
            network.stop()


class TestMPCReconstructionManager:
    """Tests for MPCReconstructionManager"""
    
    def test_manager_creation(self, local_node_configs, threshold, field_size):
        """Test creating reconstruction manager"""
        node_configs = local_node_configs
        p1 = int(node_configs[1]["port"])
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs, port=p1)
        try:
            manager = MPCReconstructionManager(network, threshold, field_size)
            assert manager.t == threshold
        finally:
            network.stop()
    
    def test_reconstruct_with_context(self, local_node_configs, n_nodes, threshold, field_size):
        """Test reconstruction with context caching"""
        node_configs = local_node_configs
        p1 = int(node_configs[1]["port"])
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs, port=p1)
        try:
            manager = MPCReconstructionManager(network, threshold, field_size)
            shamir = ShamirSecretSharing(field_size=field_size)
            
            secret = 42
            shares = shamir.share(secret, n_nodes, threshold)
            
            reconstructed = manager.get_reconstructed_value(
                shares, context="test_context"
            )
            assert reconstructed == secret
        finally:
            network.stop()
    
    def test_create_manager_helper(self, local_node_configs, threshold, field_size):
        """Test manager creation helper function"""
        node_configs = local_node_configs
        p1 = int(node_configs[1]["port"])
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs, port=p1)
        try:
            manager = create_reconstruction_manager(network, t=threshold, field_size=field_size)
            assert manager is not None
            assert manager.t == threshold
        finally:
            network.stop()


"""
Unit tests for MPC Engine
"""

import pytest
import numpy as np
from ml_training.secret_sharing import Share, ShamirSecretSharing, PackedShamirSecretSharing
from ml_training.mpc_engine import PackedMPCEngine


class TestPackedMPCEngine:
    """Tests for PackedMPCEngine"""
    
    def test_engine_creation(self, n_nodes, threshold):
        """Test creating an MPC engine"""
        engine = PackedMPCEngine(n_nodes=n_nodes, t=threshold)
        assert engine.n_nodes == n_nodes
        assert engine.t == threshold
    
    def test_secure_add(self, n_nodes, threshold):
        """Test secure addition"""
        engine = PackedMPCEngine(n_nodes=n_nodes, t=threshold)
        shamir = ShamirSecretSharing(field_size=engine.field_size)
        
        secret_a = 10
        secret_b = 20
        
        shares_a = shamir.share(secret_a, n_nodes, threshold)
        shares_b = shamir.share(secret_b, n_nodes, threshold)
        
        # Add shares using private method (homomorphic addition)
        result_shares = []
        for node_id in range(1, n_nodes + 1):
            share_a = shares_a[node_id - 1]
            share_b = shares_b[node_id - 1]
            # Use homomorphic addition: (a + b) mod field_size
            result_share = Share(
                x=share_a.x,
                y=(share_a.y + share_b.y) % engine.field_size,
                node_id=node_id
            )
            result_shares.append(result_share)
        
        result = shamir.reconstruct(result_shares)
        expected = (secret_a + secret_b) % engine.field_size
        assert result == expected
    
    def test_forward_pass(self, n_nodes, threshold):
        """Test forward pass through neural network"""
        engine = PackedMPCEngine(n_nodes=n_nodes, t=threshold)
        shamir = ShamirSecretSharing(field_size=engine.field_size)
        
        # Create input shares (one per feature)
        input_values = [1, 2, 3]
        input_shares = []
        for val in input_values:
            shares = shamir.share(val, n_nodes, threshold)
            input_shares.append(shares[0])  # Use first node's share
        
        # Create weight matrix (simplified - single layer)
        # Weights should be List[List[List[Share]]] - one matrix per layer
        weights = [[
            [Share(1, 1, 1), Share(1, 2, 1), Share(1, 3, 1)],
            [Share(1, 4, 1), Share(1, 5, 1), Share(1, 6, 1)]
        ]]
        
        # Forward pass
        output_shares = engine.forward_pass(input_shares, weights, node_id=1)
        
        # Check output structure
        assert len(output_shares) == len(weights[0])  # One output per weight row
    
    def test_backward_pass(self, n_nodes, threshold):
        """Test backward pass (gradient computation)"""
        engine = PackedMPCEngine(n_nodes=n_nodes, t=threshold)
        shamir = ShamirSecretSharing(field_size=engine.field_size)
        
        # Create loss gradient shares
        loss_grad = 1
        loss_shares = shamir.share(loss_grad, n_nodes, threshold)
        
        # Create input shares (list of Share objects, not list of lists)
        input_values = [1, 2]
        input_shares = []
        for val in input_values:
            shares = shamir.share(val, n_nodes, threshold)
            input_shares.append(shares[0])  # Use first node's share
        
        # Create predictions and targets (list of Share objects)
        predictions = input_shares
        targets = input_shares  # Simplified
        
        # Create weight shares (should be List[List[List[Share]]])
        weights = [[
            [Share(1, 1, 1), Share(1, 2, 1)]
        ]]
        
        # Backward pass (needs loss_share, predictions, targets, weights, node_id, input_shares)
        loss_share = loss_shares[0]
        gradients = engine.backward_pass(
            loss_share, predictions, targets, weights, node_id=1, input_shares=input_shares
        )
        
        # Check gradients structure
        assert len(gradients) == len(weights)
    
    def test_update_weights(self, n_nodes, threshold):
        """Test weight update"""
        engine = PackedMPCEngine(n_nodes=n_nodes, t=threshold)
        shamir = ShamirSecretSharing(field_size=engine.field_size)
        
        # Create weight shares
        weight_value = 10
        weight_shares = shamir.share(weight_value, n_nodes, threshold)
        
        # Create gradient shares
        grad_value = 5
        grad_shares = shamir.share(grad_value, n_nodes, threshold)
        
        # Update weights (expects List[List[List[Share]]])
        learning_rate = 0.01
        weights = [[[weight_shares[0]]]]  # Single weight in a layer
        gradients = [[[grad_shares[0]]]]  # Single gradient in a layer
        
        updated = engine.update_weights(weights, gradients, learning_rate, node_id=1)
        
        # Extract updated share
        updated_share = updated[0][0][0]
        
        # Reconstruct updated weight (simplified - would need all shares)
        # Check that structure is correct
        assert len(updated) == len(weights)
        assert len(updated[0]) == len(weights[0])
        assert len(updated[0][0]) == len(weights[0][0])


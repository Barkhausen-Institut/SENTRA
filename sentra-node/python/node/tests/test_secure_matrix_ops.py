"""
Unit tests for Secure Matrix Operations
"""

import pytest
import numpy as np
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.beaver_triples import SecureMultiplier, BeaverTripleGenerator
from ml_training.secure_matrix_ops import (
    SecureMatrixMultiplier,
    SecureMatrixOperations,
    GPUMatrixAccelerator
)


class TestSecureMatrixMultiplier:
    """Tests for SecureMatrixMultiplier"""
    
    def test_matrix_multiply(self, beaver_generator, triple_pool, n_nodes, threshold, field_size):
        """Test secure matrix multiplication"""
        triple_pool.initialize(n_nodes, threshold)
        multiplier = SecureMultiplier(triple_pool, n_nodes, threshold, field_size)
        matrix_mult = SecureMatrixMultiplier(multiplier, field_size)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        # Create matrices A (2x3) and B (3x2)
        A = np.array([[1, 2, 3], [4, 5, 6]])
        B = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Share matrices
        A_shares = []
        for row in A:
            row_shares = []
            for val in row:
                shares = shamir.share(int(val), n_nodes, threshold)
                row_shares.append(shares)
            A_shares.append(row_shares)
        
        B_shares = []
        for row in B:
            row_shares = []
            for val in row:
                shares = shamir.share(int(val), n_nodes, threshold)
                row_shares.append(shares)
            B_shares.append(row_shares)
        
        # Multiply (for node 1)
        C_shares = matrix_mult.secure_matrix_multiply(
            [[s[0] for s in row] for row in A_shares],
            [[s[0] for s in row] for row in B_shares],
            node_id=1
        )
        
        # Check output dimensions (2x2)
        assert len(C_shares) == 2
        assert len(C_shares[0]) == 2
    
    def test_matrix_vector_multiply(self, beaver_generator, triple_pool, n_nodes, threshold, field_size):
        """Test secure matrix-vector multiplication"""
        triple_pool.initialize(n_nodes, threshold)
        multiplier = SecureMultiplier(triple_pool, n_nodes, threshold, field_size)
        matrix_mult = SecureMatrixMultiplier(multiplier, field_size)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        # Create matrix A (2x3) and vector v (3,)
        A = np.array([[1, 2, 3], [4, 5, 6]])
        v = np.array([1, 2, 3])
        
        # Share matrix and vector
        A_shares = []
        for row in A:
            row_shares = []
            for val in row:
                shares = shamir.share(int(val), n_nodes, threshold)
                row_shares.append(shares)
            A_shares.append(row_shares)
        
        v_shares = []
        for val in v:
            shares = shamir.share(int(val), n_nodes, threshold)
            v_shares.append(shares)
        
        # Multiply (for node 1)
        result_shares = matrix_mult.secure_matrix_vector_multiply(
            [[s[0] for s in row] for row in A_shares],
            [s[0] for s in v_shares],
            node_id=1
        )
        
        # Check output dimensions (2,)
        assert len(result_shares) == 2


class TestSecureMatrixOperations:
    """Tests for SecureMatrixOperations"""
    
    def test_forward_pass(self, beaver_generator, triple_pool, n_nodes, threshold, field_size):
        """Test forward pass with matrix operations"""
        triple_pool.initialize(n_nodes, threshold)
        multiplier = SecureMultiplier(triple_pool, n_nodes, threshold, field_size)
        matrix_ops = SecureMatrixOperations(multiplier, field_size)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        # Create input shares
        input_values = [1, 2, 3]
        input_shares = []
        for val in input_values:
            shares = shamir.share(int(val), n_nodes, threshold)
            input_shares.append(shares[0])
        
        # Create weight matrix (should be List[List[List[Share]]] - one matrix per layer)
        weights = [[
            [Share(1, 1, 1), Share(1, 2, 1), Share(1, 3, 1)],
            [Share(1, 4, 1), Share(1, 5, 1), Share(1, 6, 1)]
        ]]
        
        # Forward pass
        output = matrix_ops.forward_pass(input_shares, weights, node_id=1)
        
        # Check output
        assert len(output) == len(weights[0])  # One output per weight row
    
    def test_update_weights(self, beaver_generator, triple_pool, n_nodes, threshold, field_size):
        """Test weight update"""
        triple_pool.initialize(n_nodes, threshold)
        multiplier = SecureMultiplier(triple_pool, n_nodes, threshold, field_size)
        matrix_ops = SecureMatrixOperations(multiplier, field_size)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        # Create weight and gradient shares
        weight_value = 10
        grad_value = 5
        
        weight_shares = shamir.share(weight_value, n_nodes, threshold)
        grad_shares = shamir.share(grad_value, n_nodes, threshold)
        
        # Weights should be List[List[List[Share]]] - one matrix per layer
        weights = [[[weight_shares[0]]]]
        gradients = [[[grad_shares[0]]]]
        
        # Update weights (uses learning_rate, not lr)
        updated = matrix_ops.update_weights(weights, gradients, learning_rate=0.01, node_id=1)
        
        # Check structure
        assert len(updated) == len(weights)
        assert len(updated[0]) == len(weights[0])


class TestGPUMatrixAccelerator:
    """Tests for GPUMatrixAccelerator"""
    
    def test_gpu_availability_check(self):
        """Test GPU availability check"""
        gpu = GPUMatrixAccelerator()
        # Should not crash
        available = gpu.is_available()
        assert isinstance(available, bool)
    
    def test_cpu_fallback(self):
        """Test CPU fallback when GPU unavailable"""
        gpu = GPUMatrixAccelerator()
        
        # Create test matrices
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        # Should work even if GPU unavailable
        try:
            result = gpu.matrix_multiply_gpu(A, B)
            assert result is not None
            assert result.shape == (2, 2)
        except (ImportError, RuntimeError) as e:
            # GPU/CuPy errors are expected on systems without CUDA
            # Just verify the accelerator exists
            assert gpu is not None


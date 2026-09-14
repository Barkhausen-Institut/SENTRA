"""
Performance benchmarks for ML Training Pipeline
"""

import pytest

pytest.importorskip("pytest_benchmark")
import time
import numpy as np
from ml_training.secret_sharing import ShamirSecretSharing, PackedShamirSecretSharing
from ml_training.beaver_triples import SecureMultiplier, BeaverTripleGenerator
from ml_training.secure_matrix_ops import SecureMatrixMultiplier, SecureMatrixOperations


class TestSecretSharingBenchmarks:
    """Benchmarks for secret sharing operations"""
    
    def test_shamir_sharing_benchmark(self, benchmark):
        """Benchmark Shamir secret sharing"""
        shamir = ShamirSecretSharing()
        n_nodes = 5
        threshold = 1
        
        def share_secret():
            secret = 42
            shares = shamir.share(secret, n_nodes, threshold)
            return shares
        
        result = benchmark(share_secret)
        assert len(result) == n_nodes
    
    def test_shamir_reconstruction_benchmark(self, benchmark):
        """Benchmark Shamir secret reconstruction"""
        shamir = ShamirSecretSharing()
        n_nodes = 5
        threshold = 1
        secret = 42
        shares = shamir.share(secret, n_nodes, threshold)
        
        def reconstruct():
            return shamir.reconstruct(shares)
        
        result = benchmark(reconstruct)
        assert result == secret
    
    def test_packed_sharing_benchmark(self, benchmark):
        """Benchmark Packed Shamir secret sharing"""
        packed = PackedShamirSecretSharing()
        n_nodes = 5
        threshold = 1
        secrets = list(range(10))  # 10 secrets

        def share_packed():
            chunks = packed.share_vector(secrets, n_nodes, threshold)
            return chunks

        result = benchmark(share_packed)
        assert len(result) > 0


class TestBeaverTriplesBenchmarks:
    """Benchmarks for Beaver triple operations"""
    
    def test_triple_generation_benchmark(self, benchmark):
        """Benchmark Beaver triple generation"""
        generator = BeaverTripleGenerator(field_size=2**31 - 1)
        
        def generate():
            return generator.generate_triple(n_nodes=5, t=1)
        
        result = benchmark(generate)
        assert result is not None
    
    def test_secure_multiplication_benchmark(self, benchmark, beaver_generator, triple_pool):
        """Benchmark secure multiplication"""
        triple_pool.initialize(5, 1)
        multiplier = SecureMultiplier(triple_pool, 5, 1, 2**31 - 1)
        shamir = ShamirSecretSharing()
        
        secret_a = 10
        secret_b = 20
        shares_a = shamir.share(secret_a, 5, 1)
        shares_b = shamir.share(secret_b, 5, 1)
        
        def multiply():
            share_a = shares_a[0]
            share_b = shares_b[0]
            return multiplier.multiply(share_a, share_b, node_id=1)
        
        result = benchmark(multiply)
        assert result is not None


class TestMatrixOperationsBenchmarks:
    """Benchmarks for matrix operations"""
    
    def test_matrix_multiply_benchmark(self, benchmark, beaver_generator, triple_pool):
        """Benchmark secure matrix multiplication"""
        triple_pool.initialize(5, 1)
        multiplier = SecureMultiplier(triple_pool, 5, 1, 2**31 - 1)
        matrix_mult = SecureMatrixMultiplier(multiplier, 2**31 - 1)
        shamir = ShamirSecretSharing()
        
        # Small matrices for benchmark
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        # Share matrices (simplified - just for structure)
        A_shares = [[shamir.share(int(A[i, j]), 5, 1)[0] for j in range(2)] for i in range(2)]
        B_shares = [[shamir.share(int(B[i, j]), 5, 1)[0] for j in range(2)] for i in range(2)]
        
        def multiply():
            return matrix_mult.secure_matrix_multiply(A_shares, B_shares, node_id=1)
        
        result = benchmark(multiply)
        assert result is not None


class TestEndToEndBenchmarks:
    """End-to-end performance benchmarks"""
    
    def test_small_training_benchmark(self, benchmark):
        """Benchmark small training run"""
        from ml_training.training_pipeline import SentraTrainingPipeline
        
        def train():
            pipeline = SentraTrainingPipeline(
                n_nodes=5,
                t=1,
                s=1,
                enable_network=False
            )
            X = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
            y = [np.array([0]), np.array([1]), np.array([0])]
            weight_shapes = [(2, 1)]
            # Just test initialization, not full training
            return pipeline
        
        result = benchmark(train)
        assert result is not None


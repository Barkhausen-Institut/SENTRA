"""
Unit tests for Secret Sharing (Shamir & Packed Shamir)
"""

import pytest
import numpy as np
from ml_training.secret_sharing import (
    Share,
    ShamirSecretSharing,
    PackedShamirSecretSharing
)


class TestShare:
    """Tests for Share dataclass"""
    
    def test_share_creation(self):
        """Test creating a share"""
        share = Share(x=1, y=42, node_id=1)
        assert share.x == 1
        assert share.y == 42
        assert share.node_id == 1


class TestShamirSecretSharing:
    """Tests for Shamir Secret Sharing"""
    
    def test_share_secret(self, shamir_sharing, n_nodes, threshold):
        """Test sharing a secret"""
        secret = 42
        shares = shamir_sharing.share(secret, n_nodes, threshold)
        
        assert len(shares) == n_nodes
        assert all(isinstance(s, Share) for s in shares)
        assert all(s.x == i + 1 for i, s in enumerate(shares))
        assert all(s.node_id == i + 1 for i, s in enumerate(shares))
    
    def test_reconstruct_secret(self, shamir_sharing, sample_shares, threshold):
        """Test reconstructing a secret from shares"""
        shares, original_secret = sample_shares
        
        # Reconstruct with all shares
        reconstructed = shamir_sharing.reconstruct(shares)
        assert reconstructed == original_secret
        
        # Reconstruct with threshold+1 shares
        threshold_shares = shares[:threshold + 1]
        reconstructed = shamir_sharing.reconstruct(threshold_shares)
        assert reconstructed == original_secret
    
    def test_reconstruct_insufficient_shares(self, shamir_sharing, sample_shares):
        """Test that reconstruction fails with insufficient shares"""
        shares, _ = sample_shares
        
        # Try with only 1 share (need at least 2)
        with pytest.raises(ValueError):
            shamir_sharing.reconstruct([shares[0]])
    
    def test_share_large_secret(self, shamir_sharing, n_nodes, threshold):
        """Test sharing a large secret (field wrapping)"""
        large_secret = 2**30  # Large value
        shares = shamir_sharing.share(large_secret, n_nodes, threshold)
        reconstructed = shamir_sharing.reconstruct(shares)
        assert reconstructed == large_secret % shamir_sharing.field_size
    
    def test_share_negative_secret(self, shamir_sharing, n_nodes, threshold):
        """Test sharing a negative secret (should wrap to field)"""
        negative_secret = -10
        shares = shamir_sharing.share(negative_secret, n_nodes, threshold)
        reconstructed = shamir_sharing.reconstruct(shares)
        expected = negative_secret % shamir_sharing.field_size
        assert reconstructed == expected
    
    def test_invalid_threshold(self, shamir_sharing, n_nodes):
        """Test that threshold >= n raises error"""
        with pytest.raises(ValueError):
            shamir_sharing.share(42, n_nodes, n_nodes)
    
    def test_multiple_secrets(self, shamir_sharing, n_nodes, threshold):
        """Test sharing and reconstructing multiple secrets"""
        secrets = [10, 20, 30, 40, 50]
        all_shares = []
        
        for secret in secrets:
            shares = shamir_sharing.share(secret, n_nodes, threshold)
            all_shares.append(shares)
            reconstructed = shamir_sharing.reconstruct(shares)
            assert reconstructed == secret
    
    def test_share_addition(self, shamir_sharing, n_nodes, threshold):
        """Test that shares can be added (homomorphic property)"""
        secret1 = 10
        secret2 = 20
        
        shares1 = shamir_sharing.share(secret1, n_nodes, threshold)
        shares2 = shamir_sharing.share(secret2, n_nodes, threshold)
        
        # Add shares pointwise
        sum_shares = [
            Share(x=s1.x, y=(s1.y + s2.y) % shamir_sharing.field_size, node_id=s1.node_id)
            for s1, s2 in zip(shares1, shares2)
        ]
        
        # Reconstruct should give sum
        reconstructed = shamir_sharing.reconstruct(sum_shares)
        expected = (secret1 + secret2) % shamir_sharing.field_size
        assert reconstructed == expected


class TestPackedShamirSecretSharing:
    """Tests for Packed Shamir Secret Sharing"""
    
    def test_share_secrets(self, packed_sharing, n_nodes, threshold):
        """Test sharing multiple secrets in one packed polynomial."""
        secrets = [10, 20, 30]
        packed_shares = packed_sharing.share_secrets(secrets, n_nodes, threshold)
        assert len(packed_shares) == n_nodes
        assert all(isinstance(s, Share) for s in packed_shares)
    
    def test_reconstruct_secrets(self, packed_sharing, n_nodes, threshold):
        """Test reconstructing packed secrets from party shares."""
        secrets = [10, 20, 30]
        k = len(secrets)
        shares = packed_sharing.share_secrets(secrets, n_nodes, threshold)
        need = threshold + k
        rec = packed_sharing.reconstruct_secrets(shares[:need], k=k, t=threshold)
        assert rec == secrets
    
    def test_packed_addition(self, packed_sharing, n_nodes, threshold):
        """Homomorphic addition on packed party shares."""
        secrets1 = [10, 20]
        secrets2 = [5, 15]
        k = len(secrets1)
        p = int(packed_sharing.field_size)
        s1 = packed_sharing.share_secrets(secrets1, n_nodes, threshold)
        s2 = packed_sharing.share_secrets(secrets2, n_nodes, threshold)
        sum_shares = [
            Share(x=a.x, y=(a.y + b.y) % p, node_id=a.node_id)
            for a, b in zip(s1, s2)
        ]
        need = threshold + k
        rec = packed_sharing.reconstruct_secrets(sum_shares[:need], k=k, t=threshold)
        expected = [(secrets1[i] + secrets2[i]) % p for i in range(k)]
        assert rec == expected
    
    def test_packed_share_vector_roundtrip(self, packed_sharing, n_nodes, threshold):
        """Single-chunk share_vector / reconstruct_vector roundtrip."""
        secrets = [1, 2, 3]
        k = len(secrets)
        chunks = packed_sharing.share_vector(secrets, n_nodes, threshold, packing_factor=k)
        assert len(chunks) == 1
        out = packed_sharing.reconstruct_vector(chunks, t=threshold, packing_factor=k)
        assert out == secrets


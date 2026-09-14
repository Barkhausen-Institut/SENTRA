"""
Unit tests for Secure Division and Averaging
"""

import pytest
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_division import SecureDivider, SecureAverager


class TestSecureDivider:
    """Tests for SecureDivider"""
    
    def test_secure_divide(self, secure_multiplier, n_nodes, threshold, field_size):
        """Test secure division"""
        from ml_training.secure_division import SecureDivider
        divider = SecureDivider(secure_multiplier, field_size)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        # Divide 20 / 4 = 5 (using scalar division)
        dividend = 20
        divisor = 4
        
        dividend_shares = shamir.share(dividend, n_nodes, threshold)
        
        result_shares = []
        for node_id in range(1, n_nodes + 1):
            divd_share = dividend_shares[node_id - 1]
            result_share = divider.secure_scalar_divide(divd_share, divisor, node_id)
            result_shares.append(result_share)
        
        result = shamir.reconstruct(result_shares)
        # In finite field, division is multiplication by inverse
        # Compute expected: dividend * divisor^(-1) mod field_size
        divisor_inv = pow(divisor, field_size - 2, field_size)
        expected = (dividend * divisor_inv) % field_size
        # The result should match the expected field value
        assert result == expected
    
    def test_secure_divide_by_one(self, secure_multiplier, n_nodes, threshold, field_size):
        """Test dividing by one"""
        from ml_training.secure_division import SecureDivider
        divider = SecureDivider(secure_multiplier, field_size)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        dividend = 42
        divisor = 1
        
        dividend_shares = shamir.share(dividend, n_nodes, threshold)
        
        result_shares = []
        for node_id in range(1, n_nodes + 1):
            divd_share = dividend_shares[node_id - 1]
            result_share = divider.secure_scalar_divide(divd_share, divisor, node_id)
            result_shares.append(result_share)
        
        result = shamir.reconstruct(result_shares)
        # Dividing by 1 should return the dividend (in field arithmetic)
        assert result == (dividend % field_size)


class TestSecureAverager:
    """Tests for SecureAverager"""
    
    def test_secure_average(self, secure_multiplier, n_nodes, threshold, field_size):
        """Test secure averaging"""
        from ml_training.secure_division import SecureDivider, SecureAverager
        divider = SecureDivider(secure_multiplier, field_size)
        averager = SecureAverager(divider)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        # Average of [10, 20, 30] = 20
        values = [10, 20, 30]
        
        # Create shares for each value
        all_shares = []
        for value in values:
            shares = shamir.share(value, n_nodes, threshold)
            all_shares.append(shares)
        
        # Average shares using divider's secure_average method
        avg_shares = []
        for node_id in range(1, n_nodes + 1):
            node_shares = [shares[node_id - 1] for shares in all_shares]
            avg_share = divider.secure_average(node_shares, node_id)
            avg_shares.append(avg_share)
        
        result = shamir.reconstruct(avg_shares)
        # In finite field: (10 + 20 + 30) / 3 = 60 * 3^(-1) mod field_size
        three_inv = pow(3, field_size - 2, field_size)
        expected = ((10 + 20 + 30) * three_inv) % field_size
        assert result == expected
    
    def test_average_single_value(self, secure_multiplier, n_nodes, threshold, field_size):
        """Test averaging a single value"""
        from ml_training.secure_division import SecureDivider
        divider = SecureDivider(secure_multiplier, field_size)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        value = 42
        shares = shamir.share(value, n_nodes, threshold)
        
        avg_shares = []
        for node_id in range(1, n_nodes + 1):
            share = shares[node_id - 1]
            avg_share = divider.secure_average([share], node_id)
            avg_shares.append(avg_share)
        
        result = shamir.reconstruct(avg_shares)
        # Averaging a single value should return the value (in field arithmetic)
        assert result == (value % field_size)
    
    def test_average_empty_list(self, secure_multiplier, field_size):
        """Test averaging empty list"""
        from ml_training.secure_division import SecureDivider
        divider = SecureDivider(secure_multiplier, field_size)
        
        # Should handle empty list gracefully
        avg_share = divider.secure_average([], node_id=1)
        # Result should be zero share
        assert avg_share.y == 0


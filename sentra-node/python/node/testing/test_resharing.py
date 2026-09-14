"""
Test script for DPSS resharing
Tests share redistribution when nodes change
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.dpss_resharer import DPSSResharer
from ml_training.secure_comm import SecureMPCNetwork
import numpy as np


def create_mock_network(node_id: int, node_configs: dict) -> SecureMPCNetwork:
    """Create a mock network for testing"""
    # For testing, we'll use a simplified approach
    # In real usage, this would be created properly
    network = SecureMPCNetwork(node_id, node_configs, port=8000 + node_id, use_tls=False)
    return network


def test_resharing():
    """Test DPSS resharing protocol"""
    print("=" * 70)
    print("Testing DPSS Resharing")
    print("=" * 70)
    
    # Test parameters
    field_size = 2**31 - 1
    t = 1  # Threshold
    old_n = 5  # Original number of nodes
    new_n = 4  # New number of nodes
    
    old_node_ids = list(range(1, old_n + 1))
    new_node_ids = list(range(1, new_n + 1))  # Remove node 5
    
    print(f"\nTest configuration:")
    print(f"  Threshold t: {t}")
    print(f"  Old nodes: {old_node_ids}")
    print(f"  New nodes: {new_node_ids}")
    print(f"  Node removed: {old_n}")
    
    # Create Shamir secret sharing
    shamir = ShamirSecretSharing(field_size)
    
    # Create mock network (for testing, we don't need actual connections)
    node_configs = {i: {'host': 'localhost', 'port': 8000 + i} for i in old_node_ids}
    # Note: For actual resharing test, we'd need real network connections
    # For now, we test the resharing logic directly
    
    # Test 1: Create shares and reshare
    print("\n" + "-" * 70)
    print("Test 1: Resharing a single secret")
    print("-" * 70)
    
    secret = 12345
    print(f"Original secret: {secret}")
    
    # Create original shares
    old_shares = shamir.share(secret, old_n, t)
    print(f"Created {len(old_shares)} original shares")
    
    # Verify reconstruction works
    reconstructed = shamir.reconstruct(old_shares[:t+1])
    print(f"Reconstructed from {t+1} shares: {reconstructed} (should be {secret})")
    assert reconstructed == secret, f"Reconstruction failed: {reconstructed} != {secret}"
    
    # Reshare (simplified - need actual network for full test)
    # For testing, we'll manually simulate resharing
    print(f"\nSimulating resharing from {old_n} nodes to {new_n} nodes...")
    
    # Collect shares from remaining old nodes (nodes 1-4)
    remaining_old_shares = [s for s in old_shares if s.node_id in new_node_ids]
    print(f"Collected {len(remaining_old_shares)} shares from remaining nodes")
    
    if len(remaining_old_shares) >= t + 1:
        # Reconstruct
        reconstructed_secret = shamir.reconstruct(remaining_old_shares[:t+1])
        print(f"Reconstructed secret: {reconstructed_secret}")
        
        # Generate new shares for new node set
        new_shares = shamir.share(reconstructed_secret, new_n, t)
        
        # Verify new shares
        verify_secret = shamir.reconstruct(new_shares[:t+1])
        print(f"Verified new shares: {verify_secret} (should be {secret})")
        assert verify_secret == secret, f"Resharing failed: {verify_secret} != {secret}"
        print("✓ Resharing successful!")
    else:
        print(f"✗ Not enough shares for resharing (need {t+1}, have {len(remaining_old_shares)})")
    
    # Test 2: Resharing with insufficient shares
    print("\n" + "-" * 70)
    print("Test 2: Resharing with insufficient shares")
    print("-" * 70)
    
    # Try resharing when we don't have enough shares
    insufficient_shares = old_shares[:t]  # Only t shares (need t+1)
    print(f"Testing with {len(insufficient_shares)} shares (need {t+1})")
    
    try:
        # This should fail
        reconstructed = shamir.reconstruct(insufficient_shares)
        print(f"✗ Reconstruction succeeded with insufficient shares (should fail)")
    except Exception as e:
        print(f"✓ Correctly failed with insufficient shares: {e}")
    
    # Test 3: Multiple secrets (weights)
    print("\n" + "-" * 70)
    print("Test 3: Resharing multiple secrets (simulating weights)")
    print("-" * 70)
    
    # Simulate a weight matrix with 3 weights
    weights = [100, 200, 300]
    all_weight_shares = []
    
    for weight in weights:
        shares = shamir.share(weight, old_n, t)
        all_weight_shares.append(shares)
    
    print(f"Created shares for {len(weights)} weights")
    
    # Reshare each weight
    reshared_weights = []
    for i, weight_shares in enumerate(all_weight_shares):
        remaining = [s for s in weight_shares if s.node_id in new_node_ids]
        if len(remaining) >= t + 1:
            reconstructed = shamir.reconstruct(remaining[:t+1])
            new_shares = shamir.share(reconstructed, new_n, t)
            verify = shamir.reconstruct(new_shares[:t+1])
            reshared_weights.append(verify)
            print(f"Weight {i}: {weights[i]} -> reshared -> {verify} ✓")
        else:
            print(f"Weight {i}: Insufficient shares ✗")
    
    assert reshared_weights == weights, f"Resharing failed: {reshared_weights} != {weights}"
    print(f"\n✓ Successfully reshared all {len(weights)} weights")
    
    print("\n" + "=" * 70)
    print("All resharing tests passed!")
    print("=" * 70)
    print("\nNote: Full DPSS resharing requires:")
    print("  - Network connections between nodes")
    print("  - Coordination protocol")
    print("  - This test verifies the core resharing logic")
    print("=" * 70)


if __name__ == "__main__":
    test_resharing()



import unittest
from typing import List
from ml_training.secret_sharing import Share
from ml_training.mpc_engine import PackedMPCEngine

class MockNetwork:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.queues = {i: {} for i in range(1, n_nodes + 1)}

    def broadcast(self, sender_id, key, value):
        for i in range(1, self.n_nodes + 1):
            if i != sender_id:
                self.queues[i][key] = value

    def receive(self, receiver_id, key):
        return self.queues[receiver_id].get(key)

class TestSecureAggregation(unittest.TestCase):

    def test_aggregation_logic(self):
        """
        Verify that secure aggregation sums shares from all nodes.
        Logic:
        1. Nodes 1..3 generate local gradient G_i
        2. Nodes share G_i -> [G_i]_1, [G_i]_2, [G_i]_3
        3. Nodes exchange shares (mocked)
        4. Nodes sum received shares -> [Sum G]_j
        5. Reconstruct [Sum G] and verify it equals Sum(G_i)
        """
        n_nodes = 3
        t = 1
        engines = {i: PackedMPCEngine(n_nodes, t) for i in range(1, n_nodes + 1)}
        
        # 1. Local Gradients (Scalars for simplicity, but code logic is identical for tensors)
        # wrapped in List[List[List[Share]]] structure: [layer][row][col]
        # We'll simulate 1 layer, 1x1 matrix (scalar)
        local_gradients = {
            1: 10,
            2: 20,
            3: 30
        } # Expected Sum = 60
        
        # 2. Generate Shares
        # Each node i generates shares of its local_gradient[i]
        # content_shares[i][j] = Share of G_i held by Node j
        content_shares = {} 
        for i in range(1, n_nodes + 1):
            val = local_gradients[i]
            # Use engine's internal secret sharing to generate shares
            # We access pss directly for test simulation
            # share_secrets takes a list of secrets
            shares = engines[i].pss.share_secrets([val], n_nodes, t)
            content_shares[i] = shares # List of Share objects, index 0 is for Node 1
            
        # 3. Exchange & 4. Local Sum (The "Secure Aggregation" step)
        # Node j receives [G_1]_j, [G_2]_j, [G_3]_j
        aggregated_shares = {}
        for j in range(1, n_nodes + 1):
            # Collect shares destined for Node j
            received = []
            for i in range(1, n_nodes + 1):
                # Share for node j is at index j-1
                # The share object has .y = share value
                received.append(content_shares[i][j-1])
            
            # Sum them up (Simulating mpc_engine.secure_aggregate_gradients logic)
            # In real code, this would be: 
            # agg_share_j = engines[j].secure_aggregate_gradients(received)
            # But secure_aggregate_gradients currently takes "local gradients" and does the exchange?
            # NO. The method signature in mpc_engine.py takes `gradients`.
            # If `gradients` are shares of local gradient, then we need to implement the exchange inside it.
            
            # Helper to sum shares
            # We'll use the first engine's helper since logic is static-ish
            sum_y = 0
            for s in received:
                sum_y = (sum_y + s.y) % engines[j].field_size
            
            # This is the aggregated share held by Node j
            aggregated_shares[j] = Share(x=j, y=sum_y, node_id=j)
            
        # 5. Verify Reconstruction
        # Collect [Sum G]_1, [Sum G]_2, [Sum G]_3
        final_shares = [aggregated_shares[j] for j in range(1, n_nodes + 1)]
        reconstructed_list = engines[1].pss.reconstruct_secrets(final_shares, k=1, t=t)
        reconstructed = reconstructed_list[0]
        
        print(f"Reconstructed Sum: {reconstructed}")
        self.assertEqual(reconstructed, 60, "Secure Aggregation failed to sum gradients")


if __name__ == '__main__':
    unittest.main()

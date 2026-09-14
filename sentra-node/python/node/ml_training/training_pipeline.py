"""
End-to-End Training Pipeline
Main interface for secure ML training
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from ml_training.coordinator import TrainingCoordinator
from ml_training.kvs import KVSCluster


class SentraTrainingPipeline:
    """
    Main training pipeline for SENTRA
    """
    
    def __init__(self, n_nodes: int, t: int, s: int, 
                 batch_size: int = 32, learning_rate: float = 0.01,
                 num_epochs: int = 10,
                 node_id: int = 1, node_configs: Optional[Dict[int, Dict[str, Any]]] = None,
                 enable_network: bool = False,
                 seed: int = 2026,
                 train_mode: str = "secure",
                 log_mini_batches: bool = False,
                 progress_every_batches: int = 10):
        """
        Initialize training pipeline
        Args:
            n_nodes: Total number of nodes
            t: Privacy threshold (need t+1 shares to reconstruct)
            s: Adversarial share limit
            batch_size: Mini-batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            node_id: ID of this node
            node_configs: Dictionary mapping node_id to {host, port} for network
            enable_network: Whether to enable multi-node network communication
        """
        self.n_nodes = n_nodes
        self.t = t
        self.s = s
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.node_id = node_id
        self.enable_network = enable_network
        self.seed = seed
        self.train_mode = train_mode
        self.log_mini_batches = log_mini_batches
        self.progress_every_batches = max(1, int(progress_every_batches))
        
        # Initialize KVS cluster
        node_ids = list(range(1, n_nodes + 1))
        self.kvs_cluster = KVSCluster(node_ids)
        
        # Initialize coordinator with network support (always plain SGD)
        self.coordinator = TrainingCoordinator(
            self.kvs_cluster, n_nodes, t, s, batch_size, learning_rate,
            node_id=node_id, node_configs=node_configs,
            enable_network=enable_network, seed=seed, train_mode=train_mode
        )
    
    def train(self, dataset: List[np.ndarray], labels: List[np.ndarray],
              weight_shapes: List[Tuple[int, int]]):
        """
        Main training loop
        Args:
            dataset: List of training samples
            labels: List of labels
            weight_shapes: List of (input_dim, output_dim) for each layer
        """
        print("=" * 60)
        print("SENTRA ML Training Pipeline")
        print("=" * 60)
        print(f"Nodes: {self.n_nodes}, Threshold: {self.t}, Adversarial limit: {self.s}")
        print(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Train mode: {self.train_mode}")
        print(f"Node ID: {self.node_id}")
        if self.enable_network:
            print(f"Multi-Node: Enabled (Network communication active)")
        else:
            print(f"Multi-Node: Disabled (Single-node mode)")
        print("=" * 60)
        
        # Step 1: Ingest and secret-share dataset
        print("\n[Step 1] Ingesting and secret-sharing dataset...")
        v_D = self.coordinator.ingest_dataset(dataset, labels)
        print(f"Ingested {len(dataset)} samples with version v_D={v_D}")
        
        # Step 2: Initialize and secret-share weights
        print("\n[Step 2] Initializing and secret-sharing weights...")
        v_theta = self.coordinator.initialize_weights(weight_shapes)
        print(f"Initialized weights with version v_theta={v_theta}")
        
        # Step 3: Training loop
        print("\n[Step 3] Starting training loop...")
        
        training_suspended = False
        
        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")
            
            # Check if training was suspended and if safety is restored
            # Get the packing factor that would be used and check safety with it
            # Use dynamic active node count if available
            if self.coordinator.node_manager:
                n_active = self.coordinator.node_manager.get_n_active()
            else:
                n_active = self.n_nodes  # Fallback to static count
            if training_suspended:
                # Check if safety is restored with the packing factor that would be used
                potential_packing_factor = self.coordinator.safety_checker.get_max_packing_factor(n_active)
                safety_bound = 2 * (self.coordinator.t + potential_packing_factor - 1)
                if safety_bound < n_active:
                    print(f"  [RESUME] Safety bound restored (2*(t+packing_factor-1)={safety_bound} < n_active={n_active}), resuming training")
                    training_suspended = False
                else:
                    print(f"  [SUSPENDED] Safety bound still violated (2*(t+packing_factor-1)={safety_bound} >= n_active={n_active})")
                    print(f"  Waiting for safety to be restored (e.g., node repair/join or reduce packing factor)")
                    continue  # Skip this epoch
            
            # Select mini-batches
            num_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(num_batches):
                # Select mini-batch
                batch_indices = self.coordinator.select_mini_batch(len(dataset))
                
                # Retrieve shares from KVS (simplified - would retrieve properly)
                sample_shares = []
                label_shares = []
                
                for idx in batch_indices:
                    # Get shares for this sample
                    sample_key = f"sample_{idx}_node_{self.node_id}"
                    label_key = f"label_{idx}_node_{self.node_id}"
                    
                    sample_value = self.kvs_cluster.read_with_min_version(sample_key, v_D)
                    label_value = self.kvs_cluster.read_with_min_version(label_key, v_D)
                    
                    if sample_value and label_value:
                        # sample_value.data is a list of shares (one per feature)
                        # label_value.data is either:
                        #   - list[Share] for vector labels (e.g., one-hot), or
                        #   - single Share for scalar labels.
                        sample_feature_shares = sample_value.data
                        label_data = label_value.data
                        sample_shares.append(sample_feature_shares)
                        if isinstance(label_data, list):
                            label_shares.append(label_data)
                        else:
                            label_shares.append([label_data])
                
                if not sample_shares:
                    print(f"  Batch {batch_idx + 1}/{num_batches}: [FAIL] Failed")
                    continue
                
                # Get current weights
                weight_key = f"weights_v{v_theta}"
                weight_value = self.kvs_cluster.read_with_min_version(weight_key, v_theta)
                if not weight_value:
                    print(f"  Batch {batch_idx + 1}/{num_batches}: [FAIL] Failed")
                    continue
                
                weight_shares = weight_value.data
                
                # Train mini-batch
                # Use dynamic active node count if available
                if self.coordinator.node_manager:
                    n_active = self.coordinator.node_manager.get_n_active()
                else:
                    n_active = self.n_nodes  # Fallback to static count
                
                packing_factor = self.coordinator.safety_checker.get_max_packing_factor(n_active)
                updated_weights, success = self.coordinator.train_mini_batch(
                    sample_shares, label_shares, weight_shares, packing_factor,
                    node_id=self.node_id, batch_indices=batch_indices
                )
                
                if success:
                    # Commit update
                    committed = self.coordinator.commit_model_update(updated_weights)
                    if committed:
                        v_theta = self.coordinator.v_theta
                        if self.log_mini_batches:
                            print(f"Committed mini-batch {batch_indices}, v_theta={v_theta}")
                            print(f"  Batch {batch_idx + 1}/{num_batches}: [OK] Committed")
                    else:
                        print(f"  Batch {batch_idx + 1}/{num_batches}: [FAIL] Commit failed")
                else:
                    # Mini-batch aborted (safety bound violated or other error)
                    # Check if it's a safety bound violation with the actual packing factor
                    # Use dynamic active node count if available
                    if self.coordinator.node_manager:
                        n_active = self.coordinator.node_manager.get_n_active()
                    else:
                        n_active = self.n_nodes  # Fallback to static count
                    safety_bound = 2 * (self.coordinator.t + packing_factor - 1)
                    if safety_bound >= n_active:
                        print(f"  Batch {batch_idx + 1}/{num_batches}: [ABORT] Safety bound violated")
                        print(f"    Condition: 2*(t+packing_factor-1) = 2*({self.coordinator.t}+{packing_factor}-1) = {safety_bound} >= n_active = {n_active}")
                        print(f"    Packing factor s={packing_factor} does not satisfy safety bound")
                        print(f"    Current mini-batch DISCARDED (model state unchanged)")
                        print(f"  Training SUSPENDED until safety is restored")
                        print(f"    (e.g., increase n_active via node repair/join, or reduce packing factor)")
                        # Mark training as suspended and abort current epoch
                        training_suspended = True
                        break
                    else:
                        print(f"  Batch {batch_idx + 1}/{num_batches}: [FAIL] Training failed")

                # Lightweight progress signal when mini-batch logs are disabled.
                if (not self.log_mini_batches) and ((batch_idx + 1) % self.progress_every_batches == 0):
                    print(f"  Progress: epoch {epoch + 1}/{self.num_epochs}, batch {batch_idx + 1}/{num_batches}")
            
            print(f"Epoch {epoch + 1} completed, model version: v_theta={self.coordinator.v_theta}")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)

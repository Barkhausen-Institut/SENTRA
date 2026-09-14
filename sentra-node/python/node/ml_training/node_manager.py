"""
Node Manager for Coordinating Node Failures and Resharing
Handles node churn, failure detection, and share redistribution
"""

from typing import List, Set, Optional, Callable, Dict
from ml_training.node_failure_detector import NodeFailureDetector
from ml_training.dpss_resharer import DPSSResharer
from ml_training.secret_sharing import Share, PackedShamirSecretSharing
from ml_training.secure_comm import SecureMPCNetwork
from ml_training.coordinator import SafetyBoundChecker
import threading


class NodeManager:
    """
    Manages node failures, recoveries, and share resharing
    Coordinates between failure detection and resharing protocols
    """
    
    def __init__(self, 
                 network: SecureMPCNetwork,
                 failure_detector: NodeFailureDetector,
                 safety_checker: SafetyBoundChecker,
                 pss: PackedShamirSecretSharing,
                 t: int,
                 resharing_enabled: bool = True):
        """
        Initialize node manager
        Args:
            network: SecureMPCNetwork instance
            failure_detector: NodeFailureDetector instance
            safety_checker: SafetyBoundChecker instance
            pss: PackedShamirSecretSharing instance
            t: Privacy threshold
            resharing_enabled: Whether to enable automatic resharing
        """
        self.network = network
        self.failure_detector = failure_detector
        self.safety_checker = safety_checker
        self.pss = pss
        self.t = t
        self.resharing_enabled = resharing_enabled
        self.lock = threading.Lock()
        self.is_resharing = False
        self.resharing_callbacks: List[Callable[[], None]] = []
        
        # Register failure/recovery callbacks
        self.failure_detector.register_failure_callback(self._handle_node_failures)
        self.failure_detector.register_recovery_callback(self._handle_node_recovery)
    
    def _handle_node_failures(self, failed_nodes: List[int]):
        """
        Handle node failures
        Checks safety bound and triggers resharing if needed
        """
        with self.lock:
            if self.is_resharing:
                print(f"Node {self.network.node_id}: Resharing already in progress, ignoring failure")
                return
            
            n_active = self.failure_detector.get_n_active()
            print(f"Node {self.network.node_id}: Handling {len(failed_nodes)} node failure(s), n_active={n_active}")
            
            # Check if safety bound still holds
            if self.safety_checker.check(n_active):
                print(f"Node {self.network.node_id}: Safety bound still satisfied (n_active={n_active}), continuing")
                return
            
            # Safety bound violated
            print(f"Node {self.network.node_id}: Safety bound violated (n_active={n_active}), triggering resharing")
            
            if self.resharing_enabled:
                self._trigger_resharing_required()
            else:
                print(f"Node {self.network.node_id}: Resharing disabled, training will suspend")
    
    def _handle_node_recovery(self, recovered_nodes: List[int]):
        """
        Handle node recoveries
        Checks if safety bound is restored
        """
        n_active = self.failure_detector.get_n_active()
        print(f"Node {self.network.node_id}: {len(recovered_nodes)} node(s) recovered, n_active={n_active}")
        
        if self.safety_checker.check(n_active):
            print(f"Node {self.network.node_id}: Safety bound restored (n_active={n_active})")
        else:
            print(f"Node {self.network.node_id}: Safety bound still violated (n_active={n_active})")
    
    def _trigger_resharing_required(self):
        """Trigger resharing required callback"""
        for callback in self.resharing_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in resharing callback: {e}")
    
    def register_resharing_callback(self, callback: Callable[[], None]):
        """Register callback for when resharing is required"""
        self.resharing_callbacks.append(callback)
    
    def initiate_resharing(self, 
                          old_shares_dict: Dict[str, List[List[List[Share]]]]) -> Dict[str, List[List[List[Share]]]]:
        """
        Initiate resharing for all shares in the dictionary
        Args:
            old_shares_dict: Dictionary mapping keys to weight shares (List[layer][row][col][shares])
        Returns:
            Dictionary with reshared shares
        """
        with self.lock:
            if self.is_resharing:
                raise RuntimeError("Resharing already in progress")
            
            self.is_resharing = True
        
        try:
            old_node_ids = list(self.failure_detector.get_active_nodes())
            new_node_ids = old_node_ids  # For now, keep same nodes (would add new nodes in full implementation)
            
            if len(old_node_ids) < self.t + 1:
                raise ValueError(f"Not enough active nodes for resharing (need >= {self.t + 1}, have {len(old_node_ids)})")
            
            # Create resharer
            resharer = DPSSResharer(
                self.network,
                self.pss.shamir,
                self.t,
                old_node_ids,
                new_node_ids
            )
            
            print(f"Node {self.network.node_id}: Starting resharing for {len(old_shares_dict)} key(s)")
            
            reshared_shares = {}
            for key, shares in old_shares_dict.items():
                try:
                    reshared_shares[key] = resharer.reshare_weight_shares(shares)
                    print(f"Node {self.network.node_id}: Reshared {key}")
                except Exception as e:
                    print(f"Node {self.network.node_id}: Failed to reshare {key}: {e}")
                    reshared_shares[key] = shares  # Use original shares if resharing fails
            
            print(f"Node {self.network.node_id}: Resharing completed")
            return reshared_shares
            
        finally:
            with self.lock:
                self.is_resharing = False
    
    def get_n_active(self) -> int:
        """Get number of active nodes"""
        return self.failure_detector.get_n_active()
    
    def get_active_nodes(self) -> Set[int]:
        """Get set of active nodes"""
        return self.failure_detector.get_active_nodes()


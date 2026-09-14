"""
Node Failure Detection for Multi-Node MPC
Monitors node health using heartbeats and tracks active nodes
"""

from typing import Dict, Set, List, Optional, Callable
from ml_training.secure_comm import SecureMPCNetwork, MessageType
import threading
import time


class NodeFailureDetector:
    """
    Detects node failures using heartbeat protocol
    Maintains set of active nodes and notifies on failures
    """
    
    def __init__(self, network: SecureMPCNetwork, 
                 heartbeat_interval: float = 2.0,
                 failure_timeout: float = 6.0,
                 startup_grace_period: Optional[float] = None):
        """
        Initialize node failure detector
        Args:
            network: SecureMPCNetwork instance
            heartbeat_interval: Time between heartbeat sends (seconds)
            failure_timeout: Time before considering node failed (seconds)
            startup_grace_period: Initial period where failure detection is muted.
                If None, defaults to max(2 * heartbeat_interval, failure_timeout).
        """
        self.network = network
        self.heartbeat_interval = heartbeat_interval
        self.failure_timeout = failure_timeout
        if startup_grace_period is None:
            startup_grace_period = max(2.0 * heartbeat_interval, failure_timeout)
        self.startup_grace_period = float(startup_grace_period)
        self.monitor_start_time = 0.0
        self.last_heartbeat: Dict[int, float] = {}
        self.active_nodes: Set[int] = set(network.node_configs.keys())
        self.running = False
        self.lock = threading.Lock()
        self.failure_callbacks: List[Callable[[List[int]], None]] = []
        self.recovery_callbacks: List[Callable[[List[int]], None]] = []
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._detection_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """Start heartbeat monitoring and failure detection"""
        if self.running:
            return
        
        self.running = True
        self.monitor_start_time = time.time()
        
        # Initialize last heartbeat times for all nodes
        with self.lock:
            current_time = time.time()
            for node_id in self.network.node_configs.keys():
                if node_id != self.network.node_id:
                    self.last_heartbeat[node_id] = current_time
        
        # Start heartbeat sending thread
        self._heartbeat_thread = threading.Thread(
            target=self._send_heartbeats, 
            daemon=True,
            name=f"NodeFailureDetector-Hearbeat-{self.network.node_id}"
        )
        self._heartbeat_thread.start()
        
        # Start failure detection thread
        self._detection_thread = threading.Thread(
            target=self._detect_failures,
            daemon=True,
            name=f"NodeFailureDetector-Detection-{self.network.node_id}"
        )
        self._detection_thread.start()
        
        # Register heartbeat ACK handler
        # Note: ACK messages are already handled by the channel's _handle_heartbeat
        # We'll also register a handler to update our heartbeat tracking
        from ml_training.secure_comm import MessageType
        self.network.channel.register_handler(
            MessageType.ACK.value,
            self._handle_heartbeat_ack
        )
        
        print(f"Node {self.network.node_id}: Started failure detection (interval={self.heartbeat_interval}s, timeout={self.failure_timeout}s)")
    
    def stop_monitoring(self):
        """Stop heartbeat monitoring"""
        self.running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1.0)
        if self._detection_thread:
            self._detection_thread.join(timeout=1.0)
    
    def _send_heartbeats(self):
        """Periodically send heartbeats to all other nodes"""
        while self.running:
            try:
                with self.lock:
                    nodes_to_ping = list(self.active_nodes)
                
                for node_id in nodes_to_ping:
                    if node_id != self.network.node_id:
                        try:
                            self.network.channel.send_message(
                                node_id, 
                                MessageType.HEARTBEAT, 
                                {'sender': self.network.node_id, 'timestamp': time.time()}
                            )
                        except Exception as e:
                            # Connection failed - will be detected by timeout
                            pass
                
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                if self.running:
                    print(f"Node {self.network.node_id}: Error in heartbeat sender: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _detect_failures(self):
        """Detect node failures based on heartbeat timeout"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.monitor_start_time < self.startup_grace_period:
                    time.sleep(self.heartbeat_interval)
                    continue

                failed_nodes = set()
                recovered_nodes = set()
                
                with self.lock:
                    # Check for failed nodes
                    nodes_to_check = list(self.active_nodes)
                    for node_id in nodes_to_check:
                        if node_id != self.network.node_id:
                            last_hb = self.last_heartbeat.get(node_id, 0)
                            if current_time - last_hb > self.failure_timeout:
                                failed_nodes.add(node_id)
                    
                    # Check for recovered nodes (nodes not in active set but recently heard from)
                    all_known_nodes = set(self.network.node_configs.keys())
                    for node_id in all_known_nodes:
                        if node_id != self.network.node_id:
                            last_hb = self.last_heartbeat.get(node_id, 0)
                            if node_id not in self.active_nodes and last_hb > 0:
                                if current_time - last_hb <= self.failure_timeout:
                                    recovered_nodes.add(node_id)
                
                # Handle failures
                if failed_nodes:
                    self.active_nodes -= failed_nodes
                    print(f"Node {self.network.node_id}: Detected {len(failed_nodes)} failed node(s): {failed_nodes}")
                    for callback in self.failure_callbacks:
                        try:
                            callback(list(failed_nodes))
                        except Exception as e:
                            print(f"Error in failure callback: {e}")
                
                # Handle recoveries
                if recovered_nodes:
                    self.active_nodes.update(recovered_nodes)
                    print(f"Node {self.network.node_id}: Detected {len(recovered_nodes)} recovered node(s): {recovered_nodes}")
                    for callback in self.recovery_callbacks:
                        try:
                            callback(list(recovered_nodes))
                        except Exception as e:
                            print(f"Error in recovery callback: {e}")
                
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                if self.running:
                    print(f"Node {self.network.node_id}: Error in failure detection: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _handle_heartbeat_ack(self, sender_id: int, data: Dict):
        """Handle heartbeat ACK message"""
        current_time = time.time()
        with self.lock:
            self.last_heartbeat[sender_id] = current_time
            if sender_id not in self.active_nodes:
                # Node recovered
                self.active_nodes.add(sender_id)
                print(f"Node {self.network.node_id}: Node {sender_id} recovered (received ACK)")
    
    def register_failure_callback(self, callback: Callable[[List[int]], None]):
        """Register callback for node failure events"""
        self.failure_callbacks.append(callback)

    def mark_peer_failed_immediate(self, peer_id: int) -> None:
        """
        Mark peer_id failed without waiting for heartbeat timeout (e.g. send() got connection reset).
        Removes peer from active_nodes and invokes failure callbacks once.
        """
        if int(peer_id) == int(self.network.node_id):
            return
        failed: List[int] = []
        with self.lock:
            if int(peer_id) in self.active_nodes:
                self.active_nodes.discard(int(peer_id))
                failed.append(int(peer_id))
        if not failed:
            return
        fs = set(failed)
        print(f"Node {self.network.node_id}: Detected {len(fs)} failed node(s): {fs}")
        for callback in self.failure_callbacks:
            try:
                callback(list(fs))
            except Exception as e:
                print(f"Error in failure callback: {e}")
    
    def register_recovery_callback(self, callback: Callable[[List[int]], None]):
        """Register callback for node recovery events"""
        self.recovery_callbacks.append(callback)
    
    def get_active_nodes(self) -> Set[int]:
        """Get set of currently active nodes"""
        with self.lock:
            return self.active_nodes.copy()
    
    def get_n_active(self) -> int:
        """Get number of active nodes"""
        with self.lock:
            return len(self.active_nodes)
    
    def is_node_active(self, node_id: int) -> bool:
        """Check if a specific node is active"""
        with self.lock:
            return node_id in self.active_nodes


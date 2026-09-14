"""
Test script for node failure detection
Tests heartbeat monitoring and failure detection
"""

import time
import sys
import os
import pytest
import multiprocessing
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_training.coordinator import SafetyBoundChecker
from ml_training.secure_comm import create_mpc_network
from ml_training.node_failure_detector import NodeFailureDetector


def _run_node_stub(node_id: int, node_configs, stop_event: multiprocessing.Event):
    """Run a lightweight MPC network node that stays alive until stop_event is set."""
    network = None
    try:
        network = create_mpc_network(node_id, node_configs, port=node_configs[node_id]["port"], use_tls=False)
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        if network is not None:
            try:
                network.stop()
            except Exception:
                pass


def test_failure_detection():
    """Test node failure detection with multiple nodes"""
    if os.getenv("SENTRA_RUN_INTERACTIVE", "1") != "1":
        pytest.skip("Disabled via SENTRA_RUN_INTERACTIVE=0.")
    print("=" * 70)
    print("Testing Node Failure Detection")
    print("=" * 70)
    
    # Node configuration
    node_configs = {
        1: {'host': 'localhost', 'port': 8001},
        2: {'host': 'localhost', 'port': 8002},
        3: {'host': 'localhost', 'port': 8003},
    }
    
    # Test parameters
    t = 1  # Threshold
    n_nodes = 3
    
    print(f"\nNode configuration: {node_configs}")
    print(f"Threshold: {t}, Nodes: {n_nodes}")
    print("\nAuto-starting stub nodes on ports 8002, 8003...")
    stop_event = multiprocessing.Event()
    node_procs = []
    network = None
    detector = None
    try:
        for node_id in (2, 3):
            p = multiprocessing.Process(
                target=_run_node_stub,
                args=(node_id, node_configs, stop_event),
                daemon=True,
            )
            p.start()
            node_procs.append(p)
        # Give nodes time to start and connect.
        time.sleep(2.0)

        # Test with node 1
        node_id = 1
        print(f"\nTesting from Node {node_id} perspective...")

        # Create network
        network = create_mpc_network(node_id, node_configs, port=8000 + node_id, use_tls=False)
        time.sleep(1)  # Wait for connections

        # Create failure detector
        detector = NodeFailureDetector(network, heartbeat_interval=2.0, failure_timeout=6.0)
        detector.start_monitoring()

        print(f"\nStarted failure detection on Node {node_id}")
        print(f"Initial active nodes: {detector.get_active_nodes()}")
        print(f"Initial n_active: {detector.get_n_active()}")

        # Monitor for 30 seconds
        print("\nMonitoring for 30 seconds...")
        print("Try stopping one of the other nodes to test failure detection")

        for i in range(30):
            active_nodes = detector.get_active_nodes()
            n_active = detector.get_n_active()
            print(f"[{i+1:2d}s] Active nodes: {sorted(active_nodes)}, n_active: {n_active}")
            time.sleep(1)

        print("\nTest completed")
        print(f"Final active nodes: {detector.get_active_nodes()}")
        print(f"Final n_active: {detector.get_n_active()}")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if detector is not None:
            try:
                detector.stop_monitoring()
            except Exception:
                pass
        if network is not None:
            try:
                network.stop()
            except Exception:
                pass
        stop_event.set()
        for p in node_procs:
            p.join(timeout=2.0)
        for p in node_procs:
            if p.is_alive():
                p.terminate()


def test_safety_bound_with_failures():
    """Test safety bound checking with node failures"""
    print("\n" + "=" * 70)
    print("Testing Safety Bound with Node Failures")
    print("=" * 70)
    
    t = 1  # Threshold
    s = 1  # Adversarial limit
    n_nodes = 5
    
    safety_checker = SafetyBoundChecker(t, s)
    
    print(f"t={t}, s={s}, Safety bound: 2*(t+s-1) = {2*(t+s-1)}")
    print(f"\nSafety bound requires: 2*(t+s-1) < n_active")
    
    # Test different n_active values
    for n_active in range(1, 8):
        is_safe = safety_checker.check(n_active)
        max_packing = safety_checker.get_max_packing_factor(n_active)
        status = "✓ SAFE" if is_safe else "✗ VIOLATED"
        print(f"n_active={n_active}: {status}, max_packing_factor={max_packing}")
    
    print("\nWith n_nodes=5:")
    print("- All 5 nodes active: Safety bound satisfied")
    print("- 4 nodes active: Safety bound satisfied")
    print("- 3 nodes active: Safety bound satisfied")
    print("- 2 nodes active: Safety bound VIOLATED (2*(1+1-1)=2 >= 2)")
    print("- 1 node active: Safety bound VIOLATED")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Node Failure Detection Test Suite")
    print("=" * 70)
    
    # Test 1: Safety bound checking
    test_safety_bound_with_failures()
    
    # Test 2: Actual failure detection (requires running nodes)
    print("\n" + "=" * 70)
    print("To test actual failure detection:")
    print("1. Set SENTRA_RUN_INTERACTIVE=1 to enable the test")
    print("2. Run this script or invoke pytest; the test will auto-start stub nodes")
    print("3. Stop one stub process (if desired) to observe failure detection")
    print("=" * 70)
    
    # Uncomment to run actual failure detection test
    # test_failure_detection()


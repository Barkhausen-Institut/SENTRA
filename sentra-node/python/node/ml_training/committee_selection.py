"""
Committee selection stub for SENTRA (protocol steps 2–4).

Deterministic policy: committee = sorted(live_node_ids).
Heartbeat: use NodeFailureDetector.get_active_nodes() when wired.
Later: plug BFT vote or contract event reader without changing crypto hooks.
"""

from typing import List, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ml_training.node_failure_detector import NodeFailureDetector


def get_committee(live_node_ids: Set[int]) -> List[int]:
    """
    Deterministic committee from live nodes.
    Returns sorted list of active node IDs.
    """
    return sorted(live_node_ids)


def get_committee_from_list(live_node_ids: List[int]) -> List[int]:
    """Same as get_committee but accepts list input."""
    return sorted(set(live_node_ids))


def get_committee_from_detector(detector: "NodeFailureDetector") -> List[int]:
    """
    Get committee from failure detector (heartbeat-based live set).
    Use when NodeFailureDetector is wired and started.
    """
    return get_committee(detector.get_active_nodes())

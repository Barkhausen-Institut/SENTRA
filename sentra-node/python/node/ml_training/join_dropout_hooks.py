"""
Join / dropout hooks for SENTRA (protocol steps 2–4, 6).

On join: barrier, redistribute or receive new shares, bump e.
On dropout: detect n_active, pause training, run DPSS / shrink s, resume when 2(t+s−1) < n_active.

Requires: committee selection (get_committee), failure detector (get_n_active),
          state machine (request_pause / resume), membership epoch (bump e).
"""

from typing import Optional, Callable, Any, Set
from ml_training.committee_selection import get_committee
from ml_training.packing_safety import check_packing_safety


def committee_after_dropout(live_node_ids: Set[int]) -> list:
    """Deterministic committee from remaining live nodes post-dropout."""
    return get_committee(live_node_ids)


def on_dropout_detected(
    n_active: int,
    t: int,
    s: int,
    pause_fn: Callable[[], None],
    resume_fn: Callable[[], None],
    reshare_fn: Optional[Callable[[], Any]] = None,
) -> bool:
    """
    Called when n_active drops (dropout detected).
    If 2(t+s-1) >= n_active: pause, optionally run reshare (DPSS/shrink s), resume when safe.
    Returns True if training should continue, False if should abort.
    """
    if check_packing_safety(t, s, n_active):
        return True  # Still safe, no action
    pause_fn()
    try:
        if reshare_fn:
            reshare_fn()  # Run DPSS to shrink s or redistribute
        # After reshare, caller should re-check n_active and safety before resume
        return True
    except Exception:
        return False
    finally:
        # Caller typically calls resume_fn after confirming safety
        pass


def on_join_detected(
    barrier_fn: Callable[[], None],
    redistribute_fn: Optional[Callable[[], None]] = None,
    bump_epoch_fn: Optional[Callable[[], int]] = None,
) -> None:
    """
    Called when a new node joins.
    Barrier, optionally redistribute/receive shares, bump e.
    """
    barrier_fn()
    if redistribute_fn:
        redistribute_fn()
    if bump_epoch_fn:
        bump_epoch_fn()

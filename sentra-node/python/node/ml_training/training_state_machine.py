"""
Unified training fault model for SENTRA (protocol step 12).

State machine: TRAINING <-> PAUSED_RESHARE
- TRAINING: Normal batch processing
- PAUSED_RESHARE: Suspend training, run DPSS resharing, explicit barriers and logging.
  Connect existing instability/failure detectors to transition to PAUSED_RESHARE.
"""

from enum import Enum
from typing import Optional, Callable
import threading
import time


class TrainingState(Enum):
    TRAINING = "training"
    PAUSED_RESHARE = "paused_reshare"


class TrainingStateMachine:
    """
    One state machine for unified pause/resume with explicit barriers and logging.
    Wire instability detectors and failure detectors to request PAUSED_RESHARE.
    """

    def __init__(self, log_fn: Optional[Callable[[str], None]] = None):
        self._state = TrainingState.TRAINING
        self._lock = threading.Lock()
        self._log = log_fn or (lambda msg: None)
        self._last_pause_reason: Optional[str] = None

    @property
    def state(self) -> TrainingState:
        with self._lock:
            return self._state

    def is_training(self) -> bool:
        return self.state == TrainingState.TRAINING

    def is_paused(self) -> bool:
        return self.state == TrainingState.PAUSED_RESHARE

    def request_pause(self, reason: str = "resharing") -> None:
        """Transition to PAUSED_RESHARE (e.g. from instability or failure detector)."""
        with self._lock:
            if self._state == TrainingState.PAUSED_RESHARE:
                return
            self._state = TrainingState.PAUSED_RESHARE
            self._last_pause_reason = reason
            self._log(f"[STATE] TRAINING -> PAUSED_RESHARE (reason={reason})")

    def get_last_pause_reason(self) -> Optional[str]:
        """Last reason for entering PAUSED_RESHARE (e.g. node_failure, node_rejoin)."""
        with self._lock:
            return self._last_pause_reason

    def resume(self) -> None:
        """Transition back to TRAINING after resharing complete."""
        with self._lock:
            if self._state == TrainingState.TRAINING:
                return
            self._state = TrainingState.TRAINING
            self._log("[STATE] PAUSED_RESHARE -> TRAINING")

    def wait_until_training(self, check_interval: float = 0.1) -> None:
        """
        Block until state is TRAINING.
        Call from training loop when state is PAUSED_RESHARE.
        """
        while self.is_paused():
            time.sleep(check_interval)

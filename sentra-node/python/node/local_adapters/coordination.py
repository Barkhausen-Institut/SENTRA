"""
Local coordination adapter for SENTRA local testing.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any, Deque, Dict, Iterable, List, Mapping


_EVENTS: Dict[str, Deque[Mapping[str, Any]]] = {"membership": deque(), "default": deque()}
_LOCK = Lock()


def notify(event: Mapping[str, Any]) -> None:
    kind = str(event.get("kind", "default"))
    with _LOCK:
        if kind not in _EVENTS:
            _EVENTS[kind] = deque()
        _EVENTS[kind].append(dict(event))


def get_events(kind: str = "default") -> List[Mapping[str, Any]]:
    with _LOCK:
        queue = _EVENTS.get(kind)
        if not queue:
            return []
        out = list(queue)
        queue.clear()
        return out


def listen(kind: str = "default") -> Iterable[Mapping[str, Any]]:
    return get_events(kind=kind)


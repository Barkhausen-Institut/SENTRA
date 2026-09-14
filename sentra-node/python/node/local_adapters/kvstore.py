"""
Local in-memory kvstore adapter for SENTRA local testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional


@dataclass
class _Record:
    value: Any
    version: int


class Client:
    """Minimal kvstore.Client-compatible local implementation."""

    _GLOBAL_STORE: Dict[str, _Record] = {}
    _LOCK = Lock()

    def __init__(self, endpoint: str, credentials: Any) -> None:
        self.endpoint = endpoint
        self.credentials = credentials
        self._security_context: Dict[str, Any] = {}

    def authenticate(self, payload: Dict[str, Any]) -> None:
        self._security_context = dict(payload)

    def set_security_context(self, payload: Dict[str, Any]) -> None:
        self._security_context = dict(payload)

    def get(
        self,
        *,
        key: str,
        min_version: int,
        read_quorum: int,
        epoch_token: str,
        jwt: str,
    ) -> Dict[str, Any]:
        del read_quorum, epoch_token, jwt  # unused in local adapter
        with self._LOCK:
            rec: Optional[_Record] = self._GLOBAL_STORE.get(key)
            if rec is None:
                raise KeyError(f"key not found: {key}")
            if rec.version < min_version:
                return {"value": rec.value, "version": rec.version, "quorum_ok": True}
            return {"value": rec.value, "version": rec.version, "quorum_ok": True}

    def compare_and_set(
        self,
        *,
        key: str,
        old_version: int,
        new_value: Any,
        new_version: int,
        write_quorum: int,
        epoch_token: str,
        jwt: str,
    ) -> Dict[str, Any]:
        del write_quorum, epoch_token, jwt  # unused in local adapter
        with self._LOCK:
            rec = self._GLOBAL_STORE.get(key)
            current_version = rec.version if rec is not None else 0
            if current_version != old_version:
                return {"success": False, "quorum_ok": True}
            if new_version <= current_version:
                return {"success": False, "quorum_ok": True}
            self._GLOBAL_STORE[key] = _Record(value=new_value, version=new_version)
            return {"success": True, "quorum_ok": True}


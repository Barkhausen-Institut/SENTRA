"""Optional runtime metrics for dissertation benchmarks ([BENCHMARK] lines).

Enable with environment variable ``SENTRA_RUNTIME_METRICS=1``.

Tracks per-process:
  - wire bytes sent/received (application-level, incl. framing)
  - CPU% and RSS memory samples (psutil when available)
  - secure training iterations
  - dropout/join recovery wall time
  - weight versioning (KVS persist) wall time
  - epoch loss snapshots around failure recovery
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

from ml_training.benchmark_stats import log_benchmark

_enabled: Optional[bool] = None
_lock = threading.Lock()
_state: Dict[str, Any] = {}


def metrics_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = os.environ.get("SENTRA_RUNTIME_METRICS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    return bool(_enabled)


def _ensure_state() -> None:
    if _state:
        return
    _state.update(
        {
            "bytes_sent": 0,
            "bytes_recv": 0,
            "train_iterations": 0,
            "cpu_samples": [],
            "rss_mb_peak": 0.0,
            "recovery_events": [],
            "versioning_sec_total": 0.0,
            "versioning_calls": 0,
            "loss_snapshots": [],
        }
    )


def record_wire_bytes(*, sent: int = 0, recv: int = 0) -> None:
    if not metrics_enabled():
        return
    with _lock:
        _ensure_state()
        if sent:
            _state["bytes_sent"] = int(_state["bytes_sent"]) + int(sent)
        if recv:
            _state["bytes_recv"] = int(_state["bytes_recv"]) + int(recv)


def inc_training_iteration() -> None:
    if not metrics_enabled():
        return
    with _lock:
        _ensure_state()
        _state["train_iterations"] = int(_state["train_iterations"]) + 1


def add_versioning_sec(seconds: float) -> None:
    if not metrics_enabled():
        return
    with _lock:
        _ensure_state()
        _state["versioning_sec_total"] = float(_state["versioning_sec_total"]) + float(seconds)
        _state["versioning_calls"] = int(_state["versioning_calls"]) + 1


def record_loss_snapshot(*, event: str, epoch: int, loss: Optional[float], n_active: int) -> None:
    if not metrics_enabled():
        return
    with _lock:
        _ensure_state()
        _state["loss_snapshots"].append(
            {
                "event": str(event),
                "epoch": int(epoch),
                "loss": (None if loss is None else float(loss)),
                "n_active": int(n_active),
                "t": time.time(),
            }
        )


def record_recovery_event(
    *,
    kind: str,
    wall_sec: float,
    n_active: int,
    success: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not metrics_enabled():
        return
    with _lock:
        _ensure_state()
        row = {
            "kind": str(kind),
            "wall_sec": float(wall_sec),
            "n_active": int(n_active),
            "success": bool(success),
        }
        if extra:
            row.update(extra)
        _state["recovery_events"].append(row)


def sample_resources() -> None:
    """Sample current process CPU% and RSS (MiB peak)."""
    if not metrics_enabled():
        return
    cpu_pct: Optional[float] = None
    rss_mb: Optional[float] = None
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        cpu_pct = float(proc.cpu_percent(interval=None))
        rss_mb = float(proc.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass
    with _lock:
        _ensure_state()
        if cpu_pct is not None:
            _state["cpu_samples"].append(float(cpu_pct))
        if rss_mb is not None:
            _state["rss_mb_peak"] = max(float(_state["rss_mb_peak"]), float(rss_mb))


def start_resource_sampler(
    *,
    interval_s: Optional[float] = None,
    stop_event: Optional[threading.Event] = None,
) -> Optional[threading.Thread]:
    """Background sampler; returns thread or None if disabled."""
    if not metrics_enabled():
        return None
    if interval_s is None:
        raw = os.environ.get("SENTRA_RESOURCE_SAMPLE_INTERVAL_S", "30")
        try:
            interval_s = max(5.0, float(raw))
        except ValueError:
            interval_s = 30.0
    ev = stop_event or threading.Event()

    def _loop() -> None:
        while not ev.wait(timeout=float(interval_s)):
            sample_resources()

    sample_resources()
    t = threading.Thread(target=_loop, name="sentra-resource-sampler", daemon=True)
    t.start()
    return t


def snapshot() -> Dict[str, Any]:
    with _lock:
        _ensure_state()
        cpu_samples: List[float] = list(_state.get("cpu_samples") or [])
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else None
        peak_cpu = max(cpu_samples) if cpu_samples else None
        iters = max(1, int(_state.get("train_iterations") or 0))
        bytes_sent = int(_state.get("bytes_sent") or 0)
        bytes_recv = int(_state.get("bytes_recv") or 0)
        return {
            "bytes_sent": bytes_sent,
            "bytes_recv": bytes_recv,
            "bytes_total": bytes_sent + bytes_recv,
            "comm_mb_per_iteration": (bytes_sent + bytes_recv) / float(iters) / (1024.0 * 1024.0),
            "train_iterations": int(_state.get("train_iterations") or 0),
            "cpu_avg_pct": avg_cpu,
            "cpu_peak_pct": peak_cpu,
            "rss_mb_peak": float(_state.get("rss_mb_peak") or 0.0),
            "versioning_sec_total": float(_state.get("versioning_sec_total") or 0.0),
            "versioning_calls": int(_state.get("versioning_calls") or 0),
            "recovery_events": list(_state.get("recovery_events") or []),
            "loss_snapshots": list(_state.get("loss_snapshots") or []),
        }


def emit_summary(*, role: str) -> None:
    if not metrics_enabled():
        return
    snap = snapshot()
    extra: Dict[str, Any] = {
        "bytes_sent": int(snap["bytes_sent"]),
        "bytes_recv": int(snap["bytes_recv"]),
        "bytes_total": int(snap["bytes_total"]),
        "comm_mb_per_iter": f"{float(snap['comm_mb_per_iteration']):.6f}",
        "train_iterations": int(snap["train_iterations"]),
        "rss_mb_peak": f"{float(snap['rss_mb_peak']):.3f}",
    }
    if snap.get("cpu_avg_pct") is not None:
        extra["cpu_avg_pct"] = f"{float(snap['cpu_avg_pct']):.2f}"
    if snap.get("cpu_peak_pct") is not None:
        extra["cpu_peak_pct"] = f"{float(snap['cpu_peak_pct']):.2f}"
    if int(snap.get("versioning_calls") or 0) > 0:
        extra["versioning_sec_total"] = f"{float(snap['versioning_sec_total']):.6f}"
        extra["versioning_calls"] = int(snap["versioning_calls"])
    recoveries = snap.get("recovery_events") or []
    if recoveries:
        extra["recovery_count"] = len(recoveries)
        extra["recovery_sec_total"] = f"{sum(float(r['wall_sec']) for r in recoveries):.6f}"
    log_benchmark(role=str(role), phase="runtime_metrics", extra=extra)

    for rec in recoveries:
        log_benchmark(
            role=str(role),
            phase=f"recovery_{rec.get('kind', 'event')}",
            wall_sec=float(rec.get("wall_sec", 0.0)),
            extra={
                "n_active": int(rec.get("n_active", 0)),
                "success": int(bool(rec.get("success"))),
            },
        )

    for i, loss_row in enumerate(snap.get("loss_snapshots") or []):
        loss_val = loss_row.get("loss")
        extra_loss: Dict[str, Any] = {
            "event": str(loss_row.get("event", "")),
            "epoch": int(loss_row.get("epoch", 0)),
            "n_active": int(loss_row.get("n_active", 0)),
        }
        if loss_val is not None:
            extra_loss["loss"] = f"{float(loss_val):.6f}"
        log_benchmark(role=str(role), phase=f"loss_snapshot_{i}", extra=extra_loss)

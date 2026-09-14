"""Unit tests for optional dissertation runtime metrics."""

import os

import pytest


@pytest.fixture(autouse=True)
def _reset_runtime_metrics(monkeypatch):
    import ml_training.runtime_metrics as rm

    monkeypatch.delenv("SENTRA_RUNTIME_METRICS", raising=False)
    rm._enabled = None
    rm._state.clear()
    yield
    rm._enabled = None
    rm._state.clear()


def test_disabled_by_default():
    from ml_training.runtime_metrics import metrics_enabled, record_wire_bytes, snapshot

    assert metrics_enabled() is False
    record_wire_bytes(sent=1000)
    assert snapshot()["bytes_sent"] == 0


def test_wire_bytes_and_iterations(monkeypatch):
    monkeypatch.setenv("SENTRA_RUNTIME_METRICS", "1")
    from ml_training import runtime_metrics as rm

    rm._enabled = None
    rm.record_wire_bytes(sent=100, recv=50)
    rm.inc_training_iteration()
    rm.inc_training_iteration()
    snap = rm.snapshot()
    assert snap["bytes_sent"] == 100
    assert snap["bytes_recv"] == 50
    assert snap["train_iterations"] == 2
    assert snap["comm_mb_per_iteration"] == pytest.approx(150 / 2 / (1024 * 1024))


def test_recovery_and_versioning(monkeypatch):
    monkeypatch.setenv("SENTRA_RUNTIME_METRICS", "1")
    from ml_training import runtime_metrics as rm

    rm._enabled = None
    rm.add_versioning_sec(1.5)
    rm.record_recovery_event(kind="dropout_reshare", wall_sec=2.25, n_active=2, success=True)
    snap = rm.snapshot()
    assert snap["versioning_sec_total"] == pytest.approx(1.5)
    assert len(snap["recovery_events"]) == 1


def test_summarize_from_log_lines():
    from ml_training.benchmark_stats import summarize_runtime_metrics_from_logs

    node_log = (
        '[BENCHMARK] role=node1 phase=runtime_metrics bytes_sent=1000 bytes_recv=2000 '
        'bytes_total=3000 comm_mb_per_iter=0.001 train_iterations=10 rss_mb_peak=512.000 '
        'cpu_avg_pct=12.50 cpu_peak_pct=45.00\n'
        '[BENCHMARK] role=node1 phase=recovery_dropout_reshare wall_sec=1.500000 n_active=2 success=1\n'
    )
    out = summarize_runtime_metrics_from_logs({1: node_log}, "")
    assert 1 in out["nodes"]
    assert out["nodes"][1]["bytes_total"] == "3000"
    assert len(out.get("recovery") or []) == 1

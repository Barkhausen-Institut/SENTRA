"""Wall-clock helpers for benchmark logs ([BENCHMARK] lines)."""

from __future__ import annotations

import re
from typing import Any, Dict

_BENCHMARK_RE = re.compile(
    r"\[BENCHMARK\]\s+(.+)$",
    re.MULTILINE,
)
_KV_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)")


def log_benchmark(
    *,
    role: str,
    phase: str,
    wall_sec: float | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """Emit a single parseable benchmark line to stdout (captured in headless logs)."""
    parts = [f"role={role}", f"phase={phase}"]
    if wall_sec is not None:
        parts.append(f"wall_sec={float(wall_sec):.6f}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")
    print("[BENCHMARK] " + " ".join(parts), flush=True)


def parse_benchmark_lines(log_text: str) -> list[Dict[str, str]]:
    rows: list[Dict[str, str]] = []
    for m in _BENCHMARK_RE.finditer(log_text or ""):
        row: Dict[str, str] = {}
        for km in _KV_RE.finditer(m.group(1)):
            row[km.group(1)] = km.group(2)
        if row:
            rows.append(row)
    return rows


def parse_legacy_timings(log_text: str) -> Dict[str, float]:
    """Fallback: scrape legacy timing lines if [BENCHMARK] is absent."""
    out: Dict[str, float] = {}
    patterns = [
        (r"Training Time:\s*([0-9.]+)s", "training_sec"),
        (r"Dataset Share Prep Time:\s*([0-9.]+)s", "dataset_prep_sec"),
        (r"Client Distribution Time:\s*([0-9.]+)s", "client_distribution_sec"),
        (r"Client Eval Time:\s*([0-9.]+)s", "client_eval_sec"),
        (r"Client Eval Share Wait Time:\s*([0-9.]+)s", "client_eval_share_wait_sec"),
        (r"Client Eval Reconstruct Time:\s*([0-9.]+)s", "client_eval_reconstruct_sec"),
        (r"Client Eval Upload Time:\s*([0-9.]+)s", "client_eval_upload_sec"),
        (r"Prover Time:\s*([0-9.]+)s", "prover_sec"),
    ]
    for pat, key in patterns:
        hits = re.findall(pat, log_text or "")
        if hits:
            out[key] = float(hits[-1])
    return out


def summarize_run_logs(
    *,
    node_logs: Dict[int, str],
    client_log: str,
    wall_sec: float,
) -> Dict[str, Any]:
    """Build timing summary for start_all_nodes Run summary."""
    summary: Dict[str, Any] = {"wall_sec": float(wall_sec)}
    timings: Dict[str, float] = {}

    for nid, text in sorted(node_logs.items()):
        for row in parse_benchmark_lines(text):
            phase = row.get("phase", f"node{nid}")
            if "wall_sec" in row:
                timings.setdefault(f"node{nid}_{phase}_sec", float(row["wall_sec"]))
        legacy = parse_legacy_timings(text)
        for k, v in legacy.items():
            timings.setdefault(f"node{nid}_{k}", v)
            if k == "training_sec":
                timings.setdefault("training_sec", v)

    for row in parse_benchmark_lines(client_log):
        phase = row.get("phase", "client")
        if "wall_sec" in row:
            timings.setdefault(f"client_{phase}_sec", float(row["wall_sec"]))
    for k, v in parse_legacy_timings(client_log).items():
        ck = k if k.startswith("client_") else f"client_{k}"
        timings.setdefault(ck, v)

    summary["timings"] = timings
    summary["runtime_metrics"] = summarize_runtime_metrics_from_logs(node_logs, client_log)
    return summary


def summarize_runtime_metrics_from_logs(
    node_logs: Dict[int, str],
    client_log: str,
) -> Dict[str, Any]:
    """Aggregate [BENCHMARK] phase=runtime_metrics* lines across roles."""
    out: Dict[str, Any] = {"nodes": {}, "client": {}}
    for nid, text in sorted(node_logs.items()):
        for row in parse_benchmark_lines(text):
            phase = row.get("phase", "")
            if phase == "runtime_metrics":
                out["nodes"][nid] = dict(row)
    for row in parse_benchmark_lines(client_log):
        if row.get("phase") == "runtime_metrics":
            out["client"] = dict(row)
    return out


def build_run_overhead_metrics(
    timings: Dict[str, float],
    *,
    end_to_end_sec: float,
    pre_spawn_orchestration_sec: float,
    spawn_to_last_node_exit_sec: float,
    parallel_run_sec: float,
    n_nodes: int,
    client_present: bool,
) -> Dict[str, float]:
    """Orchestrator vs per-process logged timings (cold start + MPC work)."""
    role_paths: list[float] = []
    cold_starts: list[float] = []
    mpc_totals: list[float] = []

    for nid in range(1, int(n_nodes) + 1):
        cold = timings.get(f"node{nid}_cold_start_sec")
        total = timings.get(f"node{nid}_total_sec")
        if cold is not None:
            cold_starts.append(float(cold))
        if total is not None:
            mpc_totals.append(float(total))
        if cold is not None and total is not None:
            role_paths.append(float(cold) + float(total))

    if client_present:
        cold = timings.get("client_cold_start_sec")
        total = timings.get("client_total_sec")
        if cold is not None:
            cold_starts.append(float(cold))
        if total is not None:
            mpc_totals.append(float(total))
        if cold is not None and total is not None:
            role_paths.append(float(cold) + float(total))

    critical_path = max(role_paths) if role_paths else 0.0
    max_cold = max(cold_starts) if cold_starts else 0.0
    max_mpc = max(mpc_totals) if mpc_totals else 0.0
    expected_end_to_end = float(pre_spawn_orchestration_sec) + critical_path

    return {
        "end_to_end_sec": float(end_to_end_sec),
        "pre_spawn_orchestration_sec": float(pre_spawn_orchestration_sec),
        "parallel_run_sec": float(parallel_run_sec),
        "spawn_to_last_node_exit_sec": float(spawn_to_last_node_exit_sec),
        "max_cold_start_sec": max_cold,
        "max_mpc_work_sec": max_mpc,
        "critical_path_logged_sec": critical_path,
        "post_log_process_exit_sec": max(
            0.0, float(spawn_to_last_node_exit_sec) - critical_path
        ),
        "unaccounted_orchestrator_sec": max(
            0.0, float(end_to_end_sec) - expected_end_to_end
        ),
    }

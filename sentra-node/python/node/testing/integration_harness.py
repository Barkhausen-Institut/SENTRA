import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


ProcEntry = Tuple[int, subprocess.Popen]


def integration_timeout_seconds() -> float:
    """
    Wall-clock budget for multi-node subprocess integration tests.

    Default 900s: 3× TensorFlow + MPC on WSL or /mnt/c can exceed 7 minutes.
    Override with SENTRA_INTEGRATION_TIMEOUT_SEC (e.g. 1200 on slow CI).
    """
    raw = os.environ.get("SENTRA_INTEGRATION_TIMEOUT_SEC", "900").strip()
    try:
        v = float(raw)
    except ValueError:
        v = 900.0
    return max(60.0, v)


def wait_all_processes(
    procs: Sequence[subprocess.Popen],
    *,
    timeout_sec: Optional[float] = None,
    poll_interval_s: float = 0.5,
) -> None:
    """
    Wait until every process exits. If any exits non-zero, raise immediately (avoids
    barrier deadlock where survivors wait until pytest timeout).
    """
    budget = float(timeout_sec) if timeout_sec is not None else integration_timeout_seconds()
    deadline = time.time() + budget
    pending = list(procs)
    while pending:
        now = time.time()
        if now >= deadline:
            raise AssertionError(
                f"Timed out after {budget:.0f}s waiting for node processes. "
                "Try SENTRA_INTEGRATION_TIMEOUT_SEC=1200, run from native Linux disk (not /mnt/c), "
                "or inspect testing/tmp_bench/*.log and *.err for a stuck or crashed node."
            )
        for p in list(pending):
            code = p.poll()
            if code is None:
                continue
            pending.remove(p)
            if int(code) != 0:
                raise AssertionError(
                    f"Node subprocess exited early with code {int(code)} "
                    f"(remaining peers may be stuck at a barrier; check testing/tmp_bench/*.err)."
                )
        if pending:
            time.sleep(float(poll_interval_s))


def _cleanup_legacy_backslash_tmp_bench_entries(log_dir: Path) -> None:
    """
    Remove legacy artifacts accidentally created as filenames containing backslashes,
    e.g. testing/'tmp_bench\\node1.log' on POSIX.
    """
    testing_dir = log_dir.parent
    for legacy in testing_dir.glob("tmp_bench\\*"):
        try:
            if legacy.is_file():
                legacy.unlink()
        except Exception:
            pass


def start_node_processes(
    *,
    root: Path,
    log_dir: Path,
    common_args: Sequence[str],
    node_ids: Iterable[int],
    log_prefix: str,
    startup_stagger_s: float = 0.5,
) -> List[ProcEntry]:
    _cleanup_legacy_backslash_tmp_bench_entries(log_dir)
    procs: List[ProcEntry] = []
    for node_id in node_ids:
        out_path = log_dir / f"node{node_id}_{log_prefix}.log"
        err_path = log_dir / f"node{node_id}_{log_prefix}.err"
        with open(out_path, "w", encoding="utf-8") as out_f, open(
            err_path, "w", encoding="utf-8"
        ) as err_f:
            p = subprocess.Popen(
                [sys.executable, *common_args, "--node-id", str(node_id)],
                cwd=str(root),
                stdout=out_f,
                stderr=err_f,
            )
            procs.append((int(node_id), p))
        time.sleep(float(startup_stagger_s))
    return procs


def terminate_processes(
    procs: Sequence[ProcEntry],
    *,
    terminate_timeout_s: float = 2.0,
) -> None:
    for _, p in procs:
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=float(terminate_timeout_s))
            except Exception:
                pass
    for _, p in procs:
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass


def read_node_log(log_dir: Path, *, node_id: int, log_prefix: str) -> str:
    return (log_dir / f"node{node_id}_{log_prefix}.log").read_text(
        encoding="utf-8", errors="ignore"
    )


def read_node_err(log_dir: Path, *, node_id: int, log_prefix: str) -> str:
    return (log_dir / f"node{node_id}_{log_prefix}.err").read_text(
        encoding="utf-8", errors="ignore"
    )


def get_first_failed_process(procs: Sequence[ProcEntry]) -> ProcEntry | None:
    for node_id, p in procs:
        if p.poll() is not None and int(p.returncode) != 0:
            return node_id, p
    return None

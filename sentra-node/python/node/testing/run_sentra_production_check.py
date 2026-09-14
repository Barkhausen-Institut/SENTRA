"""
Production checklist runner for SENTRA node runtime.

This script validates the core production concerns against real modules/endpoints:
1) KVS auth + version-aware read + CAS monotonicity
2) Safety-bound breach handling + resharing trigger path
3) Packed-MPC engine capability checks (+ optional tiny functional smoke)
4) Evaluation authorization hook presence/behavior

It does not require local adapters and is intended for pre-deploy validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
import time
import traceback
import sys
from typing import Any, Dict, List, Mapping, Optional

# Ensure repo root is importable when running from testing/ directory.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_training.sentra_training_node import (
    KVSOperationError,
    NodeIdentity,
    QuorumConfig,
    RetryConfig,
    SentraTrainingNode,
    StaleVersionError,
    TrainingConfig,
    VersionedKVSClient,
)

import coordination
import kvstore
import mpc


LOGGER = logging.getLogger("sentra.prodcheck")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    elapsed_s: float


def _credentials_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    creds: Dict[str, Any] = {}
    if args.mtls_cert:
        creds["cert"] = args.mtls_cert
    if args.mtls_key:
        creds["key"] = args.mtls_key
    if args.mtls_ca:
        creds["ca"] = args.mtls_ca
    if args.credentials_json:
        creds.update(json.loads(args.credentials_json))
    return creds


def _seed_key(
    raw_client: Any,
    key: str,
    value: Any,
    version: int,
    *,
    write_quorum: int,
    epoch_token: str,
    epoch_jwt: str,
) -> bool:
    result = raw_client.compare_and_set(
        key=key,
        old_version=version - 1,
        new_value=value,
        new_version=version,
        write_quorum=write_quorum,
        epoch_token=epoch_token,
        jwt=epoch_jwt,
    )
    if isinstance(result, Mapping):
        return bool(result.get("success", False))
    return bool(getattr(result, "success", bool(result)))


def check_kvs(identity: NodeIdentity, endpoint: str, quorum: QuorumConfig, retry: RetryConfig, prefix: str) -> CheckResult:
    t0 = time.time()
    key = f"{prefix}/kvs_probe"
    try:
        client = VersionedKVSClient(endpoint, identity, quorum, retry)
        ok = client.compare_and_set(key, old_version=0, new_value={"probe": True}, new_version=1)
        if not ok:
            return CheckResult("kvs_auth_get_cas", False, "initial CAS returned False", time.time() - t0)
        val = client.get(key, min_version=1)
        if val.version != 1:
            return CheckResult("kvs_auth_get_cas", False, f"unexpected version: {val.version}", time.time() - t0)
        try:
            client.get(key, min_version=2)
            return CheckResult("kvs_auth_get_cas", False, "expected stale version error, got success", time.time() - t0)
        except StaleVersionError:
            pass
        return CheckResult("kvs_auth_get_cas", True, "auth + CAS + version checks passed", time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("kvs_auth_get_cas", False, f"{type(exc).__name__}: {exc}", time.time() - t0)


def check_safety_and_reshare(
    identity: NodeIdentity,
    endpoint: str,
    quorum: QuorumConfig,
    retry: RetryConfig,
    cfg: TrainingConfig,
    prefix: str,
) -> CheckResult:
    t0 = time.time()
    try:
        node = SentraTrainingNode(
            kvs_endpoint=endpoint,
            identity=identity,
            quorum=quorum,
            retry=retry,
            config=cfg,
        )
        # Deliberately evaluate a breached safety bound scenario.
        breached_n_active = max(1, 2 * (cfg.t + cfg.packing_factor - 1))
        if node._safety_bound_holds(cfg.t, cfg.packing_factor, breached_n_active):
            return CheckResult("safety_reshare", False, "safety bound unexpectedly holds for breached case", time.time() - t0)

        node._request_resharing("prodcheck_safety_breach")
        if node.membership.is_training_allowed():
            return CheckResult("safety_reshare", False, "membership did not enter reshare pause", time.time() - t0)

        # Simulate reshare completion path.
        node.membership.mark_reshare_complete()
        if not node.membership.is_training_allowed():
            return CheckResult("safety_reshare", False, "membership did not resume after completion", time.time() - t0)

        coordination.notify(
            {
                "event": "prodcheck_safety_reshare_verified",
                "kind": "default",
                "node_id": cfg.node_id,
                "timestamp": time.time(),
            }
        )
        return CheckResult("safety_reshare", True, "safety breach + resharing pause/resume path verified", time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("safety_reshare", False, f"{type(exc).__name__}: {exc}", time.time() - t0)


def check_engine(num_gpus: int) -> CheckResult:
    t0 = time.time()
    try:
        engine = mpc.PackedEngine(num_gpus)
        required_any = [
            ("pack", ["pack_shares", "to_packed_shamir", "pack_inputs"]),
            ("forward", ["forward", "forward_pass", "infer"]),
            ("backward", ["backward", "compute_gradients", "backward_pass"]),
            ("aggregate", ["secure_aggregate_gradients", "aggregate_gradients", "secure_aggregate"]),
            ("update", ["apply_gradients", "update_model", "sgd_step"]),
        ]
        missing: List[str] = []
        for label, names in required_any:
            if not any(callable(getattr(engine, n, None)) for n in names):
                missing.append(label)
        if missing:
            return CheckResult("packed_mpc_engine", False, f"missing capability groups: {missing}", time.time() - t0)
        return CheckResult("packed_mpc_engine", True, "engine capability groups present", time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("packed_mpc_engine", False, f"{type(exc).__name__}: {exc}", time.time() - t0)


def check_eval_authz(identity: NodeIdentity, node_id: int) -> CheckResult:
    t0 = time.time()
    try:
        auth_fn_names = ["authorize_evaluation", "is_authorized", "authorize"]
        auth_fn = None
        for name in auth_fn_names:
            cand = getattr(coordination, name, None)
            if callable(cand):
                auth_fn = cand
                break
        if auth_fn is None:
            return CheckResult(
                "evaluation_authz",
                False,
                "no coordination authorization hook found (expected one of authorize_evaluation/is_authorized/authorize)",
                time.time() - t0,
            )

        # Best-effort invocation; API shapes differ by deployment.
        allowed = auth_fn(node_id=node_id, enclave_id=identity.enclave_id, action="evaluate")
        if isinstance(allowed, Mapping):
            ok = bool(allowed.get("allowed", False))
        else:
            ok = bool(allowed)
        if not ok:
            return CheckResult("evaluation_authz", False, "authorization hook denied evaluation for current identity", time.time() - t0)
        return CheckResult("evaluation_authz", True, "authorization hook present and allowed", time.time() - t0)
    except TypeError:
        # Try alternate positional form.
        try:
            alt_fn = getattr(coordination, "authorize", None)
            if callable(alt_fn):
                ok = bool(alt_fn(identity.enclave_id, "evaluate"))
                return CheckResult("evaluation_authz", ok, f"authorize(enclave, action) returned {ok}", time.time() - t0)
            return CheckResult("evaluation_authz", False, "authorization hook signature mismatch", time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            return CheckResult("evaluation_authz", False, f"{type(exc).__name__}: {exc}", time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("evaluation_authz", False, f"{type(exc).__name__}: {exc}", time.time() - t0)


def optional_train_smoke(
    identity: NodeIdentity,
    endpoint: str,
    quorum: QuorumConfig,
    retry: RetryConfig,
    cfg: TrainingConfig,
    prefix: str,
    input_dim: int,
) -> CheckResult:
    t0 = time.time()
    try:
        raw = kvstore.Client(endpoint, identity.mtls_credentials)
        auth_fn = getattr(raw, "authenticate", None)
        if callable(auth_fn):
            auth_fn({"enclave_id": identity.enclave_id, "epoch_token": identity.epoch_token, "jwt": identity.epoch_jwt})

        model = {"W": [[0.0] for _ in range(input_dim)], "b": [0.0]}
        if not _seed_key(
            raw,
            cfg.model_key,
            model,
            1,
            write_quorum=quorum.write_quorum,
            epoch_token=identity.epoch_token,
            epoch_jwt=identity.epoch_jwt,
        ):
            return CheckResult("train_smoke", False, "failed to seed model key", time.time() - t0)
        if not _seed_key(
            raw,
            cfg.model_version_key,
            1,
            1,
            write_quorum=quorum.write_quorum,
            epoch_token=identity.epoch_token,
            epoch_jwt=identity.epoch_jwt,
        ):
            return CheckResult("train_smoke", False, "failed to seed model version key", time.time() - t0)

        # Seed one minibatch.
        in_key = f"{cfg.dataset_prefix}/batch/0/inputs"
        tg_key = f"{cfg.dataset_prefix}/batch/0/targets"
        x = [[0.0 for _ in range(input_dim)] for _ in range(cfg.minibatch_size)]
        y = [[0.0] for _ in range(cfg.minibatch_size)]
        if not _seed_key(raw, in_key, x, 1, write_quorum=quorum.write_quorum, epoch_token=identity.epoch_token, epoch_jwt=identity.epoch_jwt):
            return CheckResult("train_smoke", False, "failed to seed input batch", time.time() - t0)
        if not _seed_key(raw, tg_key, y, 1, write_quorum=quorum.write_quorum, epoch_token=identity.epoch_token, epoch_jwt=identity.epoch_jwt):
            return CheckResult("train_smoke", False, "failed to seed target batch", time.time() - t0)

        node = SentraTrainingNode(
            kvs_endpoint=endpoint,
            identity=identity,
            quorum=quorum,
            retry=retry,
            config=cfg,
        )
        node.train()
        return CheckResult("train_smoke", True, "single-step train path executed", time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("train_smoke", False, f"{type(exc).__name__}: {exc}", time.time() - t0)


def main() -> int:
    parser = argparse.ArgumentParser(description="SENTRA production checklist runner")
    parser.add_argument("--kvs-endpoint", required=True)
    parser.add_argument("--enclave-id", required=True)
    parser.add_argument("--epoch-token", required=True)
    parser.add_argument("--epoch-jwt", required=True)
    parser.add_argument("--mtls-cert")
    parser.add_argument("--mtls-key")
    parser.add_argument("--mtls-ca")
    parser.add_argument("--credentials-json", help="Additional credentials JSON blob")
    parser.add_argument("--read-quorum", type=int, default=2)
    parser.add_argument("--write-quorum", type=int, default=2)
    parser.add_argument("--node-id", type=int, default=1)
    parser.add_argument("--n-nodes", type=int, default=5)
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--packing-factor", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--minibatch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--prefix", default=f"prodcheck/{int(time.time())}")
    parser.add_argument("--run-train-smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    identity = NodeIdentity(
        enclave_id=args.enclave_id,
        epoch_token=args.epoch_token,
        epoch_jwt=args.epoch_jwt,
        mtls_credentials=_credentials_from_args(args),
    )
    quorum = QuorumConfig(read_quorum=args.read_quorum, write_quorum=args.write_quorum)
    retry = RetryConfig()
    cfg = TrainingConfig(
        node_id=args.node_id,
        n_nodes=args.n_nodes,
        t=args.t,
        packing_factor=args.packing_factor,
        model_key=f"{args.prefix}/model/params",
        model_version_key=f"{args.prefix}/model/version",
        dataset_prefix=f"{args.prefix}/dataset/train",
        minibatch_size=args.minibatch_size,
        max_steps=args.max_steps,
        num_gpus=args.num_gpus,
    )

    results: List[CheckResult] = []
    results.append(check_kvs(identity, args.kvs_endpoint, quorum, retry, args.prefix))
    results.append(check_safety_and_reshare(identity, args.kvs_endpoint, quorum, retry, cfg, args.prefix))
    results.append(check_engine(args.num_gpus))
    results.append(check_eval_authz(identity, args.node_id))
    if args.run_train_smoke:
        results.append(optional_train_smoke(identity, args.kvs_endpoint, quorum, retry, cfg, args.prefix, args.input_dim))

    print("=" * 80)
    print("SENTRA Production Checklist")
    print("=" * 80)
    all_ok = True
    for res in results:
        status = "PASS" if res.ok else "FAIL"
        all_ok = all_ok and res.ok
        print(f"[{status}] {res.name} ({res.elapsed_s:.2f}s) :: {res.detail}")
    print("=" * 80)
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    print("=" * 80)
    return 0 if all_ok else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(3)

"""
Production-grade SENTRA training node runtime.

Scope:
- KVS client with mTLS + epoch JWT auth, version-aware reads, CAS writes, retries, quorum checks
- Packed Shamir minibatch workflow with safety-bound enforcement
- Packed-MPC forward/backward/secure-aggregation execution
- Monotonic model-version update loop via KVS compare-and-set
- Membership-change handling with DPSS resharing
- Evaluation API returning encrypted predictions

Assumed external libraries:
    kvstore.Client(endpoint, credentials)
    mpc.PackedEngine(num_gpus)
    coordination.notify(event)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import coordination
import kvstore
import mpc


LOGGER = logging.getLogger(__name__)


class KVSOperationError(RuntimeError):
    """Raised when a KVS operation fails after retries."""


class StaleVersionError(KVSOperationError):
    """Raised when the returned version is older than requested."""


class QuorumError(KVSOperationError):
    """Raised when quorum constraints are not satisfied."""


class MembershipEventType(str, Enum):
    """Membership event types for training coordination."""

    JOIN = "join"
    DROP = "drop"
    RESHARE_COMPLETE = "reshare_complete"


@dataclass(frozen=True)
class NodeIdentity:
    """Validated identity and auth material supplied by the platform team."""

    enclave_id: str
    epoch_token: str
    epoch_jwt: str
    mtls_credentials: Any


@dataclass(frozen=True)
class QuorumConfig:
    """Read/write quorum thresholds defined by SENTRA deployment."""

    read_quorum: int
    write_quorum: int


@dataclass(frozen=True)
class RetryConfig:
    """Retry behavior for transient failures."""

    max_attempts: int = 5
    base_delay_s: float = 0.2
    max_delay_s: float = 2.0
    jitter_s: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    """Node-local training configuration."""

    node_id: int
    n_nodes: int
    t: int
    s: int
    packing_factor: int
    model_key: str
    model_version_key: str
    dataset_prefix: str
    minibatch_size: int
    max_steps: int
    num_gpus: int = 0
    learning_rate: float = 0.05
    membership_poll_interval_s: float = 1.0


@dataclass(frozen=True)
class VersionedValue:
    """Versioned value read from KVS."""

    key: str
    value: Any
    version: int


@dataclass(frozen=True)
class MiniBatchShares:
    """Secret-shared data for one minibatch."""

    sample_keys: Sequence[str]
    input_shares: Sequence[Any]
    target_shares: Sequence[Any]
    version: int


class VersionedKVSClient:
    """Secure KVS wrapper with auth, quorum enforcement, and retries."""

    def __init__(
        self,
        endpoint: str,
        identity: NodeIdentity,
        quorum: QuorumConfig,
        retry: RetryConfig,
    ) -> None:
        self.endpoint = endpoint
        self.identity = identity
        self.quorum = quorum
        self.retry = retry
        self._client = kvstore.Client(endpoint, identity.mtls_credentials)
        self._authenticate()

    def _authenticate(self) -> None:
        """Perform epoch-bound auth handshake with KVS."""
        auth_payload = {
            "enclave_id": self.identity.enclave_id,
            "epoch_token": self.identity.epoch_token,
            "jwt": self.identity.epoch_jwt,
        }
        authenticate_fn = getattr(self._client, "authenticate", None)
        if callable(authenticate_fn):
            authenticate_fn(auth_payload)
        else:
            # Fallback for clients that accept token configuration instead of explicit auth call.
            set_ctx_fn = getattr(self._client, "set_security_context", None)
            if callable(set_ctx_fn):
                set_ctx_fn(auth_payload)
            else:
                LOGGER.warning("KVS client has no explicit auth hook; relying on constructor credentials only")

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        transient_types = (TimeoutError, ConnectionError, OSError)
        if isinstance(exc, transient_types):
            return True
        msg = str(exc).lower()
        transient_tokens = ("timeout", "temporar", "unavailable", "reset by peer", "connection refused")
        return any(token in msg for token in transient_tokens)

    def _with_retries(self, op_name: str, fn: Callable[[], Any]) -> Any:
        delay = self.retry.base_delay_s
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not self._is_transient(exc) or attempt == self.retry.max_attempts:
                    break
                sleep_s = min(self.retry.max_delay_s, delay + random.uniform(0.0, self.retry.jitter_s))
                LOGGER.warning("%s transient failure (attempt=%d/%d): %s", op_name, attempt, self.retry.max_attempts, exc)
                time.sleep(sleep_s)
                delay = min(self.retry.max_delay_s, delay * 2.0)
        raise KVSOperationError(f"{op_name} failed after retries: {last_exc}") from last_exc

    def get(self, key: str, min_version: int) -> VersionedValue:
        """Version-aware quorum read."""

        def _op() -> VersionedValue:
            result = self._client.get(
                key=key,
                min_version=min_version,
                read_quorum=self.quorum.read_quorum,
                epoch_token=self.identity.epoch_token,
                jwt=self.identity.epoch_jwt,
            )
            if isinstance(result, Mapping):
                version = int(result.get("version", -1))
                value = result.get("value")
                quorum_ok = bool(result.get("quorum_ok", True))
            else:
                version = int(getattr(result, "version"))
                value = getattr(result, "value")
                quorum_ok = bool(getattr(result, "quorum_ok", True))

            if version < min_version:
                raise StaleVersionError(f"read key={key} version={version} < min_version={min_version}")
            if not quorum_ok:
                raise QuorumError(f"read quorum not met for key={key}")
            return VersionedValue(key=key, value=value, version=version)

        return self._with_retries(f"kvs.get({key})", _op)

    def compare_and_set(self, key: str, old_version: int, new_value: Any, new_version: int) -> bool:
        """CAS write with monotonic versioning and quorum guarantees."""

        def _op() -> bool:
            result = self._client.compare_and_set(
                key=key,
                old_version=old_version,
                new_value=new_value,
                new_version=new_version,
                write_quorum=self.quorum.write_quorum,
                epoch_token=self.identity.epoch_token,
                jwt=self.identity.epoch_jwt,
            )
            if isinstance(result, Mapping):
                success = bool(result.get("success", False))
                quorum_ok = bool(result.get("quorum_ok", True))
            else:
                success = bool(getattr(result, "success", bool(result)))
                quorum_ok = bool(getattr(result, "quorum_ok", True))

            if success and not quorum_ok:
                raise QuorumError(f"write quorum not met for key={key}")
            return success

        return bool(self._with_retries(f"kvs.compare_and_set({key})", _op))


class MembershipTracker:
    """Tracks active membership and blocks training during mandatory resharing."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._active_nodes: Set[int] = set(range(1, config.n_nodes + 1))
        self._reshare_in_progress = threading.Event()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._poll_membership_events, daemon=True, name="membership-tracker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def active_count(self) -> int:
        with self._lock:
            return len(self._active_nodes)

    def is_training_allowed(self) -> bool:
        return not self._reshare_in_progress.is_set()

    def wait_until_allowed(self, sleep_s: float = 0.2) -> None:
        while self._reshare_in_progress.is_set():
            time.sleep(sleep_s)

    def set_reshare_needed(self) -> None:
        self._reshare_in_progress.set()

    def mark_reshare_complete(self) -> None:
        self._reshare_in_progress.clear()

    def _poll_membership_events(self) -> None:
        """
        Poll membership updates from coordination service.
        Expected adapters:
          - coordination.get_events(kind="membership")
          - coordination.listen(kind="membership")
        """
        while not self._stop.is_set():
            try:
                events: Iterable[Mapping[str, Any]] = ()
                listen = getattr(coordination, "get_events", None)
                if callable(listen):
                    events = listen(kind="membership")
                else:
                    stream = getattr(coordination, "listen", None)
                    if callable(stream):
                        events = stream(kind="membership")
                for event in events:
                    self._apply_event(event)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("membership polling error: %s", exc)
            time.sleep(self.config.membership_poll_interval_s)

    def _apply_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type", "")).lower()
        node_id = int(event.get("node_id", 0))
        with self._lock:
            if event_type == MembershipEventType.JOIN.value:
                self._active_nodes.add(node_id)
                self._reshare_in_progress.set()
            elif event_type == MembershipEventType.DROP.value:
                self._active_nodes.discard(node_id)
                self._reshare_in_progress.set()
            elif event_type == MembershipEventType.RESHARE_COMPLETE.value:
                self._reshare_in_progress.clear()


class SentraTrainingNode:
    """End-to-end SENTRA training node runtime."""

    def __init__(
        self,
        *,
        kvs_endpoint: str,
        identity: NodeIdentity,
        quorum: QuorumConfig,
        retry: RetryConfig,
        config: TrainingConfig,
    ) -> None:
        self.config = config
        self.kvs = VersionedKVSClient(kvs_endpoint, identity, quorum, retry)
        self.engine = mpc.PackedEngine(config.num_gpus)
        if hasattr(self.engine, "learning_rate"):
            try:
                setattr(self.engine, "learning_rate", float(config.learning_rate))
            except Exception:  # noqa: BLE001
                LOGGER.warning("unable to set engine learning_rate=%s", config.learning_rate)
        self.membership = MembershipTracker(config)
        self._model_version = 0
        self._model_params: Any = None

    @staticmethod
    def _safety_bound_holds(t: int, s: int, n_active: int) -> bool:
        return 2 * (t + s - 1) < n_active

    def _call_engine(self, names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        for name in names:
            fn = getattr(self.engine, name, None)
            if callable(fn):
                return fn(*args, **kwargs)
        raise RuntimeError(f"PackedEngine missing expected methods: {names}")

    def _request_resharing(self, reason: str) -> None:
        self.membership.set_reshare_needed()
        coordination.notify(
            {
                "event": "reshare_requested",
                "reason": reason,
                "node_id": self.config.node_id,
                "active_nodes": self.membership.active_count(),
                "timestamp": time.time(),
            }
        )
        # Invoke DPSS resharing API if runtime exposes one.
        try:
            self._call_engine(["dpss_reshare", "reshare"], reason=reason)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("DPSS resharing API call failed/unsupported: %s", exc)

    def _load_model(self) -> Tuple[Any, int]:
        # Authoritative source of truth is model_key version.
        model_val = self.kvs.get(self.config.model_key, min_version=self._model_version)
        model_version = model_val.version

        # Best-effort mirror sync for model_version_key (non-authoritative).
        try:
            version_val = self.kvs.get(self.config.model_version_key, min_version=0)
            mirror_version = int(version_val.value)
        except Exception:
            mirror_version = 0
        if mirror_version < model_version:
            try:
                self.kvs.compare_and_set(
                    key=self.config.model_version_key,
                    old_version=mirror_version,
                    new_value=model_version,
                    new_version=model_version,
                )
            except Exception:
                pass
        return model_val.value, model_version

    def _load_minibatch(self, step_idx: int) -> MiniBatchShares:
        """
        Fetch minibatch shares from KVS.
        Expects keys:
          {dataset_prefix}/batch/{step_idx}/inputs
          {dataset_prefix}/batch/{step_idx}/targets
        """
        base = f"{self.config.dataset_prefix}/batch/{step_idx}"
        input_key = f"{base}/inputs"
        target_key = f"{base}/targets"
        inp = self.kvs.get(input_key, min_version=0)
        tgt = self.kvs.get(target_key, min_version=0)
        version = max(inp.version, tgt.version)
        sample_keys = [input_key, target_key]
        return MiniBatchShares(sample_keys=sample_keys, input_shares=inp.value, target_shares=tgt.value, version=version)

    def _pack_batch(self, batch: MiniBatchShares) -> Tuple[Any, Any]:
        packed_inputs = self._call_engine(
            ["pack_shares", "pack_inputs", "to_packed_shamir"],
            batch.input_shares,
            packing_factor=self.config.packing_factor,
        )
        packed_targets = self._call_engine(
            ["pack_shares", "pack_targets", "to_packed_shamir"],
            batch.target_shares,
            packing_factor=self.config.packing_factor,
        )
        return packed_inputs, packed_targets

    def _forward_backward(self, packed_inputs: Any, packed_targets: Any, model_params: Any) -> Any:
        predictions = self._call_engine(["forward", "forward_pass"], packed_inputs, model_params)
        gradients = self._call_engine(
            ["backward", "compute_gradients", "backward_pass"],
            predictions,
            packed_targets,
            model_params,
        )
        return gradients

    def _secure_aggregate(self, gradients: Any) -> Any:
        return self._call_engine(
            ["secure_aggregate_gradients", "aggregate_gradients", "secure_aggregate"],
            gradients,
        )

    def _update_model(self, aggregated_gradients: Any) -> Any:
        return self._call_engine(
            ["apply_gradients", "update_model", "sgd_step"],
            self._model_params,
            aggregated_gradients,
        )

    def _cas_commit_model(self, new_model: Any) -> int:
        old_version = self._model_version
        new_version = old_version + 1

        # Atomic authority: model_key CAS version is the only required commit.
        model_ok = self.kvs.compare_and_set(
            key=self.config.model_key,
            old_version=old_version,
            new_value=new_model,
            new_version=new_version,
        )
        if not model_ok:
            raise KVSOperationError("model CAS failed due to contention/version conflict")

        # Best-effort mirror pointer update; do not fail committed model on mirror contention.
        try:
            self.kvs.compare_and_set(
                key=self.config.model_version_key,
                old_version=old_version,
                new_value=new_version,
                new_version=new_version,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("model-version mirror update failed (non-fatal): %s", exc)

        coordination.notify(
            {
                "event": "model_version_advanced",
                "node_id": self.config.node_id,
                "old_version": old_version,
                "new_version": new_version,
                "timestamp": time.time(),
            }
        )
        return new_version

    def train(self) -> None:
        """Main training loop with safety checks, membership handling, and CAS commits."""
        self.membership.start()
        try:
            self._model_params, self._model_version = self._load_model()
            LOGGER.info("loaded model version=%d", self._model_version)

            for step_idx in range(self.config.max_steps):
                # Pause when resharing is needed or ongoing.
                self.membership.wait_until_allowed()

                n_active = self.membership.active_count()
                if not self._safety_bound_holds(self.config.t, self.config.s, n_active):
                    self._request_resharing("safety_bound_breached")
                    self.membership.wait_until_allowed()
                    continue

                batch = self._load_minibatch(step_idx)
                packed_inputs, packed_targets = self._pack_batch(batch)
                local_grads = self._forward_backward(packed_inputs, packed_targets, self._model_params)
                agg_grads = self._secure_aggregate(local_grads)
                new_model = self._update_model(agg_grads)

                try:
                    self._model_version = self._cas_commit_model(new_model)
                    self._model_params = new_model
                    LOGGER.info("step=%d committed model_version=%d", step_idx, self._model_version)
                except KVSOperationError as exc:
                    # Lost CAS race or transient conflict: refresh and continue.
                    LOGGER.warning("step=%d commit conflict: %s; reloading model", step_idx, exc)
                    self._model_params, self._model_version = self._load_model()
        finally:
            self.membership.stop()

    def evaluate(self, test_dataset_shares: Sequence[Any]) -> Sequence[Any]:
        """
        Run secure forward inference on secret-shared test data and return encrypted predictions.

        Returned values are still secret/encrypted shares; authorized nodes can locally
        compute metrics after reconstruction according to policy.
        """
        model_params, model_version = self._load_model()
        encrypted_predictions: List[Any] = []

        for sample_shares in test_dataset_shares:
            packed = self._call_engine(
                ["pack_shares", "pack_inputs", "to_packed_shamir"],
                sample_shares,
                packing_factor=self.config.packing_factor,
            )
            pred = self._call_engine(["forward", "forward_pass", "infer"], packed, model_params)
            encrypted_predictions.append(pred)

        coordination.notify(
            {
                "event": "evaluation_completed",
                "node_id": self.config.node_id,
                "model_version": model_version,
                "num_samples": len(test_dataset_shares),
                "timestamp": time.time(),
            }
        )
        return encrypted_predictions

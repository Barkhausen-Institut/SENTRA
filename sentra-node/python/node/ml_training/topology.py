"""
YAML topology for split client / training-node deployments.

Each process receives a file generated for its environment. The other team can
emit these from their orchestration; this module only parses and builds the
structures expected by ``create_mpc_network``.

Client file (role ``client``) lists training parties and this client's listen socket.
Node file (role ``node``) lists all training parties (reachable addresses) plus optional
``client`` for post-training eval uploads.

Schema (YAML)
-------------

Client::

    role: client
    party_id: 0
    listen:
      host: "0.0.0.0"
      port: 9600
    training_nodes:
      - id: 1
        host: "node-1.example"
        port: 9601
      - id: 2
        host: "node-2.example"
        port: 9602

Training node (example party 2)::

    role: node
    party_id: 2
    listen:
      host: "0.0.0.0"
      port: 9602
    training_nodes:
      - id: 1
        host: "node-1.example"
        port: 9601
      - id: 2
        host: "node-2.example"
        port: 9602
      - id: 3
        host: "node-3.example"
        port: 9603
    client:
      party_id: 0
      host: "client.example"
      port: 9600

``training_nodes`` must contain every training party id ``1..n``. The entry for
``party_id`` uses ``listen.host`` / ``listen.port`` for the local bind (others
still see routable ``host`` in the file for that id; it is overwritten locally).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - import guard for optional envs
    yaml = None  # type: ignore
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


@dataclass(frozen=True)
class ClientTopology:
    """Parsed client-side topology (dataset owner)."""

    party_id: int
    listen_host: str
    listen_port: int
    node_configs: Dict[int, Dict[str, Any]]
    n_nodes: int


@dataclass(frozen=True)
class NodeTopology:
    """Parsed training-node topology."""

    party_id: int
    listen_host: str
    listen_port: int
    node_configs: Dict[int, Dict[str, Any]]
    n_nodes: int
    client_party_id: Optional[int]
    client_host: Optional[str]
    client_port: Optional[int]


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for topology files. Install with: pip install PyYAML"
        ) from _YAML_IMPORT_ERROR


def _sorted_training_ids(training_nodes: List[Mapping[str, Any]]) -> List[int]:
    ids = sorted(int(x["id"]) for x in training_nodes)
    if not ids or ids[0] != 1 or ids != list(range(1, len(ids) + 1)):
        raise ValueError(
            "training_nodes must list party ids 1..n contiguously (got "
            f"{[int(x['id']) for x in training_nodes]})"
        )
    return ids


def _build_node_configs_for_party(
    party_id: int,
    listen: Mapping[str, Any],
    training_nodes: List[Mapping[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], str, int]:
    listen_host = str(listen["host"])
    listen_port = int(listen["port"])
    node_cfgs: Dict[int, Dict[str, Any]] = {}
    for entry in training_nodes:
        nid = int(entry["id"])
        node_cfgs[nid] = {"host": str(entry["host"]), "port": int(entry["port"])}
    if party_id not in node_cfgs:
        raise ValueError(f"party_id={party_id} missing from training_nodes")
    node_cfgs[int(party_id)] = {"host": listen_host, "port": listen_port}
    n_nodes = len(_sorted_training_ids(training_nodes))
    return node_cfgs, listen_host, listen_port


def load_client_topology(path: str | Path) -> ClientTopology:
    _require_yaml()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("topology root must be a mapping")
    role = str(raw.get("role", "")).lower()
    if role and role != "client":
        raise ValueError(f"client topology expects role=client (got {role!r})")
    party_id = int(raw["party_id"])
    listen = raw["listen"]
    listen_host = str(listen["host"])
    listen_port = int(listen["port"])
    training_nodes = raw["training_nodes"]
    if not isinstance(training_nodes, list):
        raise ValueError("training_nodes must be a list")
    n_nodes = len(_sorted_training_ids(training_nodes))
    node_cfgs: Dict[int, Dict[str, Any]] = {}
    for entry in training_nodes:
        nid = int(entry["id"])
        node_cfgs[nid] = {"host": str(entry["host"]), "port": int(entry["port"])}
    train_ids = {int(e["id"]) for e in training_nodes}
    if int(party_id) in train_ids:
        raise ValueError("client party_id must not duplicate a training_nodes id")
    # Local bind host for SecureMPCNetwork (party_id is not a training peer).
    node_cfgs[int(party_id)] = {"host": listen_host, "port": int(listen_port)}
    return ClientTopology(
        party_id=int(party_id),
        listen_host=listen_host,
        listen_port=int(listen_port),
        node_configs=node_cfgs,
        n_nodes=n_nodes,
    )


def load_node_topology(path: str | Path, expected_party_id: int) -> NodeTopology:
    _require_yaml()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("topology root must be a mapping")
    role = str(raw.get("role", "")).lower()
    if role and role != "node":
        raise ValueError(f"node topology expects role=node (got {role!r})")
    party_id = int(raw["party_id"])
    if int(party_id) != int(expected_party_id):
        raise ValueError(f"topology party_id={party_id} does not match --node-id={expected_party_id}")
    listen = raw["listen"]
    training_nodes = raw["training_nodes"]
    if not isinstance(training_nodes, list):
        raise ValueError("training_nodes must be a list")
    node_cfgs, lh, lp = _build_node_configs_for_party(party_id, listen, training_nodes)
    n_nodes = len(_sorted_training_ids(training_nodes))

    client_party_id: Optional[int] = None
    client_host: Optional[str] = None
    client_port: Optional[int] = None
    client_block = raw.get("client")
    if isinstance(client_block, dict) and client_block:
        client_party_id = int(client_block.get("party_id", 0))
        client_host = str(client_block["host"])
        client_port = int(client_block["port"])

    return NodeTopology(
        party_id=int(party_id),
        listen_host=lh,
        listen_port=int(lp),
        node_configs=node_cfgs,
        n_nodes=n_nodes,
        client_party_id=client_party_id,
        client_host=client_host,
        client_port=client_port,
    )

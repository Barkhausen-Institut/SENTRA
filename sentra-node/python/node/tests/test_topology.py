"""Unit tests for YAML topology parsing."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from ml_training.topology import load_client_topology, load_node_topology


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return p


def test_load_client_topology(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "c.yaml",
        """
        role: client
        party_id: 0
        listen:
          host: "0.0.0.0"
          port: 9600
        training_nodes:
          - id: 1
            host: "a.example"
            port: 9601
          - id: 2
            host: "b.example"
            port: 9602
        """,
    )
    ct = load_client_topology(path)
    assert ct.party_id == 0
    assert ct.n_nodes == 2
    assert ct.listen_port == 9600
    assert ct.node_configs[1]["host"] == "a.example"
    assert ct.node_configs[1]["port"] == 9601
    assert ct.node_configs[0]["host"] == "0.0.0.0"


def test_load_node_topology_overrides_listen(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "n2.yaml",
        """
        role: node
        party_id: 2
        listen:
          host: "0.0.0.0"
          port: 9602
        training_nodes:
          - id: 1
            host: "n1.internal"
            port: 9601
          - id: 2
            host: "n2.internal"
            port: 9602
          - id: 3
            host: "n3.internal"
            port: 9603
        client:
          party_id: 0
          host: "client.internal"
          port: 9700
        """,
    )
    nt = load_node_topology(path, expected_party_id=2)
    assert nt.party_id == 2
    assert nt.n_nodes == 3
    assert nt.node_configs[2]["host"] == "0.0.0.0"
    assert nt.node_configs[2]["port"] == 9602
    assert nt.node_configs[1]["host"] == "n1.internal"
    assert nt.client_host == "client.internal"
    assert nt.client_port == 9700


def test_load_node_topology_party_mismatch(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "bad.yaml",
        """
        role: node
        party_id: 1
        listen:
          host: "0.0.0.0"
          port: 9601
        training_nodes:
          - id: 1
            host: "n1"
            port: 9601
        """,
    )
    with pytest.raises(ValueError, match="does not match"):
        load_node_topology(path, expected_party_id=2)

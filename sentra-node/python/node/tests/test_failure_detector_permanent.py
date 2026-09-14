"""Unit tests for permanent failure marking (no heartbeat false recovery)."""

from ml_training.node_failure_detector import NodeFailureDetector


class _FakeChannel:
    def __init__(self):
        self.handlers = {}

    def register_handler(self, msg_type, handler):
        self.handlers[msg_type] = handler


class _FakeNetwork:
    def __init__(self, node_id: int, peer_ids):
        self.node_id = node_id
        self.node_configs = {int(i): {} for i in peer_ids}
        self.channel = _FakeChannel()


def test_mark_peer_failed_is_permanent():
    net = _FakeNetwork(1, [1, 2, 3])
    det = NodeFailureDetector(net, heartbeat_interval=10.0, failure_timeout=10.0)
    det.mark_peer_failed_immediate(2)
    assert 2 not in det.get_active_nodes()
    det._handle_heartbeat_ack(2, {})
    assert 2 not in det.get_active_nodes()
    det.clear_permanent_failure(2)
    det._handle_heartbeat_ack(2, {})
    assert 2 in det.get_active_nodes()

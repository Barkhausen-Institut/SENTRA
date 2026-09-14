"""
DPSS Lagrange redistribution on committee change (protocol step 4).

When membership differs from previous epoch, redistribute weight shares to the
new committee without plaintext reconstruction. Uses distributed_redistribute_lagrange.
"""

from typing import List, Sequence, Any
from ml_training.secret_sharing import Share
from ml_training.dpss_distributed import distributed_redistribute_lagrange


def _weights_to_flat_ys(weights: list) -> List[int]:
    """Flatten weight shares to list of y values."""
    vals: List[int] = []
    for layer in weights:
        if isinstance(layer, list) and layer and isinstance(layer[0], list):
            for row in layer:
                for s in row:
                    vals.append(int(s.y))
        else:
            for s in layer:
                vals.append(int(s.y))
    return vals


def _flat_ys_to_weights(flat_ys: List[int], node_id: int, shapes: List[tuple]) -> list:
    """Reconstruct weight structure from flat y values."""
    idx = 0
    out = []
    for shape in shapes:
        if len(shape) == 2:
            rows, cols = shape
            layer = []
            for _ in range(rows):
                row = []
                for _ in range(cols):
                    y = flat_ys[idx]
                    idx += 1
                    row.append(Share(x=node_id, y=y, node_id=node_id))
                layer.append(row)
            out.append(layer)
        else:
            layer = []
            for _ in range(shape[0]):
                y = flat_ys[idx]
                idx += 1
                layer.append(Share(x=node_id, y=y, node_id=node_id))
            out.append(layer)
    return out


def redistribute_weights_on_committee_change(
    weights: list,
    *,
    channel: Any,
    shamir: Any,
    node_id: int,
    field_size: int,
    t: int,
    old_committee: Sequence[int],
    new_committee: Sequence[int],
    weight_shapes: List[tuple],
    context_prefix: str,
) -> list:
    """
    Redistribute weight shares from old_committee to new_committee via Lagrange.
    Requires at least t+1 nodes from old_committee (still alive) to participate.
    Returns new weights for nodes in new_committee.

    Note: Loops over each share; for large models a vectorized version would be needed.
    Caller must run barriers between rounds or use phased send/recv for synchronization.
    """
    p = int(field_size)
    sid = int(node_id)
    new_sorted = sorted(int(x) for x in new_committee)
    if sid not in new_sorted:
        raise ValueError(
            f"Node {sid} not in new committee {new_sorted}; cannot receive redistributed weights."
        )
    old_sorted = sorted(int(x) for x in old_committee)

    # subset_old must be t+1 nodes from old committee that are still in new (alive)
    overlap = [n for n in old_sorted if n in new_sorted]
    if len(overlap) < t + 1:
        raise ValueError(
            f"Need at least t+1={t+1} alive nodes from old committee for Lagrange, "
            f"got {len(overlap)} (old={old_sorted}, new={new_sorted})"
        )
    subset_old = overlap[: t + 1]

    flat_ys = _weights_to_flat_ys(weights)
    new_ys: List[int] = []

    for i, y_val in enumerate(flat_ys):
        my_share = int(y_val) if sid in subset_old else None
        ctx = f"{context_prefix}/share_{i}"
        new_y = distributed_redistribute_lagrange(
            channel=channel,
            my_node_id=sid,
            p=p,
            t=t,
            my_share_y=my_share,
            subset_old_node_ids=subset_old,
            new_committee_ids=new_sorted,
            context_prefix=ctx,
            phase="both",
        )
        if new_y is not None:
            new_ys.append(new_y)

    if len(new_ys) != len(flat_ys):
        raise RuntimeError(
            f"Lagrange redistribution length mismatch: got {len(new_ys)}, expected {len(flat_ys)}"
        )

    return _flat_ys_to_weights(new_ys, sid, weight_shapes)

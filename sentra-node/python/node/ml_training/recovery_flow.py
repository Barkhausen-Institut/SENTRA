from typing import Any, List, Tuple

from ml_training.committee_selection import get_committee_from_detector
from ml_training.dpss_lagrange_committee import redistribute_weights_on_committee_change
from ml_training.membership_epoch import MembershipEpochScope
from ml_training.packing_safety import check_packing_safety


def run_dropout_weight_reshare_recovery(
    weights: list,
    *,
    network,
    failure_detector,
    shamir,
    args: Any,
    me: MembershipEpochScope,
    field_size: int,
    pss_packing_factor: int,
    model,
    recovery_seq_holder: list,
) -> Tuple[list, bool]:
    """
    Protocol step 12 recovery: sync survivors, Lagrange redistribute weight shares to the
    live committee, re-check packing safety, then caller may resume training.

    Returns (new_weights, True) on success; (weights, False) if barrier/reshare/packing fails.
    """
    live = get_committee_from_detector(failure_detector)
    old_committee = list(range(1, int(args.n_nodes) + 1))
    recovery_seq_holder[0] += 1
    seq = int(recovery_seq_holder[0])
    tag_pre = me.ctx(f"dropout_recovery_pre_{seq}")
    n_active = len(live)

    print(
        f"Node {args.node_id}: Dropout recovery seq={seq}: survivors={live}, "
        f"n_active={n_active}, running barriers + Lagrange weight reshare..."
    )

    try:
        network.barrier_with_peers(tag_pre, set(live), timeout=300.0)
    except Exception as exc:
        print(f"Node {args.node_id}: Pre-reshare barrier failed: {exc}")
        return weights, False

    if not check_packing_safety(int(args.t), int(pss_packing_factor), n_active):
        print(
            f"Node {args.node_id}: Packing unsafe after dropout: "
            f"2*(t+s-1) >= n_active with t={args.t}, s={pss_packing_factor}, n_active={n_active}. "
            "Cannot resume; restart with fewer packing slots or more nodes."
        )
        return weights, False

    try:
        new_weights = redistribute_weights_on_committee_change(
            weights,
            channel=network.channel,
            shamir=shamir,
            node_id=int(args.node_id),
            field_size=int(field_size),
            t=int(args.t),
            old_committee=old_committee,
            new_committee=live,
            weight_shapes=model.get_weight_shapes(),
            context_prefix=me.ctx(f"lagrange_dropout_{seq}"),
        )
    except Exception as exc:
        print(f"Node {args.node_id}: Lagrange weight reshare failed: {exc}")
        return weights, False

    tag_post = me.ctx(f"dropout_recovery_post_{seq}")
    try:
        network.barrier_with_peers(tag_post, set(live), timeout=300.0)
    except Exception as exc:
        print(f"Node {args.node_id}: Post-reshare barrier failed: {exc}")
        return weights, False

    print(f"Node {args.node_id}: Dropout weight reshare complete; resuming training.")
    return new_weights, True


def run_join_recovery(
    weights: list,
    recovered_node_ids: List[int],
    *,
    network,
    failure_detector,
    shamir,
    args: Any,
    me: MembershipEpochScope,
    field_size: int,
    pss_packing_factor: int,
    model,
    join_seq_holder: list,
) -> Tuple[list, bool]:
    """
    Protocol step 6 join recovery: when a node rejoins, barrier among all active
    (including rejoiner), redistribute weight shares from old committee to new
    committee via Lagrange, then caller may resume training.
    """
    live = get_committee_from_detector(failure_detector)
    recovered_set = set(int(i) for i in recovered_node_ids)
    old_committee = sorted(n for n in live if n not in recovered_set)
    new_committee = live
    join_seq_holder[0] += 1
    seq = int(join_seq_holder[0])
    tag_pre = me.ctx(f"join_recovery_pre_{seq}")

    print(
        f"Node {args.node_id}: Join recovery seq={seq}: rejoined={recovered_node_ids}, "
        f"old_committee={old_committee}, new_committee={new_committee}..."
    )

    try:
        network.barrier_with_peers(tag_pre, set(new_committee), timeout=300.0)
    except Exception as exc:
        print(f"Node {args.node_id}: Join pre-reshare barrier failed: {exc}")
        return weights, False

    n_active = len(new_committee)
    if not check_packing_safety(int(args.t), int(pss_packing_factor), n_active):
        print(
            f"Node {args.node_id}: Packing unsafe after join: "
            f"t={args.t}, s={pss_packing_factor}, n_active={n_active}. Cannot resume."
        )
        return weights, False

    try:
        new_weights = redistribute_weights_on_committee_change(
            weights,
            channel=network.channel,
            shamir=shamir,
            node_id=int(args.node_id),
            field_size=int(field_size),
            t=int(args.t),
            old_committee=old_committee,
            new_committee=new_committee,
            weight_shapes=model.get_weight_shapes(),
            context_prefix=me.ctx(f"lagrange_join_{seq}"),
        )
    except Exception as exc:
        print(f"Node {args.node_id}: Join Lagrange reshare failed: {exc}")
        return weights, False

    tag_post = me.ctx(f"join_recovery_post_{seq}")
    try:
        network.barrier_with_peers(tag_post, set(new_committee), timeout=300.0)
    except Exception as exc:
        print(f"Node {args.node_id}: Join post-reshare barrier failed: {exc}")
        return weights, False

    network.set_active_peers(set(new_committee))
    if hasattr(me, "bump"):
        new_e = me.bump()
        network.update_membership_epoch(new_e)
        print(f"Node {args.node_id}: Bumped membership epoch e -> {new_e}.")

    print(
        f"Node {args.node_id}: Join weight reshare complete; MPC peers={sorted(new_committee)}. "
        "Resuming training."
    )
    return new_weights, True

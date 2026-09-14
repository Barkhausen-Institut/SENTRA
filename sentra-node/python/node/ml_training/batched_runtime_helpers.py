from typing import Optional, Tuple

from ml_training.membership_epoch import MembershipEpochScope
from ml_training.recovery_flow import run_dropout_weight_reshare_recovery, run_join_recovery
from ml_training.training_state_machine import TrainingStateMachine


def handle_paused_training_recovery(
    *,
    weights: list,
    state_machine: Optional[TrainingStateMachine],
    args,
    join_recovered_nodes_holder: list,
    network,
    failure_detector,
    shamir,
    me: MembershipEpochScope,
    field_size: int,
    pss_packing_factor: int,
    model,
    join_recovery_seq: list,
    dropout_recovery_seq: list,
) -> Tuple[list, int, bool, bool]:
    n_active = int(failure_detector.get_n_active()) if failure_detector is not None else int(args.n_nodes)
    if state_machine is None or not state_machine.is_paused():
        return weights, n_active, False, False

    reason = state_machine.get_last_pause_reason() or ""
    if (
        reason == "node_rejoin"
        and bool(args.enable_join_recovery)
        and join_recovered_nodes_holder
        and network is not None
        and failure_detector is not None
    ):
        new_w, joined = run_join_recovery(
            weights,
            list(join_recovered_nodes_holder),
            network=network,
            failure_detector=failure_detector,
            shamir=shamir,
            args=args,
            me=me,
            field_size=field_size,
            pss_packing_factor=pss_packing_factor,
            model=model,
            join_seq_holder=join_recovery_seq,
        )
        if joined:
            weights = new_w
            join_recovered_nodes_holder.clear()
            state_machine.resume()
            n_active = int(failure_detector.get_n_active())
            print(
                f"Node {args.node_id}: Resumed after join recovery "
                f"(n_active={n_active}). Retrying current batch."
            )
            return weights, n_active, True, False
    elif (
        bool(args.enable_dropout_reshare_recovery)
        and network is not None
        and failure_detector is not None
    ):
        new_w, recovered = run_dropout_weight_reshare_recovery(
            weights,
            network=network,
            failure_detector=failure_detector,
            shamir=shamir,
            args=args,
            me=me,
            field_size=field_size,
            pss_packing_factor=pss_packing_factor,
            model=model,
            recovery_seq_holder=dropout_recovery_seq,
        )
        if recovered:
            weights = new_w
            live_ids = failure_detector.get_active_nodes()
            network.set_active_peers(live_ids)
            state_machine.resume()
            n_active = len(live_ids)
            print(
                f"Node {args.node_id}: Resumed after dropout recovery "
                f"(n_active={n_active}, MPC peers={sorted(live_ids)}). Retrying current batch."
            )
            return weights, n_active, True, False

    print(
        f"Node {args.node_id}: Training paused due to node failure "
        f"(no recovery or recovery failed). Exiting."
    )
    return weights, n_active, False, True


def run_epoch_eval_and_stability_checks(
    *,
    epoch: int,
    args,
    model,
    weights: list,
    x_test_plain,
    y_test_plain,
    shamir,
    field_size: int,
    scale: int,
    reconstruction,
    eval_n: int,
    me: MembershipEpochScope,
    fixed_eval_indices,
    test_x_shares,
    dataset_meta: dict,
    packed_ops,
    pss,
    pss_packing_factor: int,
    test_x_lane_cache,
    lazy_unpack_metrics,
    epoch_grad_norm_estimate,
    epoch_diag: dict,
    prev_epoch_loss,
    instability_detected: bool,
    epoch_eval_x_before: dict,
    evaluate_fn,
    lazy_metrics_delta_fn,
) -> Tuple[object, bool, bool]:
    if epoch_grad_norm_estimate is not None and epoch_grad_norm_estimate > float(args.grad_norm_threshold):
        if args.node_id != 1:
            print(
                f"[WARN] Gradient norm instability: grad_norm={epoch_grad_norm_estimate:.4f} "
                f"> threshold={args.grad_norm_threshold:.4f}"
            )
        instability_detected = True
        if not args.no_abort_on_instability:
            print("[WARN] Aborting training due to instability thresholds.")
            return prev_epoch_loss, instability_detected, True

    # Shared test tensors: all nodes must run evaluate (packed unpack); only owner has y_test_plain.
    if epoch % 1 == 0 and (y_test_plain is not None or test_x_shares is not None):
        acc, loss, diag = evaluate_fn(
            model,
            weights,
            x_test_plain,
            y_test_plain,
            args.node_id,
            args.n_nodes,
            args.t,
            shamir,
            field_size,
            scale,
            reconstruction,
            n_test_samples=eval_n,
            context_prefix=me.ctx(f"eval_t{epoch}"),
            fixed_indices=fixed_eval_indices,
            x_test_shared=test_x_shares,
            dataset_meta=dataset_meta,
            packed_ops=packed_ops,
            pss=pss,
            pss_packing_factor=pss_packing_factor,
            x_test_lane_cache=test_x_lane_cache,
            unpack_metrics=lazy_unpack_metrics,
        )
        if args.node_id == 1:
            print(f"Epoch {epoch+1} Test Accuracy: {acc*100:.2f}%")
            print(f"Epoch {epoch+1} Test Loss: {loss:.4f}")
            print(
                f"Epoch {epoch+1} Diagnostics: mean|logit|={diag['mean_abs_logit']:.4f}, "
                f"max|logit|={diag['max_abs_logit']:.4f}, "
                f"mean_entropy={diag.get('mean_entropy', 0.0):.4f}"
            )
            if epoch_grad_norm_estimate is not None:
                print(f"Epoch {epoch+1} Estimated Grad Norm: {epoch_grad_norm_estimate:.4f}")
            if "update_abs_mean_sample" in epoch_diag:
                print(
                    f"Epoch {epoch+1} Update Sample Stats: mean|upd|={epoch_diag['update_abs_mean_sample']:.6f}, "
                    f"max|upd|={epoch_diag['update_abs_max_sample']:.6f}"
                )

            epoch_unstable = False
            if prev_epoch_loss is not None and float(prev_epoch_loss) > 0.0:
                loss_growth = float(loss) / float(prev_epoch_loss)
                if loss_growth > float(args.loss_growth_threshold):
                    epoch_unstable = True
                    print(
                        f"[WARN] Loss growth instability: current/prev={loss_growth:.4f} "
                        f"> threshold={args.loss_growth_threshold:.4f}"
                    )
            if epoch_grad_norm_estimate is not None and epoch_grad_norm_estimate > float(args.grad_norm_threshold):
                epoch_unstable = True
                print(
                    f"[WARN] Gradient norm instability: grad_norm={epoch_grad_norm_estimate:.4f} "
                    f"> threshold={args.grad_norm_threshold:.4f}"
                )
            if diag["max_abs_logit"] > float(args.explode_logit_threshold):
                epoch_unstable = True
                print(
                    f"[WARN] Logit explosion detected: max|logit|={diag['max_abs_logit']:.4f} "
                    f"> threshold={args.explode_logit_threshold:.4f}"
                )
            prev_epoch_loss = float(loss)

            if epoch_unstable:
                instability_detected = True
                print("[WARN] Instability detected. Triggering safety-bound check / resharing flow hint.")
                if not args.no_abort_on_instability:
                    print("[WARN] Aborting training due to instability thresholds.")
                    return prev_epoch_loss, instability_detected, True

    if args.node_id == 1 and bool(dataset_meta.get("use_pss_storage", False)):
        dev = lazy_metrics_delta_fn(lazy_unpack_metrics, "eval_x", epoch_eval_x_before)
        print(
            f"Lazy PSS eval unpack stats (epoch {epoch + 1}): "
            f"eval_x[h={dev['hits']},m={dev['misses']},rows={dev['rows_unpacked']},"
            f"batches={dev['batch_calls']},s={dev['unpack_s']:.3f}]"
        )

    if instability_detected and not args.no_abort_on_instability:
        return prev_epoch_loss, instability_detected, True
    return prev_epoch_loss, instability_detected, False


def execute_training_batch(
    *,
    weights: list,
    model,
    packed_batch_payload: Optional[dict],
    batch_idx,
    args,
    me: MembershipEpochScope,
    epoch: int,
    start_idx: int,
    train_x_lane_cache,
    train_y_lane_cache,
    train_x_shares,
    train_y_shares,
    packed_ops,
    lazy_unpack_metrics,
    softmax,
    lr: float,
    reconstruction,
    x_shares_cols: list,
    y_shares_cols: list,
    get_or_unpack_cached_rows_fn,
) -> Tuple[list, dict]:
    batch_diag = {}
    if args.debug_numerics:
        batch_diag["debug_numerics"] = True
    if packed_batch_payload is not None:
        feat_dim = int(packed_batch_payload["feat_dim"])
        cls_dim = int(packed_batch_payload["cls_dim"])
        pss_k = int(packed_batch_payload["pss_k"])
        x_rows = list(packed_batch_payload["x_rows"])
        y_rows = list(packed_batch_payload["y_rows"])
        idx_list = [int(i) for i in batch_idx.tolist()]
        if train_x_lane_cache is not None:
            x_lane_cols = get_or_unpack_cached_rows_fn(
                cache=train_x_lane_cache,
                indices=idx_list,
                packed_rows=train_x_shares,
                node_id=int(args.node_id),
                original_len=int(feat_dim),
                packing_factor=int(pss_k),
                packed_ops=packed_ops,
                context_prefix=me.ctx(f"packed_e2e_x_e{epoch}_b{start_idx}"),
                timeout_s=120.0,
                metrics=lazy_unpack_metrics,
                metrics_key="train_x",
            )
        else:
            x_lane_cols = packed_ops.unpack_packed_rows_to_lane_rows(
                packed_rows=x_rows,
                original_len=int(feat_dim),
                packing_factor=int(pss_k),
                context=me.ctx(f"packed_e2e_x_e{epoch}_b{start_idx}"),
                timeout=120.0,
            )
        if train_y_lane_cache is not None:
            y_lane_cols = get_or_unpack_cached_rows_fn(
                cache=train_y_lane_cache,
                indices=idx_list,
                packed_rows=train_y_shares,
                node_id=int(args.node_id),
                original_len=int(cls_dim),
                packing_factor=int(pss_k),
                packed_ops=packed_ops,
                context_prefix=me.ctx(f"packed_e2e_y_e{epoch}_b{start_idx}"),
                timeout_s=120.0,
                metrics=lazy_unpack_metrics,
                metrics_key="train_y",
            )
        else:
            y_lane_cols = packed_ops.unpack_packed_rows_to_lane_rows(
                packed_rows=y_rows,
                original_len=int(cls_dim),
                packing_factor=int(pss_k),
                context=me.ctx(f"packed_e2e_y_e{epoch}_b{start_idx}"),
                timeout=120.0,
            )

        weights = model.train_batch_packed(
            x_packed_rows=x_rows,
            y_packed_rows=y_rows,
            x_lane_cols=x_lane_cols,
            y_lane_cols=y_lane_cols,
            weights=weights,
            softmax_op=softmax,
            lr=lr,
            node_id=args.node_id,
            context=me.ctx(f"e{epoch}_b{start_idx}"),
            reconstruction_manager=reconstruction,
            loss_mode=args.loss_mode,
            diagnostics_out=batch_diag,
            packed_ops=packed_ops,
            packing_factor=int(pss_k),
            use_packed_forward_native=bool(args.packed_forward_native),
        )
    else:
        weights = model.train_batch(
            x_shares_cols,
            y_shares_cols,
            weights,
            softmax,
            lr,
            args.node_id,
            me.ctx(f"e{epoch}_b{start_idx}"),
            reconstruction_manager=reconstruction,
            loss_mode=args.loss_mode,
            diagnostics_out=batch_diag,
        )
    return weights, batch_diag

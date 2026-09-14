from types import SimpleNamespace

import numpy as np

from run_mnist_batched_secure import _prepare_training_batch_inputs


def test_prepare_batch_distributed_does_not_require_plaintext_locals():
    """
    Regression guard for refactor bug:
    distributed mode must not require x_train/y_train locals.
    """
    args = SimpleNamespace(node_id=1, packed_end2end=False, n_nodes=3, t=1, seed=2026)
    batch_idx = np.asarray([0, 1], dtype=np.int64)
    train_x_shares = np.asarray([[10, 11, 12], [20, 21, 22]], dtype=np.uint64)
    train_y_shares = np.asarray([[1, 0], [0, 1]], dtype=np.uint64)

    x_cols, y_cols, packed_payload = _prepare_training_batch_inputs(
        batch_idx=batch_idx,
        args=args,
        epoch=0,
        start_idx=0,
        me=SimpleNamespace(ctx=lambda s: f"m0_{s}"),
        use_distributed_dataset=True,
        dataset_meta={"feat_dim": 3, "cls_dim": 2, "use_pss_storage": False, "packing_factor": 1},
        train_x_shares=train_x_shares,
        train_y_shares=train_y_shares,
        use_kvs_dataset=False,
        local_kvs=None,
        packed_ops=None,
        train_x_lane_cache=None,
        train_y_lane_cache=None,
        lazy_unpack_metrics={},
        x_train=None,
        y_train=None,
        use_pss_storage=False,
        pss=None,
        pss_packing_factor=1,
        field_size=2**61 - 1,
        scale=65536,
        shamir=None,
    )

    assert packed_payload is None
    assert len(x_cols) == 2
    assert len(y_cols) == 2

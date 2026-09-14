import pytest

from start_all_nodes_cli import parse_and_validate_args


def test_cli_defaults_use_batched_mnist():
    args = parse_and_validate_args([])
    assert args.batched is True
    assert args.dataset == "mnist"


def test_cli_rejects_conflicting_dataset_sharing_modes():
    with pytest.raises(ValueError, match="Use only one dataset-sharing mode"):
        parse_and_validate_args(
            ["--distribute-dataset-shares", "--receive-dataset-shares-from-client"]
        )


def test_cli_requires_failure_detection_for_dropout_recovery():
    with pytest.raises(
        ValueError, match="--enable-dropout-reshare-recovery requires --enable-failure-detection"
    ):
        parse_and_validate_args(["--enable-dropout-reshare-recovery"])


def test_cli_requires_client_receive_mode_for_client_distributor():
    with pytest.raises(
        ValueError, match="--start-client-distributor requires --receive-dataset-shares-from-client"
    ):
        parse_and_validate_args(["--start-client-distributor"])


def test_cli_requires_dataset_sharing_for_kvs_dataset():
    with pytest.raises(
        ValueError,
        match="--use-kvs-dataset requires --distribute-dataset-shares or --receive-dataset-shares-from-client",
    ):
        parse_and_validate_args(["--use-kvs-dataset"])


def test_cli_rejects_non_mnist_dataset_value():
    with pytest.raises(SystemExit):
        parse_and_validate_args(["--dataset", "synthetic"])

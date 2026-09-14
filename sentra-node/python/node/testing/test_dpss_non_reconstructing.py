"""
Tests for non-reconstructing Shamir redistribution and proactive refresh (dpss_resharer).
"""

import os
import random
import sys

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_training.secret_sharing import ShamirSecretSharing
from ml_training.dpss_resharer import (
    DPSSResharer,
    lagrange_interpolate_at,
)


@pytest.fixture
def shamir_small():
    return ShamirSecretSharing(2**31 - 1)


def test_lagrange_interpolate_matches_reconstruct(shamir_small):
    shamir = shamir_small
    p = shamir.field_size
    secret = 1234567
    n, t = 7, 2
    shares = shamir.share(secret, n, t)
    subset = shares[: t + 1]
    xs = [s.x for s in subset]
    ys = [s.y for s in subset]
    for x_t in range(1, n + 1):
        v_l = lagrange_interpolate_at(xs, ys, x_t, p)
        v_direct = next(s.y for s in shares if s.x == x_t)
        assert v_l % p == v_direct % p


def test_redistribute_smaller_committee_no_reconstruct_call(shamir_small, monkeypatch):
    shamir = shamir_small
    secret = 42_4242
    old_n, t, new_n = 6, 2, 4
    old_ids = list(range(1, old_n + 1))
    new_ids = list(range(1, new_n + 1))
    old_shares = shamir.share(secret, old_n, t)

    called = {"reconstruct": 0}
    real_reconstruct = shamir.reconstruct

    def wrapped(shs):
        called["reconstruct"] += 1
        return real_reconstruct(shs)

    monkeypatch.setattr(shamir, "reconstruct", wrapped)

    class _Net:
        pass

    resharer = DPSSResharer(_Net(), shamir, t, old_ids, new_ids, rng=random.Random(0))
    new_shares = resharer.reshare_shares(old_shares)

    assert called["reconstruct"] == 0
    got = shamir.reconstruct(new_shares[: t + 1])
    assert got == secret % shamir.field_size


def test_proactive_refresh_preserves_secret(shamir_small, monkeypatch):
    shamir = shamir_small
    secret = 999_111
    n, t = 5, 2
    ids = list(range(1, n + 1))
    old_shares = shamir.share(secret, n, t)

    called = {"reconstruct": 0}
    real_reconstruct = shamir.reconstruct

    def wrapped(shs):
        called["reconstruct"] += 1
        return real_reconstruct(shs)

    monkeypatch.setattr(shamir, "reconstruct", wrapped)

    class _Net:
        pass

    resharer = DPSSResharer(_Net(), shamir, t, ids, ids, rng=random.Random(123))
    refreshed = resharer.reshare_shares(old_shares)

    assert called["reconstruct"] == 0
    assert shamir.reconstruct(refreshed[: t + 1]) == secret % shamir.field_size
    # Polynomial should change with high probability
    assert any(refreshed[i].y != old_shares[i].y for i in range(n))


def test_reshare_via_reconstruct_still_works(shamir_small):
    shamir = shamir_small
    secret = 7
    old_n, t, new_n = 4, 1, 3
    old_ids = list(range(1, old_n + 1))
    new_ids = list(range(1, new_n + 1))
    old_shares = shamir.share(secret, old_n, t)
    class _Net:
        pass
    resharer = DPSSResharer(_Net(), shamir, t, old_ids, new_ids)
    out = resharer.reshare_shares_via_reconstruct(old_shares)
    assert shamir.reconstruct(out[: t + 1]) == secret % shamir.field_size

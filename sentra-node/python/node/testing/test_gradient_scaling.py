import numpy as np


def test_fixed_point_update_scaling_matches_expectation():
    field_size = 2**32 - 5
    scale_factor = 1000
    p = field_size

    # Example: grad = -0.5 (fixed-point => -500 mod p), a1 = 1.0 (1000)
    dz2 = (-500) % p
    a1 = 1000
    batch_size = 64
    lr = 0.05
    lr_fixed = int(lr * scale_factor)  # 50

    # Accumulated dW in secure code path:
    # matrix multiply already applies one fixed-point truncation (/SCALE),
    # so each term is back at SCALE^1 before the batch sum.
    dw2_term = int(round((dz2 if dz2 <= p // 2 else dz2 - p) * a1 / scale_factor)) % p
    dw2_accum = dw2_term
    dw2_accum = (dw2_accum * batch_size) % p

    # Update path in code:
    # update = (dw2_accum * lr_fixed) / (scale_factor * batch_size)
    divisor = scale_factor * batch_size
    inv_div = pow(divisor, p - 2, p)
    update = (dw2_accum * lr_fixed) % p
    update = (update * inv_div) % p

    expected = (-25) % p  # -0.025 in SCALE=1000
    assert update == expected

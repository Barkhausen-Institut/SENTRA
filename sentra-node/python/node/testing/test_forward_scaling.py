def test_forward_fixed_point_scale_growth_is_controlled_by_single_truncation():
    # Simulate two-layer forward scaling with one truncation after each matmul.
    scale = 1000

    # Input and first-layer weight at SCALE^1.
    x = int(0.5 * scale)     # 500
    w1 = int(0.1 * scale)    # 100

    # First layer multiply then truncation by SCALE => back to SCALE^1
    z1_raw = w1 * x          # SCALE^2
    z1 = round(z1_raw / scale)
    assert z1 == 50          # 0.05 * 1000

    # Second layer
    w2 = int(-0.2 * scale)   # -200
    z2_raw = w2 * z1         # SCALE^2
    z2 = round(z2_raw / scale)
    assert z2 == -10         # -0.01 * 1000


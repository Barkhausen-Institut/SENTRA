import numpy as np

def test_initialization():
    input_dim = 784
    hidden_dim = 128
    output_dim = 10
    scale_factor = 1000
    init_gain = 1.0
    field_size = 2**32 - 5
    
    # 1. Image
    np.random.seed(42)  # different random seed just to get some image like bytes
    x = np.random.rand(input_dim)
    # in fixed point scale
    x_fp = (x * scale_factor).astype(np.int64) % field_size
    
    # 2. Weights W1
    w1_fp = np.zeros((hidden_dim, input_dim), dtype=np.int64)
    for i in range(hidden_dim):
        for j in range(input_dim):
            std = np.sqrt(2.0 / input_dim) * init_gain
            val = np.random.randn() * std
            w1_fp[i, j] = int(val * scale_factor) % field_size
            
    # W1 @ X
    z1_fp = np.zeros(hidden_dim, dtype=np.int64)
    for i in range(hidden_dim):
        dot = 0
        for j in range(input_dim):
            dot = (dot + w1_fp[i, j] * x_fp[j]) % field_size
        z1_fp[i] = dot
        
    # Scale down by 1000 (enclave division simulation)
    # First adjust for negative numbers logically
    z1_fp_adjusted = np.where(z1_fp > field_size // 2, z1_fp - field_size, z1_fp)
    # Then divide and handle rounding
    adj = np.where(z1_fp_adjusted >= 0, scale_factor // 2, -(scale_factor // 2))
    z1_fp = ((z1_fp_adjusted + adj) // scale_factor) % field_size
    
    # ReLU
    a1_fp = np.where((z1_fp > field_size // 2) | (z1_fp == 0), 0, z1_fp)
    
    # W2
    w2_fp = np.zeros((output_dim, hidden_dim), dtype=np.int64)
    for i in range(output_dim):
        for j in range(hidden_dim):
            std = np.sqrt(2.0 / hidden_dim) * init_gain
            val = np.random.randn() * std
            w2_fp[i, j] = int(val * scale_factor) % field_size
            
    # W2 @ A1
    z2_fp = np.zeros(output_dim, dtype=np.int64)
    for i in range(output_dim):
        dot = 0
        for j in range(hidden_dim):
            dot = (dot + w2_fp[i, j] * a1_fp[j]) % field_size
        z2_fp[i] = dot
        
    # Scale down
    z2_fp_adjusted = np.where(z2_fp > field_size // 2, z2_fp - field_size, z2_fp)
    adj = np.where(z2_fp_adjusted >= 0, scale_factor // 2, -(scale_factor // 2))
    z2_fp_trunc = ((z2_fp_adjusted + adj) // scale_factor) % field_size
    
    print("Z2 max unscaled raw val: ", np.max(np.abs(z2_fp_adjusted)))
    print("Z2 after trunc logic: ", z2_fp_trunc)
    
    final_floats = np.where(z2_fp_trunc > field_size // 2, z2_fp_trunc - field_size, z2_fp_trunc) / scale_factor
    print("Final floats: ", final_floats)

test_initialization()

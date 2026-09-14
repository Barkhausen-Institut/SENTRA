import numpy as np

p = 2**32 - 5
scale = 1000

def fp(x):
    return int(x * scale) % p

def un_fp(x):
    x_int = int(x)
    if x_int > p // 2:
        return (x_int - p) / scale
    return x_int / scale

def inv(x):
    return pow(int(x), p - 2, p)

np.random.seed(42)
input_dim = 784
hidden_dim = 128
output_dim = 10
B = 64
lr = 0.05

w1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0/input_dim)
w2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0/hidden_dim)
x = np.random.randn(input_dim, B)
y = np.zeros((output_dim, B))
y[0, :] = 1.0 # fake labels

# Convert to fixed point
w1_fp = np.array([[fp(w1[i,j]) for j in range(input_dim)] for i in range(hidden_dim)], dtype=object)
w2_fp = np.array([[fp(w2[i,j]) for j in range(hidden_dim)] for i in range(output_dim)], dtype=object)
x_fp = np.array([[fp(x[i,j]) for j in range(B)] for i in range(input_dim)], dtype=object)
y_fp = np.array([[fp(y[i,j]) for j in range(B)] for i in range(output_dim)], dtype=object)

def matmul_fp(A, B_mat):
    raw = (A @ B_mat) % p
    raw = (raw * inv(scale)) % p
    return raw

# Z1
# Check manual dot product for [0,0]
s = 0
s_true = 0
for k in range(input_dim):
    term = w1_fp[0,k] * x_fp[k,0]
    term_true = w1[0,k] * x[k,0]
    s += term
    s_true += term_true
    if k < 5:
        print(f"k={k}: w={w1[0,k]} (fp={w1_fp[0,k]}) * x={x[k,0]} (fp={x_fp[k,0]}) -> term={term} (mod p={term % p})")
        print(f"       expected term={term_true}, fp_expected={(int(term_true * scale * scale)) % p}")
        print(f"       s_accum={s % p}")
s = s % p
s = (s * inv(scale)) % p

z1_fp = matmul_fp(w1_fp, x_fp)
print(f"Manual dot [0,0]: {s}")
print(f"Z1 fp [0,0]: {z1_fp[0,0]}")

z1_unfp = np.array([[un_fp(z1_fp[i,j]) for j in range(B)] for i in range(hidden_dim)])
z1_expected = w1 @ x
print(f"Z1 diff: {np.max(np.abs(z1_unfp - z1_expected))}")

# A1
a1_fp = np.where(z1_fp < p//2, z1_fp, 0)
a1_unfp = np.array([[un_fp(a1_fp[i,j]) for j in range(B)] for i in range(hidden_dim)])
a1_expected = np.maximum(0, z1_expected)
print(f"A1 diff: {np.max(np.abs(a1_unfp - a1_expected))}")

# Z2
z2_fp = matmul_fp(w2_fp, a1_fp)
z2_unfp = np.array([[un_fp(z2_fp[i,j]) for j in range(B)] for i in range(output_dim)])
z2_expected = w2 @ a1_expected
print(f"Z2 diff: {np.max(np.abs(z2_unfp - z2_expected))}")

# Softmax + DZ2
a2_expected = np.exp(z2_expected) / np.sum(np.exp(z2_expected), axis=0, keepdims=True)
a2_fp = np.array([[fp(a2_expected[i,j]) for j in range(B)] for i in range(output_dim)], dtype=object)

dz2_fp = (a2_fp + p - y_fp) % p
dz2_expected = a2_expected - y

# DW2 Accum
dw2_accum_fp = matmul_fp(dz2_fp, a1_fp.T)
dw2_expected = dz2_expected @ a1_expected.T

# DA1
da1_fp = matmul_fp(w2_fp.T, dz2_fp)

# DZ1
relu_deriv = np.where(z1_fp < p//2, 1, 0)
dz1_fp = (da1_fp * relu_deriv) % p
dz1_expected = (w2.T @ dz2_expected) * (z1_expected > 0).astype(float)
dz1_unfp = np.array([[un_fp(dz1_fp[i,j]) for j in range(B)] for i in range(hidden_dim)])
print(f"DZ1 diff: {np.max(np.abs(dz1_unfp - dz1_expected))}")

# DW1 Accum
dw1_accum_fp = matmul_fp(dz1_fp, x_fp.T)
dw1_expected = dz1_expected @ x.T

# Update W1
lr_fp = fp(lr)
inv_B = inv(B)
factor = (lr_fp * inv_B) % p
inv_scale1 = inv(scale)
update_fp = (dw1_accum_fp * factor) % p
update_fp = (update_fp * inv_scale1) % p
new_w1_fp = (w1_fp + p - update_fp) % p

expected_w1_update = dw1_expected * lr / B
expected_new_w1 = w1 - expected_w1_update
new_w1_unfp = np.array([[un_fp(new_w1_fp[i,j]) for j in range(input_dim)] for i in range(hidden_dim)])
print(f"W1 Update diff: {np.max(np.abs(new_w1_unfp - expected_new_w1))}")


import numpy as np

p = 2**32 - 5
divisor = 1000

# Simulate a reconstructed dot product that was negative.
# e.g., actual value = -1,500,000
val_true = -1500000
opened_u64 = np.array([val_true % p], dtype=np.uint64)
print(f"opened_u64: {opened_u64}")

opened = opened_u64.astype(np.int64, copy=False)
print(f"opened before p/2 check: {opened}")

opened = np.where(opened > (p // 2), opened - p, opened)
print(f"opened after p/2 check: {opened}")

d = int(divisor)
# Round-to-nearest integer division
adj = np.where(opened >= 0, d // 2, -(d // 2))
q = (opened + adj) // d
print(f"q (expected -1500): {q}")

q_mod = np.mod(q, p).astype(np.uint64, copy=False)
print(f"q_mod (expected p - 1500 = {p - 1500}): {q_mod}")

x_points = [1, 2, 3]
secrets_mod_p_u64 = q_mod

t = 1
n = 3
L = 1
rng = np.random.default_rng()
coeffs = rng.integers(0, p, size=(t, L), dtype=np.uint64)
# Let's say coeff is 500
coeffs[0, 0] = 500

xs = [int(x) for x in x_points]
x_pows = {x: [x % p] for x in xs}

opener = 3
for nid in range(1, n + 1):
    x = int(x_points[nid - 1])
    y = (secrets_mod_p_u64.astype(np.uint64, copy=False) % np.uint64(p)).copy()
    if t > 0:
        for k in range(t):
            y = (y + (coeffs[k] * np.uint64(x_pows[x][k])) % np.uint64(p)) % np.uint64(p)
    
    print(f"nid: {nid}, share: {y}")


import numpy as np
import tensorflow as tf
import time

# Load data
from ml_training.util import loadMNISTDataset

(x_train, y_train), (x_test, y_test) = loadMNISTDataset()
x_train = x_train.reshape(-1, 784).astype(np.float64) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float64) / 255.0

# Convert labels to one-hot
y_train_oh = np.zeros((y_train.shape[0], 10), dtype=np.float64)
y_train_oh[np.arange(y_train.shape[0]), y_train] = 1.0
y_test_oh = np.zeros((y_test.shape[0], 10), dtype=np.float64)
y_test_oh[np.arange(y_test.shape[0]), y_test] = 1.0

# Scale data explicitly to simulate fixed-point ranges (e.g. 1000)
# But keep as floats for perfect mathematical tracking
x_train = x_train[:1000]
y_train_oh = y_train_oh[:1000]

# Network params
input_dim = 784
hidden_dim = 128
output_dim = 10
batch_size = 64
lr = 0.05
epochs = 5

# Init weights precisely how the Secure implementation does: He init
np.random.seed(42)
w1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
b1 = np.zeros(hidden_dim)
w2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
b2 = np.zeros(output_dim)

def relu(x):
    return np.maximum(0, x)
def relu_deriv(x):
    return (x > 0).astype(np.float64)
def softmax(x):
    # Batched softmax across cols: shapes are (10, 64)
    # Simulate secure_softmax which subtracts mean
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    exp_x = np.exp(x_centered)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

print("Starting plaintext simulation...")
for ep in range(epochs):
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    
    total_loss = 0
    correct = 0
    t0 = time.time()
    for start_idx in range(0, len(x_train), batch_size):
        batch_idx = indices[start_idx:start_idx+batch_size]
        X = x_train[batch_idx].T  # (784, 64)
        Y = y_train_oh[batch_idx].T # (10, 64)
        
        # Forward
        z1 = w1 @ X + b1[:, None] # (128, 64)
        a1 = relu(z1)             # (128, 64)
        
        z2 = w2 @ a1 + b2[:, None] # (10, 64)
        
        # DEBUG: Print max centered logit
        z2_centered = z2 - np.mean(z2, axis=0, keepdims=True)
        max_z2_centered = np.max(z2_centered)
        if start_idx == 0:
            print(f"Epoch {ep+1} Max Centered Logit: {max_z2_centered:.2f}")

        a2 = softmax(z2)           # (10, 64)
        
        # Loss
        loss = -np.sum(Y * np.log(a2 + 1e-8)) / batch_size
        total_loss += loss
        preds = np.argmax(a2, axis=0)
        true = np.argmax(Y, axis=0)
        correct += np.sum(preds == true)
        
        # Backward (dz2 = A2 - Y)
        dz2 = a2 - Y               # (10, 64)
        dw2 = dz2 @ a1.T / batch_size # (10, 128)
        db2 = np.sum(dz2, axis=1) / batch_size # (10,)
        
        da1 = w2.T @ dz2           # (128, 64)
        dz1 = da1 * relu_deriv(z1) # (128, 64)
        dw1 = dz1 @ X.T / batch_size  # (128, 784)
        db1 = np.sum(dz1, axis=1) / batch_size # (128,)
        
        if ep == 0 and start_idx == 0:
            print(f"DEBUG PLAINTEXT DW1_ACCUM: {dw1[0][:10].tolist()}", flush=True)
        
        # Update
        # Apply learning rate decay to match secure impl
        curr_lr = lr * (0.95 ** ep)
        w1 -= curr_lr * dw1
        b1 -= curr_lr * db1
        w2 -= curr_lr * dw2
        b2 -= curr_lr * db2
        
    print(f"Epoch {ep+1} | Loss: {total_loss/(len(x_train)/batch_size):.4f} | Train Acc: {correct/len(x_train):.4f} | Time: {time.time()-t0:.2f}s")
    
# Evaluate
X_t = x_test[:100].T
Y_t = y_test_oh[:100].T
z1 = w1 @ X_t + b1[:, None]
a1 = relu(z1)
z2 = w2 @ a1 + b2[:, None]
a2 = softmax(z2)
preds = np.argmax(a2, axis=0)
true = np.argmax(Y_t, axis=0)
acc = np.mean(preds == true)
print(f"Final Test Accuracy (100 samples): {acc*100:.2f}%")

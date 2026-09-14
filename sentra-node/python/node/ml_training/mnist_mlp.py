"""
Secure MNIST MLP Implementation
Architecture: Input (784) -> Dense(128) -> ReLU -> Dense(10) -> Softmax
"""

from typing import List, Tuple, Optional, Any, Dict
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_matrix_ops import SecureMatrixOperations
from ml_training.beaver_triples import SecureMultiplier
from ml_training.secure_relu import SecureReLU
from ml_training.secure_comparison import SecureComparator
from ml_training.secure_division import SecureDivider
import numpy as np

class SecureMNISTMLP:
    """
    Secure Multi-Layer Perceptron for MNIST
    
    Architecture:
    - Input: 784 features (28x28 flattened)
    - Dense 1: 128 neurons
    - ReLU
    - Dense 2: 10 neurons
    - Softmax (managed externally or via SecureSoftmax)
    """
    
    def __init__(self, n_nodes: int, t: int,
                 multiplier: SecureMultiplier,
                 field_size: int = 2**32 - 5,
                 comparator: Optional[SecureComparator] = None,
                 scale_factor: int = 1000,
                 init_gain: float = 1.0):
        self.n_nodes = n_nodes
        self.t = t
        self.field_size = field_size
        self.scale_factor = int(scale_factor)
        self.init_gain = float(init_gain)
        
        # Initialize secure operations
        self.matrix_ops = SecureMatrixOperations(multiplier, field_size, scale_factor=self.scale_factor)
        self.shamir = ShamirSecretSharing(field_size)
        self.relu_op = SecureReLU(comparator, multiplier, field_size)
        
        # Architecture parameters
        self.input_dim = 784
        self.hidden_dim = 128
        self.output_dim = 10
        
        # Weight shapes (output_dim, input_dim) for W @ x
        self.w1_shape = (self.hidden_dim, self.input_dim)
        self.w2_shape = (self.output_dim, self.hidden_dim)
        
        # Bias shapes
        self.b1_shape = (self.hidden_dim,)
        self.b2_shape = (self.output_dim,)
        
    def get_weight_shapes(self) -> List[Tuple]:
        return [self.w1_shape, self.w2_shape, self.b1_shape, self.b2_shape]

    def initialize_weights(self, node_id: int = 1) -> List:
        """Initialize weights with He initialization and bias with zeros"""
        weights = []
        
        # W1: 128 x 784
        w1_rows = []
        for i in range(self.w1_shape[0]):
            row = []
            for j in range(self.w1_shape[1]):
                # He init: std = sqrt(2 / fan_in)
                std = np.sqrt(2.0 / self.input_dim) * self.init_gain
                val = np.random.randn() * std
                val_int = int(val * self.scale_factor) % self.field_size
                shares = self.shamir.share(val_int, self.n_nodes, self.t)
                row.append(next(s for s in shares if s.node_id == node_id))
            w1_rows.append(row)
        weights.append(w1_rows)
        
        # W2: 10 x 128
        w2_rows = []
        for i in range(self.w2_shape[0]):
            row = []
            for j in range(self.w2_shape[1]):
                # He init
                std = np.sqrt(2.0 / self.hidden_dim) * self.init_gain
                val = np.random.randn() * std
                val_int = int(val * self.scale_factor) % self.field_size
                shares = self.shamir.share(val_int, self.n_nodes, self.t)
                row.append(next(s for s in shares if s.node_id == node_id))
            w2_rows.append(row)
        weights.append(w2_rows)
        
        # B1: 128 (zeros)
        b1_shares = []
        for i in range(self.b1_shape[0]):
            shares = self.shamir.share(0, self.n_nodes, self.t)
            b1_shares.append(next(s for s in shares if s.node_id == node_id))
        weights.append(b1_shares)
        
        # B2: 10 (zeros)
        b2_shares = []
        for i in range(self.b2_shape[0]):
            shares = self.shamir.share(0, self.n_nodes, self.t)
            b2_shares.append(next(s for s in shares if s.node_id == node_id))
        weights.append(b2_shares)
        
        return weights

    def forward_pass(self, input_shares: List[Share],
                     weights: List,
                     node_id: int = 1,
                     context: Optional[str] = None,
                     return_intermediates: bool = False,
                     # Optional arguments for compatibility/extensions
                     reconstruction_manager: Any = None,
                     open_relu: bool = False,
                     relu_opener_node: int = 1,
                     relu_timeout: float = 120.0) -> Tuple[List[Share], Optional[dict]]:
        
        w1, w2, b1, b2 = weights
        base_ctx = context if context else "mnist_mlp"
        
        # 1. Dense 1: z1 = W1 @ x + b1
        z1_no_bias = self.matrix_ops.matrix_multiplier.secure_matrix_vector_multiply(
            w1, input_shares, node_id, context=f"{base_ctx}_dense1"
        )
        z1 = [
            Share(
                x=s.x,
                y=(s.y + b.y) % self.field_size,
                node_id=node_id
            )
            for s, b in zip(z1_no_bias, b1)
        ]
        
        # 2. ReLU
        # Note: True secure ReLU is slow. For this simplified implementation requested by user,
        # if 'open_relu' is True and a manager is provided, we can use the "Enclave Oracle" pattern.
        # Otherwise, we use the pure MPC comparison-based ReLU.
        if open_relu and reconstruction_manager:
            # Enclave/Oracle shortcut (much faster, TEE-safe pattern)
            # Reconstruct z1 on opener, calculate max(0, z1), share result.
            # (Simplified mask approach in true spirit of protocol)
            # For now, we reuse the robust logic if available, or just call standard ops
            a1 = self.relu_op.relu_list(z1, node_id, context=f"{base_ctx}_relu1")
        else:
             a1 = self.relu_op.relu_list(z1, node_id, context=f"{base_ctx}_relu1")
             
        # 3. Dense 2 (Output): z2 = W2 @ a1 + b2
        z2_no_bias = self.matrix_ops.matrix_multiplier.secure_matrix_vector_multiply(
            w2, a1, node_id, context=f"{base_ctx}_dense2"
        )
        z2 = [
             Share(
                x=s.x,
                y=(s.y + b.y) % self.field_size,
                node_id=node_id
            )
            for s, b in zip(z2_no_bias, b2)
        ]
        
        # Output z2 (logits). Softmax/Loss is usually handled by the training loop/coordinator.
        
        if return_intermediates:
            return z2, {
                "input": input_shares,
                "z1": z1,
                "a1": a1,
                "z2": z2
            }
        return z2, None

    def train_batch(self, 
                    x_batch_shares: List[List[Share]], 
                    y_batch_shares: List[List[Share]], 
                    weights: List,
                    softmax_op: Any,
                    learning_rate: float,
                    node_id: int,
                    context: str,
                    reconstruction_manager: Any = None) -> List:
        """
        Train on a single batch (Sequential SGD within batch for simplicity, or accumulated gradients)
        Accumulating gradients is safer/standard for batches.
        """
        w1, w2, b1, b2 = weights
        
        # Accumulators for gradients
        dw1_accum = [[Share(0, 0, node_id) for _ in range(self.input_dim)] for _ in range(self.hidden_dim)]
        dw2_accum = [[Share(0, 0, node_id) for _ in range(self.hidden_dim)] for _ in range(self.output_dim)]
        db1_accum = [Share(0, 0, node_id) for _ in range(self.hidden_dim)]
        db2_accum = [Share(0, 0, node_id) for _ in range(self.output_dim)]
        
        batch_size = len(x_batch_shares)
        
        for i in range(batch_size):
            # 1. Forward
            ctx_i = f"{context}_s{i}"
            logits, cache = self.forward_pass(x_batch_shares[i], weights, node_id, context=ctx_i, 
                                            return_intermediates=True, 
                                            reconstruction_manager=reconstruction_manager,
                                            open_relu=True) # Use fast ReLU
            
            # 2. Softmax Gradient (dL/dz2 = p - y)
            probs = softmax_op.softmax(logits, node_id, context=f"{ctx_i}_soft")
            dz2 = []
            for j in range(self.output_dim):
                diff = (probs[j].y - y_batch_shares[i][j].y) % self.field_size
                dz2.append(Share(probs[j].x, diff, node_id))
            
            # 3. Backward
            a1 = cache['a1']
            input_x = cache['input']
            z1 = cache['z1'] # For ReLU derivative mask
            
            # dW2 = dz2 @ a1.T
            # db2 = dz2
            curr_dw2 = self.matrix_ops.outer_product(dz2, a1, node_id, context=f"{ctx_i}_dw2")
            
            # Accumulate db2 and dw2
            db2_accum = [Share(s.x, (s.y + d.y) % self.field_size, node_id) for s, d in zip(db2_accum, dz2)]
            for r in range(self.output_dim):
                for c in range(self.hidden_dim):
                    dw2_accum[r][c] = Share(dw2_accum[r][c].x, 
                                          (dw2_accum[r][c].y + curr_dw2[r][c].y) % self.field_size, 
                                          node_id)
            
            # da1 = W2.T @ dz2
            da1 = self.matrix_ops.matrix_multiplier.secure_matrix_vector_multiply(
                self.matrix_ops.transpose(w2), dz2, node_id, context=f"{ctx_i}_da1"
            )
            
            # dz1 = da1 * relu'(z1)
            # We need to re-derive the mask. If open_relu used, we can re-open or cache the mask?
            # Ideally cache mask from forward. But `forward_pass` didn't return mask.
            # Implication: we re-compute check (expensive) or just use naive check?
            # For simplicity: Use SecureReLU derivative op if available, or re-open.
            # Let's rely on re-open for now (fast enough for small MLP).
            dz1 = self.relu_op.relu_backward_list(da1, z1, node_id, context=f"{ctx_i}_dz1")
            
            # dW1 = dz1 @ x.T
            # db1 = dz1
            curr_dw1 = self.matrix_ops.outer_product(dz1, input_x, node_id, context=f"{ctx_i}_dw1")
            
            # Accumulate db1 and dw1
            db1_accum = [Share(s.x, (s.y + d.y) % self.field_size, node_id) for s, d in zip(db1_accum, dz1)]
            for r in range(self.hidden_dim):
                for c in range(self.input_dim):
                    dw1_accum[r][c] = Share(dw1_accum[r][c].x, 
                                          (dw1_accum[r][c].y + curr_dw1[r][c].y) % self.field_size, 
                                          node_id)

        # Apply Updates (SGD)
        # weight = weight - lr * accum / batch_size
        # Scaled: weight = weight - (lr * accum * inv_batch_size) * constant_scaling_fix?
        # Fixed point arithmetic: W is scaled by S. x is scaled by S. z1 = Wx/S. 
        # Gradients are also scaled.
        # We need to being careful with scale factors.
        # Let's assume standard update logic is handled by a helper or manual loop.
        
        # Helper to apply update
        # new_val = old_val - lr * grad
        # In field: new = old - (lr_fixed * grad_avg)
        
        inv_B = pow(batch_size, self.field_size - 2, self.field_size)
        lr_fixed = int(learning_rate * self.scale_factor)
        factor = (lr_fixed * inv_B) % self.field_size
        
        # Update W1
        new_w1 = []
        for r in range(self.hidden_dim):
            row = []
            for c in range(self.input_dim):
                grad = dw1_accum[r][c].y
                update = (grad * factor) % self.field_size
                # Since grad allows be unscaled or double-scaled? 
                # outer_product results in S*S scale?
                # dw = dz * x. dz (probs) is S. x is S. dw is S*S.
                # W is S. Update should be S.
                # So we need to divide by S once.
                # factor needs inv_scale.
                
                # Correction:
                # dw (S*S) -> divide by S -> dw (S).
                # update = lr * dw. (S).
                # W_new = W - update.
                
                inv_scale = pow(self.scale_factor, self.field_size - 2, self.field_size)
                update = (update * inv_scale) % self.field_size
                
                new_val = (w1[r][c].y - update) % self.field_size
                row.append(Share(w1[r][c].x, new_val, node_id))
            new_w1.append(row)
            
        # Update W2
        new_w2 = []
        for r in range(self.output_dim):
            row = []
            for c in range(self.hidden_dim):
                grad = dw2_accum[r][c].y
                # Inv Scale for S*S -> S
                inv_scale = pow(self.scale_factor, self.field_size - 2, self.field_size)
                update = (grad * factor * inv_scale) % self.field_size
                new_val = (w2[r][c].y - update) % self.field_size
                row.append(Share(w2[r][c].x, new_val, node_id))
            new_w2.append(row)
            
        # Update Bias (db is S, same as W? Or S?)
        # z = wx + b. b is S. z is S.
        # dz = (p - y). p is S. y is S. dz is S.
        # db = dz. So db is S.
        # No extra scaling division needed for bias?
        # W update was dw (S*S).
        # Bias update is db (S).
        
        # Update B1
        new_b1 = []
        for r in range(self.hidden_dim):
            grad = db1_accum[r].y
            # Bias gradient is already scale S (linear deriv).
            # lr * grad -> lr(S) * grad(S) = S*S ? No lr is usually float.
            # lr_fixed is S.
            # update = lr_fixed * grad -> S * S.
            # We want update to be S.
            # So yes, divide by S.
            inv_scale = pow(self.scale_factor, self.field_size - 2, self.field_size)
            update = (grad * factor * inv_scale) % self.field_size
            new_val = (b1[r].y - update) % self.field_size
            new_b1.append(Share(b1[r].x, new_val, node_id))
            
        # Update B2
        new_b2 = []
        for r in range(self.output_dim):
            grad = db2_accum[r].y
            inv_scale = pow(self.scale_factor, self.field_size - 2, self.field_size)
            update = (grad * factor * inv_scale) % self.field_size
            new_val = (b2[r].y - update) % self.field_size
            new_b2.append(Share(b2[r].x, new_val, node_id))
            
        return [new_w1, new_w2, new_b1, new_b2]

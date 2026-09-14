"""
Secure Matrix Operations
Optimized matrix operations on secret-shared values for efficient neural network training
"""

from typing import List, Optional
from ml_training.secret_sharing import Share
from ml_training.beaver_triples import SecureMultiplier
from ml_training.secure_division import SecureDivider
import numpy as np
import time


class SecureMatrixMultiplier:
    """
    Performs secure matrix multiplication on secret-shared matrices
    Optimized for neural network operations
    """
    
    def __init__(self, multiplier: SecureMultiplier, field_size: int = 2**31 - 1, scale_factor: int = 1000):
        """
        Initialize secure matrix multiplier
        Args:
            multiplier: SecureMultiplier instance
            field_size: Prime field size
        """
        self.multiplier = multiplier
        self.field_size = field_size
        self.scale_factor = int(scale_factor)
    
    def secure_matrix_multiply(self, A_shares: List[List[Share]], 
                               B_shares: List[List[Share]], 
                               node_id: int,
                               context: Optional[str] = None) -> List[List[Share]]:
        """
        Multiply two matrices securely: C = A @ B
        Args:
            A_shares: Matrix A as shares (m x k)
            B_shares: Matrix B as shares (k x n)
            node_id: Node ID
            context: Optional context for multi-node operations
        Returns:
            Matrix C as shares (m x n)
        """
        m = len(A_shares)
        k = len(A_shares[0]) if A_shares else 0
        n = len(B_shares[0]) if B_shares and B_shares[0] else 0
        
        if k != len(B_shares):
            raise ValueError(f"Matrix dimension mismatch: A is {m}x{k}, B is {len(B_shares)}x{n}")
        
        # Initialize result matrix
        C_shares = []
        
        for i in range(m):
            row = []
            for j in range(n):
                # Compute dot product: C[i][j] = sum(A[i][k] * B[k][j])
                dot_product = Share(x=A_shares[0][0].x, y=0, node_id=node_id)
                
                for k_idx in range(k):
                    # Generate unique context for each multiplication
                    mult_context = f"{context}_m{i}_{j}_k{k_idx}" if context else None
                    # Multiply A[i][k_idx] * B[k_idx][j]
                    product = self.multiplier.multiply(
                        A_shares[i][k_idx], 
                        B_shares[k_idx][j], 
                        node_id,
                        context=mult_context
                    )
                    # Add to sum
                    dot_product = Share(
                        x=dot_product.x,
                        y=(dot_product.y + product.y) % self.field_size,
                        node_id=node_id
                    )
                
                row.append(dot_product)
            C_shares.append(row)
        
        return C_shares
    
    def secure_matrix_vector_multiply(self, A_shares: List[List[Share]], 
                                      v_shares: List[Share], 
                                      node_id: int,
                                      context: Optional[str] = None) -> List[Share]:
        """
        Multiply matrix by vector securely: y = A @ v
        Optimized for neural network forward pass
        Args:
            A_shares: Matrix A as shares (m x n)
            v_shares: Vector v as shares (n,)
            node_id: Node ID
            context: Optional context for multi-node operations
        Returns:
            Vector y as shares (m,)
        """
        if not A_shares or not v_shares:
            return []
        
        m = len(A_shares)
        n = len(v_shares)
        A_n = len(A_shares[0]) if A_shares else 0
        
        # Check dimension compatibility
        if A_n != n:
            # Try to handle dimension mismatch gracefully
            # If input is larger, truncate; if smaller, pad with zeros
            if n > A_n:
                v_shares = v_shares[:A_n]
            elif n < A_n:
                # Pad with zero shares
                zero_share = Share(x=v_shares[0].x, y=0, node_id=node_id)
                v_shares = v_shares + [zero_share] * (A_n - n)
            n = A_n
        
        # Stage-2 speedup: use array-based multiply_batch_values to avoid per-element Share objects.
        x0 = v_shares[0].x
        A_y = np.array([[A_shares[i][j].y for j in range(n)] for i in range(m)], dtype=np.uint64)
        v_y = np.array([v_shares[j].y for j in range(n)], dtype=np.uint64)

        # Build (m*n) elementwise products in one go: flatten A and tile v
        y1 = A_y.reshape(-1)
        y2 = np.tile(v_y, m)

        base = context if context else "matvec"
        # Larger chunk = fewer network round-trips (2 per chunk). 32768 or 65536 can speed up.
        chunk = 32768
        prod = np.empty_like(y1, dtype=np.uint64)
        import time
        for start in range(0, y1.size, chunk):
            time.sleep(0.001) # Yield to network thread
            end = min(start + chunk, y1.size)
            prod[start:end] = self.multiplier.multiply_batch_values_fixed_point(
                y1[start:end], y2[start:end], x=x0, node_id=node_id, context_prefix=f"{base}_mv_{start}"
                , scale_factor=self.scale_factor
            )
        prod = prod.reshape(m, n)
        out = (prod.sum(axis=1) % self.field_size)

        return [Share(x=x0, y=int(out[i] % self.field_size), node_id=node_id) for i in range(m)]

    def secure_matrix_matrix_multiply_fixed_point(
        self,
        A_shares: List[List[Share]],
        B_cols: List[List[Share]],
        *,
        node_id: int,
        context: Optional[str] = None,
        chunk: int = 32768,
    ) -> List[List[Share]]:
        """
        Multiply matrix by multiple vectors (matrix-matrix): Y = A @ B, where:
          - A_shares is (m x n)
          - B_cols is a list of b vectors, each length n (so B is n x b)

        Fixed-point convention:
          - A entries are SCALE-scaled
          - B entries are SCALE-scaled
          - Output entries are SCALE-scaled, using Beaver multiply then divide by SCALE.

        This is a SIMD-style optimization: all elementwise multiplications for the whole
        mini-batch are performed via batched Beaver openings.
        """
        if not A_shares:
            return [[] for _ in range(len(B_cols))]
        if not B_cols:
            return []

        m = len(A_shares)
        n = len(A_shares[0]) if A_shares[0] else 0
        b = len(B_cols)
        if n == 0:
            return [[Share(x=0, y=0, node_id=node_id) for _ in range(m)] for _ in range(b)]

        # Validate B shapes
        for col in B_cols:
            if len(col) != n:
                raise ValueError("B_cols vectors must all have length equal to number of A columns")

        # Use the node's x-coordinate for all shares
        x0 = int(A_shares[0][0].x)

        # Build numeric arrays
        A_y = np.asarray([[int(A_shares[i][j].y) for j in range(n)] for i in range(m)], dtype=np.uint64)
        B_y = np.asarray([[int(B_cols[c][j].y) for c in range(b)] for j in range(n)], dtype=np.uint64)  # (n x b)

        base_ctx = context if context else "matmat"
        p = np.uint64(self.field_size)
        p_int = int(self.field_size)

        def _mod_matmul_small_field(lhs: np.ndarray, rhs: np.ndarray, mod_p: int, rhs_block: int = 128) -> np.ndarray:
            """
            Fast exact modular matmul for fields <= 32-bit prime.

            Preconditions:
              - lhs, rhs entries are already reduced mod p
              - mod_p <= 0xFFFFFFFF
            Rationale:
              - Single multiply fits in uint64 exactly since (p-1)^2 < 2^64.
              - Accumulator is reduced mod p each step, so no overflow drift.
            """
            m_local, k_local = lhs.shape
            if k_local != rhs.shape[0]:
                raise ValueError("matmul shape mismatch")
            b_local = rhs.shape[1]
            mod_u64 = np.uint64(mod_p)
            out = np.zeros((m_local, b_local), dtype=np.uint64)
            block = max(1, int(rhs_block))
            for b0 in range(0, b_local, block):
                b1 = min(b_local, b0 + block)
                acc = np.zeros((m_local, b1 - b0), dtype=np.uint64)
                rhs_blk = rhs[:, b0:b1].astype(np.uint64, copy=False)
                for kk in range(k_local):
                    # term = lhs[:,kk] outer rhs[kk,:] (all exact in uint64 for <=32-bit field)
                    term = (
                        lhs[:, kk].reshape(m_local, 1).astype(np.uint64, copy=False)
                        * rhs_blk[kk, :].reshape(1, b1 - b0).astype(np.uint64, copy=False)
                    ) % mod_u64
                    acc = (acc + term) % mod_u64
                out[:, b0:b1] = acc
            return out
        
        # 1. Get deterministic matrix triples
        A_mac, B_mac, C_mac = self.multiplier.get_prss_matrix_triple(
            context_prefix=f"{base_ctx}_mm",
            m=m, n=n, b_dim=b,
            node_id=int(node_id),
            node_x=x0
        )
        
        # 2. Local D = W - A_mac, E = X - B_mac
        D_local = (A_y + (p - A_mac)) % p
        E_local = (B_y + (p - B_mac)) % p
        
        # 3. Batch Reconstruct D and E
        if self.multiplier.reconstruction_manager:
            D_flat = D_local.reshape(-1)
            E_flat = E_local.reshape(-1)
            
            # Pad so they have the same length for reconstruct_for_multiplication_batch_values
            max_len = max(len(D_flat), len(E_flat))
            D_padded = np.pad(D_flat, (0, max_len - len(D_flat)), mode='constant')
            E_padded = np.pad(E_flat, (0, max_len - len(E_flat)), mode='constant')
            
            D_recon_pad, E_recon_pad = self.multiplier.reconstruction_manager.reconstruct_for_multiplication_batch_values(
                d_vals_local=D_padded.astype(np.uint64),
                e_vals_local=E_padded.astype(np.uint64),
                x=x0,
                context_prefix=f"{base_ctx}_mm",
                timeout=120.0
            )
            
            D = np.asarray(D_recon_pad[:len(D_flat)], dtype=np.uint64).reshape(m, n) % p
            E = np.asarray(E_recon_pad[:len(E_flat)], dtype=np.uint64).reshape(n, b) % p
        else:
            D = D_local
            E = E_local
            
        # 4. Y = C_mac + D @ B_mac + A_mac @ E + D @ E  (modulo p)
        # For <=32-bit fields (e.g. 2^32-5), use fast exact uint64 modular matmul.
        if p_int <= 0xFFFFFFFF:
            DB = _mod_matmul_small_field(D.astype(np.uint64, copy=False), B_mac.astype(np.uint64, copy=False), p_int, rhs_block=int(chunk))
            AE = _mod_matmul_small_field(A_mac.astype(np.uint64, copy=False), E.astype(np.uint64, copy=False), p_int, rhs_block=int(chunk))
            DE = _mod_matmul_small_field(D.astype(np.uint64, copy=False), E.astype(np.uint64, copy=False), p_int, rhs_block=int(chunk))
        else:
            # Fallback for wider fields: object arithmetic avoids overflow before modulo.
            D_obj = D.astype(object)
            E_obj = E.astype(object)
            A_mac_obj = A_mac.astype(object)
            B_mac_obj = B_mac.astype(object)
            DB = (np.dot(D_obj, B_mac_obj) % p).astype(np.uint64)
            AE = (np.dot(A_mac_obj, E_obj) % p).astype(np.uint64)
            DE = (np.dot(D_obj, E_obj) % p).astype(np.uint64)
        
        Y_share = C_mac.copy()
        Y_share = (Y_share + DB) % p
        Y_share = (Y_share + AE) % p
        Y_share = (Y_share + DE) % p
        
        # 5. Fixed-point scaling (divide by scale_factor)
        # We CANNOT blindly multiply by inv_scale here! The sum is evaluated first,
        # so it has random fractional noise that makes it indivisible by scale_factor.
        # Multiplying an indivisible ring element by inv_scale spreads fractional noise
        # across the entire modulo p field, throwing the network off into random noise!
        if self.multiplier._opened_fp_enabled():
            Y_flat = Y_share.reshape(-1)
            Y_scaled_flat = self.multiplier._opened_divide_and_reshare_vector(
                values_local_u64=Y_flat,
                divisor=int(self.scale_factor),
                node_id=int(node_id),
                x=int(x0),
                context_prefix=f"{base_ctx}_mm_trunc",
                timeout=120.0
            ).astype(np.uint64, copy=False) % p
            out_mat = Y_scaled_flat.reshape(m, b)
        else:
            inv_scale = np.uint64(pow(self.scale_factor, int(p) - 2, int(p)))
            Y_share = (Y_share * inv_scale) % p
            out_mat = Y_share
        
        # Return as list-of-columns: outputs[col][i]
        out_cols: List[List[Share]] = []
        for c in range(b):
            vec = [Share(x=x0, y=int(out_mat[i, c] % p), node_id=int(node_id)) for i in range(m)]
            out_cols.append(vec)
        return out_cols
    
    def secure_matrix_add(self, A_shares: List[List[Share]], 
                         B_shares: List[List[Share]]) -> List[List[Share]]:
        """
        Add two matrices securely: C = A + B
        Args:
            A_shares: Matrix A as shares
            B_shares: Matrix B as shares
            node_id: Node ID
        Returns:
            Matrix C as shares
        """
        m = len(A_shares)
        n = len(A_shares[0]) if A_shares else 0
        
        if len(B_shares) != m or (B_shares and len(B_shares[0]) != n):
            raise ValueError("Matrix dimension mismatch for addition")
        
        C_shares = []
        for i in range(m):
            row = []
            for j in range(n):
                # Homomorphic addition
                sum_share = Share(
                    x=A_shares[i][j].x,
                    y=(A_shares[i][j].y + B_shares[i][j].y) % self.field_size,
                    node_id=A_shares[i][j].node_id
                )
                row.append(sum_share)
            C_shares.append(row)
        
        return C_shares
    
    def secure_matrix_scalar_multiply(self, A_shares: List[List[Share]], 
                                     scalar: int, 
                                     node_id: int) -> List[List[Share]]:
        """
        Multiply matrix by scalar securely: B = scalar * A
        Args:
            A_shares: Matrix A as shares
            scalar: Scalar value
            node_id: Node ID
        Returns:
            Matrix B as shares
        """
        B_shares = []
        for row in A_shares:
            new_row = []
            for share in row:
                new_share = Share(
                    x=share.x,
                    y=(share.y * scalar) % self.field_size,
                    node_id=node_id
                )
                new_row.append(new_share)
            B_shares.append(new_row)
        
        return B_shares
    
    def secure_matrix_transpose(self, A_shares: List[List[Share]]) -> List[List[Share]]:
        """
        Transpose matrix securely (no computation needed, just rearrange)
        Args:
            A_shares: Matrix A as shares
        Returns:
            Transposed matrix as shares
        """
        if not A_shares:
            return []
        
        m = len(A_shares)
        n = len(A_shares[0])
        
        A_T = []
        for j in range(n):
            row = []
            for i in range(m):
                row.append(A_shares[i][j])
            A_T.append(row)
        
        return A_T

    
    def secure_outer_product(self, 
                             v1_shares: List[Share], 
                             v2_shares: List[Share], 
                             node_id: int, 
                             context: Optional[str] = None) -> List[List[Share]]:
        """
        Compute outer product of two share vectors: M = v1 @ v2.T
        Args:
            v1_shares: Vector 1 (m,)
            v2_shares: Vector 2 (n,)
            node_id: Node ID
            context: Optional context
        Returns:
            Matrix M (m x n) as shares
        """
        if not v1_shares or not v2_shares:
            return []
            
        m = len(v1_shares)
        n = len(v2_shares)
        
        # Use simple coordinate for all shares
        x0 = v1_shares[0].x
        
        # Extract values
        y1 = np.array([s.y for s in v1_shares], dtype=np.uint64)
        y2 = np.array([s.y for s in v2_shares], dtype=np.uint64)
        
        # Tiling for outer product: 
        # We want res[i][j] = v1[i] * v2[j]
        # Flattened: repeat v1 each element n times, tile v2 m times
        # v1: [a, b] -> [a, a, b, b] (repeat_elements)
        # v2: [c, d] -> [c, d, c, d] (tile)
        
        v1_flat = np.repeat(y1, n)
        v2_flat = np.tile(y2, m)
        
        base = context if context else "outer"
        chunk = 32768
        prod = np.empty_like(v1_flat, dtype=np.uint64)
        
        for start in range(0, v1_flat.size, chunk):
            time.sleep(0.001) 
            end = min(start + chunk, v1_flat.size)
            prod[start:end] = self.multiplier.multiply_batch_values_fixed_point(
                v1_flat[start:end], 
                v2_flat[start:end], 
                x=x0, 
                node_id=node_id, 
                context_prefix=f"{base}_{start}",
                scale_factor=self.scale_factor
            )
            
        # Reshape to (m, n)
        prod_mat = prod.reshape(m, n)
        
        # Convert back to Shares
        result = []
        for i in range(m):
            row = []
            for j in range(n):
                row.append(Share(x=x0, y=int(prod_mat[i, j]), node_id=node_id))
            result.append(row)
            
        return result


class SecureMatrixOperations:
    """
    High-level secure matrix operations for neural networks
    Optimized for forward/backward pass
    """
    
    def __init__(self, multiplier: SecureMultiplier, field_size: int = 2**31 - 1, scale_factor: int = 1000):
        """
        Initialize secure matrix operations
        Args:
            multiplier: SecureMultiplier instance
            field_size: Prime field size
        """
        self.matrix_multiplier = SecureMatrixMultiplier(multiplier, field_size, scale_factor=scale_factor)
        self.multiplier = multiplier
        self.field_size = field_size
        self.scale_factor = int(scale_factor)
        
    def outer_product(self, v1: List[Share], v2: List[Share], node_id: int, context: Optional[str] = None) -> List[List[Share]]:
        """Wrapper for secure outer product"""
        return self.matrix_multiplier.secure_outer_product(v1, v2, node_id, context)

    def transpose(self, A_shares: List[List[Share]]) -> List[List[Share]]:
        """Wrapper for secure matrix transpose"""
        return self.matrix_multiplier.secure_matrix_transpose(A_shares)
    
    def forward_pass(self, input_shares: List[Share], 
                    weights: List[List[List[Share]]], 
                    node_id: int,
                    context: Optional[str] = None) -> List[Share]:
        """
        Perform optimized forward pass through neural network
        Args:
            input_shares: Input vector as shares
            weights: Weight matrices (one per layer)
            node_id: Node ID
            context: Optional context for multi-node operations
        Returns:
            Output vector as shares
        """
        current = input_shares
        
        for layer_idx, layer_weights in enumerate(weights):
            # Generate context for this layer
            layer_context = f"{context}_L{layer_idx}" if context else None
            # Matrix-vector multiplication: output = weights @ input
            current = self.matrix_multiplier.secure_matrix_vector_multiply(
                layer_weights, current, node_id, context=layer_context
            )
            # Note: Activation function would be applied here
            # For now, we skip it (would need secure comparison)
        
        return current
    
    def backward_pass(self, loss_grad_shares: List[Share], 
                     input_shares: List[Share], 
                     weights: List[List[List[Share]]], 
                     node_id: int,
                     context: Optional[str] = None) -> List[List[List[Share]]]:
        """
        Perform optimized backward pass (gradient computation)
        Args:
            loss_grad_shares: Loss gradients as shares
            input_shares: Input vector as shares
            weights: Weight matrices
            node_id: Node ID
            context: Optional context for multi-node operations
        Returns:
            Gradient matrices (same shape as weights)
        """
        gradients = []
        
        # Simplified backward pass
        # In production, would compute proper gradients using chain rule
        current_grad = loss_grad_shares
        
        for layer_idx in range(len(weights) - 1, -1, -1):
            layer_weights = weights[layer_idx]
            layer_gradients = []
            layer_context = f"{context}_L{layer_idx}" if context else None
            
            # Get input for this layer
            if layer_idx == 0:
                layer_input = input_shares
            else:
                # Use current_grad as input for hidden layers (simplified)
                layer_input = current_grad
            
            # Compute gradients for this layer
            for i, weight_row in enumerate(layer_weights):
                row_gradients = []
                for j, weight_share in enumerate(weight_row):
                    # For weight[i][j] (out i, in j), grad should be output_grad[i] * input[j].
                    if i < len(current_grad) and j < len(layer_input):
                        input_share = layer_input[j]
                        out_grad_share = current_grad[i]
                        # Generate unique context for each multiplication
                        mult_context = f"{layer_context}_g{i}_{j}" if layer_context else None
                        grad_share = self.multiplier.multiply_fixed_point(
                            input_share,
                            out_grad_share,
                            node_id=node_id,
                            scale_factor=self.scale_factor,
                            context=mult_context
                        )
                    else:
                        # Zero gradient if indices out of range
                        grad_share = Share(x=weight_share.x, y=0, node_id=node_id)
                    row_gradients.append(grad_share)
                layer_gradients.append(row_gradients)
            
            gradients.insert(0, layer_gradients)
            
            # Update current_grad for next layer (simplified)
            if layer_idx > 0:
                # Transpose weights for next layer gradient
                W_T = self.matrix_multiplier.secure_matrix_transpose(layer_weights)
                # Ensure current_grad size matches
                if len(current_grad) <= len(W_T[0]) if W_T else 0:
                    current_grad = self.matrix_multiplier.secure_matrix_vector_multiply(
                        W_T, current_grad, node_id, context=f"{layer_context}_bp" if layer_context else None
                    )
                else:
                    # Truncate if needed
                    current_grad = self.matrix_multiplier.secure_matrix_vector_multiply(
                        W_T, current_grad[:len(W_T[0])] if W_T else [], node_id, context=f"{layer_context}_bp" if layer_context else None
                    )
        
        return gradients
    
    def update_weights(self, weights: List[List[List[Share]]], 
                      gradients: List[List[List[Share]]], 
                      learning_rate: float, 
                      node_id: int) -> List[List[List[Share]]]:
        """
        Update weights using gradients: W_new = W - lr * grad
        Args:
            weights: Current weight matrices
            gradients: Gradient matrices
            learning_rate: Learning rate
            node_id: Node ID
        Returns:
            Updated weight matrices
        """
        updated_weights = []
        
        for layer_weights, layer_grads in zip(weights, gradients):
            layer_updated = []
            
            for weight_row, grad_row in zip(layer_weights, layer_grads):
                row_updated = []
                
                for weight_share, grad_share in zip(weight_row, grad_row):
                    # Scale gradient by learning rate
                    lr_grad = self.matrix_multiplier.secure_matrix_scalar_multiply(
                        [[grad_share]], int(learning_rate * 1000), node_id
                    )[0][0]
                    
                    # Subtract: weight - lr * grad
                    new_weight = Share(
                        x=weight_share.x,
                        y=(weight_share.y - lr_grad.y) % self.field_size,
                        node_id=node_id
                    )
                    row_updated.append(new_weight)
                
                layer_updated.append(row_updated)
            
            updated_weights.append(layer_updated)
        
        return updated_weights
    
    def batch_forward_pass(self, batch_inputs: List[List[Share]], 
                          weights: List[List[List[Share]]], 
                          node_id: int) -> List[List[Share]]:
        """
        Perform forward pass on a batch of inputs
        Optimized for batch processing
        Args:
            batch_inputs: List of input vectors (batch)
            weights: Weight matrices
            node_id: Node ID
        Returns:
            List of output vectors (batch)
        """
        batch_outputs = []
        for input_shares in batch_inputs:
            output = self.forward_pass(input_shares, weights, node_id)
            batch_outputs.append(output)
        return batch_outputs


class GPUMatrixAccelerator:
    """
    GPU acceleration for matrix operations using CuPy
    Falls back to CPU if GPU unavailable
    """
    
    def __init__(self):
        """Initialize GPU accelerator"""
        self.use_gpu = False
        self.cp = None
        
        try:
            import cupy as cp
            self.cp = cp
            self.use_gpu = True
            print("GPU acceleration enabled (CuPy)")
        except ImportError:
            print("Warning: CuPy not available, using CPU")
            self.use_gpu = False
    
    def matrix_multiply_gpu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication on GPU if available
        Args:
            A: Matrix A
            B: Matrix B
        Returns:
            Product matrix
        """
        if self.use_gpu:
            A_gpu = self.cp.asarray(A)
            B_gpu = self.cp.asarray(B)
            C_gpu = self.cp.dot(A_gpu, B_gpu)
            return self.cp.asnumpy(C_gpu)
        else:
            return np.dot(A, B)
    
    def is_available(self) -> bool:
        """Check if GPU acceleration is available"""
        return self.use_gpu

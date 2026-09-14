"""
Secure Convolutional Operations
Implements secure 2D convolution on secret-shared images
"""

from typing import List, Tuple, Optional
from ml_training.secret_sharing import Share
from ml_training.beaver_triples import SecureMultiplier
import numpy as np
from numpy.lib.stride_tricks import as_strided


class SecureConvolution:
    """
    Secure 2D convolution operations on secret-shared images
    """
    
    def __init__(self, multiplier: SecureMultiplier, field_size: int = 2**31 - 1, scale_factor: int = 1000):
        """
        Initialize secure convolution
        Args:
            multiplier: SecureMultiplier instance
            field_size: Prime field size
        """
        self.multiplier = multiplier
        self.field_size = field_size
        self.scale_factor = int(scale_factor)
    
    def conv2d(self, input_shares: List[List[List[Share]]],
               kernel_shares: List[List[List[Share]]],
               stride: int = 1,
               padding: int = 0,
               node_id: int = 1,
               context: Optional[str] = None) -> List[List[List[Share]]]:
        """
        Perform 2D convolution on secret-shared images
        
        Args:
            input_shares: Input image as shares [H x W x C]
            kernel_shares: Convolution kernel as shares [K_h x K_w x C_in x C_out]
            stride: Stride for convolution
            padding: Padding size
            node_id: Node ID
            context: Optional context for multi-node operations
        
        Returns:
            Output feature maps as shares [H_out x W_out x C_out]
        """
        # Get dimensions
        H, W, C_in = len(input_shares), len(input_shares[0]), len(input_shares[0][0])
        K_h, K_w = len(kernel_shares), len(kernel_shares[0])
        C_out = len(kernel_shares[0][0][0]) if kernel_shares else 1
        
        # Calculate output dimensions
        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1
        
        # Stage-2 speedup: operate on numpy arrays of share values.
        # Extract y-values
        x0 = input_shares[0][0][0].x
        input_y = np.array([[[input_shares[h][w][c].y for c in range(C_in)] for w in range(W)] for h in range(H)],
                           dtype=np.uint64)
        kernel_y = np.array(
            [[[[kernel_shares[kh][kw][ci][co].y for co in range(C_out)] for ci in range(C_in)] for kw in range(K_w)]
             for kh in range(K_h)],
            dtype=np.uint64
        )

        # Pad input
        if padding > 0:
            input_p = np.pad(input_y, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
        else:
            input_p = input_y

        Hp, Wp = input_p.shape[0], input_p.shape[1]
        H_out = (Hp - K_h) // stride + 1
        W_out = (Wp - K_w) // stride + 1
        P = H_out * W_out
        F = K_h * K_w * C_in

        # Sliding window view: (H_out, W_out, K_h, K_w, C_in)
        s0, s1, s2 = input_p.strides
        patches = as_strided(
            input_p,
            shape=(H_out, W_out, K_h, K_w, C_in),
            strides=(s0 * stride, s1 * stride, s0, s1, s2),
            writeable=False,
        )
        patches_flat = patches.reshape(P, F)  # (P,F)
        kernel_flat = kernel_y.reshape(F, C_out)  # (F,C_out)

        # Build elementwise multiply inputs for all (pos, c_out, feat)
        # Shape (P*C_out, F)
        s1_mat = np.repeat(patches_flat, C_out, axis=0)
        s2_mat = np.tile(kernel_flat.T, (P, 1))
        y1 = s1_mat.reshape(-1)
        y2 = s2_mat.reshape(-1)

        base = context if context else "conv2d"
        # Multiply in chunks to limit peak memory / message size. Larger chunk = fewer round-trips.
        # In privacy_mode, use smaller chunks to avoid dealer/reconstruction timeouts.
        chunk = 4096 if bool(getattr(self.multiplier, "privacy_mode", False)) else 32768
        prod = np.empty_like(y1, dtype=np.uint64)
        for start in range(0, y1.size, chunk):
            end = min(start + chunk, y1.size)
            prod[start:end] = self.multiplier.multiply_batch_values_fixed_point(
                y1[start:end],
                y2[start:end],
                x=x0,
                node_id=node_id,
                context_prefix=f"{base}_mul_{start}",
                scale_factor=self.scale_factor,
            )

        prod = prod.reshape(P * C_out, F)
        out_flat = (prod.sum(axis=1) % self.field_size).reshape(P, C_out)
        out = out_flat.reshape(H_out, W_out, C_out)

        # Convert back to Share objects (small tensor)
        output_shares: List[List[List[Share]]] = []
        for h in range(H_out):
            row = []
            for w in range(W_out):
                chans = []
                for co in range(C_out):
                    chans.append(Share(x=x0, y=int(out[h, w, co] % self.field_size), node_id=node_id))
                row.append(chans)
            output_shares.append(row)

        return output_shares
    
    def conv2d_simple(self, input_shares: List[List[Share]],
                     kernel_shares: List[List[Share]],
                     stride: int = 1,
                     padding: int = 0,
                     node_id: int = 1,
                     context: Optional[str] = None) -> List[List[Share]]:
        """
        Simplified 2D convolution for single-channel input/output
        
        Args:
            input_shares: Input image as shares [H x W] (single channel)
            kernel_shares: Convolution kernel as shares [K_h x K_w]
            stride: Stride for convolution
            padding: Padding size
            node_id: Node ID
            context: Optional context
        
        Returns:
            Output feature map as shares [H_out x W_out]
        """
        # Get dimensions
        H, W = len(input_shares), len(input_shares[0])
        K_h, K_w = len(kernel_shares), len(kernel_shares[0])
        
        # Calculate output dimensions
        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1
        
        # Initialize output
        output_shares = []
        
        for h_out in range(H_out):
            output_row = []
            for w_out in range(W_out):
                # Calculate input window position
                h_start = h_out * stride - padding
                w_start = w_out * stride - padding
                
                # Convolve
                conv_sum = Share(x=input_shares[0][0].x, y=0, node_id=node_id)
                
                for k_h in range(K_h):
                    for k_w in range(K_w):
                        h_in = h_start + k_h
                        w_in = w_start + k_w
                        
                        # Check bounds (handle padding)
                        if h_in < 0 or h_in >= H or w_in < 0 or w_in >= W:
                            continue  # Padding: skip
                        
                        # Get shares
                        input_share = input_shares[h_in][w_in]
                        kernel_share = kernel_shares[k_h][k_w]
                        
                        # Multiply
                        conv_context = f"{context}_h{h_out}_w{w_out}_k{k_h}_{k_w}" if context else None
                        product = self.multiplier.multiply_fixed_point(
                            input_share, kernel_share, node_id=node_id, scale_factor=self.scale_factor, context=conv_context
                        )
                        
                        # Accumulate
                        conv_sum = Share(
                            x=conv_sum.x,
                            y=(conv_sum.y + product.y) % self.field_size,
                            node_id=node_id
                        )
                
                output_row.append(conv_sum)
            output_shares.append(output_row)
        
        return output_shares
    
    def conv2d_backward(self, output_grad_shares: List[List[List[Share]]],
                        input_shares: List[List[List[Share]]],
                        kernel_shares: List[List[List[Share]]],
                        stride: int = 1,
                        padding: int = 0,
                        node_id: int = 1,
                        context: Optional[str] = None) -> Tuple[List[List[List[Share]]], List[List[List[List[Share]]]]]:
        """
        Backward pass for 2D convolution
        
        Computes gradients w.r.t. input and kernel
        
        Args:
            output_grad_shares: Gradient w.r.t. output [H_out x W_out x C_out]
            input_shares: Original input [H x W x C_in]
            kernel_shares: Convolution kernel [K_h x K_w x C_in x C_out]
            stride: Stride used in forward pass
            padding: Padding used in forward pass
            node_id: Node ID
            context: Optional context
        
        Returns:
            Tuple of (input_grad_shares, kernel_grad_shares)
            - input_grad_shares: Gradient w.r.t. input [H x W x C_in]
            - kernel_grad_shares: Gradient w.r.t. kernel [K_h x K_w x C_in x C_out]
        """
        H, W, C_in = len(input_shares), len(input_shares[0]), len(input_shares[0][0])
        H_out, W_out, C_out = len(output_grad_shares), len(output_grad_shares[0]), len(output_grad_shares[0][0])
        K_h, K_w = len(kernel_shares), len(kernel_shares[0])
        
        # Initialize input gradients (zero)
        input_grad_shares = []
        for h in range(H):
            input_row = []
            for w in range(W):
                input_channels = []
                for c_in in range(C_in):
                    zero_share = Share(x=input_shares[0][0][0].x, y=0, node_id=node_id)
                    input_channels.append(zero_share)
                input_row.append(input_channels)
            input_grad_shares.append(input_row)
        
        # Initialize kernel gradients (zero)
        kernel_grad_shares = []
        for k_h in range(K_h):
            kernel_row = []
            for k_w in range(K_w):
                kernel_ch_in = []
                for c_in in range(C_in):
                    kernel_ch_out = []
                    for c_out in range(C_out):
                        zero_share = Share(x=kernel_shares[0][0][0][0].x, y=0, node_id=node_id)
                        kernel_ch_out.append(zero_share)
                    kernel_ch_in.append(kernel_ch_out)
                kernel_row.append(kernel_ch_in)
            kernel_grad_shares.append(kernel_row)
        
        # Stage-2 speedup: express conv backward using im2col-style array math and batched secure multiplies.
        x0 = input_shares[0][0][0].x
        input_y = np.array([[[input_shares[h][w][c].y for c in range(C_in)] for w in range(W)] for h in range(H)],
                           dtype=np.uint64)
        kernel_y = np.array(
            [[[[kernel_shares[kh][kw][ci][co].y for co in range(C_out)] for ci in range(C_in)] for kw in range(K_w)]
             for kh in range(K_h)],
            dtype=np.uint64
        )
        outg_y = np.array([[[output_grad_shares[h][w][c].y for c in range(C_out)] for w in range(W_out)] for h in range(H_out)],
                          dtype=np.uint64)

        # Pad input for patch extraction
        if padding > 0:
            input_p = np.pad(input_y, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
        else:
            input_p = input_y

        Hp, Wp = input_p.shape[0], input_p.shape[1]
        P = H_out * W_out
        F = K_h * K_w * C_in

        s0, s1, s2 = input_p.strides
        patches = as_strided(
            input_p,
            shape=(H_out, W_out, K_h, K_w, C_in),
            strides=(s0 * stride, s1 * stride, s0, s1, s2),
            writeable=False,
        )
        patches_flat = patches.reshape(P, F)  # (P,F)
        outg_flat = outg_y.reshape(P, C_out)  # (P,C_out)
        kernel_flat = kernel_y.reshape(F, C_out)  # (F,C_out)

        base = context if context else "conv2d_bwd"
        # Larger chunk = fewer round-trips. In privacy_mode, keep smaller to avoid timeouts.
        chunk = 4096 if bool(getattr(self.multiplier, "privacy_mode", False)) else 32768

        # ---- kernel_grad_flat = patches_flat.T @ outg_flat  (F,C_out)
        A_T = patches_flat.T  # (F,P)
        s1_mat = np.repeat(A_T, C_out, axis=0)        # (F*C_out,P)
        s2_mat = np.tile(outg_flat.T, (F, 1))         # (F*C_out,P)
        y1 = s1_mat.reshape(-1)
        y2 = s2_mat.reshape(-1)
        prod = np.empty_like(y1, dtype=np.uint64)
        for start in range(0, y1.size, chunk):
            end = min(start + chunk, y1.size)
            prod[start:end] = self.multiplier.multiply_batch_values_fixed_point(
                y1[start:end], y2[start:end], x=x0, node_id=node_id, context_prefix=f"{base}_kg_{start}",
                scale_factor=self.scale_factor
            )
        prod = prod.reshape(F * C_out, P)
        kg_flat = (prod.sum(axis=1) % self.field_size).reshape(F, C_out)
        kernel_grad = kg_flat.reshape(K_h, K_w, C_in, C_out)

        # ---- d_patches = outg_flat @ kernel_flat.T  (P,F)
        s1_mat = np.repeat(outg_flat, F, axis=0)      # (P*F,C_out)
        s2_mat = np.tile(kernel_flat, (P, 1))         # (P*F,C_out)
        y1 = s1_mat.reshape(-1)
        y2 = s2_mat.reshape(-1)
        prod = np.empty_like(y1, dtype=np.uint64)
        for start in range(0, y1.size, chunk):
            end = min(start + chunk, y1.size)
            prod[start:end] = self.multiplier.multiply_batch_values_fixed_point(
                y1[start:end], y2[start:end], x=x0, node_id=node_id, context_prefix=f"{base}_ig_{start}",
                scale_factor=self.scale_factor
            )
        prod = prod.reshape(P * F, C_out)
        d_patches = (prod.sum(axis=1) % self.field_size).reshape(H_out, W_out, K_h, K_w, C_in)

        # Scatter-add patches into input_grad (padded)
        input_grad_p = np.zeros((Hp, Wp, C_in), dtype=np.uint64)
        for h_out in range(H_out):
            for w_out in range(W_out):
                h0 = h_out * stride
                w0 = w_out * stride
                input_grad_p[h0:h0 + K_h, w0:w0 + K_w, :] = (
                    input_grad_p[h0:h0 + K_h, w0:w0 + K_w, :] + d_patches[h_out, w_out]
                ) % self.field_size

        if padding > 0:
            input_grad = input_grad_p[padding:padding + H, padding:padding + W, :]
        else:
            input_grad = input_grad_p

        # Convert to Share objects
        input_grad_shares = [
            [[Share(x=x0, y=int(input_grad[h, w, c] % self.field_size), node_id=node_id) for c in range(C_in)]
             for w in range(W)]
            for h in range(H)
        ]
        kernel_grad_shares = [
            [
                [
                    [Share(x=x0, y=int(kernel_grad[kh, kw, ci, co] % self.field_size), node_id=node_id) for co in range(C_out)]
                    for ci in range(C_in)
                ]
                for kw in range(K_w)
            ]
            for kh in range(K_h)
        ]

        return input_grad_shares, kernel_grad_shares

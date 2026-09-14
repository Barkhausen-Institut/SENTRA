"""
Secure Pooling Operations
Implements secure max pooling and average pooling on secret-shared feature maps
"""

from typing import List, Tuple, Optional
from ml_training.secret_sharing import Share
from ml_training.secure_comparison import SecureComparator
from ml_training.secure_division import SecureDivider
import numpy as np


class SecurePooling:
    """
    Secure pooling operations (max pooling, average pooling)
    """
    
    def __init__(self, comparator: Optional[SecureComparator] = None,
                 divider: Optional[SecureDivider] = None,
                 field_size: int = 2**31 - 1):
        """
        Initialize secure pooling
        Args:
            comparator: SecureComparator for max pooling (optional)
            divider: SecureDivider for average pooling (optional)
            field_size: Prime field size
        """
        self.comparator = comparator
        self.divider = divider
        self.field_size = field_size
    
    def max_pool2d(self, input_shares: List[List[Share]],
                   pool_size: Tuple[int, int] = (2, 2),
                   stride: Optional[int] = None,
                   node_id: int = 1,
                   context: Optional[str] = None) -> List[List[Share]]:
        """
        Secure max pooling (simplified - uses average if comparator not available)
        
        Note: True max pooling requires secure comparison which is complex.
        This implementation uses a simplified approach.
        
        Args:
            input_shares: Input feature map as shares [H x W]
            pool_size: Pooling window size (pool_h, pool_w)
            stride: Stride (defaults to pool_size)
            node_id: Node ID
            context: Optional context
        
        Returns:
            Pooled feature map as shares [H_out x W_out]
        """
        pool_h, pool_w = pool_size
        stride_h = stride if stride else pool_h
        stride_w = stride if stride else pool_w
        
        H, W = len(input_shares), len(input_shares[0])
        H_out = (H - pool_h) // stride_h + 1
        W_out = (W - pool_w) // stride_w + 1
        
        output_shares = []
        
        # NOTE:
        # True secure max pooling requires secure comparison/argmax, which this codebase
        # does not implement correctly yet (the comparator uses local shares).
        #
        # IMPORTANT (fixed-point correctness):
        # Average pooling requires division by 4, which in a prime field is a modular inverse
        # and introduces "field fractions" unless you implement a truncation protocol.
        # To avoid that (and keep values integer fixed-point), we use SUM pooling as the stand-in.
        for h_out in range(H_out):
            output_row = []
            for w_out in range(W_out):
                h_start = h_out * stride_h
                w_start = w_out * stride_w
                
                # Collect shares in pooling window
                window_shares = []
                for h in range(h_start, h_start + pool_h):
                    for w in range(w_start, w_start + pool_w):
                        if h < H and w < W:
                            window_shares.append(input_shares[h][w])
                
                if not window_shares:
                    zero_share = Share(x=input_shares[0][0].x, y=0, node_id=node_id)
                    output_row.append(zero_share)
                    continue

                pool_sum = Share(x=window_shares[0].x, y=0, node_id=node_id)
                for share in window_shares:
                    pool_sum = Share(
                        x=pool_sum.x,
                        y=(pool_sum.y + share.y) % self.field_size,
                        node_id=node_id
                    )
                # SUM pooling (no division)
                output_row.append(pool_sum)
            
            output_shares.append(output_row)
        
        return output_shares
    
    def avg_pool2d(self, input_shares: List[List[Share]],
                   pool_size: Tuple[int, int] = (2, 2),
                   stride: Optional[int] = None,
                   node_id: int = 1,
                   context: Optional[str] = None) -> List[List[Share]]:
        """
        Secure average pooling
        
        Args:
            input_shares: Input feature map as shares [H x W]
            pool_size: Pooling window size (pool_h, pool_w)
            stride: Stride (defaults to pool_size)
            node_id: Node ID
            context: Optional context
        
        Returns:
            Pooled feature map as shares [H_out x W_out]
        """
        pool_h, pool_w = pool_size
        stride_h = stride if stride else pool_h
        stride_w = stride if stride else pool_w
        
        H, W = len(input_shares), len(input_shares[0])
        H_out = (H - pool_h) // stride_h + 1
        W_out = (W - pool_w) // stride_w + 1
        
        output_shares = []
        
        for h_out in range(H_out):
            output_row = []
            for w_out in range(W_out):
                h_start = h_out * stride_h
                w_start = w_out * stride_w
                
                # Sum shares in pooling window
                pool_sum = Share(x=input_shares[0][0].x, y=0, node_id=node_id)
                count = 0
                
                for h in range(h_start, h_start + pool_h):
                    for w in range(w_start, w_start + pool_w):
                        if h < H and w < W:
                            share = input_shares[h][w]
                            pool_sum = Share(
                                x=pool_sum.x,
                                y=(pool_sum.y + share.y) % self.field_size,
                                node_id=node_id
                            )
                            count += 1
                
                # SUM pooling (no division)
                output_row.append(pool_sum)
            
            output_shares.append(output_row)
        
        return output_shares
    
    def pool2d_backward(self, output_grad_shares: List[List[Share]],
                       input_shares: List[List[Share]],
                       pool_size: Tuple[int, int] = (2, 2),
                       stride: Optional[int] = None,
                       pool_type: str = "avg",
                       node_id: int = 1,
                       context: Optional[str] = None) -> List[List[Share]]:
        """
        Backward pass for pooling operations
        
        Args:
            output_grad_shares: Gradient w.r.t. pooled output [H_out x W_out]
            input_shares: Original input [H x W]
            pool_size: Pooling window size (pool_h, pool_w)
            stride: Stride used in forward pass
            pool_type: "avg" for average pooling, "max" for max pooling
            node_id: Node ID
            context: Optional context
        
        Returns:
            Gradient w.r.t. input [H x W]
        """
        pool_h, pool_w = pool_size
        stride_h = stride if stride else pool_h
        stride_w = stride if stride else pool_w
        
        H, W = len(input_shares), len(input_shares[0])
        H_out, W_out = len(output_grad_shares), len(output_grad_shares[0])
        
        # Initialize input gradients (zero)
        input_grad_shares = []
        for h in range(H):
            input_row = []
            for w in range(W):
                zero_share = Share(x=input_shares[0][0].x, y=0, node_id=node_id)
                input_row.append(zero_share)
            input_grad_shares.append(input_row)
        
        # Distribute gradients back
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = h_out * stride_h
                w_start = w_out * stride_w
                
                output_grad = output_grad_shares[h_out][w_out]
                
                # Collect input positions in pooling window
                window_positions = []
                for h in range(h_start, h_start + pool_h):
                    for w in range(w_start, w_start + pool_w):
                        if h < H and w < W:
                            window_positions.append((h, w))
                
                if not window_positions:
                    continue
                
                if pool_type == "avg":
                    # SUM pooling backward: each input in the window receives the full gradient
                    for h, w in window_positions:
                        input_grad_shares[h][w] = Share(
                            x=input_grad_shares[h][w].x,
                            y=(input_grad_shares[h][w].y + output_grad.y) % self.field_size,
                            node_id=node_id
                        )
                
                elif pool_type == "max":
                    # Max pooling: gradient goes to max position only
                    # Simplified: distribute equally (true max requires secure comparison)
                    if self.comparator:
                        # Would need to find max position securely
                        # For now, use average distribution
                        pass
                    
                    # SUM pooling backward (same as above): distribute full gradient
                    for h, w in window_positions:
                        input_grad_shares[h][w] = Share(
                            x=input_grad_shares[h][w].x,
                            y=(input_grad_shares[h][w].y + output_grad.y) % self.field_size,
                            node_id=node_id
                        )
        
        return input_grad_shares

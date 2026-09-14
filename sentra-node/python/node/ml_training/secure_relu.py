"""
Secure ReLU Activation Function
Implements ReLU(x) = max(0, x) using secure comparison and multiplication
"""

from typing import List
from ml_training.secret_sharing import Share
from ml_training.secure_comparison import SecureComparator
from ml_training.beaver_triples import SecureMultiplier


class SecureReLU:
    """
    Performs secure ReLU activation: ReLU(x) = max(0, x)
    """
    
    def __init__(self, comparator: SecureComparator, multiplier: SecureMultiplier,
                 field_size: int = 2**32 - 5):
        """
        Initialize secure ReLU
        
        Args:
            comparator: SecureComparator instance for secure comparison
            multiplier: SecureMultiplier instance for secure multiplication
            field_size: Prime field size
        """
        self.comparator = comparator
        self.multiplier = multiplier
        self.field_size = field_size
    
    def relu(self, x_share: Share, node_id: int, context: str = None) -> Share:
        """
        Compute ReLU(x) = max(0, x) securely
        
        ReLU(x) = x if x > 0, else 0
        This is computed as: x * (x > 0)
        
        Args:
            x_share: Share of input value
            node_id: Node ID
            context: Optional context for debugging
        
        Returns:
            Share of ReLU(x)
        """
        # Create zero share (0 in the field)
        zero_share = Share(x=x_share.x, y=0, node_id=node_id)
        
        # Securely compare x > 0
        # Returns 1 if x > 0, 0 if x <= 0
        is_positive = self.comparator.secure_greater_than(x_share, zero_share, node_id)
        
        # ReLU(x) = x * (x > 0)
        # If x > 0: is_positive = 1, so result = x * 1 = x
        # If x <= 0: is_positive = 0, so result = x * 0 = 0
        relu_share = self.multiplier.multiply(
            x_share, is_positive, node_id,
            context=context if context else "relu"
        )
        
        return relu_share
    
    def relu_list(self, x_shares: List[Share], node_id: int, context: str = None) -> List[Share]:
        """
        Apply ReLU to a list of shares using true batching.
        """
        relu_vals, _mask = self.relu_list_with_mask(x_shares, node_id=node_id, context=context)
        return relu_vals

    def relu_list_with_mask(self, x_shares: List[Share], node_id: int, context: str = None):
        """
        Apply ReLU to a list of shares and also return the (x > 0) mask shares.
        Useful for reusing the same mask during backward pass.
        """
        if not x_shares:
            return [], []
            
        import numpy as np
        
        # 1. Check if x > 0 for all elements
        zero_shares = [Share(x=x_shares[0].x, y=0, node_id=node_id) for _ in range(len(x_shares))]
        cmp_ctx = f"{context}_cmp" if context else "relu_cmp_batch"
        is_positive_shares = self.comparator.secure_greater_than_batch(
            x_shares, zero_shares, node_id, context=cmp_ctx
        )
        
        # 2. Extract arrays
        y1 = np.array([s.y for s in x_shares], dtype=np.uint64)
        y2 = np.array([s.y for s in is_positive_shares], dtype=np.uint64)
        if y1.shape != y2.shape:
            raise RuntimeError(
                f"relu_list mask length mismatch: x={y1.shape} mask={y2.shape} context={context}"
            )
        
        # 3. Batched multiplication: x * (x > 0)
        prod = self.multiplier.multiply_batch_values(
            y1, y2, x=x_shares[0].x, node_id=node_id, context_prefix=context if context else "relu_batch"
        )
        
        if self.multiplier.reconstruction_manager is not None and context and "e0_b0" in context[:5]:
             is_pos_dbg = [self.multiplier.reconstruction_manager.get_reconstructed_value([is_positive_shares[i]], f"{context}_is_pos_dbg_{i}", use_cache=False) for i in range(5)]
             print(f"DEBUG BATCH 1 IS_POS RECON: {is_pos_dbg}", flush=True)

        # 4. Convert back to shares
        relu_shares = [Share(x=x_shares[0].x, y=int(v), node_id=node_id) for v in prod]
        return relu_shares, is_positive_shares
    
    def relu_2d(self, x_shares_2d: List[List[Share]], node_id: int, context: str = None) -> List[List[Share]]:
        """
        Apply ReLU to a 2D array of shares (e.g., feature map)
        
        Args:
            x_shares_2d: 2D list of input shares [H x W]
            node_id: Node ID
            context: Optional context for debugging
        
        Returns:
            2D list of ReLU output shares [H x W]
        """
        relu_outputs = []
        for h, row in enumerate(x_shares_2d):
            relu_row = []
            for w, x_share in enumerate(row):
                relu_ctx = f"{context}_{h}_{w}" if context else None
                relu_out = self.relu(x_share, node_id, context=relu_ctx)
                relu_row.append(relu_out)
            relu_outputs.append(relu_row)
        
        return relu_outputs
    
    def relu_3d(self, x_shares_3d: List[List[List[Share]]], node_id: int, context: str = None) -> List[List[List[Share]]]:
        """
        Apply ReLU to a 3D array of shares (e.g., multi-channel feature map)
        
        Args:
            x_shares_3d: 3D list of input shares [H x W x C]
            node_id: Node ID
            context: Optional context for debugging
        
        Returns:
            3D list of ReLU output shares [H x W x C]
        """
        relu_outputs = []
        for h, row in enumerate(x_shares_3d):
            relu_row = []
            for w, col in enumerate(row):
                relu_col = []
                for c, x_share in enumerate(col):
                    relu_ctx = f"{context}_{h}_{w}_{c}" if context else None
                    relu_out = self.relu(x_share, node_id, context=relu_ctx)
                    relu_col.append(relu_out)
                relu_row.append(relu_col)
            relu_outputs.append(relu_row)
        
        return relu_outputs
    
    def relu_backward(self, output_grad_share: Share, input_share: Share,
                     node_id: int, context: str = None) -> Share:
        """
        Compute gradient of ReLU w.r.t. input
        
        dReLU/dx = 1 if x > 0, else 0
        So: grad_input = output_grad * (input > 0)
        
        Args:
            output_grad_share: Gradient w.r.t. ReLU output
            input_share: Original input to ReLU (before activation)
            node_id: Node ID
            context: Optional context for debugging
        
        Returns:
            Share of gradient w.r.t. input
        """
        # Create zero share
        zero_share = Share(x=input_share.x, y=0, node_id=node_id)
        
        # Check if input > 0
        is_positive = self.comparator.secure_greater_than(input_share, zero_share, node_id)
        
        # grad_input = output_grad * (input > 0)
        input_grad = self.multiplier.multiply(
            output_grad_share, is_positive, node_id,
            context=context if context else "relu_backward"
        )
        
        return input_grad
    
    def relu_backward_list(self, output_grad_shares: List[Share], input_shares: List[Share],
                          node_id: int, context: str = None) -> List[Share]:
        """
        Compute ReLU backward for a list of shares using true batching.
        """
        if not input_shares or not output_grad_shares:
            return []
            
        import numpy as np
        
        # 1. Check if input > 0
        zero_shares = [Share(x=input_shares[0].x, y=0, node_id=node_id) for _ in range(len(input_shares))]
        cmp_ctx = f"{context}_cmp" if context else "relu_bw_cmp_batch"
        is_positive_shares = self.comparator.secure_greater_than_batch(
            input_shares, zero_shares, node_id, context=cmp_ctx
        )
        
        # 2. Extract arrays
        y_grad = np.array([s.y for s in output_grad_shares], dtype=np.uint64)
        y_pos = np.array([s.y for s in is_positive_shares], dtype=np.uint64)
        if y_grad.shape != y_pos.shape:
            raise RuntimeError(
                f"relu_backward_list mask length mismatch: grad={y_grad.shape} mask={y_pos.shape} context={context}"
            )
        
        return self.relu_backward_with_mask_list(
            output_grad_shares=output_grad_shares,
            is_positive_shares=is_positive_shares,
            node_id=node_id,
            context=context,
        )

    def relu_backward_with_mask_list(
        self,
        output_grad_shares: List[Share],
        is_positive_shares: List[Share],
        node_id: int,
        context: str = None,
    ) -> List[Share]:
        """
        Compute ReLU backward given a precomputed (input > 0) mask.
        This avoids repeating secure comparison in backward pass.
        """
        if not output_grad_shares or not is_positive_shares:
            return []
        if len(output_grad_shares) != len(is_positive_shares):
            raise ValueError("output_grad_shares and is_positive_shares length mismatch")

        import numpy as np
        y_grad = np.array([s.y for s in output_grad_shares], dtype=np.uint64)
        y_pos = np.array([s.y for s in is_positive_shares], dtype=np.uint64)
        if y_grad.shape != y_pos.shape:
            raise RuntimeError(
                f"relu_backward_with_mask_list shape mismatch: grad={y_grad.shape} mask={y_pos.shape} context={context}"
            )

        prod = self.multiplier.multiply_batch_values(
            y_grad, y_pos, x=output_grad_shares[0].x, node_id=node_id, context_prefix=context if context else "relu_bw_batch"
        )
        return [Share(x=output_grad_shares[0].x, y=int(v), node_id=node_id) for v in prod]

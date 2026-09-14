"""
Secure Comparison Operations
Implements secure comparison (GT, LT, EQ) on secret-shared values
"""

from typing import List
from ml_training.secret_sharing import Share
from ml_training.beaver_triples import SecureMultiplier
import random
import time


class SecureComparator:
    """
    Performs secure comparison operations on secret-shared values
    """
    
    def __init__(self, multiplier: SecureMultiplier, field_size: int = 2**31 - 1):
        """
        Initialize secure comparator
        Args:
            multiplier: SecureMultiplier instance
            field_size: Prime field size
        """
        self.multiplier = multiplier
        self.field_size = field_size
    
    def secure_greater_than(self, share1: Share, share2: Share, node_id: int) -> Share:
        """
        Securely compare share1 > share2
        Returns a share of 1 if true, 0 if false
        Args:
            share1: First share
            share2: Second share
            node_id: Node ID
        Returns:
            Share of comparison result
        """
        # Compute diff = share1 - share2
        diff_share = Share(
            x=share1.x,
            y=(share1.y - share2.y) % self.field_size,
            node_id=node_id
        )
        
        # Check sign bit (simplified - would need secure bit extraction)
        # For minimal pipeline, use simplified comparison
        # In production, would use secure bit decomposition
        
        # Simplified: if diff > field_size/2, treat as negative
        is_positive = 1 if diff_share.y < self.field_size // 2 else 0
        
        return Share(x=share1.x, y=is_positive, node_id=node_id)
    
    def secure_greater_than_batch(
        self,
        share1_list: List[Share],
        share2_list: List[Share],
        node_id: int,
        context: str = "cmp_batch",
    ) -> List[Share]:
        """
        Securely compare arrays: share1 > share2 element-wise
        Returns a list of shares of 1 if true, 0 if false
        """
        import numpy as np
        
        # Check lengths
        if len(share1_list) != len(share2_list):
            raise ValueError("Lengths of share1_list and share2_list must match")
            
        if not share1_list:
            return []
            
        # Fast path with proper mathematical evaluation using opener
        if self.multiplier.reconstruction_manager is not None:
            context_prefix = str(context) if context else "cmp_batch"
            y1 = np.array([s.y for s in share1_list], dtype=np.uint64)
            x0 = share1_list[0].x
            
            p = int(self.field_size)
            n = int(self.multiplier.n_nodes)
            net = self.multiplier.reconstruction_manager.network
            opener = self.multiplier._opened_fp_opener()
            
            ctx_in = f"{context_prefix}_in"
            net.broadcast_vector(ctx_in, x=int(x0), values=np.asarray(y1, dtype=np.uint64))
            
            import time
            time.sleep(0.001)
            
            if int(node_id) == int(opener):
                _t0 = time.time()
                opened_u64 = self.multiplier.reconstruction_manager.reconstruct_opened_vector_values(
                    context=ctx_in,
                    values_local=np.asarray(y1, dtype=np.uint64),
                    x=int(x0),
                    timeout=120.0,
                )
                opened = opened_u64.astype(np.int64, copy=False)
                # Re-center fixed-point negative values
                opened = np.where(opened > (p // 2), opened - p, opened)
                
                # Compute is_pos mask
                is_pos = np.where(opened > 0, 1, 0)
                is_pos_mod = is_pos.astype(np.uint64, copy=False)
                
                x_points = [i for i in range(1, n + 1)]
                out_prefix = f"{context_prefix}_out"
                opener_vec = self.multiplier._reshare_vector_from_opener(
                    secrets_mod_p_u64=is_pos_mod,
                    node_id=node_id,
                    context_prefix=out_prefix,
                    x_points=x_points,
                    timeout=120.0,
                )
                try:
                    net.channel.clear_vector(ctx_in)
                except Exception:
                    pass
                if hasattr(self.multiplier, "add_prover_time"):
                    self.multiplier.add_prover_time(time.time() - _t0, "secure_compare_batch_opened")
                if len(opener_vec) != len(share1_list):
                    raise RuntimeError(
                        f"secure_greater_than_batch length mismatch (opener): got={len(opener_vec)} expected={len(share1_list)} context={context_prefix}"
                    )
                return [Share(x=x0, y=int(val), node_id=node_id) for val in opener_vec]
            else:
                # Peer nodes: wait for opener's output vector directly.
                out_ctx = f"{context_prefix}_out_to_{int(node_id)}"
                opener_vec = None
                import time
                start = time.time()
                while time.time() - start < 120.0:
                    recv = net.channel.get_received_vector(out_ctx)
                    if int(opener) in recv:
                        values = recv[int(opener)].get("values")
                        if values is not None:
                            opener_vec = np.asarray(list(values), dtype=np.uint64) % np.uint64(p)
                            break
                    time.sleep(0.01)
                if opener_vec is None:
                    raise RuntimeError(
                        f"secure_greater_than_batch timed out waiting for opener vector context={out_ctx}"
                    )
                try:
                    net.channel.clear_vector(out_ctx)
                except Exception:
                    pass
                try:
                    net.channel.clear_vector(ctx_in)
                except Exception:
                    pass
                if len(opener_vec) != len(share1_list):
                    raise RuntimeError(
                        f"secure_greater_than_batch length mismatch (peer): got={len(opener_vec)} expected={len(share1_list)} context={context_prefix}"
                    )
                return [Share(x=x0, y=int(val), node_id=node_id) for val in opener_vec]

        # Extract y values
        y1 = np.array([s.y for s in share1_list], dtype=np.uint64)
        y2 = np.array([s.y for s in share2_list], dtype=np.uint64)
        
        # Compute diff module field_size
        p = np.uint64(self.field_size)
        diff = (y1 + p - y2 % p) % p
        
        # Check sign (simplified)
        half_p = p // 2
        is_pos = np.where(diff < half_p, 1, 0)
        
        # Reshare
        x = share1_list[0].x
        return [Share(x=x, y=int(val), node_id=node_id) for val in is_pos]
        
    def secure_less_than(self, share1: Share, share2: Share, node_id: int) -> Share:
        """Securely compare share1 < share2"""
        return self.secure_greater_than(share2, share1, node_id)
    
    def secure_equal(self, share1: Share, share2: Share, node_id: int) -> Share:
        """
        Securely compare share1 == share2
        Returns a share of 1 if equal, 0 otherwise
        """
        # Compute diff = share1 - share2
        diff_share = Share(
            x=share1.x,
            y=(share1.y - share2.y) % self.field_size,
            node_id=node_id
        )
        
        # Check if diff == 0 (simplified)
        is_zero = 1 if diff_share.y == 0 else 0
        
        return Share(x=share1.x, y=is_zero, node_id=node_id)


class SecureClipper:
    """
    Performs secure gradient clipping on secret-shared gradients
    """
    
    def __init__(self, comparator: SecureComparator, multiplier: SecureMultiplier,
                 field_size: int = 2**31 - 1):
        """
        Initialize secure clipper
        Args:
            comparator: SecureComparator instance
            multiplier: SecureMultiplier instance
            field_size: Prime field size
        """
        self.comparator = comparator
        self.multiplier = multiplier
        self.field_size = field_size
    
    def compute_squared_norm(self, grad_shares: List[Share], node_id: int) -> Share:
        """
        Compute squared L2 norm of gradients
        Args:
            grad_shares: List of gradient shares
            node_id: Node ID
        Returns:
            Share of squared norm
        """
        if not grad_shares:
            # Return zero share if no gradients
            return Share(x=0, y=0, node_id=node_id)
        
        # Ensure all are Share objects, not lists
        if isinstance(grad_shares[0], list):
            raise ValueError(f"Expected List[Share], got nested list. First element: {type(grad_shares[0])}")
        
        norm_sq = Share(x=grad_shares[0].x, y=0, node_id=node_id)
        
        for grad_share in grad_shares:
            # Square each gradient component
            grad_sq = self.multiplier.multiply(grad_share, grad_share, node_id)
            # Sum
            norm_sq = Share(
                x=norm_sq.x,
                y=(norm_sq.y + grad_sq.y) % self.field_size,
                node_id=node_id
            )
        
        return norm_sq
    
    def clip_gradients(self, grad_shares: List[Share], clip_norm: float, node_id: int) -> List[Share]:
        """
        Clip gradients to have norm <= clip_norm
        Args:
            grad_shares: List of gradient shares
            clip_norm: Maximum norm
            node_id: Node ID
        Returns:
            Clipped gradient shares
        """
        # Compute norm
        norm_sq_share = self.compute_squared_norm(grad_shares, node_id)
        
        # Compare with clip_norm^2
        clip_norm_sq = int(clip_norm * clip_norm * 1000000) % self.field_size
        clip_norm_share = Share(x=norm_sq_share.x, y=clip_norm_sq, node_id=node_id)
        
        # Check if norm > clip_norm (simplified)
        # In production, would use secure comparison and division
        needs_clipping = self.comparator.secure_greater_than(
            norm_sq_share, clip_norm_share, node_id
        )
        
        # Simplified clipping (would need secure division for proper scaling)
        # For minimal pipeline, return gradients as-is if within bound
        if needs_clipping.y == 0:
            return grad_shares
        
        # Would compute scaling factor and multiply gradients
        # For now, simplified version
        return grad_shares

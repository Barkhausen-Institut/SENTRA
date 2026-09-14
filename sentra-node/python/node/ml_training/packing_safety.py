"""
Packing safety for SENTRA (protocol step 5).

Central guard: 2*(t + s - 1) < n_active.
- t: privacy threshold (need t+1 shares to reconstruct)
- s: packing factor (PSS packing factor, or adversarial share limit depending on context)
- n_active: number of currently active nodes

On failure: lower s, re-pack via DPSS, or stop with a clear error.
"""


def check_packing_safety(t: int, s: int, n_active: int) -> bool:
    """
    Check packing safety bound: 2*(t + s - 1) < n_active.
    Returns True if safe to proceed with packing factor s.
    """
    return 2 * (t + s - 1) < n_active


def get_max_safe_packing_factor(t: int, n_active: int, cap: int = 16) -> int:
    """
    Maximum packing factor s such that 2*(t+s-1) < n_active.
    Returns 1 if bound cannot be satisfied.
    """
    if not check_packing_safety(t, 1, n_active):
        return 1
    # Strict: 2*(t+s-1) < n_active  =>  s < (n_active - 2t + 2) / 2  =>  max s = floor((n_active - 2t + 1) / 2)
    max_s = (n_active - 2 * t + 1) // 2
    return max(1, min(cap, max_s))


class PackingSafetyError(RuntimeError):
    """Raised when packing safety bound is violated."""

    def __init__(self, t: int, s: int, n_active: int):
        bound = 2 * (t + s - 1)
        super().__init__(
            f"Packing safety violated: 2*(t+s-1)={bound} >= n_active={n_active}. "
            f"Lower packing factor s, run DPSS resharing with smaller s, or increase n_active."
        )

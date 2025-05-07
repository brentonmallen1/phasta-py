"""Acceleration module for PHASTA-Py.

This module provides hardware acceleration backends for PHASTA-Py using
CUDA, Metal, or CPU as a fallback option.
"""

from .base import AccelerationBackend, AccelerationContext, get_current_backend, set_current_backend
from .cuda import CUDABackend
from .metal import MetalBackend
from .cpu import CPUBackend


def get_best_available_backend() -> AccelerationBackend:
    """Get the best available acceleration backend.
    
    This function checks for available acceleration backends in the following order:
    1. CUDA (if available)
    2. Metal (if available)
    3. CPU (always available as fallback)
    
    Returns:
        AccelerationBackend: The best available acceleration backend
    """
    # Try CUDA first
    cuda_backend = CUDABackend()
    if cuda_backend.is_available():
        return cuda_backend
    
    # Try Metal next
    metal_backend = MetalBackend()
    if metal_backend.is_available():
        return metal_backend
    
    # Fall back to CPU
    return CPUBackend()


__all__ = [
    'AccelerationBackend',
    'AccelerationContext',
    'get_current_backend',
    'set_current_backend',
    'CUDABackend',
    'MetalBackend',
    'CPUBackend',
    'get_best_available_backend'
]

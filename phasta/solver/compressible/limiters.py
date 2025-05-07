"""
Limiters for compressible flow solver.

This module implements various limiting strategies for the compressible
Navier-Stokes equations, including:
- Slope limiters
- Flux limiters
- Pressure limiters
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class LimiterConfig:
    """Configuration for limiting strategies."""
    scheme: str  # Name of the limiting scheme
    beta: float = 1.0  # Parameter for slope limiters
    epsilon: float = 1e-6  # Small number to prevent division by zero
    params: Optional[Dict[str, float]] = None  # Additional scheme parameters

class Limiter:
    """Base class for limiting strategies."""
    
    def __init__(self, config: LimiterConfig):
        """Initialize limiter with configuration."""
        self.config = config
    
    def compute_slope_limiter(self, u: np.ndarray, dx: float) -> np.ndarray:
        """Compute slope limiter."""
        raise NotImplementedError
    
    def compute_flux_limiter(self, r: np.ndarray) -> np.ndarray:
        """Compute flux limiter."""
        raise NotImplementedError
    
    def compute_pressure_limiter(self, p: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Compute pressure limiter."""
        raise NotImplementedError

class SlopeLimiter(Limiter):
    """Slope limiting strategies."""
    
    def compute_slope_limiter(self, u: np.ndarray, dx: float) -> np.ndarray:
        """
        Compute slope limiter.
        
        Args:
            u: Solution values
            dx: Grid spacing
            
        Returns:
            np.ndarray: Limited slopes
        """
        # Compute gradients
        du = np.zeros_like(u)
        du[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        
        # Compute ratio of consecutive gradients
        r = np.zeros_like(u)
        r[1:-1] = (u[2:] - u[1:-1]) / (u[1:-1] - u[:-2] + self.config.epsilon)
        
        # Compute limiter
        phi = self.compute_flux_limiter(r)
        
        # Apply limiter to slopes
        du_limited = du * phi
        
        return du_limited
    
    def compute_flux_limiter(self, r: np.ndarray) -> np.ndarray:
        """
        Compute flux limiter for slope limiting.
        
        Args:
            r: Ratio of consecutive gradients
            
        Returns:
            np.ndarray: Limiter values
        """
        # Minmod limiter
        phi = np.zeros_like(r)
        mask = r > 0
        phi[mask] = np.minimum(1, r[mask])
        return phi

class FluxLimiter(Limiter):
    """Flux limiting strategies."""
    
    def compute_flux_limiter(self, r: np.ndarray) -> np.ndarray:
        """
        Compute flux limiter.
        
        Args:
            r: Ratio of consecutive gradients
            
        Returns:
            np.ndarray: Limiter values
        """
        # Superbee limiter
        phi = np.zeros_like(r)
        mask = r > 0
        phi[mask] = np.maximum(
            np.minimum(2 * r[mask], 1),
            np.minimum(r[mask], 2)
        )
        return phi

class PressureLimiter(Limiter):
    """Pressure limiting strategies."""
    
    def compute_pressure_limiter(self, p: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """
        Compute pressure limiter.
        
        Args:
            p: Pressure values
            rho: Density values
            
        Returns:
            np.ndarray: Limited pressure values
        """
        # Compute pressure ratio
        p_ratio = np.zeros_like(p)
        p_ratio[1:-1] = p[1:-1] / (p[2:] + p[:-2] + self.config.epsilon)
        
        # Compute limiter
        phi = self.compute_flux_limiter(p_ratio)
        
        # Apply limiter to pressure
        p_limited = p * phi
        
        # Ensure positive pressure
        p_limited = np.maximum(p_limited, 0.0)
        
        return p_limited
    
    def compute_flux_limiter(self, r: np.ndarray) -> np.ndarray:
        """
        Compute flux limiter for pressure limiting.
        
        Args:
            r: Ratio of consecutive values
            
        Returns:
            np.ndarray: Limiter values
        """
        # Van Leer limiter
        phi = np.zeros_like(r)
        mask = r > 0
        phi[mask] = 2 * r[mask] / (1 + r[mask])
        return phi

def create_limiter(config: LimiterConfig) -> Limiter:
    """Factory function to create limiter based on configuration."""
    if config.scheme.lower() == "slope":
        return SlopeLimiter(config)
    elif config.scheme.lower() == "flux":
        return FluxLimiter(config)
    elif config.scheme.lower() == "pressure":
        return PressureLimiter(config)
    else:
        raise ValueError(f"Unknown limiting scheme: {config.scheme}") 
"""
Shock capturing schemes for compressible flow solver.

This module implements various shock capturing schemes for the compressible
Navier-Stokes equations, including:
- TVD schemes
- WENO reconstruction
- Artificial viscosity
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ShockCapturingConfig:
    """Configuration for shock capturing schemes."""
    scheme: str  # Name of the shock capturing scheme
    kappa: float = 1.0  # Parameter for TVD/WENO schemes
    epsilon: float = 1e-6  # Small number to prevent division by zero
    c_vis: float = 0.1  # Coefficient for artificial viscosity
    params: Optional[Dict[str, float]] = None  # Additional scheme parameters

class ShockCapturing:
    """Base class for shock capturing schemes."""
    
    def __init__(self, config: ShockCapturingConfig):
        """Initialize shock capturing scheme."""
        self.config = config
    
    def compute_limiter(self, r: np.ndarray) -> np.ndarray:
        """Compute limiter function."""
        raise NotImplementedError
    
    def compute_reconstruction(self, u: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute reconstructed values at cell interfaces."""
        raise NotImplementedError
    
    def compute_artificial_viscosity(self, u: np.ndarray, dx: float) -> np.ndarray:
        """Compute artificial viscosity."""
        raise NotImplementedError

class TVDScheme(ShockCapturing):
    """Total Variation Diminishing scheme."""
    
    def compute_limiter(self, r: np.ndarray) -> np.ndarray:
        """
        Compute TVD limiter function.
        
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
    
    def compute_reconstruction(self, u: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute TVD reconstruction at cell interfaces.
        
        Args:
            u: Solution values
            dx: Grid spacing
            
        Returns:
            tuple: (u_left, u_right) reconstructed values
        """
        # Compute gradients
        du = np.zeros_like(u)
        du[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        
        # Compute ratio of consecutive gradients
        r = np.zeros_like(u)
        r[1:-1] = (u[2:] - u[1:-1]) / (u[1:-1] - u[:-2] + self.config.epsilon)
        
        # Compute limiter
        phi = self.compute_limiter(r)
        
        # Compute reconstructed values
        u_left = u + 0.5 * dx * du * phi
        u_right = u - 0.5 * dx * du * phi
        
        return u_left, u_right

class WENOScheme(ShockCapturing):
    """Weighted Essentially Non-Oscillatory scheme."""
    
    def compute_smoothness_indicators(self, u: np.ndarray, dx: float) -> np.ndarray:
        """
        Compute smoothness indicators for WENO reconstruction.
        
        Args:
            u: Solution values
            dx: Grid spacing
            
        Returns:
            np.ndarray: Smoothness indicators
        """
        beta = np.zeros((3, len(u)))
        
        # First stencil
        beta[0, 2:-2] = (
            13/12 * (u[2:-2] - 2*u[1:-3] + u[:-4])**2 +
            0.25 * (3*u[2:-2] - 4*u[1:-3] + u[:-4])**2
        )
        
        # Second stencil
        beta[1, 2:-2] = (
            13/12 * (u[3:-1] - 2*u[2:-2] + u[1:-3])**2 +
            0.25 * (u[3:-1] - u[1:-3])**2
        )
        
        # Third stencil
        beta[2, 2:-2] = (
            13/12 * (u[4:] - 2*u[3:-1] + u[2:-2])**2 +
            0.25 * (3*u[3:-1] - 4*u[2:-2] + u[1:-3])**2
        )
        
        return beta
    
    def compute_weights(self, beta: np.ndarray) -> np.ndarray:
        """
        Compute WENO weights.
        
        Args:
            beta: Smoothness indicators
            
        Returns:
            np.ndarray: WENO weights
        """
        # Linear weights
        gamma = np.array([0.1, 0.6, 0.3])
        
        # Compute nonlinear weights
        alpha = gamma / (self.config.epsilon + beta)**2
        weights = alpha / np.sum(alpha, axis=0)
        
        return weights
    
    def compute_reconstruction(self, u: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute WENO reconstruction at cell interfaces.
        
        Args:
            u: Solution values
            dx: Grid spacing
            
        Returns:
            tuple: (u_left, u_right) reconstructed values
        """
        # Compute smoothness indicators
        beta = self.compute_smoothness_indicators(u, dx)
        
        # Compute weights
        weights = self.compute_weights(beta)
        
        # Compute reconstructed values
        u_left = np.zeros_like(u)
        u_right = np.zeros_like(u)
        
        # Left interface
        u_left[2:-2] = (
            weights[0, 2:-2] * (2*u[1:-3] - u[:-4]) +
            weights[1, 2:-2] * (2*u[2:-2] - u[1:-3]) +
            weights[2, 2:-2] * (2*u[3:-1] - u[2:-2])
        )
        
        # Right interface
        u_right[2:-2] = (
            weights[0, 2:-2] * (2*u[2:-2] - u[1:-3]) +
            weights[1, 2:-2] * (2*u[3:-1] - u[2:-2]) +
            weights[2, 2:-2] * (2*u[4:] - u[3:-1])
        )
        
        return u_left, u_right

class ArtificialViscosity(ShockCapturing):
    """Artificial viscosity for shock capturing."""
    
    def compute_artificial_viscosity(self, u: np.ndarray, dx: float) -> np.ndarray:
        """
        Compute artificial viscosity.
        
        Args:
            u: Solution values
            dx: Grid spacing
            
        Returns:
            np.ndarray: Artificial viscosity
        """
        # Compute second derivative
        d2u = np.zeros_like(u)
        d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        
        # Compute artificial viscosity
        nu = self.config.c_vis * dx**2 * np.abs(d2u)
        
        return nu

def create_shock_capturing(config: ShockCapturingConfig) -> ShockCapturing:
    """Factory function to create shock capturing scheme."""
    if config.scheme.lower() == "tvd":
        return TVDScheme(config)
    elif config.scheme.lower() == "weno":
        return WENOScheme(config)
    elif config.scheme.lower() == "artificial_viscosity":
        return ArtificialViscosity(config)
    else:
        raise ValueError(f"Unknown shock capturing scheme: {config.scheme}") 
"""
Flux assembly module for the compressible flow solver.

This module implements the flux calculations and assembly from the original PHASTA codebase.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FluxConfig:
    """Configuration for flux calculations."""
    gamma: float = 1.4  # Ratio of specific heats
    prandtl: float = 0.72  # Prandtl number
    reynolds: float = 1.0  # Reynolds number
    mach: float = 0.1  # Mach number
    viscosity_model: str = "constant"  # Viscosity model type

class FluxAssembler:
    """
    Implements the flux calculations and assembly for compressible flow.
    
    This class implements the flux calculations and assembly from the original PHASTA codebase.
    """
    
    def __init__(self, config: FluxConfig):
        """
        Initialize the flux assembler.
        
        Args:
            config: Flux configuration parameters
        """
        self.config = config
        
    def compute_convective_flux(self, 
                              u: np.ndarray,  # Velocity
                              p: np.ndarray,  # Pressure
                              rho: np.ndarray,  # Density
                              T: np.ndarray,  # Temperature
                              normal: np.ndarray  # Face normal
                              ) -> np.ndarray:
        """
        Compute the convective flux.
        
        Args:
            u: Velocity vector
            p: Pressure
            rho: Density
            T: Temperature
            normal: Face normal vector
            
        Returns:
            np.ndarray: Convective flux vector
        """
        # Compute velocity normal to face
        un = np.dot(u, normal)
        
        # Compute total energy
        e = p / ((self.config.gamma - 1) * rho) + 0.5 * np.sum(u**2)
        
        # Compute convective flux
        flux = np.zeros(5)  # [rho, rho*u, rho*v, rho*w, rho*e]
        flux[0] = rho * un
        flux[1:4] = rho * u * un + p * normal
        flux[4] = rho * e * un + p * un
        
        return flux
        
    def compute_viscous_flux(self,
                           u: np.ndarray,  # Velocity
                           T: np.ndarray,  # Temperature
                           grad_u: np.ndarray,  # Velocity gradient
                           grad_T: np.ndarray,  # Temperature gradient
                           normal: np.ndarray  # Face normal
                           ) -> np.ndarray:
        """
        Compute the viscous flux.
        
        Args:
            u: Velocity vector
            T: Temperature
            grad_u: Velocity gradient tensor
            grad_T: Temperature gradient vector
            normal: Face normal vector
            
        Returns:
            np.ndarray: Viscous flux vector
        """
        # Compute viscosity
        mu = self._compute_viscosity(T)
        
        # Compute stress tensor
        tau = self._compute_stress_tensor(grad_u, mu)
        
        # Compute heat flux
        q = -self._compute_heat_flux(grad_T, mu)
        
        # Compute viscous flux
        flux = np.zeros(5)  # [0, tau_x, tau_y, tau_z, q + tau·u]
        flux[1:4] = np.dot(tau, normal)
        flux[4] = np.dot(q, normal) + np.dot(np.dot(tau, u), normal)
        
        return flux
        
    def _compute_viscosity(self, T: np.ndarray) -> float:
        """
        Compute the dynamic viscosity.
        
        Args:
            T: Temperature
            
        Returns:
            float: Dynamic viscosity
        """
        if self.config.viscosity_model == "constant":
            return 1.0 / self.config.reynolds
        elif self.config.viscosity_model == "sutherland":
            # Sutherland's law
            T0 = 273.15  # Reference temperature
            S = 110.4  # Sutherland's constant
            mu0 = 1.0 / self.config.reynolds  # Reference viscosity
            return mu0 * (T/T0)**1.5 * (T0 + S)/(T + S)
        else:
            raise ValueError(f"Unknown viscosity model: {self.config.viscosity_model}")
            
    def _compute_stress_tensor(self, 
                             grad_u: np.ndarray, 
                             mu: float) -> np.ndarray:
        """
        Compute the stress tensor.
        
        Args:
            grad_u: Velocity gradient tensor
            mu: Dynamic viscosity
            
        Returns:
            np.ndarray: Stress tensor
        """
        # Compute strain rate tensor
        S = 0.5 * (grad_u + grad_u.T)
        
        # Compute divergence
        div_u = np.trace(grad_u)
        
        # Compute stress tensor
        tau = 2 * mu * (S - (1/3) * div_u * np.eye(3))
        
        return tau
        
    def _compute_heat_flux(self, 
                          grad_T: np.ndarray, 
                          mu: float) -> np.ndarray:
        """
        Compute the heat flux vector.
        
        Args:
            grad_T: Temperature gradient
            mu: Dynamic viscosity
            
        Returns:
            np.ndarray: Heat flux vector
        """
        # Compute thermal conductivity
        k = mu * self.config.gamma / (self.config.prandtl * (self.config.gamma - 1))
        
        # Compute heat flux
        q = -k * grad_T
        
        return q
        
    def assemble_flux(self,
                     u: np.ndarray,
                     p: np.ndarray,
                     rho: np.ndarray,
                     T: np.ndarray,
                     grad_u: np.ndarray,
                     grad_T: np.ndarray,
                     normal: np.ndarray
                     ) -> np.ndarray:
        """
        Assemble the total flux (convective + viscous).
        
        Args:
            u: Velocity vector
            p: Pressure
            rho: Density
            T: Temperature
            grad_u: Velocity gradient tensor
            grad_T: Temperature gradient vector
            normal: Face normal vector
            
        Returns:
            np.ndarray: Total flux vector
        """
        # Compute convective flux
        conv_flux = self.compute_convective_flux(u, p, rho, T, normal)
        
        # Compute viscous flux
        visc_flux = self.compute_viscous_flux(u, T, grad_u, grad_T, normal)
        
        # Assemble total flux
        total_flux = conv_flux + visc_flux
        
        return total_flux 
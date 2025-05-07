"""
Element calculations module for the compressible flow solver.

This module implements the element-level computations from the original PHASTA codebase.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class ElementConfig:
    """Configuration for element calculations."""
    element_type: str = "hex"  # Element type (hex, tet, prism)
    quadrature_order: int = 2  # Quadrature order
    stabilization: str = "SUPG"  # Stabilization method
    shock_capturing: bool = True  # Enable shock capturing
    artificial_viscosity: float = 0.0  # Artificial viscosity coefficient

class ElementCalculator:
    """
    Implements the element-level computations for compressible flow.
    
    This class implements the element-level computations from the original PHASTA codebase.
    """
    
    def __init__(self, config: ElementConfig):
        """
        Initialize the element calculator.
        
        Args:
            config: Element configuration parameters
        """
        self.config = config
        self._setup_quadrature()
        self._setup_shape_functions()
        
    def _setup_quadrature(self):
        """Set up quadrature points and weights."""
        if self.config.element_type == "hex":
            self.quad_points, self.quad_weights = self._setup_hex_quadrature()
        elif self.config.element_type == "tet":
            self.quad_points, self.quad_weights = self._setup_tet_quadrature()
        elif self.config.element_type == "prism":
            self.quad_points, self.quad_weights = self._setup_prism_quadrature()
        else:
            raise ValueError(f"Unknown element type: {self.config.element_type}")
            
    def _setup_shape_functions(self):
        """Set up shape functions and their derivatives."""
        if self.config.element_type == "hex":
            self.shape_functions = self._setup_hex_shape_functions()
        elif self.config.element_type == "tet":
            self.shape_functions = self._setup_tet_shape_functions()
        elif self.config.element_type == "prism":
            self.shape_functions = self._setup_prism_shape_functions()
        else:
            raise ValueError(f"Unknown element type: {self.config.element_type}")
            
    def compute_element_residual(self,
                               element_nodes: np.ndarray,
                               element_solution: np.ndarray,
                               element_gradients: np.ndarray,
                               time_step: float
                               ) -> np.ndarray:
        """
        Compute the element residual.
        
        Args:
            element_nodes: Element node coordinates
            element_solution: Element solution vector
            element_gradients: Element solution gradients
            time_step: Time step size
            
        Returns:
            np.ndarray: Element residual vector
        """
        # Initialize residual
        residual = np.zeros_like(element_solution)
        
        # Loop over quadrature points
        for qp, weight in zip(self.quad_points, self.quad_weights):
            # Compute shape functions and derivatives at quadrature point
            N, dN = self.shape_functions(qp)
            
            # Compute Jacobian and its inverse
            J = np.dot(dN, element_nodes)
            J_inv = np.linalg.inv(J)
            
            # Compute physical derivatives
            dN_phys = np.dot(dN, J_inv)
            
            # Compute solution and gradients at quadrature point
            qp_solution = np.dot(N, element_solution)
            qp_gradients = np.dot(dN_phys, element_gradients)
            
            # Compute flux at quadrature point
            flux = self._compute_flux(qp_solution, qp_gradients)
            
            # Add contribution to residual
            residual += weight * np.abs(J) * (
                self._compute_time_derivative(qp_solution, time_step) +
                self._compute_flux_divergence(flux, dN_phys) +
                self._compute_stabilization(qp_solution, qp_gradients, dN_phys)
            )
            
        return residual
        
    def compute_element_jacobian(self,
                               element_nodes: np.ndarray,
                               element_solution: np.ndarray,
                               element_gradients: np.ndarray,
                               time_step: float
                               ) -> np.ndarray:
        """
        Compute the element Jacobian matrix.
        
        Args:
            element_nodes: Element node coordinates
            element_solution: Element solution vector
            element_gradients: Element solution gradients
            time_step: Time step size
            
        Returns:
            np.ndarray: Element Jacobian matrix
        """
        # Initialize Jacobian
        n_dof = element_solution.shape[0]
        jacobian = np.zeros((n_dof, n_dof))
        
        # Loop over quadrature points
        for qp, weight in zip(self.quad_points, self.quad_weights):
            # Compute shape functions and derivatives at quadrature point
            N, dN = self.shape_functions(qp)
            
            # Compute Jacobian and its inverse
            J = np.dot(dN, element_nodes)
            J_inv = np.linalg.inv(J)
            
            # Compute physical derivatives
            dN_phys = np.dot(dN, J_inv)
            
            # Compute solution and gradients at quadrature point
            qp_solution = np.dot(N, element_solution)
            qp_gradients = np.dot(dN_phys, element_gradients)
            
            # Compute flux Jacobian at quadrature point
            flux_jacobian = self._compute_flux_jacobian(qp_solution, qp_gradients)
            
            # Add contribution to Jacobian
            jacobian += weight * np.abs(J) * (
                self._compute_time_derivative_jacobian(time_step) +
                self._compute_flux_divergence_jacobian(flux_jacobian, dN_phys) +
                self._compute_stabilization_jacobian(qp_solution, qp_gradients, dN_phys)
            )
            
        return jacobian
        
    def _compute_flux(self,
                     solution: np.ndarray,
                     gradients: np.ndarray
                     ) -> np.ndarray:
        """
        Compute the flux at a point.
        
        Args:
            solution: Solution vector at point
            gradients: Solution gradients at point
            
        Returns:
            np.ndarray: Flux vector
        """
        # Extract solution components
        rho = solution[0]
        u = solution[1:4]
        e = solution[4]
        
        # Compute pressure
        p = (self.config.gamma - 1) * (rho * e - 0.5 * rho * np.sum(u**2))
        
        # Compute convective flux
        conv_flux = self._compute_convective_flux(rho, u, p, e)
        
        # Compute viscous flux
        visc_flux = self._compute_viscous_flux(u, gradients)
        
        # Add artificial viscosity if enabled
        if self.config.shock_capturing:
            art_visc = self._compute_artificial_viscosity(solution, gradients)
            visc_flux += art_visc
            
        return conv_flux + visc_flux
        
    def _compute_stabilization(self,
                             solution: np.ndarray,
                             gradients: np.ndarray,
                             dN_phys: np.ndarray
                             ) -> np.ndarray:
        """
        Compute the stabilization term.
        
        Args:
            solution: Solution vector at point
            gradients: Solution gradients at point
            dN_phys: Physical derivatives of shape functions
            
        Returns:
            np.ndarray: Stabilization term
        """
        if self.config.stabilization == "SUPG":
            return self._compute_supg_stabilization(solution, gradients, dN_phys)
        elif self.config.stabilization == "GLS":
            return self._compute_gls_stabilization(solution, gradients, dN_phys)
        else:
            raise ValueError(f"Unknown stabilization method: {self.config.stabilization}")
            
    def _compute_artificial_viscosity(self,
                                    solution: np.ndarray,
                                    gradients: np.ndarray
                                    ) -> np.ndarray:
        """
        Compute the artificial viscosity term.
        
        Args:
            solution: Solution vector at point
            gradients: Solution gradients at point
            
        Returns:
            np.ndarray: Artificial viscosity term
        """
        # Compute solution gradient magnitude
        grad_mag = np.linalg.norm(gradients)
        
        # Compute artificial viscosity coefficient
        nu = self.config.artificial_viscosity * grad_mag
        
        # Compute artificial viscosity term
        return -nu * gradients 
"""
Shape functions module for the compressible flow solver.

This module implements the finite element shape functions from the original PHASTA codebase.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

class ElementType(Enum):
    """Types of finite elements."""
    HEX = 1  # Hexahedral element
    TET = 2  # Tetrahedral element
    PRISM = 3  # Prismatic element
    QUAD = 4  # Quadrilateral element
    TRI = 5  # Triangular element

@dataclass
class ShapeFunctionConfig:
    """Configuration for shape functions."""
    element_type: ElementType
    order: int = 1  # Polynomial order
    quadrature_order: int = 2  # Quadrature order

class ShapeFunctions:
    """
    Implements the finite element shape functions.
    
    This class implements the shape functions from the original PHASTA codebase.
    """
    
    def __init__(self, config: ShapeFunctionConfig):
        """
        Initialize the shape functions.
        
        Args:
            config: Shape function configuration parameters
        """
        self.config = config
        self._setup_quadrature()
        
    def _setup_quadrature(self):
        """Set up quadrature points and weights."""
        if self.config.element_type == ElementType.HEX:
            self.quad_points, self.quad_weights = self._setup_hex_quadrature()
        elif self.config.element_type == ElementType.TET:
            self.quad_points, self.quad_weights = self._setup_tet_quadrature()
        elif self.config.element_type == ElementType.PRISM:
            self.quad_points, self.quad_weights = self._setup_prism_quadrature()
        elif self.config.element_type == ElementType.QUAD:
            self.quad_points, self.quad_weights = self._setup_quad_quadrature()
        elif self.config.element_type == ElementType.TRI:
            self.quad_points, self.quad_weights = self._setup_tri_quadrature()
        else:
            raise ValueError(f"Unknown element type: {self.config.element_type}")
            
    def evaluate(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate shape functions and their derivatives at a point.
        
        Args:
            xi: Natural coordinates
            
        Returns:
            tuple: (N, dN) shape functions and their derivatives
        """
        if self.config.element_type == ElementType.HEX:
            return self._evaluate_hex(xi)
        elif self.config.element_type == ElementType.TET:
            return self._evaluate_tet(xi)
        elif self.config.element_type == ElementType.PRISM:
            return self._evaluate_prism(xi)
        elif self.config.element_type == ElementType.QUAD:
            return self._evaluate_quad(xi)
        elif self.config.element_type == ElementType.TRI:
            return self._evaluate_tri(xi)
        else:
            raise ValueError(f"Unknown element type: {self.config.element_type}")
            
    def _evaluate_hex(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate hexahedral shape functions.
        
        Args:
            xi: Natural coordinates (xi, eta, zeta)
            
        Returns:
            tuple: (N, dN) shape functions and their derivatives
        """
        xi1 = 1.0 - xi[0]
        xi2 = 1.0 + xi[0]
        eta1 = 1.0 - xi[1]
        eta2 = 1.0 + xi[1]
        zeta1 = 1.0 - xi[2]
        zeta2 = 1.0 + xi[2]
        
        # Shape functions
        N = np.array([
            0.125 * xi1 * eta1 * zeta1,
            0.125 * xi2 * eta1 * zeta1,
            0.125 * xi2 * eta2 * zeta1,
            0.125 * xi1 * eta2 * zeta1,
            0.125 * xi1 * eta1 * zeta2,
            0.125 * xi2 * eta1 * zeta2,
            0.125 * xi2 * eta2 * zeta2,
            0.125 * xi1 * eta2 * zeta2
        ])
        
        # Derivatives
        dN = np.array([
            [-0.125 * eta1 * zeta1, -0.125 * xi1 * zeta1, -0.125 * xi1 * eta1],
            [ 0.125 * eta1 * zeta1, -0.125 * xi2 * zeta1, -0.125 * xi2 * eta1],
            [ 0.125 * eta2 * zeta1,  0.125 * xi2 * zeta1, -0.125 * xi2 * eta2],
            [-0.125 * eta2 * zeta1,  0.125 * xi1 * zeta1, -0.125 * xi1 * eta2],
            [-0.125 * eta1 * zeta2, -0.125 * xi1 * zeta2,  0.125 * xi1 * eta1],
            [ 0.125 * eta1 * zeta2, -0.125 * xi2 * zeta2,  0.125 * xi2 * eta1],
            [ 0.125 * eta2 * zeta2,  0.125 * xi2 * zeta2,  0.125 * xi2 * eta2],
            [-0.125 * eta2 * zeta2,  0.125 * xi1 * zeta2,  0.125 * xi1 * eta2]
        ])
        
        return N, dN
        
    def _evaluate_tet(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate tetrahedral shape functions.
        
        Args:
            xi: Natural coordinates (xi, eta, zeta)
            
        Returns:
            tuple: (N, dN) shape functions and their derivatives
        """
        # Shape functions
        N = np.array([
            1.0 - xi[0] - xi[1] - xi[2],
            xi[0],
            xi[1],
            xi[2]
        ])
        
        # Derivatives
        dN = np.array([
            [-1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0]
        ])
        
        return N, dN
        
    def _evaluate_prism(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate prismatic shape functions.
        
        Args:
            xi: Natural coordinates (xi, eta, zeta)
            
        Returns:
            tuple: (N, dN) shape functions and their derivatives
        """
        xi1 = 1.0 - xi[0] - xi[1]
        
        # Shape functions
        N = np.array([
            xi1 * (1.0 - xi[2]),
            xi[0] * (1.0 - xi[2]),
            xi[1] * (1.0 - xi[2]),
            xi1 * (1.0 + xi[2]),
            xi[0] * (1.0 + xi[2]),
            xi[1] * (1.0 + xi[2])
        ])
        
        # Derivatives
        dN = np.array([
            [-(1.0 - xi[2]), -(1.0 - xi[2]), -xi1],
            [ (1.0 - xi[2]),  0.0,          -xi[0]],
            [ 0.0,           (1.0 - xi[2]), -xi[1]],
            [-(1.0 + xi[2]), -(1.0 + xi[2]),  xi1],
            [ (1.0 + xi[2]),  0.0,           xi[0]],
            [ 0.0,           (1.0 + xi[2]),  xi[1]]
        ])
        
        return N, dN
        
    def _evaluate_quad(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate quadrilateral shape functions.
        
        Args:
            xi: Natural coordinates (xi, eta)
            
        Returns:
            tuple: (N, dN) shape functions and their derivatives
        """
        xi1 = 1.0 - xi[0]
        xi2 = 1.0 + xi[0]
        eta1 = 1.0 - xi[1]
        eta2 = 1.0 + xi[1]
        
        # Shape functions
        N = np.array([
            0.25 * xi1 * eta1,
            0.25 * xi2 * eta1,
            0.25 * xi2 * eta2,
            0.25 * xi1 * eta2
        ])
        
        # Derivatives
        dN = np.array([
            [-0.25 * eta1, -0.25 * xi1],
            [ 0.25 * eta1, -0.25 * xi2],
            [ 0.25 * eta2,  0.25 * xi2],
            [-0.25 * eta2,  0.25 * xi1]
        ])
        
        return N, dN
        
    def _evaluate_tri(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate triangular shape functions.
        
        Args:
            xi: Natural coordinates (xi, eta)
            
        Returns:
            tuple: (N, dN) shape functions and their derivatives
        """
        # Shape functions
        N = np.array([
            1.0 - xi[0] - xi[1],
            xi[0],
            xi[1]
        ])
        
        # Derivatives
        dN = np.array([
            [-1.0, -1.0],
            [ 1.0,  0.0],
            [ 0.0,  1.0]
        ])
        
        return N, dN
        
    def _setup_hex_quadrature(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up quadrature for hexahedral elements.
        
        Returns:
            tuple: (points, weights) quadrature points and weights
        """
        # Gauss-Legendre quadrature
        if self.config.quadrature_order == 1:
            points = np.array([[0.0, 0.0, 0.0]])
            weights = np.array([8.0])
        elif self.config.quadrature_order == 2:
            g = 1.0 / np.sqrt(3.0)
            points = np.array([
                [-g, -g, -g], [g, -g, -g], [g, g, -g], [-g, g, -g],
                [-g, -g,  g], [g, -g,  g], [g, g,  g], [-g, g,  g]
            ])
            weights = np.ones(8)
        else:
            raise ValueError(f"Unsupported quadrature order: {self.config.quadrature_order}")
            
        return points, weights
        
    def _setup_tet_quadrature(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up quadrature for tetrahedral elements.
        
        Returns:
            tuple: (points, weights) quadrature points and weights
        """
        if self.config.quadrature_order == 1:
            points = np.array([[0.25, 0.25, 0.25]])
            weights = np.array([1.0/6.0])
        elif self.config.quadrature_order == 2:
            a = 0.5854101966249685
            b = 0.1381966011250105
            points = np.array([
                [a, b, b], [b, a, b], [b, b, a], [b, b, b]
            ])
            weights = np.array([1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0])
        else:
            raise ValueError(f"Unsupported quadrature order: {self.config.quadrature_order}")
            
        return points, weights
        
    def _setup_prism_quadrature(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up quadrature for prismatic elements.
        
        Returns:
            tuple: (points, weights) quadrature points and weights
        """
        if self.config.quadrature_order == 1:
            points = np.array([[1.0/3.0, 1.0/3.0, 0.0]])
            weights = np.array([1.0])
        elif self.config.quadrature_order == 2:
            g = 1.0 / np.sqrt(3.0)
            points = np.array([
                [1.0/3.0, 1.0/3.0, -g],
                [1.0/3.0, 1.0/3.0,  g]
            ])
            weights = np.array([0.5, 0.5])
        else:
            raise ValueError(f"Unsupported quadrature order: {self.config.quadrature_order}")
            
        return points, weights
        
    def _setup_quad_quadrature(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up quadrature for quadrilateral elements.
        
        Returns:
            tuple: (points, weights) quadrature points and weights
        """
        if self.config.quadrature_order == 1:
            points = np.array([[0.0, 0.0]])
            weights = np.array([4.0])
        elif self.config.quadrature_order == 2:
            g = 1.0 / np.sqrt(3.0)
            points = np.array([
                [-g, -g], [g, -g], [g, g], [-g, g]
            ])
            weights = np.ones(4)
        else:
            raise ValueError(f"Unsupported quadrature order: {self.config.quadrature_order}")
            
        return points, weights
        
    def _setup_tri_quadrature(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up quadrature for triangular elements.
        
        Returns:
            tuple: (points, weights) quadrature points and weights
        """
        if self.config.quadrature_order == 1:
            points = np.array([[1.0/3.0, 1.0/3.0]])
            weights = np.array([0.5])
        elif self.config.quadrature_order == 2:
            points = np.array([
                [1.0/6.0, 1.0/6.0],
                [2.0/3.0, 1.0/6.0],
                [1.0/6.0, 2.0/3.0]
            ])
            weights = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
        else:
            raise ValueError(f"Unsupported quadrature order: {self.config.quadrature_order}")
            
        return points, weights 
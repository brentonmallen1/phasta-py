"""Finite element basis functions and shape functions."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class ShapeFunction:
    """Base class for shape functions."""
    
    def __init__(self, order: int):
        """Initialize shape function.
        
        Args:
            order: Polynomial order of the shape function
        """
        self.order = order
    
    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate shape function at given points.
        
        Args:
            xi: Natural coordinates, shape (n_points, dim)
            
        Returns:
            Shape function values, shape (n_points, n_basis)
        """
        raise NotImplementedError
    
    def evaluate_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate shape function derivatives at given points.
        
        Args:
            xi: Natural coordinates, shape (n_points, dim)
            
        Returns:
            Shape function derivatives, shape (n_points, n_basis, dim)
        """
        raise NotImplementedError


class LagrangeShapeFunction(ShapeFunction):
    """Lagrange shape functions for various element types."""
    
    def __init__(self, element_type: str, order: int):
        """Initialize Lagrange shape function.
        
        Args:
            element_type: Type of element ('line', 'tri', 'quad', 'tet', 'hex')
            order: Polynomial order of the shape function
        """
        super().__init__(order)
        self.element_type = element_type
        self._validate()
    
    def _validate(self) -> None:
        """Validate element type and order."""
        valid_types = {'line', 'tri', 'quad', 'tet', 'hex'}
        if self.element_type not in valid_types:
            raise ValueError(f"Invalid element type: {self.element_type}")
        
        if self.order < 1:
            raise ValueError("Order must be at least 1")
    
    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate Lagrange shape functions.
        
        Args:
            xi: Natural coordinates, shape (n_points, dim)
            
        Returns:
            Shape function values, shape (n_points, n_basis)
        """
        if self.element_type == 'line':
            return self._evaluate_line(xi)
        elif self.element_type == 'tri':
            return self._evaluate_tri(xi)
        elif self.element_type == 'quad':
            return self._evaluate_quad(xi)
        elif self.element_type == 'tet':
            return self._evaluate_tet(xi)
        elif self.element_type == 'hex':
            return self._evaluate_hex(xi)
        else:
            raise ValueError(f"Unsupported element type: {self.element_type}")
    
    def evaluate_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate Lagrange shape function derivatives.
        
        Args:
            xi: Natural coordinates, shape (n_points, dim)
            
        Returns:
            Shape function derivatives, shape (n_points, n_basis, dim)
        """
        if self.element_type == 'line':
            return self._evaluate_line_derivatives(xi)
        elif self.element_type == 'tri':
            return self._evaluate_tri_derivatives(xi)
        elif self.element_type == 'quad':
            return self._evaluate_quad_derivatives(xi)
        elif self.element_type == 'tet':
            return self._evaluate_tet_derivatives(xi)
        elif self.element_type == 'hex':
            return self._evaluate_hex_derivatives(xi)
        else:
            raise ValueError(f"Unsupported element type: {self.element_type}")
    
    def _evaluate_line(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate 1D Lagrange shape functions."""
        n_points = xi.shape[0]
        n_basis = self.order + 1
        N = np.zeros((n_points, n_basis))
        
        # For first order (linear)
        if self.order == 1:
            N[:, 0] = 0.5 * (1 - xi[:, 0])
            N[:, 1] = 0.5 * (1 + xi[:, 0])
        else:
            # Higher order implementation
            raise NotImplementedError("Higher order line elements not implemented")
        
        return N
    
    def _evaluate_line_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate derivatives of 1D Lagrange shape functions."""
        n_points = xi.shape[0]
        n_basis = self.order + 1
        dN = np.zeros((n_points, n_basis, 1))
        
        # For first order (linear)
        if self.order == 1:
            dN[:, 0, 0] = -0.5
            dN[:, 1, 0] = 0.5
        else:
            # Higher order implementation
            raise NotImplementedError("Higher order line elements not implemented")
        
        return dN
    
    def _evaluate_tri(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate 2D triangular Lagrange shape functions."""
        n_points = xi.shape[0]
        n_basis = (self.order + 1) * (self.order + 2) // 2
        N = np.zeros((n_points, n_basis))
        
        # For first order (linear)
        if self.order == 1:
            N[:, 0] = 1 - xi[:, 0] - xi[:, 1]
            N[:, 1] = xi[:, 0]
            N[:, 2] = xi[:, 1]
        else:
            # Higher order implementation
            raise NotImplementedError("Higher order triangular elements not implemented")
        
        return N
    
    def _evaluate_tri_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate derivatives of 2D triangular Lagrange shape functions."""
        n_points = xi.shape[0]
        n_basis = (self.order + 1) * (self.order + 2) // 2
        dN = np.zeros((n_points, n_basis, 2))
        
        # For first order (linear)
        if self.order == 1:
            dN[:, 0, 0] = -1.0
            dN[:, 0, 1] = -1.0
            dN[:, 1, 0] = 1.0
            dN[:, 1, 1] = 0.0
            dN[:, 2, 0] = 0.0
            dN[:, 2, 1] = 1.0
        else:
            # Higher order implementation
            raise NotImplementedError("Higher order triangular elements not implemented")
        
        return dN
    
    def _evaluate_quad(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate 2D quadrilateral Lagrange shape functions."""
        n_points = xi.shape[0]
        n_basis = (self.order + 1) ** 2
        N = np.zeros((n_points, n_basis))
        
        # For first order (bilinear)
        if self.order == 1:
            N[:, 0] = 0.25 * (1 - xi[:, 0]) * (1 - xi[:, 1])
            N[:, 1] = 0.25 * (1 + xi[:, 0]) * (1 - xi[:, 1])
            N[:, 2] = 0.25 * (1 + xi[:, 0]) * (1 + xi[:, 1])
            N[:, 3] = 0.25 * (1 - xi[:, 0]) * (1 + xi[:, 1])
        else:
            # Higher order implementation
            raise NotImplementedError("Higher order quadrilateral elements not implemented")
        
        return N
    
    def _evaluate_quad_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate derivatives of 2D quadrilateral Lagrange shape functions."""
        n_points = xi.shape[0]
        n_basis = (self.order + 1) ** 2
        dN = np.zeros((n_points, n_basis, 2))
        
        # For first order (bilinear)
        if self.order == 1:
            dN[:, 0, 0] = -0.25 * (1 - xi[:, 1])
            dN[:, 0, 1] = -0.25 * (1 - xi[:, 0])
            dN[:, 1, 0] = 0.25 * (1 - xi[:, 1])
            dN[:, 1, 1] = -0.25 * (1 + xi[:, 0])
            dN[:, 2, 0] = 0.25 * (1 + xi[:, 1])
            dN[:, 2, 1] = 0.25 * (1 + xi[:, 0])
            dN[:, 3, 0] = -0.25 * (1 + xi[:, 1])
            dN[:, 3, 1] = 0.25 * (1 - xi[:, 0])
        else:
            # Higher order implementation
            raise NotImplementedError("Higher order quadrilateral elements not implemented")
        
        return dN
    
    def _evaluate_tet(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate 3D tetrahedral Lagrange shape functions."""
        n_points = xi.shape[0]
        n_basis = (self.order + 1) * (self.order + 2) * (self.order + 3) // 6
        N = np.zeros((n_points, n_basis))
        
        # For first order (linear)
        if self.order == 1:
            N[:, 0] = 1 - xi[:, 0] - xi[:, 1] - xi[:, 2]
            N[:, 1] = xi[:, 0]
            N[:, 2] = xi[:, 1]
            N[:, 3] = xi[:, 2]
        else:
            # Higher order implementation
            raise NotImplementedError("Higher order tetrahedral elements not implemented")
        
        return N
    
    def _evaluate_tet_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate derivatives of 3D tetrahedral Lagrange shape functions."""
        n_points = xi.shape[0]
        n_basis = (self.order + 1) * (self.order + 2) * (self.order + 3) // 6
        dN = np.zeros((n_points, n_basis, 3))
        
        # For first order (linear)
        if self.order == 1:
            dN[:, 0, 0] = -1.0
            dN[:, 0, 1] = -1.0
            dN[:, 0, 2] = -1.0
            dN[:, 1, 0] = 1.0
            dN[:, 1, 1] = 0.0
            dN[:, 1, 2] = 0.0
            dN[:, 2, 0] = 0.0
            dN[:, 2, 1] = 1.0
            dN[:, 2, 2] = 0.0
            dN[:, 3, 0] = 0.0
            dN[:, 3, 1] = 0.0
            dN[:, 3, 2] = 1.0
        else:
            # Higher order implementation
            raise NotImplementedError("Higher order tetrahedral elements not implemented")
        
        return dN
    
    def _evaluate_hex(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate 3D hexahedral Lagrange shape functions."""
        n_points = xi.shape[0]
        n_basis = (self.order + 1) ** 3
        N = np.zeros((n_points, n_basis))
        
        # For first order (trilinear)
        if self.order == 1:
            N[:, 0] = 0.125 * (1 - xi[:, 0]) * (1 - xi[:, 1]) * (1 - xi[:, 2])
            N[:, 1] = 0.125 * (1 + xi[:, 0]) * (1 - xi[:, 1]) * (1 - xi[:, 2])
            N[:, 2] = 0.125 * (1 + xi[:, 0]) * (1 + xi[:, 1]) * (1 - xi[:, 2])
            N[:, 3] = 0.125 * (1 - xi[:, 0]) * (1 + xi[:, 1]) * (1 - xi[:, 2])
            N[:, 4] = 0.125 * (1 - xi[:, 0]) * (1 - xi[:, 1]) * (1 + xi[:, 2])
            N[:, 5] = 0.125 * (1 + xi[:, 0]) * (1 - xi[:, 1]) * (1 + xi[:, 2])
            N[:, 6] = 0.125 * (1 + xi[:, 0]) * (1 + xi[:, 1]) * (1 + xi[:, 2])
            N[:, 7] = 0.125 * (1 - xi[:, 0]) * (1 + xi[:, 1]) * (1 + xi[:, 2])
        else:
            # Higher order implementation
            raise NotImplementedError("Higher order hexahedral elements not implemented")
        
        return N
    
    def _evaluate_hex_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate derivatives of 3D hexahedral Lagrange shape functions."""
        n_points = xi.shape[0]
        n_basis = (self.order + 1) ** 3
        dN = np.zeros((n_points, n_basis, 3))
        
        # For first order (trilinear)
        if self.order == 1:
            # d/dxi
            dN[:, 0, 0] = -0.125 * (1 - xi[:, 1]) * (1 - xi[:, 2])
            dN[:, 1, 0] = 0.125 * (1 - xi[:, 1]) * (1 - xi[:, 2])
            dN[:, 2, 0] = 0.125 * (1 + xi[:, 1]) * (1 - xi[:, 2])
            dN[:, 3, 0] = -0.125 * (1 + xi[:, 1]) * (1 - xi[:, 2])
            dN[:, 4, 0] = -0.125 * (1 - xi[:, 1]) * (1 + xi[:, 2])
            dN[:, 5, 0] = 0.125 * (1 - xi[:, 1]) * (1 + xi[:, 2])
            dN[:, 6, 0] = 0.125 * (1 + xi[:, 1]) * (1 + xi[:, 2])
            dN[:, 7, 0] = -0.125 * (1 + xi[:, 1]) * (1 + xi[:, 2])
            
            # d/deta
            dN[:, 0, 1] = -0.125 * (1 - xi[:, 0]) * (1 - xi[:, 2])
            dN[:, 1, 1] = -0.125 * (1 + xi[:, 0]) * (1 - xi[:, 2])
            dN[:, 2, 1] = 0.125 * (1 + xi[:, 0]) * (1 - xi[:, 2])
            dN[:, 3, 1] = 0.125 * (1 - xi[:, 0]) * (1 - xi[:, 2])
            dN[:, 4, 1] = -0.125 * (1 - xi[:, 0]) * (1 + xi[:, 2])
            dN[:, 5, 1] = -0.125 * (1 + xi[:, 0]) * (1 + xi[:, 2])
            dN[:, 6, 1] = 0.125 * (1 + xi[:, 0]) * (1 + xi[:, 2])
            dN[:, 7, 1] = 0.125 * (1 - xi[:, 0]) * (1 + xi[:, 2])
            
            # d/dzeta
            dN[:, 0, 2] = -0.125 * (1 - xi[:, 0]) * (1 - xi[:, 1])
            dN[:, 1, 2] = -0.125 * (1 + xi[:, 0]) * (1 - xi[:, 1])
            dN[:, 2, 2] = -0.125 * (1 + xi[:, 0]) * (1 + xi[:, 1])
            dN[:, 3, 2] = -0.125 * (1 - xi[:, 0]) * (1 + xi[:, 1])
            dN[:, 4, 2] = 0.125 * (1 - xi[:, 0]) * (1 - xi[:, 1])
            dN[:, 5, 2] = 0.125 * (1 + xi[:, 0]) * (1 - xi[:, 1])
            dN[:, 6, 2] = 0.125 * (1 + xi[:, 0]) * (1 + xi[:, 1])
            dN[:, 7, 2] = 0.125 * (1 - xi[:, 0]) * (1 + xi[:, 1])
        else:
            # Higher order implementation
            raise NotImplementedError("Higher order hexahedral elements not implemented")
        
        return dN

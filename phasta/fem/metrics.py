"""Element metrics and Jacobian calculations."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from .basis import LagrangeShapeFunction


class ElementMetrics:
    """Class for computing element metrics and Jacobians."""
    
    def __init__(self, element_type: str, order: int = 1):
        """Initialize element metrics calculator.
        
        Args:
            element_type: Type of element ('line', 'tri', 'quad', 'tet', 'hex')
            order: Polynomial order of the element
        """
        self.element_type = element_type
        self.order = order
        self.shape_function = LagrangeShapeFunction(element_type, order)
    
    def compute_jacobian(self, xi: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jacobian matrix and its determinant at given points.
        
        Args:
            xi: Natural coordinates, shape (n_points, dim)
            nodes: Physical coordinates of element nodes, shape (n_nodes, dim)
            
        Returns:
            Tuple containing:
            - Jacobian matrices, shape (n_points, dim, dim)
            - Jacobian determinants, shape (n_points,)
        """
        # Get shape function derivatives
        dN = self.shape_function.evaluate_derivatives(xi)
        
        # Compute Jacobian matrices
        n_points = xi.shape[0]
        dim = nodes.shape[1]
        J = np.zeros((n_points, dim, dim))
        
        for i in range(n_points):
            J[i] = np.dot(dN[i].T, nodes)
        
        # Compute determinants
        if dim == 1:
            detJ = np.abs(J[:, 0, 0])
        elif dim == 2:
            detJ = np.abs(J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])
        else:  # dim == 3
            detJ = np.abs(
                J[:, 0, 0] * (J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1]) -
                J[:, 0, 1] * (J[:, 1, 0] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 0]) +
                J[:, 0, 2] * (J[:, 1, 0] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 0])
            )
        
        return J, detJ
    
    def compute_inverse_jacobian(self, J: np.ndarray) -> np.ndarray:
        """Compute inverse of Jacobian matrices.
        
        Args:
            J: Jacobian matrices, shape (n_points, dim, dim)
            
        Returns:
            Inverse Jacobian matrices, shape (n_points, dim, dim)
        """
        n_points = J.shape[0]
        dim = J.shape[1]
        invJ = np.zeros_like(J)
        
        if dim == 1:
            invJ[:, 0, 0] = 1.0 / J[:, 0, 0]
        elif dim == 2:
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            invJ[:, 0, 0] = J[:, 1, 1] / det
            invJ[:, 0, 1] = -J[:, 0, 1] / det
            invJ[:, 1, 0] = -J[:, 1, 0] / det
            invJ[:, 1, 1] = J[:, 0, 0] / det
        else:  # dim == 3
            for i in range(n_points):
                invJ[i] = np.linalg.inv(J[i])
        
        return invJ
    
    def compute_physical_derivatives(self, xi: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        """Compute derivatives with respect to physical coordinates.
        
        Args:
            xi: Natural coordinates, shape (n_points, dim)
            nodes: Physical coordinates of element nodes, shape (n_nodes, dim)
            
        Returns:
            Derivatives with respect to physical coordinates, shape (n_points, n_basis, dim)
        """
        # Get shape function derivatives in natural coordinates
        dN = self.shape_function.evaluate_derivatives(xi)
        
        # Compute Jacobian matrices and their inverses
        J, _ = self.compute_jacobian(xi, nodes)
        invJ = self.compute_inverse_jacobian(J)
        
        # Transform derivatives to physical coordinates
        n_points = xi.shape[0]
        n_basis = dN.shape[1]
        dim = nodes.shape[1]
        dN_phys = np.zeros((n_points, n_basis, dim))
        
        for i in range(n_points):
            for j in range(n_basis):
                dN_phys[i, j] = np.dot(invJ[i], dN[i, j])
        
        return dN_phys
    
    def compute_element_volume(self, nodes: np.ndarray) -> float:
        """Compute element volume/area/length.
        
        Args:
            nodes: Physical coordinates of element nodes, shape (n_nodes, dim)
            
        Returns:
            Element volume/area/length
        """
        # Use Gaussian quadrature to compute volume
        if self.element_type == 'line':
            xi = np.array([[0.0]])  # Single point for line
            weights = np.array([2.0])
        elif self.element_type == 'tri':
            xi = np.array([[1/3, 1/3]])  # Single point for triangle
            weights = np.array([0.5])
        elif self.element_type == 'quad':
            xi = np.array([[0.0, 0.0]])  # Single point for quad
            weights = np.array([4.0])
        elif self.element_type == 'tet':
            xi = np.array([[0.25, 0.25, 0.25]])  # Single point for tet
            weights = np.array([1/6])
        elif self.element_type == 'hex':
            xi = np.array([[0.0, 0.0, 0.0]])  # Single point for hex
            weights = np.array([8.0])
        else:
            raise ValueError(f"Unsupported element type: {self.element_type}")
        
        # Compute Jacobian determinant
        _, detJ = self.compute_jacobian(xi, nodes)
        
        # Compute volume
        volume = np.sum(weights * detJ)
        
        return volume

"""CPU acceleration backend for PHASTA-Py.

This module implements the CPU acceleration backend using NumPy and SciPy.
It serves as a fallback option when GPU acceleration is not available.
"""

import numpy as np
from scipy import sparse
from typing import List, Tuple, Union
from .base import AccelerationBackend


class CPUBackend(AccelerationBackend):
    """CPU acceleration backend using NumPy and SciPy."""
    
    def __init__(self):
        """Initialize the CPU backend."""
        pass
    
    def is_available(self) -> bool:
        """Check if CPU backend is available.
        
        Returns:
            bool: Always True as CPU is always available
        """
        return True
    
    def matvec(self, A: Union[sparse.spmatrix, np.ndarray], 
               x: np.ndarray) -> np.ndarray:
        """Compute matrix-vector product using CPU.
        
        Args:
            A: Sparse or dense matrix
            x: Vector to multiply with
            
        Returns:
            Result of matrix-vector multiplication
        """
        return A @ x
    
    def assemble_element_matrix(self, element_nodes: np.ndarray,
                              basis_functions: List[callable],
                              quadrature_points: List[Tuple[float, float]],
                              quadrature_weights: List[float]) -> np.ndarray:
        """Assemble element matrix using CPU.
        
        Args:
            element_nodes: Node coordinates for the element
            basis_functions: List of basis functions
            quadrature_points: List of quadrature point coordinates
            quadrature_weights: List of quadrature weights
            
        Returns:
            Element matrix
        """
        n_basis = len(basis_functions)
        element_matrix = np.zeros((n_basis, n_basis))
        
        # Evaluate basis functions at quadrature points
        basis_values = []
        for basis in basis_functions:
            values = np.array([basis(xi, eta) for xi, eta in quadrature_points])
            basis_values.append(values)
        
        # Compute element matrix
        for i in range(n_basis):
            for j in range(n_basis):
                element_matrix[i, j] = np.sum(
                    quadrature_weights * basis_values[i] * basis_values[j]
                )
        
        return element_matrix
    
    def compute_element_metrics(self, element_nodes: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute element metrics using CPU.
        
        Args:
            element_nodes: Node coordinates for the element
            
        Returns:
            Tuple of (Jacobian matrix, determinant)
        """
        # Compute Jacobian matrix
        # For a 2D element, we need to compute derivatives of shape functions
        # This is a simplified version - in practice, you'd use proper shape functions
        dx_dxi = element_nodes[1, 0] - element_nodes[0, 0]
        dx_deta = element_nodes[3, 0] - element_nodes[0, 0]
        dy_dxi = element_nodes[1, 1] - element_nodes[0, 1]
        dy_deta = element_nodes[3, 1] - element_nodes[0, 1]
        
        J = np.array([[dx_dxi, dx_deta],
                     [dy_dxi, dy_deta]])
        
        # Compute determinant
        det = np.linalg.det(J)
        
        return J, det
    
    def parallel_reduce(self, local_data: np.ndarray, 
                       operation: str = 'sum') -> np.ndarray:
        """Perform parallel reduction operation using CPU.
        
        Args:
            local_data: Local data to reduce
            operation: Reduction operation ('sum', 'min', 'max')
            
        Returns:
            Result of reduction operation
        """
        if operation == 'sum':
            return np.sum(local_data, axis=0)
        elif operation == 'min':
            return np.min(local_data, axis=0)
        elif operation == 'max':
            return np.max(local_data, axis=0)
        else:
            raise ValueError(f"Unknown reduction operation: {operation}")
    
    def synchronize(self) -> None:
        """Synchronize CPU operations.
        
        Note: This is a no-op for CPU backend as operations are synchronous.
        """
        pass

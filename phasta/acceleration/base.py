"""Base classes for hardware acceleration.

This module defines the base classes and interfaces for hardware acceleration
in PHASTA-Py. It provides a common interface for different acceleration backends
(CPU, CUDA, Metal) and defines the core functionality that each backend must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple
import numpy as np
from scipy import sparse


class AccelerationBackend(ABC):
    """Base class for all acceleration backends.
    
    This abstract base class defines the interface that all acceleration backends
    must implement. It provides methods for basic operations like matrix-vector
    multiplication, element assembly, and other performance-critical operations.
    """
    
    @abstractmethod
    def __init__(self):
        """Initialize the acceleration backend."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this acceleration backend is available.
        
        Returns:
            bool: True if the backend is available, False otherwise.
        """
        pass
    
    @abstractmethod
    def matvec(self, A: Union[sparse.spmatrix, np.ndarray], 
               x: np.ndarray) -> np.ndarray:
        """Compute matrix-vector product.
        
        Args:
            A: Sparse or dense matrix
            x: Vector to multiply with
            
        Returns:
            Result of matrix-vector multiplication
        """
        pass
    
    @abstractmethod
    def assemble_element_matrix(self, element_nodes: np.ndarray,
                              basis_functions: List[callable],
                              quadrature_points: List[Tuple[float, float]],
                              quadrature_weights: List[float]) -> np.ndarray:
        """Assemble element matrix.
        
        Args:
            element_nodes: Node coordinates for the element
            basis_functions: List of basis functions
            quadrature_points: List of quadrature point coordinates
            quadrature_weights: List of quadrature weights
            
        Returns:
            Element matrix
        """
        pass
    
    @abstractmethod
    def compute_element_metrics(self, element_nodes: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute element metrics.
        
        Args:
            element_nodes: Node coordinates for the element
            
        Returns:
            Tuple of (Jacobian matrix, determinant)
        """
        pass
    
    @abstractmethod
    def parallel_reduce(self, local_data: np.ndarray, 
                       operation: str = 'sum') -> np.ndarray:
        """Perform parallel reduction operation.
        
        Args:
            local_data: Local data to reduce
            operation: Reduction operation ('sum', 'min', 'max')
            
        Returns:
            Result of reduction operation
        """
        pass
    
    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize the acceleration device."""
        pass


class AccelerationContext:
    """Context manager for acceleration backends.
    
    This class manages the lifecycle of acceleration backends and provides
    a convenient way to switch between different backends.
    """
    
    def __init__(self, backend: Optional[AccelerationBackend] = None):
        """Initialize the acceleration context.
        
        Args:
            backend: Optional acceleration backend to use
        """
        self.backend = backend
        self._previous_backend = None
    
    def __enter__(self):
        """Enter the acceleration context."""
        if self.backend is not None:
            self._previous_backend = get_current_backend()
            set_current_backend(self.backend)
        return self.backend
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the acceleration context."""
        if self.backend is not None:
            set_current_backend(self._previous_backend)


# Global state for current backend
_current_backend: Optional[AccelerationBackend] = None


def get_current_backend() -> Optional[AccelerationBackend]:
    """Get the current acceleration backend.
    
    Returns:
        Current acceleration backend or None if no backend is set
    """
    return _current_backend


def set_current_backend(backend: Optional[AccelerationBackend]) -> None:
    """Set the current acceleration backend.
    
    Args:
        backend: Acceleration backend to set as current
    """
    global _current_backend
    _current_backend = backend

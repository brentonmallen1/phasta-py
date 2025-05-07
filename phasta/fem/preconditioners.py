"""Preconditioners for linear solvers."""

from typing import Optional, Union
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spilu, LinearOperator


class Preconditioner:
    """Base class for preconditioners."""
    
    def __init__(self):
        """Initialize preconditioner."""
        self.M = None
    
    def setup(self, A: sparse.spmatrix) -> None:
        """Set up preconditioner for matrix A.
        
        Args:
            A: System matrix
        """
        raise NotImplementedError("Subclasses must implement setup()")
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply preconditioner to vector x.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector
        """
        raise NotImplementedError("Subclasses must implement apply()")


class DiagonalPreconditioner(Preconditioner):
    """Diagonal (Jacobi) preconditioner."""
    
    def setup(self, A: sparse.spmatrix) -> None:
        """Set up diagonal preconditioner.
        
        Args:
            A: System matrix
        """
        self.M = 1.0 / A.diagonal()
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply diagonal preconditioner.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector
        """
        return self.M * x


class ILUPreconditioner(Preconditioner):
    """Incomplete LU preconditioner."""
    
    def __init__(self, fill_factor: float = 10.0, drop_tol: float = 1e-4):
        """Initialize ILU preconditioner.
        
        Args:
            fill_factor: Maximum fill factor
            drop_tol: Drop tolerance for small elements
        """
        super().__init__()
        self.fill_factor = fill_factor
        self.drop_tol = drop_tol
        self.ilu = None
    
    def setup(self, A: sparse.spmatrix) -> None:
        """Set up ILU preconditioner.
        
        Args:
            A: System matrix
        """
        self.ilu = spilu(A.tocsc(), fill_factor=self.fill_factor,
                        drop_tol=self.drop_tol)
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply ILU preconditioner.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector
        """
        return self.ilu.solve(x)


class BlockJacobiPreconditioner(Preconditioner):
    """Block Jacobi preconditioner."""
    
    def __init__(self, block_size: int = 4):
        """Initialize block Jacobi preconditioner.
        
        Args:
            block_size: Size of diagonal blocks
        """
        super().__init__()
        self.block_size = block_size
        self.blocks = []
    
    def setup(self, A: sparse.spmatrix) -> None:
        """Set up block Jacobi preconditioner.
        
        Args:
            A: System matrix
        """
        n = A.shape[0]
        self.blocks = []
        
        # Extract and factorize diagonal blocks
        for i in range(0, n, self.block_size):
            end = min(i + self.block_size, n)
            block = A[i:end, i:end].toarray()
            self.blocks.append(np.linalg.inv(block))
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply block Jacobi preconditioner.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector
        """
        y = np.zeros_like(x)
        for i, block in enumerate(self.blocks):
            start = i * self.block_size
            end = min(start + self.block_size, len(x))
            y[start:end] = block @ x[start:end]
        return y


class AMGPreconditioner(Preconditioner):
    """Algebraic multigrid preconditioner."""
    
    def __init__(self, strength: float = 0.25, max_levels: int = 10):
        """Initialize AMG preconditioner.
        
        Args:
            strength: Strength of connection threshold
            max_levels: Maximum number of multigrid levels
        """
        super().__init__()
        self.strength = strength
        self.max_levels = max_levels
        self.levels = []
    
    def setup(self, A: sparse.spmatrix) -> None:
        """Set up AMG preconditioner.
        
        Args:
            A: System matrix
        """
        # This is a simplified version of AMG setup
        # In practice, you would use a library like PyAMG
        
        # For now, we'll use a simple two-level method
        n = A.shape[0]
        
        # Create coarse grid (every other point)
        fine_to_coarse = np.zeros(n, dtype=int)
        fine_to_coarse[::2] = np.arange(n//2 + n%2)
        fine_to_coarse[1::2] = fine_to_coarse[::2]
        
        # Create restriction operator
        R = sparse.coo_matrix(
            (np.ones(n), (fine_to_coarse, np.arange(n))),
            shape=(n//2 + n%2, n)
        )
        
        # Create prolongation operator
        P = R.T
        
        # Create coarse grid operator
        Ac = R @ A @ P
        
        # Store operators
        self.levels = [(A, P, R, Ac)]
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply AMG preconditioner.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector
        """
        # This is a simplified version of AMG solve
        # In practice, you would use a library like PyAMG
        
        # For now, we'll use a simple two-level method
        A, P, R, Ac = self.levels[0]
        
        # Pre-smoothing
        y = x.copy()
        for _ in range(3):  # 3 iterations of Jacobi
            y = y - 0.5 * (A @ y - x) / A.diagonal()
        
        # Coarse grid correction
        rc = R @ (x - A @ y)
        ec = sparse.linalg.spsolve(Ac, rc)
        y = y + P @ ec
        
        # Post-smoothing
        for _ in range(3):  # 3 iterations of Jacobi
            y = y - 0.5 * (A @ y - x) / A.diagonal()
        
        return y


class PreconditionerFactory:
    """Factory for creating preconditioners."""
    
    @staticmethod
    def create_preconditioner(precond_type: str, **kwargs) -> Preconditioner:
        """Create a preconditioner of the specified type.
        
        Args:
            precond_type: Type of preconditioner ('diagonal', 'ilu', 'block_jacobi', 'amg')
            **kwargs: Additional arguments for preconditioner initialization
            
        Returns:
            Preconditioner instance
        """
        precond_types = {
            'diagonal': DiagonalPreconditioner,
            'ilu': ILUPreconditioner,
            'block_jacobi': BlockJacobiPreconditioner,
            'amg': AMGPreconditioner
        }
        
        if precond_type not in precond_types:
            raise ValueError(f"Unknown preconditioner type: {precond_type}")
        
        return precond_types[precond_type](**kwargs) 
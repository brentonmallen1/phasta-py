"""Direct solvers module.

This module provides direct solution methods for linear systems, including
LU decomposition, Cholesky decomposition, and sparse direct solvers.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, factorized
from typing import List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class DirectSolver:
    """Base class for direct solvers."""
    
    def __init__(self, matrix: Union[np.ndarray, sp.spmatrix]):
        """Initialize direct solver.
        
        Args:
            matrix: System matrix (dense or sparse)
        """
        self.matrix = matrix
        self.factorized = None
    
    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve linear system Ax = b.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Solution vector x
        """
        raise NotImplementedError


class LUSolver(DirectSolver):
    """LU decomposition solver."""
    
    def __init__(self, matrix: Union[np.ndarray, sp.spmatrix]):
        """Initialize LU solver.
        
        Args:
            matrix: System matrix (dense or sparse)
        """
        super().__init__(matrix)
        self._factorize()
    
    def _factorize(self):
        """Factorize the matrix."""
        if sp.issparse(self.matrix):
            self.factorized = factorized(self.matrix)
        else:
            from scipy.linalg import lu_factor
            self.lu, self.piv = lu_factor(self.matrix)
    
    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve linear system using LU decomposition.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Solution vector x
        """
        if sp.issparse(self.matrix):
            return self.factorized(b)
        else:
            from scipy.linalg import lu_solve
            return lu_solve((self.lu, self.piv), b)


class CholeskySolver(DirectSolver):
    """Cholesky decomposition solver for symmetric positive definite systems."""
    
    def __init__(self, matrix: Union[np.ndarray, sp.spmatrix]):
        """Initialize Cholesky solver.
        
        Args:
            matrix: Symmetric positive definite matrix
        """
        super().__init__(matrix)
        self._factorize()
    
    def _factorize(self):
        """Factorize the matrix."""
        if sp.issparse(self.matrix):
            from scipy.sparse.linalg import cholesky
            self.factorized = cholesky(self.matrix)
        else:
            from scipy.linalg import cholesky
            self.L = cholesky(self.matrix, lower=True)
    
    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve linear system using Cholesky decomposition.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Solution vector x
        """
        if sp.issparse(self.matrix):
            return self.factorized(b)
        else:
            from scipy.linalg import solve_triangular
            y = solve_triangular(self.L, b, lower=True)
            return solve_triangular(self.L.T, y, lower=False)


class SparseDirectSolver(DirectSolver):
    """Sparse direct solver using various methods."""
    
    def __init__(self, matrix: sp.spmatrix, method: str = 'umfpack'):
        """Initialize sparse direct solver.
        
        Args:
            matrix: Sparse system matrix
            method: Solution method ('umfpack', 'superlu', or 'pardiso')
        """
        super().__init__(matrix)
        self.method = method
        self._factorize()
    
    def _factorize(self):
        """Factorize the matrix."""
        if self.method == 'umfpack':
            self.factorized = factorized(self.matrix)
        elif self.method == 'superlu':
            from scipy.sparse.linalg import spilu
            self.factorized = spilu(self.matrix)
        elif self.method == 'pardiso':
            try:
                from pypardiso import factorized as pardiso_factorized
                self.factorized = pardiso_factorized(self.matrix)
            except ImportError:
                logger.warning("PyPardiso not available, falling back to UMFPACK")
                self.factorized = factorized(self.matrix)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve sparse linear system.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Solution vector x
        """
        return self.factorized(b)


class BlockDirectSolver:
    """Block direct solver for block-structured systems."""
    
    def __init__(self, blocks: List[Union[np.ndarray, sp.spmatrix]]):
        """Initialize block direct solver.
        
        Args:
            blocks: List of block matrices
        """
        self.blocks = blocks
        self.solvers = [LUSolver(block) for block in blocks]
    
    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve block-structured system.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Solution vector x
        """
        n = len(self.blocks)
        block_size = b.shape[0] // n
        x = np.zeros_like(b)
        
        # Solve each block
        for i in range(n):
            start = i * block_size
            end = (i + 1) * block_size
            x[start:end] = self.solvers[i].solve(b[start:end])
        
        return x


def solve_direct(A: Union[np.ndarray, sp.spmatrix], b: np.ndarray,
                method: str = 'auto') -> np.ndarray:
    """Convenience function for direct solution.
    
    Args:
        A: System matrix
        b: Right-hand side vector
        method: Solution method ('auto', 'lu', 'cholesky', 'sparse')
        
    Returns:
        Solution vector x
    """
    if method == 'auto':
        if sp.issparse(A):
            method = 'sparse'
        elif np.allclose(A, A.T):
            method = 'cholesky'
        else:
            method = 'lu'
    
    if method == 'lu':
        solver = LUSolver(A)
    elif method == 'cholesky':
        solver = CholeskySolver(A)
    elif method == 'sparse':
        solver = SparseDirectSolver(A)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return solver.solve(b) 
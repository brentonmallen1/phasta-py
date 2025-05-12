"""Advanced linear solvers module.

This module provides functionality for:
- Krylov subspace methods
- Preconditioned iterative solvers
- Direct solvers
- Matrix-free solvers
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from abc import ABC, abstractmethod
from scipy.sparse import spmatrix
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


class LinearSolver(ABC):
    """Base class for linear solvers."""
    
    def __init__(self, 
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 verbose: bool = False):
        """Initialize linear solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            verbose: Whether to print convergence information
        """
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.iterations = 0
        self.residual = 0.0
    
    @abstractmethod
    def solve(self, A: Union[np.ndarray, spmatrix], b: np.ndarray) -> np.ndarray:
        """Solve linear system Ax = b.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        pass


class ConjugateGradient(LinearSolver):
    """Conjugate Gradient solver."""
    
    def solve(self, A: Union[np.ndarray, spmatrix], b: np.ndarray) -> np.ndarray:
        """Solve linear system using Conjugate Gradient method.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        n = len(b)
        x = np.zeros(n)
        r = b.copy()
        p = r.copy()
        rsold = np.dot(r, r)
        
        for i in range(self.max_iter):
            Ap = A @ p
            alpha = rsold / np.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = np.dot(r, r)
            
            if self.verbose and i % 10 == 0:
                logger.info(f"Iteration {i}: residual = {np.sqrt(rsnew)}")
            
            if np.sqrt(rsnew) < self.tol:
                self.iterations = i + 1
                self.residual = np.sqrt(rsnew)
                return x
            
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        
        self.iterations = self.max_iter
        self.residual = np.sqrt(rsold)
        return x


class GMRES(LinearSolver):
    """GMRES solver."""
    
    def __init__(self,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 verbose: bool = False,
                 restart: int = 30):
        """Initialize GMRES solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            verbose: Whether to print convergence information
            restart: Number of iterations before restart
        """
        super().__init__(max_iter, tol, verbose)
        self.restart = restart
    
    def solve(self, A: Union[np.ndarray, spmatrix], b: np.ndarray) -> np.ndarray:
        """Solve linear system using GMRES method.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        n = len(b)
        x = np.zeros(n)
        r = b.copy()
        beta = np.linalg.norm(r)
        
        if beta < self.tol:
            return x
        
        # Arnoldi process
        V = np.zeros((n, self.restart + 1))
        H = np.zeros((self.restart + 1, self.restart))
        V[:, 0] = r / beta
        
        for j in range(self.restart):
            w = A @ V[:, j]
            
            for i in range(j + 1):
                H[i, j] = np.dot(w, V[:, i])
                w -= H[i, j] * V[:, i]
            
            H[j + 1, j] = np.linalg.norm(w)
            
            if H[j + 1, j] < self.tol:
                break
            
            V[:, j + 1] = w / H[j + 1, j]
        
        # Solve least squares problem
        e1 = np.zeros(j + 2)
        e1[0] = beta
        y = np.linalg.lstsq(H[:j+2, :j+1], e1, rcond=None)[0]
        
        # Update solution
        x += V[:, :j+1] @ y
        
        self.iterations = j + 1
        self.residual = np.linalg.norm(b - A @ x)
        
        return x


class DirectSolver(LinearSolver):
    """Direct solver using sparse LU decomposition."""
    
    def solve(self, A: Union[np.ndarray, spmatrix], b: np.ndarray) -> np.ndarray:
        """Solve linear system using direct method.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        if isinstance(A, spmatrix):
            x = spsolve(A, b)
        else:
            x = np.linalg.solve(A, b)
        
        self.iterations = 1
        self.residual = np.linalg.norm(b - A @ x)
        
        return x


class MatrixFreeSolver(LinearSolver):
    """Matrix-free solver using function-based matrix-vector products."""
    
    def __init__(self,
                 matvec: Callable[[np.ndarray], np.ndarray],
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 verbose: bool = False):
        """Initialize matrix-free solver.
        
        Args:
            matvec: Function that computes matrix-vector product
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            verbose: Whether to print convergence information
        """
        super().__init__(max_iter, tol, verbose)
        self.matvec = matvec
    
    def solve(self, A: Union[np.ndarray, spmatrix], b: np.ndarray) -> np.ndarray:
        """Solve linear system using matrix-free method.
        
        Args:
            A: System matrix (ignored, using matvec function instead)
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        n = len(b)
        x = np.zeros(n)
        r = b.copy()
        p = r.copy()
        rsold = np.dot(r, r)
        
        for i in range(self.max_iter):
            Ap = self.matvec(p)
            alpha = rsold / np.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = np.dot(r, r)
            
            if self.verbose and i % 10 == 0:
                logger.info(f"Iteration {i}: residual = {np.sqrt(rsnew)}")
            
            if np.sqrt(rsnew) < self.tol:
                self.iterations = i + 1
                self.residual = np.sqrt(rsnew)
                return x
            
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        
        self.iterations = self.max_iter
        self.residual = np.sqrt(rsold)
        return x 
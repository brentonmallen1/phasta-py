"""
Linear solver module for the compressible flow solver.

This module implements the GMRES solver from the original PHASTA codebase.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

@dataclass
class SolverConfig:
    """Configuration for the linear solver."""
    max_iter: int = 100  # Maximum number of iterations
    tolerance: float = 1e-6  # Convergence tolerance
    restart: int = 30  # GMRES restart parameter
    preconditioner: str = "none"  # Preconditioner type
    preconditioner_params: Optional[Dict[str, Any]] = None  # Preconditioner parameters

class LinearSolver:
    """
    Implements the GMRES solver for compressible flow.
    
    This class implements the GMRES solver from the original PHASTA codebase.
    """
    
    def __init__(self, config: SolverConfig):
        """
        Initialize the linear solver.
        
        Args:
            config: Solver configuration parameters
        """
        self.config = config
        self._setup_preconditioner()
        
    def _setup_preconditioner(self):
        """Set up the preconditioner."""
        if self.config.preconditioner == "none":
            self.preconditioner = None
        elif self.config.preconditioner == "jacobi":
            self.preconditioner = self._setup_jacobi_preconditioner()
        elif self.config.preconditioner == "ilu":
            self.preconditioner = self._setup_ilu_preconditioner()
        else:
            raise ValueError(f"Unknown preconditioner: {self.config.preconditioner}")
            
    def solve(self, 
             A: LinearOperator,
             b: np.ndarray,
             x0: Optional[np.ndarray] = None
             ) -> np.ndarray:
        """
        Solve the linear system Ax = b using GMRES.
        
        Args:
            A: Linear operator representing the system matrix
            b: Right-hand side vector
            x0: Initial guess (optional)
            
        Returns:
            np.ndarray: Solution vector
        """
        if x0 is None:
            x0 = np.zeros_like(b)
            
        # Apply preconditioner if available
        if self.preconditioner is not None:
            A = self.preconditioner(A)
            b = self.preconditioner(b)
            
        # GMRES iteration
        x = x0.copy()
        r = b - A.matvec(x)
        beta = np.linalg.norm(r)
        
        if beta < self.config.tolerance:
            return x
            
        # Initialize Krylov subspace
        V = np.zeros((len(b), self.config.restart + 1))
        H = np.zeros((self.config.restart + 1, self.config.restart))
        V[:, 0] = r / beta
        
        for j in range(self.config.restart):
            # Arnoldi process
            w = A.matvec(V[:, j])
            for i in range(j + 1):
                H[i, j] = np.dot(w, V[:, i])
                w = w - H[i, j] * V[:, i]
                
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < self.config.tolerance:
                break
                
            V[:, j + 1] = w / H[j + 1, j]
            
            # Solve least squares problem
            e1 = np.zeros(j + 2)
            e1[0] = beta
            y = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)[0]
            
            # Update solution
            x = x0 + np.dot(V[:, :j + 1], y)
            
            # Check convergence
            r = b - A.matvec(x)
            beta = np.linalg.norm(r)
            if beta < self.config.tolerance:
                break
                
        return x
        
    def _setup_jacobi_preconditioner(self) -> Callable:
        """
        Set up Jacobi preconditioner.
        
        Returns:
            Callable: Preconditioner function
        """
        def preconditioner(A):
            if isinstance(A, LinearOperator):
                # Extract diagonal
                n = A.shape[0]
                diag = np.zeros(n)
                for i in range(n):
                    ei = np.zeros(n)
                    ei[i] = 1.0
                    diag[i] = A.matvec(ei)[i]
                    
                # Create preconditioner
                M = LinearOperator(
                    (n, n),
                    matvec=lambda x: x / diag,
                    rmatvec=lambda x: x / diag
                )
                return M
            else:
                # For vector input
                return A / self.diag
                
        return preconditioner
        
    def _setup_ilu_preconditioner(self) -> Callable:
        """
        Set up ILU preconditioner.
        
        Returns:
            Callable: Preconditioner function
        """
        from scipy.sparse.linalg import spilu
        
        def preconditioner(A):
            if isinstance(A, LinearOperator):
                # Convert to sparse matrix
                n = A.shape[0]
                rows = []
                cols = []
                data = []
                for i in range(n):
                    ei = np.zeros(n)
                    ei[i] = 1.0
                    row = A.matvec(ei)
                    for j in range(n):
                        if abs(row[j]) > 1e-10:
                            rows.append(i)
                            cols.append(j)
                            data.append(row[j])
                            
                A_sparse = csr_matrix((data, (rows, cols)), shape=(n, n))
                
                # Compute ILU factorization
                self.ilu = spilu(A_sparse)
                
                # Create preconditioner
                M = LinearOperator(
                    (n, n),
                    matvec=lambda x: self.ilu.solve(x),
                    rmatvec=lambda x: self.ilu.solve(x)
                )
                return M
            else:
                # For vector input
                return self.ilu.solve(A)
                
        return preconditioner 
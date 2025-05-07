"""
AMG solver module for the compressible flow solver.

This module implements the algebraic multigrid solver from the original PHASTA codebase.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import LinearOperator

@dataclass
class AMGConfig:
    """Configuration for the AMG solver."""
    max_levels: int = 10  # Maximum number of multigrid levels
    max_coarse_size: int = 100  # Maximum size of coarsest level
    strength_threshold: float = 0.25  # Threshold for strong connections
    max_iter: int = 100  # Maximum number of iterations
    tolerance: float = 1e-6  # Convergence tolerance
    smoother: str = "jacobi"  # Smoother type
    smoother_params: Optional[Dict[str, Any]] = None  # Smoother parameters

class AMGSolver:
    """
    Implements the algebraic multigrid solver.
    
    This class implements the AMG solver from the original PHASTA codebase.
    """
    
    def __init__(self, config: AMGConfig):
        """
        Initialize the AMG solver.
        
        Args:
            config: AMG configuration parameters
        """
        self.config = config
        self._setup_smoother()
        
    def _setup_smoother(self):
        """Set up the smoother."""
        if self.config.smoother == "jacobi":
            self.smoother = self._setup_jacobi_smoother()
        elif self.config.smoother == "gauss_seidel":
            self.smoother = self._setup_gauss_seidel_smoother()
        else:
            raise ValueError(f"Unknown smoother: {self.config.smoother}")
            
    def solve(self, 
             A: csr_matrix,
             b: np.ndarray,
             x0: Optional[np.ndarray] = None
             ) -> np.ndarray:
        """
        Solve the linear system Ax = b using AMG.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x0: Initial guess (optional)
            
        Returns:
            np.ndarray: Solution vector
        """
        if x0 is None:
            x0 = np.zeros_like(b)
            
        # Build multigrid hierarchy
        self._build_hierarchy(A)
        
        # V-cycle iteration
        x = x0.copy()
        for iter in range(self.config.max_iter):
            # Pre-smoothing
            x = self._smooth(A, b, x)
            
            # Compute residual
            r = b - A.dot(x)
            
            # Restrict residual
            r_coarse = self._restrict(r)
            
            # Solve coarse system
            e_coarse = self._solve_coarse(r_coarse)
            
            # Prolongate error
            e = self._prolongate(e_coarse)
            
            # Update solution
            x = x + e
            
            # Post-smoothing
            x = self._smooth(A, b, x)
            
            # Check convergence
            r = b - A.dot(x)
            if np.linalg.norm(r) < self.config.tolerance:
                break
                
        return x
        
    def _build_hierarchy(self, A: csr_matrix):
        """
        Build the multigrid hierarchy.
        
        Args:
            A: System matrix
        """
        self.levels = []
        current_A = A
        
        while len(self.levels) < self.config.max_levels:
            # Compute strength of connections
            S = self._compute_strength(current_A)
            
            # Select coarse grid points
            C, F = self._select_coarse_points(S)
            
            # Build interpolation operator
            P = self._build_interpolation(current_A, C, F)
            
            # Build coarse grid operator
            R = P.T
            A_coarse = R.dot(current_A.dot(P))
            
            # Store level information
            self.levels.append({
                'A': current_A,
                'P': P,
                'R': R,
                'C': C,
                'F': F
            })
            
            current_A = A_coarse
            
            # Check if coarsest level is small enough
            if current_A.shape[0] <= self.config.max_coarse_size:
                break
                
    def _compute_strength(self, A: csr_matrix) -> csr_matrix:
        """
        Compute strength of connections.
        
        Args:
            A: System matrix
            
        Returns:
            csr_matrix: Strength matrix
        """
        # Get matrix data
        rows, cols, data = A.nonzero()
        
        # Compute strength
        strength = np.zeros_like(data)
        for i in range(len(data)):
            if rows[i] != cols[i]:
                strength[i] = abs(data[i]) / max(
                    abs(A[rows[i], rows[i]]),
                    abs(A[cols[i], cols[i]])
                )
                
        # Create strength matrix
        S = csr_matrix((strength, (rows, cols)), shape=A.shape)
        
        # Apply threshold
        S.data[S.data < self.config.strength_threshold] = 0
        S.eliminate_zeros()
        
        return S
        
    def _select_coarse_points(self, S: csr_matrix) -> Tuple[List[int], List[int]]:
        """
        Select coarse grid points.
        
        Args:
            S: Strength matrix
            
        Returns:
            tuple: (C, F) coarse and fine point lists
        """
        n = S.shape[0]
        C = []  # Coarse points
        F = []  # Fine points
        
        # Compute measure of importance
        measure = np.zeros(n)
        for i in range(n):
            measure[i] = S[i, :].nnz
            
        # Select coarse points
        while len(C) + len(F) < n:
            # Find point with maximum measure
            i = np.argmax(measure)
            
            if measure[i] > 0:
                C.append(i)
                measure[i] = -1  # Mark as selected
                
                # Update measures of neighbors
                for j in S[i, :].nonzero()[1]:
                    if measure[j] > 0:
                        measure[j] = -1
                        F.append(j)
            else:
                break
                
        return C, F
        
    def _build_interpolation(self,
                           A: csr_matrix,
                           C: List[int],
                           F: List[int]
                           ) -> csr_matrix:
        """
        Build interpolation operator.
        
        Args:
            A: System matrix
            C: Coarse point list
            F: Fine point list
            
        Returns:
            csr_matrix: Interpolation operator
        """
        n = A.shape[0]
        nc = len(C)
        
        # Create mapping from global to coarse indices
        c_map = {c: i for i, c in enumerate(C)}
        
        # Build interpolation matrix
        rows = []
        cols = []
        data = []
        
        # Interpolate from coarse points
        for i in C:
            rows.append(i)
            cols.append(c_map[i])
            data.append(1.0)
            
        # Interpolate from fine points
        for i in F:
            # Find strong connections to coarse points
            strong_c = []
            strong_a = []
            for j in A[i, :].nonzero()[1]:
                if j in C:
                    strong_c.append(j)
                    strong_a.append(A[i, j])
                    
            if strong_c:
                # Compute interpolation weights
                denom = sum(strong_a)
                for j, a in zip(strong_c, strong_a):
                    rows.append(i)
                    cols.append(c_map[j])
                    data.append(a / denom)
                    
        return csr_matrix((data, (rows, cols)), shape=(n, nc))
        
    def _restrict(self, r: np.ndarray) -> np.ndarray:
        """
        Restrict residual to coarser level.
        
        Args:
            r: Fine level residual
            
        Returns:
            np.ndarray: Coarse level residual
        """
        r_coarse = r.copy()
        for level in self.levels:
            r_coarse = level['R'].dot(r_coarse)
        return r_coarse
        
    def _prolongate(self, e_coarse: np.ndarray) -> np.ndarray:
        """
        Prolongate error to finer level.
        
        Args:
            e_coarse: Coarse level error
            
        Returns:
            np.ndarray: Fine level error
        """
        e = e_coarse.copy()
        for level in reversed(self.levels):
            e = level['P'].dot(e)
        return e
        
    def _solve_coarse(self, r_coarse: np.ndarray) -> np.ndarray:
        """
        Solve the coarsest level system.
        
        Args:
            r_coarse: Coarse level right-hand side
            
        Returns:
            np.ndarray: Coarse level solution
        """
        A_coarse = self.levels[-1]['A']
        return np.linalg.solve(A_coarse.toarray(), r_coarse)
        
    def _smooth(self,
               A: csr_matrix,
               b: np.ndarray,
               x: np.ndarray
               ) -> np.ndarray:
        """
        Apply smoother.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x: Current solution
            
        Returns:
            np.ndarray: Smoothed solution
        """
        return self.smoother(A, b, x)
        
    def _setup_jacobi_smoother(self) -> Callable:
        """
        Set up Jacobi smoother.
        
        Returns:
            Callable: Smoother function
        """
        def smoother(A: csr_matrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
            # Extract diagonal
            diag = A.diagonal()
            
            # Compute residual
            r = b - A.dot(x)
            
            # Update solution
            x = x + r / diag
            
            return x
            
        return smoother
        
    def _setup_gauss_seidel_smoother(self) -> Callable:
        """
        Set up Gauss-Seidel smoother.
        
        Returns:
            Callable: Smoother function
        """
        def smoother(A: csr_matrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
            # Convert to COO format for easier access
            A_coo = A.tocoo()
            
            # Sort by row
            sort_idx = np.argsort(A_coo.row)
            rows = A_coo.row[sort_idx]
            cols = A_coo.col[sort_idx]
            data = A_coo.data[sort_idx]
            
            # Forward sweep
            for i in range(A.shape[0]):
                # Find entries in row i
                row_idx = np.where(rows == i)[0]
                row_cols = cols[row_idx]
                row_data = data[row_idx]
                
                # Compute residual
                r = b[i]
                for j, a in zip(row_cols, row_data):
                    if j < i:  # Already updated
                        r -= a * x[j]
                    else:  # Not yet updated
                        r -= a * x[j]
                        
                # Update solution
                diag_idx = np.where(row_cols == i)[0]
                if len(diag_idx) > 0:
                    x[i] = r / row_data[diag_idx[0]]
                    
            return x
            
        return smoother 
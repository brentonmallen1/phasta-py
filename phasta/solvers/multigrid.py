"""Multi-grid solvers module.

This module provides multi-grid solution methods for linear systems, including
geometric and algebraic multi-grid with various smoothers and transfer operators.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import List, Optional, Tuple, Union, Callable
import logging

logger = logging.getLogger(__name__)


class MultiGridSolver:
    """Base class for multi-grid solvers."""
    
    def __init__(self, A: sp.spmatrix, max_levels: int = 5,
                 smoother: str = 'jacobi', n_smooth: int = 2):
        """Initialize multi-grid solver.
        
        Args:
            A: System matrix
            max_levels: Maximum number of grid levels
            smoother: Smoother type ('jacobi', 'gauss_seidel', or 'sor')
            n_smooth: Number of smoothing iterations
        """
        self.A = A
        self.max_levels = max_levels
        self.smoother = smoother
        self.n_smooth = n_smooth
        self.levels = []
        self._setup_levels()
    
    def _setup_levels(self):
        """Set up multi-grid levels."""
        raise NotImplementedError
    
    def _smooth(self, A: sp.spmatrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Apply smoothing iteration.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x: Initial guess
            
        Returns:
            Smoothed solution
        """
        if self.smoother == 'jacobi':
            return self._jacobi_smooth(A, b, x)
        elif self.smoother == 'gauss_seidel':
            return self._gauss_seidel_smooth(A, b, x)
        elif self.smoother == 'sor':
            return self._sor_smooth(A, b, x)
        else:
            raise ValueError(f"Unknown smoother: {self.smoother}")
    
    def _jacobi_smooth(self, A: sp.spmatrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Apply Jacobi smoothing.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x: Initial guess
            
        Returns:
            Smoothed solution
        """
        D = sp.diags(A.diagonal())
        D_inv = sp.diags(1.0 / A.diagonal())
        R = A - D
        
        for _ in range(self.n_smooth):
            x = D_inv @ (b - R @ x)
        
        return x
    
    def _gauss_seidel_smooth(self, A: sp.spmatrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Apply Gauss-Seidel smoothing.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x: Initial guess
            
        Returns:
            Smoothed solution
        """
        A_csr = A.tocsr()
        for _ in range(self.n_smooth):
            for i in range(len(x)):
                x[i] = (b[i] - A_csr[i, :i] @ x[:i] - A_csr[i, i+1:] @ x[i+1:]) / A_csr[i, i]
        
        return x
    
    def _sor_smooth(self, A: sp.spmatrix, b: np.ndarray, x: np.ndarray,
                   omega: float = 1.5) -> np.ndarray:
        """Apply SOR smoothing.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            x: Initial guess
            omega: Relaxation parameter
            
        Returns:
            Smoothed solution
        """
        A_csr = A.tocsr()
        for _ in range(self.n_smooth):
            for i in range(len(x)):
                x[i] = (1 - omega) * x[i] + omega * (
                    b[i] - A_csr[i, :i] @ x[:i] - A_csr[i, i+1:] @ x[i+1:]
                ) / A_csr[i, i]
        
        return x
    
    def solve(self, b: np.ndarray, x0: Optional[np.ndarray] = None,
             max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Solve linear system using multi-grid method.
        
        Args:
            b: Right-hand side vector
            x0: Initial guess
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Solution vector x
        """
        if x0 is None:
            x0 = np.zeros_like(b)
        
        x = x0.copy()
        for _ in range(max_iter):
            x = self._v_cycle(b, x)
            residual = np.linalg.norm(b - self.A @ x)
            if residual < tol:
                break
        
        return x
    
    def _v_cycle(self, b: np.ndarray, x: np.ndarray, level: int = 0) -> np.ndarray:
        """Perform V-cycle.
        
        Args:
            b: Right-hand side vector
            x: Initial guess
            level: Current grid level
            
        Returns:
            Updated solution
        """
        raise NotImplementedError


class GeometricMultiGrid(MultiGridSolver):
    """Geometric multi-grid solver."""
    
    def __init__(self, A: sp.spmatrix, grid_hierarchy: List[Tuple[np.ndarray, np.ndarray]],
                 max_levels: int = 5, smoother: str = 'jacobi', n_smooth: int = 2):
        """Initialize geometric multi-grid solver.
        
        Args:
            A: System matrix
            grid_hierarchy: List of (nodes, elements) for each level
            max_levels: Maximum number of grid levels
            smoother: Smoother type
            n_smooth: Number of smoothing iterations
        """
        self.grid_hierarchy = grid_hierarchy
        super().__init__(A, max_levels, smoother, n_smooth)
    
    def _setup_levels(self):
        """Set up multi-grid levels."""
        for i in range(min(self.max_levels, len(self.grid_hierarchy))):
            nodes, elements = self.grid_hierarchy[i]
            if i == 0:
                self.levels.append({
                    'A': self.A,
                    'nodes': nodes,
                    'elements': elements
                })
            else:
                # Compute coarse grid matrix
                A_coarse = self._compute_coarse_matrix(
                    self.levels[-1]['A'],
                    self.levels[-1]['nodes'],
                    nodes
                )
                self.levels.append({
                    'A': A_coarse,
                    'nodes': nodes,
                    'elements': elements
                })
    
    def _compute_coarse_matrix(self, A_fine: sp.spmatrix,
                             nodes_fine: np.ndarray,
                             nodes_coarse: np.ndarray) -> sp.spmatrix:
        """Compute coarse grid matrix.
        
        Args:
            A_fine: Fine grid matrix
            nodes_fine: Fine grid nodes
            nodes_coarse: Coarse grid nodes
            
        Returns:
            Coarse grid matrix
        """
        # Compute interpolation matrix
        P = self._compute_interpolation(nodes_fine, nodes_coarse)
        
        # Compute coarse grid matrix
        return P.T @ A_fine @ P
    
    def _compute_interpolation(self, nodes_fine: np.ndarray,
                             nodes_coarse: np.ndarray) -> sp.spmatrix:
        """Compute interpolation matrix.
        
        Args:
            nodes_fine: Fine grid nodes
            nodes_coarse: Coarse grid nodes
            
        Returns:
            Interpolation matrix
        """
        # Use linear interpolation
        n_fine = len(nodes_fine)
        n_coarse = len(nodes_coarse)
        P = sp.lil_matrix((n_fine, n_coarse))
        
        for i in range(n_fine):
            # Find nearest coarse nodes
            distances = np.linalg.norm(nodes_coarse - nodes_fine[i], axis=1)
            nearest = np.argsort(distances)[:4]  # Use 4 nearest nodes
            
            # Compute interpolation weights
            weights = 1.0 / (distances[nearest] + 1e-10)
            weights /= np.sum(weights)
            
            # Set interpolation weights
            P[i, nearest] = weights
        
        return P.tocsr()
    
    def _v_cycle(self, b: np.ndarray, x: np.ndarray, level: int = 0) -> np.ndarray:
        """Perform V-cycle.
        
        Args:
            b: Right-hand side vector
            x: Initial guess
            level: Current grid level
            
        Returns:
            Updated solution
        """
        if level == len(self.levels) - 1:
            # Solve on coarsest grid
            return spsolve(self.levels[level]['A'], b)
        
        # Pre-smoothing
        x = self._smooth(self.levels[level]['A'], b, x)
        
        # Compute residual
        r = b - self.levels[level]['A'] @ x
        
        # Restrict residual
        P = self._compute_interpolation(
            self.levels[level]['nodes'],
            self.levels[level + 1]['nodes']
        )
        r_coarse = P.T @ r
        
        # Recursive solve
        e_coarse = self._v_cycle(r_coarse, np.zeros_like(r_coarse), level + 1)
        
        # Prolongate error
        e = P @ e_coarse
        
        # Correct solution
        x = x + e
        
        # Post-smoothing
        x = self._smooth(self.levels[level]['A'], b, x)
        
        return x


class AlgebraicMultiGrid(MultiGridSolver):
    """Algebraic multi-grid solver."""
    
    def __init__(self, A: sp.spmatrix, max_levels: int = 5,
                 smoother: str = 'jacobi', n_smooth: int = 2,
                 strength_threshold: float = 0.25):
        """Initialize algebraic multi-grid solver.
        
        Args:
            A: System matrix
            max_levels: Maximum number of grid levels
            smoother: Smoother type
            n_smooth: Number of smoothing iterations
            strength_threshold: Threshold for strong connections
        """
        self.strength_threshold = strength_threshold
        super().__init__(A, max_levels, smoother, n_smooth)
    
    def _setup_levels(self):
        """Set up multi-grid levels."""
        A = self.A
        for _ in range(self.max_levels):
            if A.shape[0] <= 100:  # Stop if matrix is small enough
                break
            
            # Compute strength of connection
            S = self._compute_strength(A)
            
            # Compute interpolation
            P = self._compute_interpolation(A, S)
            
            # Compute coarse grid matrix
            A = P.T @ A @ P
            
            self.levels.append({
                'A': A,
                'P': P
            })
    
    def _compute_strength(self, A: sp.spmatrix) -> sp.spmatrix:
        """Compute strength of connection matrix.
        
        Args:
            A: System matrix
            
        Returns:
            Strength matrix
        """
        A_csr = A.tocsr()
        S = sp.lil_matrix(A.shape)
        
        for i in range(A.shape[0]):
            row = A_csr[i].toarray().flatten()
            max_off_diag = np.max(np.abs(row[row != 0]))
            threshold = self.strength_threshold * max_off_diag
            
            # Mark strong connections
            strong = np.abs(row) > threshold
            S[i, strong] = 1
        
        return S.tocsr()
    
    def _compute_interpolation(self, A: sp.spmatrix,
                             S: sp.spmatrix) -> sp.spmatrix:
        """Compute interpolation matrix.
        
        Args:
            A: System matrix
            S: Strength matrix
            
        Returns:
            Interpolation matrix
        """
        # Use standard interpolation
        A_csr = A.tocsr()
        S_csr = S.tocsr()
        n = A.shape[0]
        
        # Find C-points (coarse grid points)
        is_c_point = np.zeros(n, dtype=bool)
        for i in range(n):
            if not is_c_point[i]:
                is_c_point[i] = True
                # Mark neighbors as F-points
                neighbors = S_csr[i].indices
                is_c_point[neighbors] = False
        
        # Build interpolation
        P = sp.lil_matrix((n, np.sum(is_c_point)))
        c_points = np.where(is_c_point)[0]
        
        for i in range(n):
            if is_c_point[i]:
                # C-points interpolate to themselves
                j = np.searchsorted(c_points, i)
                P[i, j] = 1
            else:
                # F-points interpolate from strong C-points
                strong_c = np.intersect1d(S_csr[i].indices, c_points)
                if len(strong_c) > 0:
                    weights = A_csr[i, strong_c].toarray().flatten()
                    weights = np.abs(weights)
                    weights /= np.sum(weights)
                    for j, c in enumerate(strong_c):
                        P[i, np.searchsorted(c_points, c)] = weights[j]
        
        return P.tocsr()
    
    def _v_cycle(self, b: np.ndarray, x: np.ndarray, level: int = 0) -> np.ndarray:
        """Perform V-cycle.
        
        Args:
            b: Right-hand side vector
            x: Initial guess
            level: Current grid level
            
        Returns:
            Updated solution
        """
        if level == len(self.levels):
            # Solve on coarsest grid
            return spsolve(self.levels[level-1]['A'], b)
        
        # Pre-smoothing
        x = self._smooth(self.levels[level]['A'], b, x)
        
        # Compute residual
        r = b - self.levels[level]['A'] @ x
        
        # Restrict residual
        r_coarse = self.levels[level]['P'].T @ r
        
        # Recursive solve
        e_coarse = self._v_cycle(r_coarse, np.zeros_like(r_coarse), level + 1)
        
        # Prolongate error
        e = self.levels[level]['P'] @ e_coarse
        
        # Correct solution
        x = x + e
        
        # Post-smoothing
        x = self._smooth(self.levels[level]['A'], b, x)
        
        return x 
"""Advanced preconditioners module.

This module provides functionality for:
- Algebraic multigrid (AMG) preconditioners
- Incomplete factorization preconditioners
- Block preconditioners
- Hybrid preconditioners
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spilu

logger = logging.getLogger(__name__)


class Preconditioner(ABC):
    """Base class for preconditioners."""
    
    def __init__(self):
        """Initialize preconditioner."""
        pass
    
    @abstractmethod
    def setup(self, A: csr_matrix) -> None:
        """Setup preconditioner.
        
        Args:
            A: System matrix
        """
        pass
    
    @abstractmethod
    def apply(self, b: np.ndarray) -> np.ndarray:
        """Apply preconditioner.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Preconditioned vector
        """
        pass


class AMGPreconditioner(Preconditioner):
    """Algebraic multigrid preconditioner."""
    
    def __init__(self,
                 max_levels: int = 10,
                 coarsening_type: str = 'RS',
                 interpolation_type: str = 'classical',
                 smoother_type: str = 'jacobi',
                 smoothing_steps: int = 2):
        """Initialize AMG preconditioner.
        
        Args:
            max_levels: Maximum number of multigrid levels
            coarsening_type: Coarsening strategy ('RS', 'PMIS', 'HMIS')
            interpolation_type: Interpolation strategy ('classical', 'direct', 'standard')
            smoother_type: Smoother type ('jacobi', 'gauss_seidel', 'ilu')
            smoothing_steps: Number of smoothing steps
        """
        super().__init__()
        self.max_levels = max_levels
        self.coarsening_type = coarsening_type
        self.interpolation_type = interpolation_type
        self.smoother_type = smoother_type
        self.smoothing_steps = smoothing_steps
        self.levels = []
    
    def setup(self, A: csr_matrix) -> None:
        """Setup AMG preconditioner.
        
        Args:
            A: System matrix
        """
        self.levels = []
        current_A = A
        
        # Build multigrid hierarchy
        for level in range(self.max_levels):
            # Coarsen matrix
            P, R = self._coarsen(current_A)
            
            # Store level information
            self.levels.append({
                'A': current_A,
                'P': P,
                'R': R,
                'smoother': self._create_smoother(current_A)
            })
            
            # Compute coarse grid matrix
            current_A = R @ current_A @ P
            
            # Check if coarsening is sufficient
            if current_A.shape[0] < 10:
                break
    
    def apply(self, b: np.ndarray) -> np.ndarray:
        """Apply AMG preconditioner.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Preconditioned vector
        """
        return self._v_cycle(b, 0)
    
    def _v_cycle(self, b: np.ndarray, level: int) -> np.ndarray:
        """Perform V-cycle.
        
        Args:
            b: Right-hand side vector
            level: Current level
            
        Returns:
            Solution vector
        """
        if level == len(self.levels) - 1:
            # Direct solve on coarsest level
            return np.linalg.solve(self.levels[level]['A'].toarray(), b)
        
        # Pre-smoothing
        x = np.zeros_like(b)
        for _ in range(self.smoothing_steps):
            x = self.levels[level]['smoother'](b, x)
        
        # Compute residual
        r = b - self.levels[level]['A'] @ x
        
        # Restrict residual
        r_coarse = self.levels[level]['R'] @ r
        
        # Recursive solve
        e_coarse = self._v_cycle(r_coarse, level + 1)
        
        # Prolongate error
        e = self.levels[level]['P'] @ e_coarse
        
        # Update solution
        x = x + e
        
        # Post-smoothing
        for _ in range(self.smoothing_steps):
            x = self.levels[level]['smoother'](b, x)
        
        return x
    
    def _coarsen(self, A: csr_matrix) -> Tuple[csr_matrix, csr_matrix]:
        """Coarsen matrix.
        
        Args:
            A: System matrix
            
        Returns:
            Tuple of (prolongation matrix, restriction matrix)
        """
        if self.coarsening_type == 'RS':
            return self._rs_coarsening(A)
        elif self.coarsening_type == 'PMIS':
            return self._pmis_coarsening(A)
        elif self.coarsening_type == 'HMIS':
            return self._hmis_coarsening(A)
        else:
            raise ValueError(f"Unknown coarsening type: {self.coarsening_type}")
    
    def _rs_coarsening(self, A: csr_matrix) -> Tuple[csr_matrix, csr_matrix]:
        """Ruge-Stuben coarsening.
        
        Args:
            A: System matrix
            
        Returns:
            Tuple of (prolongation matrix, restriction matrix)
        """
        n = A.shape[0]
        
        # Compute strength matrix
        S = self._compute_strength_matrix(A)
        
        # Find coarse points
        coarse_points = self._find_coarse_points(S)
        
        # Build prolongation matrix
        P = self._build_prolongation(A, S, coarse_points)
        
        # Restriction matrix is transpose of prolongation
        R = P.T
        
        return P, R
    
    def _compute_strength_matrix(self, A: csr_matrix) -> csr_matrix:
        """Compute strength matrix.
        
        Args:
            A: System matrix
            
        Returns:
            Strength matrix
        """
        # Compute row sums
        row_sums = np.abs(A).sum(axis=1).A1
        
        # Compute strength threshold
        theta = 0.25
        
        # Build strength matrix
        S = A.copy()
        for i in range(A.shape[0]):
            for j in range(A.indptr[i], A.indptr[i+1]):
                if np.abs(A.data[j]) < theta * row_sums[i]:
                    S.data[j] = 0
        
        S.eliminate_zeros()
        return S
    
    def _find_coarse_points(self, S: csr_matrix) -> np.ndarray:
        """Find coarse points.
        
        Args:
            S: Strength matrix
            
        Returns:
            Array of coarse point indices
        """
        n = S.shape[0]
        coarse_points = []
        fine_points = set(range(n))
        
        # First pass: strong connections
        for i in range(n):
            if i in fine_points:
                # Count strong connections
                strong_connections = set(S[i].nonzero()[1])
                if len(strong_connections) > 0:
                    coarse_points.append(i)
                    fine_points.remove(i)
                    # Remove strongly connected points
                    fine_points -= strong_connections
        
        # Second pass: remaining points
        for i in range(n):
            if i in fine_points:
                coarse_points.append(i)
                fine_points.remove(i)
        
        return np.array(coarse_points)
    
    def _build_prolongation(self,
                          A: csr_matrix,
                          S: csr_matrix,
                          coarse_points: np.ndarray) -> csr_matrix:
        """Build prolongation matrix.
        
        Args:
            A: System matrix
            S: Strength matrix
            coarse_points: Array of coarse point indices
            
        Returns:
            Prolongation matrix
        """
        n = A.shape[0]
        n_coarse = len(coarse_points)
        
        # Create mapping from fine to coarse points
        fine_to_coarse = np.full(n, -1)
        fine_to_coarse[coarse_points] = np.arange(n_coarse)
        
        # Build prolongation matrix
        P = csr_matrix((n, n_coarse))
        
        # Direct interpolation
        for i in range(n):
            if i in coarse_points:
                # Coarse point
                P[i, fine_to_coarse[i]] = 1.0
            else:
                # Fine point
                neighbors = S[i].nonzero()[1]
                coarse_neighbors = [j for j in neighbors if j in coarse_points]
                
                if len(coarse_neighbors) > 0:
                    # Compute interpolation weights
                    weights = np.abs(A[i, coarse_neighbors].toarray()[0])
                    weights = weights / np.sum(weights)
                    
                    # Set prolongation entries
                    for j, w in zip(coarse_neighbors, weights):
                        P[i, fine_to_coarse[j]] = w
        
        return P
    
    def _create_smoother(self, A: csr_matrix) -> Callable:
        """Create smoother.
        
        Args:
            A: System matrix
            
        Returns:
            Smoother function
        """
        if self.smoother_type == 'jacobi':
            # Jacobi smoother
            D = spdiags(1.0 / A.diagonal(), 0, A.shape[0], A.shape[0])
            return lambda b, x: x + D @ (b - A @ x)
        
        elif self.smoother_type == 'gauss_seidel':
            # Gauss-Seidel smoother
            L = A.tocsr()
            U = A.tocsr()
            D = spdiags(A.diagonal(), 0, A.shape[0], A.shape[0])
            
            # Split into lower and upper triangular parts
            for i in range(A.shape[0]):
                for j in range(A.indptr[i], A.indptr[i+1]):
                    if A.indices[j] < i:
                        L.data[j] = A.data[j]
                        U.data[j] = 0
                    elif A.indices[j] > i:
                        L.data[j] = 0
                        U.data[j] = A.data[j]
            
            L = L + D
            return lambda b, x: np.linalg.solve(L.toarray(), b - U @ x)
        
        elif self.smoother_type == 'ilu':
            # ILU smoother
            ilu = spilu(A.tocsc())
            return lambda b, x: x + ilu.solve(b - A @ x)
        
        else:
            raise ValueError(f"Unknown smoother type: {self.smoother_type}")


class ILUPreconditioner(Preconditioner):
    """Incomplete LU preconditioner."""
    
    def __init__(self,
                 fill_factor: int = 10,
                 drop_tol: float = 1e-4):
        """Initialize ILU preconditioner.
        
        Args:
            fill_factor: Maximum fill-in factor
            drop_tol: Drop tolerance
        """
        super().__init__()
        self.fill_factor = fill_factor
        self.drop_tol = drop_tol
        self.ilu = None
    
    def setup(self, A: csr_matrix) -> None:
        """Setup ILU preconditioner.
        
        Args:
            A: System matrix
        """
        self.ilu = spilu(A.tocsc(),
                        fill_factor=self.fill_factor,
                        drop_tol=self.drop_tol)
    
    def apply(self, b: np.ndarray) -> np.ndarray:
        """Apply ILU preconditioner.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Preconditioned vector
        """
        if self.ilu is None:
            raise RuntimeError("Preconditioner not set up")
        return self.ilu.solve(b)


class BlockPreconditioner(Preconditioner):
    """Block preconditioner."""
    
    def __init__(self,
                 block_size: int,
                 preconditioner_type: str = 'ilu'):
        """Initialize block preconditioner.
        
        Args:
            block_size: Size of blocks
            preconditioner_type: Type of block preconditioner ('ilu', 'amg')
        """
        super().__init__()
        self.block_size = block_size
        self.preconditioner_type = preconditioner_type
        self.block_preconditioners = []
    
    def setup(self, A: csr_matrix) -> None:
        """Setup block preconditioner.
        
        Args:
            A: System matrix
        """
        n = A.shape[0]
        n_blocks = n // self.block_size
        
        # Create block preconditioners
        self.block_preconditioners = []
        for i in range(n_blocks):
            # Extract block
            start = i * self.block_size
            end = (i + 1) * self.block_size
            block = A[start:end, start:end]
            
            # Create preconditioner
            if self.preconditioner_type == 'ilu':
                preconditioner = ILUPreconditioner()
            elif self.preconditioner_type == 'amg':
                preconditioner = AMGPreconditioner()
            else:
                raise ValueError(f"Unknown preconditioner type: {self.preconditioner_type}")
            
            # Setup preconditioner
            preconditioner.setup(block)
            self.block_preconditioners.append(preconditioner)
    
    def apply(self, b: np.ndarray) -> np.ndarray:
        """Apply block preconditioner.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Preconditioned vector
        """
        if not self.block_preconditioners:
            raise RuntimeError("Preconditioner not set up")
        
        n = len(b)
        n_blocks = n // self.block_size
        x = np.zeros_like(b)
        
        # Apply block preconditioners
        for i in range(n_blocks):
            start = i * self.block_size
            end = (i + 1) * self.block_size
            x[start:end] = self.block_preconditioners[i].apply(b[start:end])
        
        return x 
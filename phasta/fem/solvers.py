"""Linear solvers for finite element method."""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, spsolve


class LinearSolver:
    """Base class for linear solvers."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-8):
        """Initialize linear solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
        self.iterations = 0
        self.residual = 0.0
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
        """Solve linear system Ax = b.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        raise NotImplementedError("Subclasses must implement solve()")


class DirectSolver(LinearSolver):
    """Direct solver using sparse LU decomposition."""
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
        """Solve linear system using direct method.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        return spsolve(A, b)


class GMRESSolver(LinearSolver):
    """GMRES iterative solver."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-8,
                 restart: int = 30, preconditioner: Optional[Callable] = None):
        """Initialize GMRES solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            restart: Number of iterations before restart
            preconditioner: Preconditioner function
        """
        super().__init__(max_iter, tol)
        self.restart = restart
        self.preconditioner = preconditioner
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
        """Solve linear system using GMRES.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        # Create linear operator
        if self.preconditioner is not None:
            M = LinearOperator(A.shape, matvec=self.preconditioner)
        else:
            M = None
        
        # Solve system
        x, info = gmres(A, b, M=M, restart=self.restart,
                       maxiter=self.max_iter, tol=self.tol)
        
        # Store convergence information
        self.iterations = info
        self.residual = np.linalg.norm(A @ x - b)
        
        return x


class ConjugateGradientSolver(LinearSolver):
    """Conjugate gradient solver for symmetric positive definite systems."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-8,
                 preconditioner: Optional[Callable] = None):
        """Initialize conjugate gradient solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            preconditioner: Preconditioner function
        """
        super().__init__(max_iter, tol)
        self.preconditioner = preconditioner
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
        """Solve linear system using conjugate gradient.
        
        Args:
            A: System matrix (must be symmetric positive definite)
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        # Create linear operator
        if self.preconditioner is not None:
            M = LinearOperator(A.shape, matvec=self.preconditioner)
        else:
            M = None
        
        # Solve system
        x, info = sparse.linalg.cg(A, b, M=M,
                                 maxiter=self.max_iter, tol=self.tol)
        
        # Store convergence information
        self.iterations = info
        self.residual = np.linalg.norm(A @ x - b)
        
        return x


class BiCGSTABSolver(LinearSolver):
    """BiCGSTAB solver for non-symmetric systems."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-8,
                 preconditioner: Optional[Callable] = None):
        """Initialize BiCGSTAB solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            preconditioner: Preconditioner function
        """
        super().__init__(max_iter, tol)
        self.preconditioner = preconditioner
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
        """Solve linear system using BiCGSTAB.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        # Create linear operator
        if self.preconditioner is not None:
            M = LinearOperator(A.shape, matvec=self.preconditioner)
        else:
            M = None
        
        # Solve system
        x, info = sparse.linalg.bicgstab(A, b, M=M,
                                       maxiter=self.max_iter, tol=self.tol)
        
        # Store convergence information
        self.iterations = info
        self.residual = np.linalg.norm(A @ x - b)
        
        return x


class SolverFactory:
    """Factory for creating linear solvers."""
    
    @staticmethod
    def create_solver(solver_type: str, **kwargs) -> LinearSolver:
        """Create a linear solver of the specified type.
        
        Args:
            solver_type: Type of solver ('direct', 'gmres', 'cg', 'bicgstab')
            **kwargs: Additional arguments for solver initialization
            
        Returns:
            Linear solver instance
        """
        solvers = {
            'direct': DirectSolver,
            'gmres': GMRESSolver,
            'cg': ConjugateGradientSolver,
            'bicgstab': BiCGSTABSolver
        }
        
        if solver_type not in solvers:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
        return solvers[solver_type](**kwargs) 
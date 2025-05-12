"""Advanced nonlinear solvers module.

This module provides functionality for:
- Newton-Krylov methods
- Trust-region methods
- Line search methods
- Globalization strategies
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from abc import ABC, abstractmethod
from scipy.sparse import spmatrix
from scipy.sparse.linalg import spsolve
from .linear import LinearSolver, ConjugateGradient, GMRES

logger = logging.getLogger(__name__)


class NonlinearSolver(ABC):
    """Base class for nonlinear solvers."""
    
    def __init__(self, 
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 verbose: bool = False):
        """Initialize nonlinear solver.
        
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
    def solve(self, 
             residual: Callable[[np.ndarray], np.ndarray],
             jacobian: Callable[[np.ndarray], Union[np.ndarray, spmatrix]],
             x0: np.ndarray) -> np.ndarray:
        """Solve nonlinear system F(x) = 0.
        
        Args:
            residual: Function that computes residual F(x)
            jacobian: Function that computes Jacobian dF/dx
            x0: Initial guess
            
        Returns:
            Solution vector
        """
        pass


class NewtonKrylov(NonlinearSolver):
    """Newton-Krylov solver with line search."""
    
    def __init__(self,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 verbose: bool = False,
                 linear_solver: Optional[LinearSolver] = None,
                 max_line_search: int = 10,
                 min_step: float = 1e-4):
        """Initialize Newton-Krylov solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            verbose: Whether to print convergence information
            linear_solver: Linear solver for Newton step
            max_line_search: Maximum number of line search steps
            min_step: Minimum step size in line search
        """
        super().__init__(max_iter, tol, verbose)
        self.linear_solver = linear_solver or GMRES()
        self.max_line_search = max_line_search
        self.min_step = min_step
    
    def solve(self, 
             residual: Callable[[np.ndarray], np.ndarray],
             jacobian: Callable[[np.ndarray], Union[np.ndarray, spmatrix]],
             x0: np.ndarray) -> np.ndarray:
        """Solve nonlinear system using Newton-Krylov method.
        
        Args:
            residual: Function that computes residual F(x)
            jacobian: Function that computes Jacobian dF/dx
            x0: Initial guess
            
        Returns:
            Solution vector
        """
        x = x0.copy()
        F = residual(x)
        F_norm = np.linalg.norm(F)
        
        for i in range(self.max_iter):
            if self.verbose:
                logger.info(f"Iteration {i}: residual = {F_norm}")
            
            if F_norm < self.tol:
                self.iterations = i
                self.residual = F_norm
                return x
            
            # Compute Newton step
            J = jacobian(x)
            dx = self.linear_solver.solve(J, -F)
            
            # Line search
            alpha = 1.0
            F_new = residual(x + alpha * dx)
            F_new_norm = np.linalg.norm(F_new)
            
            for _ in range(self.max_line_search):
                if F_new_norm < F_norm:
                    break
                alpha *= 0.5
                if alpha < self.min_step:
                    break
                F_new = residual(x + alpha * dx)
                F_new_norm = np.linalg.norm(F_new)
            
            x += alpha * dx
            F = F_new
            F_norm = F_new_norm
        
        self.iterations = self.max_iter
        self.residual = F_norm
        return x


class TrustRegion(NonlinearSolver):
    """Trust-region solver."""
    
    def __init__(self,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 verbose: bool = False,
                 max_radius: float = 1.0,
                 min_radius: float = 1e-4,
                 eta1: float = 0.1,
                 eta2: float = 0.75):
        """Initialize trust-region solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            verbose: Whether to print convergence information
            max_radius: Maximum trust region radius
            min_radius: Minimum trust region radius
            eta1: Lower threshold for radius update
            eta2: Upper threshold for radius update
        """
        super().__init__(max_iter, tol, verbose)
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.eta1 = eta1
        self.eta2 = eta2
    
    def solve(self, 
             residual: Callable[[np.ndarray], np.ndarray],
             jacobian: Callable[[np.ndarray], Union[np.ndarray, spmatrix]],
             x0: np.ndarray) -> np.ndarray:
        """Solve nonlinear system using trust-region method.
        
        Args:
            residual: Function that computes residual F(x)
            jacobian: Function that computes Jacobian dF/dx
            x0: Initial guess
            
        Returns:
            Solution vector
        """
        x = x0.copy()
        F = residual(x)
        F_norm = np.linalg.norm(F)
        radius = self.max_radius
        
        for i in range(self.max_iter):
            if self.verbose:
                logger.info(f"Iteration {i}: residual = {F_norm}, radius = {radius}")
            
            if F_norm < self.tol:
                self.iterations = i
                self.residual = F_norm
                return x
            
            # Compute step
            J = jacobian(x)
            dx = self._compute_step(J, F, radius)
            
            # Evaluate step
            F_new = residual(x + dx)
            F_new_norm = np.linalg.norm(F_new)
            
            # Compute reduction ratio
            actual_reduction = F_norm - F_new_norm
            predicted_reduction = F_norm - np.linalg.norm(F + J @ dx)
            rho = actual_reduction / predicted_reduction if predicted_reduction > 0 else 0
            
            # Update trust region radius
            if rho < self.eta1:
                radius *= 0.25
            elif rho > self.eta2:
                radius = min(2 * radius, self.max_radius)
            
            # Update iterate
            if rho > 0:
                x += dx
                F = F_new
                F_norm = F_new_norm
        
        self.iterations = self.max_iter
        self.residual = F_norm
        return x
    
    def _compute_step(self, 
                     J: Union[np.ndarray, spmatrix],
                     F: np.ndarray,
                     radius: float) -> np.ndarray:
        """Compute trust-region step.
        
        Args:
            J: Jacobian matrix
            F: Residual vector
            radius: Trust region radius
            
        Returns:
            Step vector
        """
        # Solve trust-region subproblem
        n = len(F)
        I = np.eye(n)
        dx = np.linalg.solve(J.T @ J + radius * I, -J.T @ F)
        
        # Project step onto trust region
        dx_norm = np.linalg.norm(dx)
        if dx_norm > radius:
            dx *= radius / dx_norm
        
        return dx 
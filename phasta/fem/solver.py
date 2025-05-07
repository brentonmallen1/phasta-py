"""Solver module for finite element method."""

from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
from .global_assembly import GlobalAssembly


class Solver:
    """Base class for finite element solvers."""
    
    def __init__(self, assembly: GlobalAssembly):
        """Initialize solver.
        
        Args:
            assembly: GlobalAssembly instance for matrix assembly
        """
        self.assembly = assembly
    
    def solve(self, nodes: np.ndarray, elements: np.ndarray, 
              dirichlet_nodes: Optional[np.ndarray] = None,
              dirichlet_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve the system.
        
        Args:
            nodes: Physical coordinates of all nodes
            elements: Element connectivity
            dirichlet_nodes: Indices of nodes with Dirichlet BC
            dirichlet_values: Values at Dirichlet nodes
            
        Returns:
            Solution vector
        """
        raise NotImplementedError("Subclasses must implement solve()")


class LinearSolver(Solver):
    """Linear solver for finite element problems."""
    
    def __init__(self, assembly: GlobalAssembly, solver_type: str = 'direct'):
        """Initialize linear solver.
        
        Args:
            assembly: GlobalAssembly instance for matrix assembly
            solver_type: Type of solver ('direct' or 'iterative')
        """
        super().__init__(assembly)
        self.solver_type = solver_type
    
    def solve(self, nodes: np.ndarray, elements: np.ndarray,
              dirichlet_nodes: Optional[np.ndarray] = None,
              dirichlet_values: Optional[np.ndarray] = None,
              k: Union[float, np.ndarray] = 1.0,
              f: Union[float, Callable] = 0.0) -> np.ndarray:
        """Solve linear system.
        
        Args:
            nodes: Physical coordinates of all nodes
            elements: Element connectivity
            dirichlet_nodes: Indices of nodes with Dirichlet BC
            dirichlet_values: Values at Dirichlet nodes
            k: Material property
            f: Source term
            
        Returns:
            Solution vector
        """
        # Assemble system
        K = self.assembly.assemble_stiffness_matrix(nodes, elements, k)
        F = self.assembly.assemble_load_vector(nodes, elements, f)
        
        # Apply boundary conditions
        if dirichlet_nodes is not None and dirichlet_values is not None:
            K, F = self.assembly.apply_dirichlet_bc(K, F, dirichlet_nodes, dirichlet_values)
        
        # Solve system
        if self.solver_type == 'direct':
            u = spsolve(K, F)
        else:  # iterative
            # Define matrix-vector product
            def matvec(x):
                return K @ x
            
            # Create linear operator
            A = LinearOperator(K.shape, matvec=matvec)
            
            # Solve using GMRES
            u, _ = gmres(A, F, tol=1e-8, maxiter=1000)
        
        return u


class TimeDependentSolver(Solver):
    """Time-dependent solver for finite element problems."""
    
    def __init__(self, assembly: GlobalAssembly, dt: float, theta: float = 0.5):
        """Initialize time-dependent solver.
        
        Args:
            assembly: GlobalAssembly instance for matrix assembly
            dt: Time step size
            theta: Time integration parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit)
        """
        super().__init__(assembly)
        self.dt = dt
        self.theta = theta
    
    def solve(self, nodes: np.ndarray, elements: np.ndarray,
              u0: np.ndarray, t_end: float,
              dirichlet_nodes: Optional[np.ndarray] = None,
              dirichlet_values: Optional[np.ndarray] = None,
              k: Union[float, np.ndarray] = 1.0,
              f: Union[float, Callable] = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Solve time-dependent system.
        
        Args:
            nodes: Physical coordinates of all nodes
            elements: Element connectivity
            u0: Initial solution
            t_end: End time
            dirichlet_nodes: Indices of nodes with Dirichlet BC
            dirichlet_values: Values at Dirichlet nodes
            k: Material property
            f: Source term
            
        Returns:
            Tuple of (time points, solution vectors)
        """
        # Assemble matrices
        M = self.assembly.assemble_mass_matrix(nodes, elements)
        K = self.assembly.assemble_stiffness_matrix(nodes, elements, k)
        
        # Compute time steps
        n_steps = int(np.ceil(t_end / self.dt))
        times = np.linspace(0, t_end, n_steps + 1)
        
        # Initialize solution array
        u = np.zeros((n_steps + 1, len(u0)))
        u[0] = u0
        
        # Time stepping
        for n in range(n_steps):
            # Current time
            t = times[n]
            
            # Compute load vector
            if callable(f):
                def f_t(x):
                    return f(x, t)
                F = self.assembly.assemble_load_vector(nodes, elements, f_t)
            else:
                F = self.assembly.assemble_load_vector(nodes, elements, f)
            
            # Apply boundary conditions
            if dirichlet_nodes is not None and dirichlet_values is not None:
                K_mod, F_mod = self.assembly.apply_dirichlet_bc(K, F, dirichlet_nodes, dirichlet_values)
            else:
                K_mod, F_mod = K, F
            
            # Compute system matrix and right-hand side
            A = M + self.theta * self.dt * K_mod
            b = (M - (1 - self.theta) * self.dt * K_mod) @ u[n] + self.dt * F_mod
            
            # Solve system
            u[n + 1] = spsolve(A, b)
        
        return times, u


class NonlinearSolver(Solver):
    """Nonlinear solver for finite element problems."""
    
    def __init__(self, assembly: GlobalAssembly, max_iter: int = 50, tol: float = 1e-8):
        """Initialize nonlinear solver.
        
        Args:
            assembly: GlobalAssembly instance for matrix assembly
            max_iter: Maximum number of Newton iterations
            tol: Convergence tolerance
        """
        super().__init__(assembly)
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, nodes: np.ndarray, elements: np.ndarray,
              u0: np.ndarray,
              dirichlet_nodes: Optional[np.ndarray] = None,
              dirichlet_values: Optional[np.ndarray] = None,
              k: Union[float, np.ndarray] = 1.0,
              f: Union[float, Callable] = 0.0) -> np.ndarray:
        """Solve nonlinear system using Newton's method.
        
        Args:
            nodes: Physical coordinates of all nodes
            elements: Element connectivity
            u0: Initial guess
            dirichlet_nodes: Indices of nodes with Dirichlet BC
            dirichlet_values: Values at Dirichlet nodes
            k: Material property
            f: Source term
            
        Returns:
            Solution vector
        """
        u = u0.copy()
        
        for iter in range(self.max_iter):
            # Compute residual
            K = self.assembly.assemble_stiffness_matrix(nodes, elements, k)
            F = self.assembly.assemble_load_vector(nodes, elements, f)
            R = K @ u - F
            
            # Apply boundary conditions
            if dirichlet_nodes is not None and dirichlet_values is not None:
                K_mod, R_mod = self.assembly.apply_dirichlet_bc(K, R, dirichlet_nodes, dirichlet_values)
            else:
                K_mod, R_mod = K, R
            
            # Check convergence
            if np.linalg.norm(R_mod) < self.tol:
                break
            
            # Compute update
            du = spsolve(K_mod, -R_mod)
            u += du
        
        if iter == self.max_iter - 1:
            raise RuntimeError("Newton iteration did not converge")
        
        return u 
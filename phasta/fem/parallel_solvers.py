"""Parallel solvers module for distributed systems."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import sparse
from mpi4py import MPI
from .solvers import LinearSolver, GMRESSolver, ConjugateGradientSolver, BiCGSTABSolver
from .preconditioners import Preconditioner, PreconditionerFactory
from .parallel import ParallelMesh


class ParallelLinearSolver(LinearSolver):
    """Base class for parallel linear solvers."""
    
    def __init__(self, comm: Optional[MPI.Comm] = None):
        """Initialize parallel solver.
        
        Args:
            comm: MPI communicator (default: MPI.COMM_WORLD)
        """
        super().__init__()
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
    def _communicate_ghost_values(self, x: np.ndarray, mesh: ParallelMesh) -> None:
        """Communicate ghost node values between processes.
        
        Args:
            x: Solution vector
            mesh: Parallel mesh
        """
        # Send ghost node values
        for rank, send_indices in mesh.send_buffers.items():
            send_data = x[send_indices]
            self.comm.Send(send_data, dest=rank, tag=0)
        
        # Receive ghost node values
        for rank, recv_indices in mesh.recv_buffers.items():
            recv_data = np.empty(len(recv_indices), dtype=x.dtype)
            self.comm.Recv(recv_data, source=rank, tag=0)
            x[recv_indices] = recv_data


class ParallelGMRES(ParallelLinearSolver, GMRESSolver):
    """Parallel GMRES solver."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6, 
                 restart: int = 30, comm: Optional[MPI.Comm] = None):
        """Initialize parallel GMRES solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            restart: Restart parameter
            comm: MPI communicator
        """
        ParallelLinearSolver.__init__(self, comm)
        GMRESSolver.__init__(self, max_iter, tol, restart)
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray, 
              mesh: ParallelMesh, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve linear system in parallel.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            mesh: Parallel mesh
            x0: Initial guess
            
        Returns:
            Solution vector
        """
        # Initialize solution
        x = x0.copy() if x0 is not None else np.zeros_like(b)
        
        # Main GMRES iteration
        for k in range(self.max_iter):
            # Compute residual
            r = b - A @ x
            self._communicate_ghost_values(r, mesh)
            
            # Check convergence
            norm_r = np.linalg.norm(r)
            if norm_r < self.tol:
                break
            
            # Arnoldi process
            V = np.zeros((len(b), self.restart + 1))
            H = np.zeros((self.restart + 1, self.restart))
            V[:, 0] = r / norm_r
            
            for j in range(self.restart):
                # Matrix-vector product
                w = A @ V[:, j]
                self._communicate_ghost_values(w, mesh)
                
                # Modified Gram-Schmidt
                for i in range(j + 1):
                    H[i, j] = np.dot(V[:, i], w)
                    w -= H[i, j] * V[:, i]
                
                H[j + 1, j] = np.linalg.norm(w)
                if H[j + 1, j] < self.tol:
                    break
                
                V[:, j + 1] = w / H[j + 1, j]
            
            # Solve least squares problem
            e1 = np.zeros(j + 2)
            e1[0] = norm_r
            y = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)[0]
            
            # Update solution
            x += V[:, :j + 1] @ y
            self._communicate_ghost_values(x, mesh)
        
        return x


class ParallelCG(ParallelLinearSolver, ConjugateGradientSolver):
    """Parallel Conjugate Gradient solver."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6,
                 comm: Optional[MPI.Comm] = None):
        """Initialize parallel CG solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            comm: MPI communicator
        """
        ParallelLinearSolver.__init__(self, comm)
        ConjugateGradientSolver.__init__(self, max_iter, tol)
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray,
              mesh: ParallelMesh, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve linear system in parallel.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            mesh: Parallel mesh
            x0: Initial guess
            
        Returns:
            Solution vector
        """
        # Initialize solution
        x = x0.copy() if x0 is not None else np.zeros_like(b)
        
        # Compute initial residual
        r = b - A @ x
        self._communicate_ghost_values(r, mesh)
        p = r.copy()
        
        # Main CG iteration
        for k in range(self.max_iter):
            # Matrix-vector product
            Ap = A @ p
            self._communicate_ghost_values(Ap, mesh)
            
            # Compute step length
            alpha = np.dot(r, r) / np.dot(p, Ap)
            
            # Update solution and residual
            x += alpha * p
            r_new = r - alpha * Ap
            self._communicate_ghost_values(r_new, mesh)
            
            # Check convergence
            if np.linalg.norm(r_new) < self.tol:
                break
            
            # Update search direction
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new
        
        return x


class ParallelBiCGSTAB(ParallelLinearSolver, BiCGSTABSolver):
    """Parallel BiCGSTAB solver."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6,
                 comm: Optional[MPI.Comm] = None):
        """Initialize parallel BiCGSTAB solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            comm: MPI communicator
        """
        ParallelLinearSolver.__init__(self, comm)
        BiCGSTABSolver.__init__(self, max_iter, tol)
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray,
              mesh: ParallelMesh, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve linear system in parallel.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            mesh: Parallel mesh
            x0: Initial guess
            
        Returns:
            Solution vector
        """
        # Initialize solution
        x = x0.copy() if x0 is not None else np.zeros_like(b)
        
        # Compute initial residual
        r = b - A @ x
        self._communicate_ghost_values(r, mesh)
        r0 = r.copy()
        
        # Initialize variables
        p = r.copy()
        v = np.zeros_like(r)
        rho = 1.0
        alpha = 1.0
        omega = 1.0
        
        # Main BiCGSTAB iteration
        for k in range(self.max_iter):
            # Compute rho
            rho_old = rho
            rho = np.dot(r0, r)
            
            # Check for breakdown
            if abs(rho) < self.tol:
                break
            
            # Compute beta
            beta = (rho / rho_old) * (alpha / omega)
            
            # Update p
            p = r + beta * (p - omega * v)
            
            # Compute v
            v = A @ p
            self._communicate_ghost_values(v, mesh)
            
            # Compute alpha
            alpha = rho / np.dot(r0, v)
            
            # Update solution and residual
            s = r - alpha * v
            t = A @ s
            self._communicate_ghost_values(t, mesh)
            
            # Compute omega
            omega = np.dot(t, s) / np.dot(t, t)
            
            # Update solution
            x += alpha * p + omega * s
            r = s - omega * t
            self._communicate_ghost_values(r, mesh)
            
            # Check convergence
            if np.linalg.norm(r) < self.tol:
                break
        
        return x


class ParallelSolverFactory:
    """Factory for creating parallel solvers."""
    
    @staticmethod
    def create(solver_type: str, **kwargs) -> ParallelLinearSolver:
        """Create a parallel solver.
        
        Args:
            solver_type: Type of solver ('gmres', 'cg', or 'bicgstab')
            **kwargs: Additional solver parameters
            
        Returns:
            Parallel solver instance
        """
        solver_type = solver_type.lower()
        if solver_type == 'gmres':
            return ParallelGMRES(**kwargs)
        elif solver_type == 'cg':
            return ParallelCG(**kwargs)
        elif solver_type == 'bicgstab':
            return ParallelBiCGSTAB(**kwargs)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}") 
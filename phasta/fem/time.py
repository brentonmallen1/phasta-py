"""Time integration module for finite element method."""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import sparse
from .global_assembly import GlobalAssembly


class TimeIntegrator:
    """Base class for time integration schemes."""
    
    def __init__(self, dt: float):
        """Initialize time integrator.
        
        Args:
            dt: Time step size
        """
        self.dt = dt
        self.time = 0.0
        self.step = 0
    
    def advance(self, solution: np.ndarray, rhs: np.ndarray,
                mass_matrix: sparse.csr_matrix,
                stiffness_matrix: sparse.csr_matrix) -> np.ndarray:
        """Advance solution in time.
        
        Args:
            solution: Current solution vector
            rhs: Right-hand side vector
            mass_matrix: Mass matrix
            stiffness_matrix: Stiffness matrix
            
        Returns:
            Updated solution vector
        """
        raise NotImplementedError("Subclasses must implement advance()")


class ExplicitEuler(TimeIntegrator):
    """Explicit Euler time integration scheme."""
    
    def advance(self, solution: np.ndarray, rhs: np.ndarray,
                mass_matrix: sparse.csr_matrix,
                stiffness_matrix: sparse.csr_matrix) -> np.ndarray:
        """Advance solution using explicit Euler.
        
        Args:
            solution: Current solution vector
            rhs: Right-hand side vector
            mass_matrix: Mass matrix
            stiffness_matrix: Stiffness matrix
            
        Returns:
            Updated solution vector
        """
        # Compute right-hand side
        f = rhs - stiffness_matrix @ solution
        
        # Update solution
        solution_new = solution + self.dt * (mass_matrix @ f)
        
        # Update time and step
        self.time += self.dt
        self.step += 1
        
        return solution_new


class ImplicitEuler(TimeIntegrator):
    """Implicit Euler time integration scheme."""
    
    def advance(self, solution: np.ndarray, rhs: np.ndarray,
                mass_matrix: sparse.csr_matrix,
                stiffness_matrix: sparse.csr_matrix) -> np.ndarray:
        """Advance solution using implicit Euler.
        
        Args:
            solution: Current solution vector
            rhs: Right-hand side vector
            mass_matrix: Mass matrix
            stiffness_matrix: Stiffness matrix
            
        Returns:
            Updated solution vector
        """
        # Compute system matrix
        A = mass_matrix + self.dt * stiffness_matrix
        
        # Compute right-hand side
        b = mass_matrix @ solution + self.dt * rhs
        
        # Solve linear system
        solution_new = sparse.linalg.spsolve(A, b)
        
        # Update time and step
        self.time += self.dt
        self.step += 1
        
        return solution_new


class CrankNicolson(TimeIntegrator):
    """Crank-Nicolson time integration scheme."""
    
    def advance(self, solution: np.ndarray, rhs: np.ndarray,
                mass_matrix: sparse.csr_matrix,
                stiffness_matrix: sparse.csr_matrix) -> np.ndarray:
        """Advance solution using Crank-Nicolson.
        
        Args:
            solution: Current solution vector
            rhs: Right-hand side vector
            mass_matrix: Mass matrix
            stiffness_matrix: Stiffness matrix
            
        Returns:
            Updated solution vector
        """
        # Compute system matrix
        A = mass_matrix + 0.5 * self.dt * stiffness_matrix
        
        # Compute right-hand side
        b = (mass_matrix - 0.5 * self.dt * stiffness_matrix) @ solution + self.dt * rhs
        
        # Solve linear system
        solution_new = sparse.linalg.spsolve(A, b)
        
        # Update time and step
        self.time += self.dt
        self.step += 1
        
        return solution_new


class BDF2(TimeIntegrator):
    """Second-order backward differentiation formula."""
    
    def __init__(self, dt: float):
        """Initialize BDF2 integrator.
        
        Args:
            dt: Time step size
        """
        super().__init__(dt)
        self.solution_prev = None
    
    def advance(self, solution: np.ndarray, rhs: np.ndarray,
                mass_matrix: sparse.csr_matrix,
                stiffness_matrix: sparse.csr_matrix) -> np.ndarray:
        """Advance solution using BDF2.
        
        Args:
            solution: Current solution vector
            rhs: Right-hand side vector
            mass_matrix: Mass matrix
            stiffness_matrix: Stiffness matrix
            
        Returns:
            Updated solution vector
        """
        # First step uses implicit Euler
        if self.step == 0:
            solution_new = ImplicitEuler(self.dt).advance(
                solution, rhs, mass_matrix, stiffness_matrix)
            self.solution_prev = solution.copy()
        else:
            # Compute system matrix
            A = 1.5 * mass_matrix + self.dt * stiffness_matrix
            
            # Compute right-hand side
            b = 2.0 * mass_matrix @ solution - 0.5 * mass_matrix @ self.solution_prev + self.dt * rhs
            
            # Solve linear system
            solution_new = sparse.linalg.spsolve(A, b)
            
            # Update previous solution
            self.solution_prev = solution.copy()
        
        # Update time and step
        self.time += self.dt
        self.step += 1
        
        return solution_new


class TimeDependentProblem:
    """Class for solving time-dependent problems."""
    
    def __init__(self, integrator: TimeIntegrator,
                 mass_matrix: sparse.csr_matrix,
                 stiffness_matrix: sparse.csr_matrix,
                 initial_condition: np.ndarray,
                 rhs_function: Optional[Callable[[float], np.ndarray]] = None):
        """Initialize time-dependent problem.
        
        Args:
            integrator: Time integrator
            mass_matrix: Mass matrix
            stiffness_matrix: Stiffness matrix
            initial_condition: Initial solution
            rhs_function: Function computing right-hand side at given time
        """
        self.integrator = integrator
        self.mass_matrix = mass_matrix
        self.stiffness_matrix = stiffness_matrix
        self.solution = initial_condition.copy()
        self.rhs_function = rhs_function or (lambda t: np.zeros_like(initial_condition))
        
        # Solution history
        self.time_history = [0.0]
        self.solution_history = [initial_condition.copy()]
    
    def solve(self, end_time: float) -> Tuple[List[float], List[np.ndarray]]:
        """Solve problem up to end time.
        
        Args:
            end_time: End time for simulation
            
        Returns:
            Tuple of (time history, solution history)
        """
        while self.integrator.time < end_time:
            # Compute right-hand side
            rhs = self.rhs_function(self.integrator.time)
            
            # Advance solution
            self.solution = self.integrator.advance(
                self.solution, rhs, self.mass_matrix, self.stiffness_matrix)
            
            # Store history
            self.time_history.append(self.integrator.time)
            self.solution_history.append(self.solution.copy())
        
        return self.time_history, self.solution_history 
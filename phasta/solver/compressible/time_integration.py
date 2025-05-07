"""
Time integration schemes for compressible flow solver.

This module implements various time integration schemes for the compressible
Navier-Stokes equations, including:
- Explicit Runge-Kutta methods (RK2, RK3, RK4)
- Implicit methods (Backward Euler, Crank-Nicolson)
- Multi-stage methods (SSP-RK3, TVD-RK3)
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class TimeIntegrationConfig:
    """Configuration for time integration schemes."""
    scheme: str  # Name of the time integration scheme
    order: int   # Order of accuracy
    cfl: float   # CFL number for explicit schemes
    max_iter: int = 100  # Maximum iterations for implicit schemes
    tol: float = 1e-6    # Convergence tolerance for implicit schemes
    params: Optional[Dict[str, float]] = None  # Additional scheme parameters

class TimeIntegrator:
    """Base class for time integration schemes."""
    
    def __init__(self, config: TimeIntegrationConfig):
        """Initialize time integrator with configuration."""
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.cfl <= 0:
            raise ValueError("CFL number must be positive")
        if self.config.order < 1:
            raise ValueError("Order of accuracy must be at least 1")
    
    def compute_timestep(self, solution: np.ndarray, mesh: Dict) -> float:
        """Compute stable timestep based on CFL condition."""
        raise NotImplementedError
    
    def integrate(self, solution: np.ndarray, mesh: Dict,
                 residual: Callable[[np.ndarray, Dict], np.ndarray]) -> np.ndarray:
        """Integrate solution forward in time."""
        raise NotImplementedError

class ExplicitRK(TimeIntegrator):
    """Explicit Runge-Kutta time integration schemes."""
    
    def __init__(self, config: TimeIntegrationConfig):
        """Initialize explicit RK scheme."""
        super().__init__(config)
        self._setup_coefficients()
    
    def _setup_coefficients(self):
        """Setup Runge-Kutta coefficients based on order."""
        if self.config.order == 2:
            # Second-order RK (Heun's method)
            self.a = np.array([[0, 0],
                             [1, 0]])
            self.b = np.array([0.5, 0.5])
            self.c = np.array([0, 1])
        elif self.config.order == 3:
            # Third-order RK
            self.a = np.array([[0, 0, 0],
                             [0.5, 0, 0],
                             [-1, 2, 0]])
            self.b = np.array([1/6, 2/3, 1/6])
            self.c = np.array([0, 0.5, 1])
        elif self.config.order == 4:
            # Fourth-order RK
            self.a = np.array([[0, 0, 0, 0],
                             [0.5, 0, 0, 0],
                             [0, 0.5, 0, 0],
                             [0, 0, 1, 0]])
            self.b = np.array([1/6, 1/3, 1/3, 1/6])
            self.c = np.array([0, 0.5, 0.5, 1])
        else:
            raise ValueError(f"Unsupported order: {self.config.order}")
    
    def compute_timestep(self, solution: np.ndarray, mesh: Dict) -> float:
        """Compute stable timestep based on CFL condition."""
        # Compute maximum wave speed
        gamma = 1.4  # Ratio of specific heats
        p = (gamma - 1) * (solution[:, 4] - 0.5 * np.sum(solution[:, 1:4]**2, axis=1))
        c = np.sqrt(gamma * p / solution[:, 0])  # Speed of sound
        v = np.sqrt(np.sum(solution[:, 1:4]**2, axis=1))  # Velocity magnitude
        max_speed = np.max(v + c)
        
        # Compute minimum cell size
        dx = np.min(np.linalg.norm(mesh["nodes"][mesh["elements"][:, 1:]] - 
                                  mesh["nodes"][mesh["elements"][:, 0:1]], axis=2))
        
        return self.config.cfl * dx / max_speed
    
    def integrate(self, solution: np.ndarray, mesh: Dict,
                 residual: Callable[[np.ndarray, Dict], np.ndarray]) -> np.ndarray:
        """Integrate solution forward in time using explicit RK method."""
        dt = self.compute_timestep(solution, mesh)
        stages = len(self.b)
        k = np.zeros((stages, *solution.shape))
        
        # Compute stages
        for i in range(stages):
            stage_solution = solution.copy()
            for j in range(i):
                stage_solution += dt * self.a[i, j] * k[j]
            k[i] = residual(stage_solution, mesh)
        
        # Update solution
        for i in range(stages):
            solution += dt * self.b[i] * k[i]
        
        return solution

class SSPRK3(TimeIntegrator):
    """Strong Stability Preserving Runge-Kutta 3rd order method."""
    
    def __init__(self, config: TimeIntegrationConfig):
        """Initialize SSP-RK3 scheme."""
        super().__init__(config)
        if self.config.order != 3:
            raise ValueError("SSP-RK3 must be 3rd order")
    
    def compute_timestep(self, solution: np.ndarray, mesh: Dict) -> float:
        """Compute stable timestep based on CFL condition."""
        # Same as ExplicitRK
        gamma = 1.4
        p = (gamma - 1) * (solution[:, 4] - 0.5 * np.sum(solution[:, 1:4]**2, axis=1))
        c = np.sqrt(gamma * p / solution[:, 0])
        v = np.sqrt(np.sum(solution[:, 1:4]**2, axis=1))
        max_speed = np.max(v + c)
        dx = np.min(np.linalg.norm(mesh["nodes"][mesh["elements"][:, 1:]] - 
                                  mesh["nodes"][mesh["elements"][:, 0:1]], axis=2))
        return self.config.cfl * dx / max_speed
    
    def integrate(self, solution: np.ndarray, mesh: Dict,
                 residual: Callable[[np.ndarray, Dict], np.ndarray]) -> np.ndarray:
        """Integrate solution forward in time using SSP-RK3 method."""
        dt = self.compute_timestep(solution, mesh)
        
        # Stage 1
        k1 = residual(solution, mesh)
        u1 = solution + dt * k1
        
        # Stage 2
        k2 = residual(u1, mesh)
        u2 = 0.75 * solution + 0.25 * (u1 + dt * k2)
        
        # Stage 3
        k3 = residual(u2, mesh)
        solution = (1/3) * solution + (2/3) * (u2 + dt * k3)
        
        return solution

class TVDRK3(TimeIntegrator):
    """Total Variation Diminishing Runge-Kutta 3rd order method."""
    
    def __init__(self, config: TimeIntegrationConfig):
        """Initialize TVD-RK3 scheme."""
        super().__init__(config)
        if self.config.order != 3:
            raise ValueError("TVD-RK3 must be 3rd order")
    
    def compute_timestep(self, solution: np.ndarray, mesh: Dict) -> float:
        """Compute stable timestep based on CFL condition."""
        # Same as ExplicitRK
        gamma = 1.4
        p = (gamma - 1) * (solution[:, 4] - 0.5 * np.sum(solution[:, 1:4]**2, axis=1))
        c = np.sqrt(gamma * p / solution[:, 0])
        v = np.sqrt(np.sum(solution[:, 1:4]**2, axis=1))
        max_speed = np.max(v + c)
        dx = np.min(np.linalg.norm(mesh["nodes"][mesh["elements"][:, 1:]] - 
                                  mesh["nodes"][mesh["elements"][:, 0:1]], axis=2))
        return self.config.cfl * dx / max_speed
    
    def integrate(self, solution: np.ndarray, mesh: Dict,
                 residual: Callable[[np.ndarray, Dict], np.ndarray]) -> np.ndarray:
        """Integrate solution forward in time using TVD-RK3 method."""
        dt = self.compute_timestep(solution, mesh)
        
        # Stage 1
        k1 = residual(solution, mesh)
        u1 = solution + dt * k1
        
        # Stage 2
        k2 = residual(u1, mesh)
        u2 = 0.5 * (solution + u1 + dt * k2)
        
        # Stage 3
        k3 = residual(u2, mesh)
        solution = (1/3) * solution + (2/3) * (u2 + dt * k3)
        
        return solution

class ImplicitTimeIntegrator(TimeIntegrator):
    """Base class for implicit time integration schemes."""
    
    def __init__(self, config: TimeIntegrationConfig):
        """Initialize implicit time integrator."""
        super().__init__(config)
        self._setup_coefficients()
    
    def _setup_coefficients(self):
        """Setup time integration coefficients."""
        if self.config.scheme.lower() == "backward_euler":
            self.theta = 1.0  # Fully implicit
        elif self.config.scheme.lower() == "crank_nicolson":
            self.theta = 0.5  # Second-order accurate
        else:
            raise ValueError(f"Unknown implicit scheme: {self.config.scheme}")
    
    def compute_timestep(self, solution: np.ndarray, mesh: Dict) -> float:
        """Compute stable timestep based on CFL condition."""
        # For implicit methods, we can use larger timesteps
        # but still need to ensure stability
        gamma = 1.4
        p = (gamma - 1) * (solution[:, 4] - 0.5 * np.sum(solution[:, 1:4]**2, axis=1))
        c = np.sqrt(gamma * p / solution[:, 0])
        v = np.sqrt(np.sum(solution[:, 1:4]**2, axis=1))
        max_speed = np.max(v + c)
        dx = np.min(np.linalg.norm(mesh["nodes"][mesh["elements"][:, 1:]] - 
                                  mesh["nodes"][mesh["elements"][:, 0:1]], axis=2))
        return self.config.cfl * dx / max_speed
    
    def integrate(self, solution: np.ndarray, mesh: Dict,
                 residual: Callable[[np.ndarray, Dict], np.ndarray],
                 jacobian: Callable[[np.ndarray, Dict], np.ndarray],
                 linear_solver: Any) -> np.ndarray:
        """
        Integrate solution forward in time using implicit method.
        
        Args:
            solution: Current solution vector
            mesh: Mesh data
            residual: Function to compute residuals
            jacobian: Function to compute Jacobian matrix
            linear_solver: Linear solver instance
            
        Returns:
            np.ndarray: Updated solution vector
        """
        dt = self.compute_timestep(solution, mesh)
        solution_new = solution.copy()
        
        # Nonlinear iteration
        for iter in range(self.config.max_iter):
            # Compute residual
            R = residual(solution_new, mesh)
            
            # Compute Jacobian
            J = jacobian(solution_new, mesh)
            
            # Add time derivative contribution
            J += np.eye(J.shape[0]) / dt
            
            # Solve linear system
            delta = linear_solver.solve(J, R)
            
            # Update solution
            solution_new -= delta
            
            # Check convergence
            if np.linalg.norm(delta) < self.config.tol:
                break
        
        return solution_new

class DualTimeStepper(TimeIntegrator):
    """Dual time-stepping for steady-state problems."""
    
    def __init__(self, config: TimeIntegrationConfig):
        """Initialize dual time-stepper."""
        super().__init__(config)
        self.pseudo_time_integrator = ExplicitRK(config)
    
    def integrate(self, solution: np.ndarray, mesh: Dict,
                 residual: Callable[[np.ndarray, Dict], np.ndarray],
                 jacobian: Callable[[np.ndarray, Dict], np.ndarray],
                 linear_solver: Any) -> np.ndarray:
        """
        Integrate solution using dual time-stepping.
        
        Args:
            solution: Current solution vector
            mesh: Mesh data
            residual: Function to compute residuals
            jacobian: Function to compute Jacobian matrix
            linear_solver: Linear solver instance
            
        Returns:
            np.ndarray: Updated solution vector
        """
        dt = self.compute_timestep(solution, mesh)
        solution_new = solution.copy()
        
        # Pseudo-time iteration
        for iter in range(self.config.max_iter):
            # Compute physical time residual
            R_phys = residual(solution_new, mesh)
            
            # Compute pseudo-time residual
            R_pseudo = R_phys + (solution_new - solution) / dt
            
            # Update solution using explicit method
            solution_new = self.pseudo_time_integrator.integrate(
                solution_new, mesh, lambda x, m: R_pseudo
            )
            
            # Check convergence
            if np.linalg.norm(R_pseudo) < self.config.tol:
                break
        
        return solution_new

class LocalTimeStepper(TimeIntegrator):
    """Local time-stepping for multi-scale problems."""
    
    def __init__(self, config: TimeIntegrationConfig):
        """Initialize local time-stepper."""
        super().__init__(config)
        self.base_integrator = create_time_integrator(config)
    
    def compute_local_timesteps(self, solution: np.ndarray, mesh: Dict) -> np.ndarray:
        """Compute local timesteps for each element."""
        gamma = 1.4
        p = (gamma - 1) * (solution[:, 4] - 0.5 * np.sum(solution[:, 1:4]**2, axis=1))
        c = np.sqrt(gamma * p / solution[:, 0])
        v = np.sqrt(np.sum(solution[:, 1:4]**2, axis=1))
        max_speeds = v + c
        
        # Compute element sizes
        elements = mesh["elements"]
        nodes = mesh["nodes"]
        element_sizes = np.zeros(len(elements))
        for i, element in enumerate(elements):
            element_nodes = nodes[element]
            min_size = float('inf')
            for j in range(len(element)):
                for k in range(j + 1, len(element)):
                    edge = element_nodes[j] - element_nodes[k]
                    size = np.linalg.norm(edge)
                    min_size = min(min_size, size)
            element_sizes[i] = min_size
        
        # Compute local timesteps
        local_dt = self.config.cfl * element_sizes / max_speeds
        return local_dt
    
    def integrate(self, solution: np.ndarray, mesh: Dict,
                 residual: Callable[[np.ndarray, Dict], np.ndarray]) -> np.ndarray:
        """
        Integrate solution using local time-stepping.
        
        Args:
            solution: Current solution vector
            mesh: Mesh data
            residual: Function to compute residuals
            
        Returns:
            np.ndarray: Updated solution vector
        """
        # Compute local timesteps
        local_dt = self.compute_local_timesteps(solution, mesh)
        
        # Update each element with its local timestep
        elements = mesh["elements"]
        solution_new = solution.copy()
        
        for i, element in enumerate(elements):
            element_solution = solution[element]
            element_dt = local_dt[i]
            
            # Update element solution
            element_solution_new = self.base_integrator.integrate(
                element_solution, mesh, residual
            )
            
            solution_new[element] = element_solution_new
        
        return solution_new

def create_time_integrator(config: TimeIntegrationConfig) -> TimeIntegrator:
    """Factory function to create time integrator based on configuration."""
    if config.scheme.lower() == "explicit_rk":
        return ExplicitRK(config)
    elif config.scheme.lower() == "ssp_rk3":
        return SSPRK3(config)
    elif config.scheme.lower() == "tvd_rk3":
        return TVDRK3(config)
    elif config.scheme.lower() in ["backward_euler", "crank_nicolson"]:
        return ImplicitTimeIntegrator(config)
    elif config.scheme.lower() == "dual_time":
        return DualTimeStepper(config)
    elif config.scheme.lower() == "local_time":
        return LocalTimeStepper(config)
    else:
        raise ValueError(f"Unknown time integration scheme: {config.scheme}") 
"""
Main solver module for compressible flow.

This module implements the main solver class that orchestrates the compressible flow simulation.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from .shape_functions import ShapeFunctions, ShapeFunctionConfig, ElementType
from .amg_solver import AMGSolver, AMGConfig
from .io import IOHandler, IOConfig

@dataclass
class SolverConfig:
    """Configuration for the compressible flow solver."""
    # Time stepping
    dt: float = 0.001  # Time step size
    max_steps: int = 1000  # Maximum number of time steps
    cfl: float = 0.5  # CFL number
    
    # Physics
    gamma: float = 1.4  # Ratio of specific heats
    mu: float = 1.0e-5  # Dynamic viscosity
    k: float = 1.0e-5  # Thermal conductivity
    pr: float = 0.72  # Prandtl number
    
    # Solver settings
    max_iter: int = 100  # Maximum number of nonlinear iterations
    tolerance: float = 1e-6  # Convergence tolerance
    
    # Output settings
    output_dir: str = "output"  # Output directory
    save_frequency: int = 100  # Save frequency in time steps
    
    # Element settings
    element_type: ElementType = ElementType.HEX  # Element type
    order: int = 1  # Polynomial order
    
    # AMG settings
    amg_max_levels: int = 10  # Maximum number of multigrid levels
    amg_max_coarse_size: int = 100  # Maximum size of coarsest level
    amg_strength_threshold: float = 0.25  # Threshold for strong connections
    amg_smoother: str = "jacobi"  # Smoother type

class CompressibleSolver:
    """
    Implements the compressible flow solver.
    
    This class orchestrates the compressible flow simulation, including:
    - Time stepping
    - Spatial discretization
    - Linear system solution
    - I/O operations
    """
    
    def __init__(self, config: SolverConfig):
        """
        Initialize the solver.
        
        Args:
            config: Solver configuration parameters
        """
        self.config = config
        
        # Initialize components
        self._init_components()
        
        # Initialize solution
        self.solution = None
        self.mesh = None
        self.time = 0.0
        self.time_step = 0
        
    def _init_components(self):
        """Initialize solver components."""
        # Shape functions
        shape_config = ShapeFunctionConfig(
            element_type=self.config.element_type,
            order=self.config.order
        )
        self.shape_functions = ShapeFunctions(shape_config)
        
        # AMG solver
        amg_config = AMGConfig(
            max_levels=self.config.amg_max_levels,
            max_coarse_size=self.config.amg_max_coarse_size,
            strength_threshold=self.config.amg_strength_threshold,
            smoother=self.config.amg_smoother
        )
        self.amg_solver = AMGSolver(amg_config)
        
        # I/O handler
        io_config = IOConfig(
            output_dir=self.config.output_dir,
            save_frequency=self.config.save_frequency
        )
        self.io_handler = IOHandler(io_config)
        
    def set_mesh(self, mesh: Dict[str, np.ndarray]):
        """
        Set the computational mesh.
        
        Args:
            mesh: Mesh data dictionary containing:
                - nodes: Node coordinates
                - elements: Element connectivity
                - element_type: Type of elements
                - boundary_conditions: Boundary condition data
        """
        self.mesh = mesh
        
        # Initialize solution
        n_nodes = len(mesh['nodes'])
        self.solution = np.zeros((n_nodes, 5))  # [rho, rho*u, rho*v, rho*w, rho*E]
        
    def set_initial_conditions(self, initial_conditions: Dict[str, np.ndarray]):
        """
        Set initial conditions.
        
        Args:
            initial_conditions: Dictionary containing initial values for:
                - density: Density field
                - velocity: Velocity field (3 components)
                - pressure: Pressure field
        """
        if self.mesh is None:
            raise ValueError("Mesh must be set before initial conditions")
            
        n_nodes = len(self.mesh['nodes'])
        
        # Set density
        self.solution[:, 0] = initial_conditions['density']
        
        # Set momentum
        self.solution[:, 1:4] = (
            initial_conditions['density'][:, np.newaxis] *
            initial_conditions['velocity']
        )
        
        # Set total energy
        v2 = np.sum(initial_conditions['velocity']**2, axis=1)
        self.solution[:, 4] = (
            initial_conditions['pressure'] / (self.config.gamma - 1.0) +
            0.5 * initial_conditions['density'] * v2
        )
        
    def run(self):
        """Run the simulation."""
        if self.mesh is None or self.solution is None:
            raise ValueError("Mesh and initial conditions must be set before running")
            
        # Main time stepping loop
        for step in range(self.config.max_steps):
            # Compute time step
            dt = self._compute_time_step()
            
            # Update solution
            self._time_step(dt)
            
            # Update time
            self.time += dt
            self.time_step += 1
            
            # Save solution
            self.io_handler.save_solution(
                self.solution,
                self.mesh,
                self.time_step,
                self.time
            )
            
            # Check for completion
            if self._check_completion():
                break
                
    def _compute_time_step(self) -> float:
        """
        Compute time step size based on CFL condition.
        
        Returns:
            float: Time step size
        """
        # Get solution variables
        rho = self.solution[:, 0]
        u = self.solution[:, 1] / rho
        v = self.solution[:, 2] / rho
        w = self.solution[:, 3] / rho
        p = self._compute_pressure()
        
        # Compute sound speed
        c = np.sqrt(self.config.gamma * p / rho)
        
        # Compute maximum wave speed
        max_speed = np.max(np.abs(u) + c)
        
        # Compute minimum element size
        min_size = self._compute_min_element_size()
        
        # Compute time step
        dt = self.config.cfl * min_size / max_speed
        
        return min(dt, self.config.dt)
        
    def _compute_min_element_size(self) -> float:
        """
        Compute minimum element size.
        
        Returns:
            float: Minimum element size
        """
        # Get element nodes
        elements = self.mesh['elements']
        nodes = self.mesh['nodes']
        
        min_size = float('inf')
        for element in elements:
            element_nodes = nodes[element]
            
            # Compute element size (e.g., minimum edge length)
            for i in range(len(element)):
                for j in range(i + 1, len(element)):
                    edge = element_nodes[i] - element_nodes[j]
                    size = np.linalg.norm(edge)
                    min_size = min(min_size, size)
                    
        return min_size
        
    def _time_step(self, dt: float):
        """
        Perform a single time step.
        
        Args:
            dt: Time step size
        """
        # Nonlinear iteration
        for iter in range(self.config.max_iter):
            # Compute residual
            residual = self._compute_residual()
            
            # Check convergence
            if np.linalg.norm(residual) < self.config.tolerance:
                break
                
            # Compute Jacobian
            jacobian = self._compute_jacobian()
            
            # Solve linear system
            delta = self.amg_solver.solve(jacobian, residual)
            
            # Update solution
            self.solution += delta
            
    def _compute_residual(self) -> np.ndarray:
        """
        Compute residual vector.
        
        Returns:
            np.ndarray: Residual vector
        """
        # Initialize residual
        residual = np.zeros_like(self.solution)
        
        # Loop over elements
        elements = self.mesh['elements']
        for element in elements:
            # Get element solution
            element_solution = self.solution[element]
            
            # Get element nodes
            element_nodes = self.mesh['nodes'][element]
            
            # Compute element residual
            element_residual = self._compute_element_residual(
                element_solution,
                element_nodes
            )
            
            # Assemble into global residual
            residual[element] += element_residual
            
        return residual
        
    def _compute_element_residual(self,
                                element_solution: np.ndarray,
                                element_nodes: np.ndarray
                                ) -> np.ndarray:
        """
        Compute element residual.
        
        Args:
            element_solution: Solution at element nodes
            element_nodes: Element node coordinates
            
        Returns:
            np.ndarray: Element residual
        """
        # Initialize element residual
        n_nodes = len(element_nodes)
        element_residual = np.zeros((n_nodes, 5))
        
        # Loop over quadrature points
        for q in range(len(self.shape_functions.quad_points)):
            # Get quadrature point
            xi = self.shape_functions.quad_points[q]
            w = self.shape_functions.quad_weights[q]
            
            # Evaluate shape functions
            N, dN = self.shape_functions.evaluate(xi)
            
            # Compute Jacobian
            J = dN.dot(element_nodes)
            detJ = np.linalg.det(J)
            
            # Compute solution at quadrature point
            q_solution = N.dot(element_solution)
            
            # Compute fluxes
            inviscid_flux = self._compute_inviscid_flux(q_solution)
            viscous_flux = self._compute_viscous_flux(q_solution, dN, J)
            
            # Compute residual contribution
            for i in range(n_nodes):
                element_residual[i] += (
                    w * detJ * (
                        inviscid_flux.dot(dN[i]) +
                        viscous_flux.dot(dN[i])
                    )
                )
                
        return element_residual
        
    def _compute_inviscid_flux(self, q: np.ndarray) -> np.ndarray:
        """
        Compute inviscid flux.
        
        Args:
            q: Solution vector [rho, rho*u, rho*v, rho*w, rho*E]
            
        Returns:
            np.ndarray: Inviscid flux
        """
        # Extract variables
        rho = q[0]
        u = q[1] / rho
        v = q[2] / rho
        w = q[3] / rho
        E = q[4] / rho
        
        # Compute pressure
        p = (self.config.gamma - 1.0) * rho * (E - 0.5 * (u**2 + v**2 + w**2))
        
        # Compute flux
        flux = np.zeros((3, 5))
        
        # x-direction
        flux[0, 0] = rho * u
        flux[0, 1] = rho * u**2 + p
        flux[0, 2] = rho * u * v
        flux[0, 3] = rho * u * w
        flux[0, 4] = rho * u * E + p * u
        
        # y-direction
        flux[1, 0] = rho * v
        flux[1, 1] = rho * u * v
        flux[1, 2] = rho * v**2 + p
        flux[1, 3] = rho * v * w
        flux[1, 4] = rho * v * E + p * v
        
        # z-direction
        flux[2, 0] = rho * w
        flux[2, 1] = rho * u * w
        flux[2, 2] = rho * v * w
        flux[2, 3] = rho * w**2 + p
        flux[2, 4] = rho * w * E + p * w
        
        return flux
        
    def _compute_viscous_flux(self,
                            q: np.ndarray,
                            dN: np.ndarray,
                            J: np.ndarray
                            ) -> np.ndarray:
        """
        Compute viscous flux.
        
        Args:
            q: Solution vector
            dN: Shape function derivatives
            J: Jacobian matrix
            
        Returns:
            np.ndarray: Viscous flux
        """
        # Extract variables
        rho = q[0]
        u = q[1] / rho
        v = q[2] / rho
        w = q[3] / rho
        E = q[4] / rho
        
        # Compute temperature
        p = (self.config.gamma - 1.0) * rho * (E - 0.5 * (u**2 + v**2 + w**2))
        T = p / (rho * self.config.gamma)
        
        # Compute velocity gradients
        dudx = dN.dot(u)
        dudy = dN.dot(v)
        dudz = dN.dot(w)
        
        # Compute stress tensor
        tau = np.zeros((3, 3))
        tau[0, 0] = 2.0 * self.config.mu * dudx[0]
        tau[0, 1] = self.config.mu * (dudx[1] + dudy[0])
        tau[0, 2] = self.config.mu * (dudx[2] + dudz[0])
        tau[1, 0] = tau[0, 1]
        tau[1, 1] = 2.0 * self.config.mu * dudy[1]
        tau[1, 2] = self.config.mu * (dudy[2] + dudz[1])
        tau[2, 0] = tau[0, 2]
        tau[2, 1] = tau[1, 2]
        tau[2, 2] = 2.0 * self.config.mu * dudz[2]
        
        # Compute heat flux
        dTdx = dN.dot(T)
        qx = -self.config.k * dTdx[0]
        qy = -self.config.k * dTdx[1]
        qz = -self.config.k * dTdx[2]
        
        # Compute flux
        flux = np.zeros((3, 5))
        
        # x-direction
        flux[0, 1] = tau[0, 0]
        flux[0, 2] = tau[0, 1]
        flux[0, 3] = tau[0, 2]
        flux[0, 4] = (
            tau[0, 0] * u + tau[0, 1] * v + tau[0, 2] * w + qx
        )
        
        # y-direction
        flux[1, 1] = tau[1, 0]
        flux[1, 2] = tau[1, 1]
        flux[1, 3] = tau[1, 2]
        flux[1, 4] = (
            tau[1, 0] * u + tau[1, 1] * v + tau[1, 2] * w + qy
        )
        
        # z-direction
        flux[2, 1] = tau[2, 0]
        flux[2, 2] = tau[2, 1]
        flux[2, 3] = tau[2, 2]
        flux[2, 4] = (
            tau[2, 0] * u + tau[2, 1] * v + tau[2, 2] * w + qz
        )
        
        return flux
        
    def _compute_jacobian(self) -> np.ndarray:
        """
        Compute Jacobian matrix.
        
        Returns:
            np.ndarray: Jacobian matrix
        """
        # This is a simplified implementation
        # In practice, this would be computed using automatic differentiation
        # or numerical differentiation
        
        # For now, return identity matrix
        n_dof = self.solution.size
        return np.eye(n_dof)
        
    def _compute_pressure(self) -> np.ndarray:
        """
        Compute pressure field.
        
        Returns:
            np.ndarray: Pressure field
        """
        rho = self.solution[:, 0]
        u = self.solution[:, 1] / rho
        v = self.solution[:, 2] / rho
        w = self.solution[:, 3] / rho
        E = self.solution[:, 4] / rho
        
        return (self.config.gamma - 1.0) * rho * (
            E - 0.5 * (u**2 + v**2 + w**2)
        )
        
    def _check_completion(self) -> bool:
        """
        Check if simulation should be completed.
        
        Returns:
            bool: True if simulation should be completed
        """
        return self.time_step >= self.config.max_steps 
"""
Boundary conditions for compressible flow.

This module implements various boundary conditions for the compressible flow solver,
including wall, inlet, outlet, periodic, and symmetry conditions.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class BoundaryConfig:
    """Configuration for boundary conditions."""
    gamma: float = 1.4  # Ratio of specific heats
    R: float = 287.0    # Gas constant
    wall_temperature: Optional[float] = None  # Wall temperature for isothermal walls
    wall_heat_flux: Optional[float] = None   # Wall heat flux for specified heat flux
    slip_condition: bool = False  # Whether to use slip condition at walls
    periodic_offset: Optional[np.ndarray] = None  # Offset for periodic boundaries
    periodic_rotation: Optional[np.ndarray] = None  # Rotation matrix for periodic boundaries

class BoundaryCondition:
    """Base class for all boundary conditions."""
    
    def __init__(self, name: str, config: BoundaryConfig):
        """Initialize boundary condition.
        
        Args:
            name: Name of the boundary condition
            config: Boundary condition configuration
        """
        self.name = name
        self.config = config
        
    def apply(self, solution: np.ndarray, mesh: Dict[str, Any], 
              bc_nodes: np.ndarray) -> np.ndarray:
        """Apply boundary condition to solution.
        
        Args:
            solution: Current solution array
            mesh: Mesh data
            bc_nodes: Array of boundary node indices
            
        Returns:
            Modified solution array
        """
        raise NotImplementedError("Subclasses must implement apply()")
        
    def _get_normals(self, mesh: Dict[str, Any], bc_nodes: np.ndarray) -> np.ndarray:
        """Get normal vectors at boundary nodes.
        
        Args:
            mesh: Mesh data
            bc_nodes: Array of boundary node indices
            
        Returns:
            Normal vectors at boundary nodes
        """
        # Implementation depends on mesh data structure
        # This is a placeholder
        return np.zeros((len(bc_nodes), 3))
        
    def _compute_pressure(self, state: np.ndarray) -> np.ndarray:
        """Compute pressure from conservative variables.
        
        Args:
            state: Conservative state vector
            
        Returns:
            Pressure array
        """
        rho = state[:, 0]
        u = state[:, 1:4] / rho[:, np.newaxis]
        e = state[:, 4] / rho
        p = (self.config.gamma - 1.0) * rho * (e - 0.5 * np.sum(u * u, axis=1))
        return p

class WallBoundary(BoundaryCondition):
    """Wall boundary condition with no-slip/slip and temperature options."""
    
    def __init__(self, config: BoundaryConfig):
        """Initialize wall boundary condition.
        
        Args:
            config: Boundary condition configuration
        """
        super().__init__("wall", config)
        
    def apply(self, solution: np.ndarray, mesh: Dict[str, Any], 
              bc_nodes: np.ndarray) -> np.ndarray:
        """Apply wall condition.
        
        For compressible flow, this sets:
        - Velocity to zero (no-slip) or normal component to zero (slip)
        - Temperature to specified value (isothermal) or extrapolated (adiabatic)
        - Pressure extrapolated from interior
        """
        # Extract solution components
        rho = solution[bc_nodes, 0]
        u = solution[bc_nodes, 1:4] / rho[:, np.newaxis]
        p = self._compute_pressure(solution[bc_nodes])
        
        # Get normal vectors
        normals = self._get_normals(mesh, bc_nodes)
        
        if self.config.slip_condition:
            # Slip condition - set normal velocity to zero
            u_normal = np.sum(u * normals, axis=1)
            u = u - u_normal[:, np.newaxis] * normals
        else:
            # No-slip condition - set all velocity components to zero
            u.fill(0.0)
        
        # Update momentum
        solution[bc_nodes, 1:4] = rho[:, np.newaxis] * u
        
        # Handle temperature
        if self.config.wall_temperature is not None:
            # Isothermal wall
            T = self.config.wall_temperature
            p_new = p  # Keep pressure from extrapolation
            rho_new = p_new / (self.config.R * T)
            solution[bc_nodes, 0] = rho_new
        elif self.config.wall_heat_flux is not None:
            # Specified heat flux
            # Implementation depends on heat flux model
            pass
        else:
            # Adiabatic wall - extrapolate temperature
            pass
            
        return solution

class InletBoundary(BoundaryCondition):
    """Supersonic/subsonic inlet boundary condition."""
    
    def __init__(self, config: BoundaryConfig, mach: float, pressure: float, temperature: float):
        """Initialize inlet boundary condition.
        
        Args:
            config: Boundary condition configuration
            mach: Inlet Mach number
            pressure: Inlet pressure
            temperature: Inlet temperature
        """
        super().__init__("inlet", config)
        self.mach = mach
        self.pressure = pressure
        self.temperature = temperature
        
    def apply(self, solution: np.ndarray, mesh: Dict[str, Any], 
              bc_nodes: np.ndarray) -> np.ndarray:
        """Apply inlet condition.
        
        For supersonic flow, all variables are specified.
        For subsonic flow, one characteristic is extrapolated.
        """
        # Compute speed of sound
        a = np.sqrt(self.config.gamma * self.config.R * self.temperature)
        
        # Compute velocity magnitude
        u_mag = self.mach * a
        
        # Get normal vectors at boundary nodes
        normals = self._get_normals(mesh, bc_nodes)
        
        # Set velocity in normal direction
        u = u_mag * normals
        
        # Set density from equation of state
        rho = self.pressure / (self.config.R * self.temperature)
        
        # Set total energy
        e = self.pressure / ((self.config.gamma - 1.0) * rho) + 0.5 * np.sum(u * u, axis=1)
        
        # Update solution
        solution[bc_nodes, 0] = rho
        solution[bc_nodes, 1:4] = rho[:, np.newaxis] * u
        solution[bc_nodes, 4] = rho * e
        
        return solution

class OutletBoundary(BoundaryCondition):
    """Supersonic/subsonic outlet boundary condition."""
    
    def __init__(self, config: BoundaryConfig, pressure: Optional[float] = None):
        """Initialize outlet boundary condition.
        
        Args:
            config: Boundary condition configuration
            pressure: Back pressure (None for supersonic outlet)
        """
        super().__init__("outlet", config)
        self.pressure = pressure
        
    def apply(self, solution: np.ndarray, mesh: Dict[str, Any], 
              bc_nodes: np.ndarray) -> np.ndarray:
        """Apply outlet condition.
        
        For supersonic flow, all variables are extrapolated.
        For subsonic flow, pressure is specified and other variables extrapolated.
        """
        if self.pressure is None:
            # Supersonic outlet - extrapolate all variables
            return solution
        else:
            # Subsonic outlet - specify pressure, extrapolate others
            # Get current state
            rho = solution[bc_nodes, 0]
            u = solution[bc_nodes, 1:4] / rho[:, np.newaxis]
            e = solution[bc_nodes, 4] / rho
            
            # Update pressure
            p = self.pressure
            
            # Update density from equation of state
            T = p / (self.config.R * rho)
            rho_new = p / (self.config.R * T)
            
            # Update solution
            solution[bc_nodes, 0] = rho_new
            solution[bc_nodes, 1:4] = rho_new[:, np.newaxis] * u
            solution[bc_nodes, 4] = rho_new * e
            
            return solution

class SymmetryBoundary(BoundaryCondition):
    """Symmetry boundary condition."""
    
    def __init__(self, config: BoundaryConfig):
        """Initialize symmetry boundary condition.
        
        Args:
            config: Boundary condition configuration
        """
        super().__init__("symmetry", config)
        
    def apply(self, solution: np.ndarray, mesh: Dict[str, Any], 
              bc_nodes: np.ndarray) -> np.ndarray:
        """Apply symmetry condition.
        
        This sets:
        - Normal velocity to zero
        - Normal gradients of other variables to zero
        """
        # Get normal vectors
        normals = self._get_normals(mesh, bc_nodes)
        
        # Get current state
        rho = solution[bc_nodes, 0]
        u = solution[bc_nodes, 1:4] / rho[:, np.newaxis]
        
        # Project velocity onto normal
        u_normal = np.sum(u * normals, axis=1)
        
        # Subtract normal component
        u = u - u_normal[:, np.newaxis] * normals
        
        # Update solution
        solution[bc_nodes, 1:4] = rho[:, np.newaxis] * u
        
        return solution

class PeriodicBoundary(BoundaryCondition):
    """Periodic boundary condition with translation and rotation support."""
    
    def __init__(self, config: BoundaryConfig, 
                 source_nodes: np.ndarray,
                 target_nodes: np.ndarray):
        """Initialize periodic boundary condition.
        
        Args:
            config: Boundary condition configuration
            source_nodes: Source boundary node indices
            target_nodes: Target boundary node indices
        """
        super().__init__("periodic", config)
        self.source_nodes = source_nodes
        self.target_nodes = target_nodes
        
    def apply(self, solution: np.ndarray, mesh: Dict[str, Any], 
              bc_nodes: np.ndarray) -> np.ndarray:
        """Apply periodic condition by copying solution from corresponding nodes.
        
        This handles both translation and rotation periodicity.
        """
        # Get node coordinates
        source_coords = mesh['nodes'][self.source_nodes]
        target_coords = mesh['nodes'][self.target_nodes]
        
        if self.config.periodic_rotation is not None:
            # Rotation periodicity
            R = self.config.periodic_rotation
            # Rotate source coordinates
            source_coords = np.dot(source_coords, R.T)
            
        if self.config.periodic_offset is not None:
            # Translation periodicity
            offset = self.config.periodic_offset
            # Translate source coordinates
            source_coords = source_coords + offset
            
        # Find corresponding nodes
        for i, target_coord in enumerate(target_coords):
            # Find closest source node
            distances = np.linalg.norm(source_coords - target_coord, axis=1)
            closest = np.argmin(distances)
            
            # Copy solution
            solution[self.target_nodes[i]] = solution[self.source_nodes[closest]]
            
        return solution 
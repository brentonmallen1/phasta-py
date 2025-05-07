"""
Unit tests for boundary conditions.

This module contains tests for all boundary condition types in the compressible flow solver.
"""

import numpy as np
import pytest
from ..boundary_conditions import (
    BoundaryConfig,
    WallBoundary,
    InletBoundary,
    OutletBoundary,
    SymmetryBoundary,
    PeriodicBoundary
)

@pytest.fixture
def config():
    """Create a default boundary configuration."""
    return BoundaryConfig()

@pytest.fixture
def mesh():
    """Create a simple test mesh."""
    return {
        'nodes': np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ]),
        'elements': np.array([[0, 1, 2, 3]]),
        'boundary_faces': {
            'wall': np.array([0, 1]),
            'inlet': np.array([2]),
            'outlet': np.array([3]),
            'symmetry': np.array([0]),
            'periodic': {
                'source': np.array([0]),
                'target': np.array([1])
            }
        }
    }

@pytest.fixture
def solution():
    """Create a test solution array."""
    n_nodes = 4
    solution = np.zeros((n_nodes, 5))
    # Set some initial values
    solution[:, 0] = 1.0  # density
    solution[:, 1:4] = 1.0  # velocity
    solution[:, 4] = 2.0  # energy
    return solution

def test_wall_boundary_no_slip(config, mesh, solution):
    """Test no-slip wall boundary condition."""
    wall = WallBoundary(config)
    bc_nodes = mesh['boundary_faces']['wall']
    
    # Apply boundary condition
    result = wall.apply(solution.copy(), mesh, bc_nodes)
    
    # Check velocity is zero
    assert np.allclose(result[bc_nodes, 1:4], 0.0)
    
    # Check density and energy are preserved
    assert np.allclose(result[bc_nodes, 0], solution[bc_nodes, 0])
    assert np.allclose(result[bc_nodes, 4], solution[bc_nodes, 4])

def test_wall_boundary_slip(config, mesh, solution):
    """Test slip wall boundary condition."""
    config.slip_condition = True
    wall = WallBoundary(config)
    bc_nodes = mesh['boundary_faces']['wall']
    
    # Apply boundary condition
    result = wall.apply(solution.copy(), mesh, bc_nodes)
    
    # Check normal velocity is zero
    normals = wall._get_normals(mesh, bc_nodes)
    u = result[bc_nodes, 1:4] / result[bc_nodes, 0:1]
    u_normal = np.sum(u * normals, axis=1)
    assert np.allclose(u_normal, 0.0)

def test_wall_boundary_isothermal(config, mesh, solution):
    """Test isothermal wall boundary condition."""
    config.wall_temperature = 300.0
    wall = WallBoundary(config)
    bc_nodes = mesh['boundary_faces']['wall']
    
    # Apply boundary condition
    result = wall.apply(solution.copy(), mesh, bc_nodes)
    
    # Check temperature is correct
    p = wall._compute_pressure(result[bc_nodes])
    T = p / (config.R * result[bc_nodes, 0])
    assert np.allclose(T, config.wall_temperature)

def test_inlet_boundary(config, mesh, solution):
    """Test inlet boundary condition."""
    inlet = InletBoundary(config, mach=0.5, pressure=101325.0, temperature=300.0)
    bc_nodes = mesh['boundary_faces']['inlet']
    
    # Apply boundary condition
    result = inlet.apply(solution.copy(), mesh, bc_nodes)
    
    # Check density
    expected_rho = inlet.pressure / (config.R * inlet.temperature)
    assert np.allclose(result[bc_nodes, 0], expected_rho)
    
    # Check velocity magnitude
    u = result[bc_nodes, 1:4] / result[bc_nodes, 0:1]
    u_mag = np.sqrt(np.sum(u * u, axis=1))
    expected_u = inlet.mach * np.sqrt(config.gamma * config.R * inlet.temperature)
    assert np.allclose(u_mag, expected_u)

def test_outlet_boundary_supersonic(config, mesh, solution):
    """Test supersonic outlet boundary condition."""
    outlet = OutletBoundary(config)
    bc_nodes = mesh['boundary_faces']['outlet']
    
    # Apply boundary condition
    result = outlet.apply(solution.copy(), mesh, bc_nodes)
    
    # Check solution is unchanged
    assert np.allclose(result[bc_nodes], solution[bc_nodes])

def test_outlet_boundary_subsonic(config, mesh, solution):
    """Test subsonic outlet boundary condition."""
    outlet = OutletBoundary(config, pressure=101325.0)
    bc_nodes = mesh['boundary_faces']['outlet']
    
    # Apply boundary condition
    result = outlet.apply(solution.copy(), mesh, bc_nodes)
    
    # Check pressure
    p = outlet._compute_pressure(result[bc_nodes])
    assert np.allclose(p, outlet.pressure)
    
    # Check velocity is preserved
    u_orig = solution[bc_nodes, 1:4] / solution[bc_nodes, 0:1]
    u_new = result[bc_nodes, 1:4] / result[bc_nodes, 0:1]
    assert np.allclose(u_new, u_orig)

def test_symmetry_boundary(config, mesh, solution):
    """Test symmetry boundary condition."""
    symmetry = SymmetryBoundary(config)
    bc_nodes = mesh['boundary_faces']['symmetry']
    
    # Apply boundary condition
    result = symmetry.apply(solution.copy(), mesh, bc_nodes)
    
    # Check normal velocity is zero
    normals = symmetry._get_normals(mesh, bc_nodes)
    u = result[bc_nodes, 1:4] / result[bc_nodes, 0:1]
    u_normal = np.sum(u * normals, axis=1)
    assert np.allclose(u_normal, 0.0)

def test_periodic_boundary(config, mesh, solution):
    """Test periodic boundary condition."""
    source_nodes = mesh['boundary_faces']['periodic']['source']
    target_nodes = mesh['boundary_faces']['periodic']['target']
    periodic = PeriodicBoundary(config, source_nodes, target_nodes)
    
    # Apply boundary condition
    result = periodic.apply(solution.copy(), mesh, target_nodes)
    
    # Check solution is copied correctly
    assert np.allclose(result[target_nodes], solution[source_nodes])

def test_periodic_boundary_with_rotation(config, mesh, solution):
    """Test periodic boundary condition with rotation."""
    source_nodes = mesh['boundary_faces']['periodic']['source']
    target_nodes = mesh['boundary_faces']['periodic']['target']
    
    # Set up rotation matrix (90 degrees around z-axis)
    config.periodic_rotation = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    periodic = PeriodicBoundary(config, source_nodes, target_nodes)
    
    # Apply boundary condition
    result = periodic.apply(solution.copy(), mesh, target_nodes)
    
    # Check solution is transformed correctly
    # This is a simplified check - in practice, you'd need to verify
    # the actual transformation of the solution variables
    assert not np.allclose(result[target_nodes], solution[source_nodes]) 
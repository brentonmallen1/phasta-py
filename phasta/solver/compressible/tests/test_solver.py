"""
Test module for the compressible flow solver.

This module contains tests to validate the compressible flow solver implementation.
"""

import numpy as np
import pytest
from ...solver.compressible.solver import CompressibleSolver, SolverConfig
from ...solver.compressible.shape_functions import ElementType

def create_test_mesh():
    """Create a simple test mesh."""
    # Create a 2x2x2 hexahedral mesh
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ])
    
    elements = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7]  # Single hex element
    ])
    
    return {
        'nodes': nodes,
        'elements': elements,
        'element_type': 'hex'
    }

def create_test_initial_conditions():
    """Create test initial conditions."""
    n_nodes = 8  # Number of nodes in test mesh
    
    # Uniform flow
    density = np.ones(n_nodes)
    velocity = np.zeros((n_nodes, 3))
    velocity[:, 0] = 1.0  # x-velocity
    pressure = np.ones(n_nodes)
    
    return {
        'density': density,
        'velocity': velocity,
        'pressure': pressure
    }

def test_solver_initialization():
    """Test solver initialization."""
    config = SolverConfig()
    solver = CompressibleSolver(config)
    
    assert solver.solution is None
    assert solver.mesh is None
    assert solver.time == 0.0
    assert solver.time_step == 0

def test_mesh_setup():
    """Test mesh setup."""
    config = SolverConfig()
    solver = CompressibleSolver(config)
    
    mesh = create_test_mesh()
    solver.set_mesh(mesh)
    
    assert solver.mesh is not None
    assert solver.solution is not None
    assert solver.solution.shape == (8, 5)  # 8 nodes, 5 variables

def test_initial_conditions():
    """Test setting initial conditions."""
    config = SolverConfig()
    solver = CompressibleSolver(config)
    
    mesh = create_test_mesh()
    solver.set_mesh(mesh)
    
    ic = create_test_initial_conditions()
    solver.set_initial_conditions(ic)
    
    # Check density
    np.testing.assert_array_almost_equal(
        solver.solution[:, 0],
        ic['density']
    )
    
    # Check momentum
    np.testing.assert_array_almost_equal(
        solver.solution[:, 1:4],
        ic['density'][:, np.newaxis] * ic['velocity']
    )
    
    # Check total energy
    v2 = np.sum(ic['velocity']**2, axis=1)
    expected_energy = (
        ic['pressure'] / (config.gamma - 1.0) +
        0.5 * ic['density'] * v2
    )
    np.testing.assert_array_almost_equal(
        solver.solution[:, 4],
        expected_energy
    )

def test_time_step_computation():
    """Test time step computation."""
    config = SolverConfig()
    solver = CompressibleSolver(config)
    
    mesh = create_test_mesh()
    solver.set_mesh(mesh)
    
    ic = create_test_initial_conditions()
    solver.set_initial_conditions(ic)
    
    dt = solver._compute_time_step()
    assert dt > 0
    assert dt <= config.dt

def test_residual_computation():
    """Test residual computation."""
    config = SolverConfig()
    solver = CompressibleSolver(config)
    
    mesh = create_test_mesh()
    solver.set_mesh(mesh)
    
    ic = create_test_initial_conditions()
    solver.set_initial_conditions(ic)
    
    residual = solver._compute_residual()
    assert residual.shape == solver.solution.shape

def test_pressure_computation():
    """Test pressure computation."""
    config = SolverConfig()
    solver = CompressibleSolver(config)
    
    mesh = create_test_mesh()
    solver.set_mesh(mesh)
    
    ic = create_test_initial_conditions()
    solver.set_initial_conditions(ic)
    
    pressure = solver._compute_pressure()
    np.testing.assert_array_almost_equal(pressure, ic['pressure'])

def test_inviscid_flux():
    """Test inviscid flux computation."""
    config = SolverConfig()
    solver = CompressibleSolver(config)
    
    # Test state
    q = np.array([1.0, 1.0, 0.0, 0.0, 2.5])  # [rho, rho*u, rho*v, rho*w, rho*E]
    
    flux = solver._compute_inviscid_flux(q)
    assert flux.shape == (3, 5)
    
    # Check mass flux
    np.testing.assert_almost_equal(flux[0, 0], 1.0)  # rho*u
    np.testing.assert_almost_equal(flux[1, 0], 0.0)  # rho*v
    np.testing.assert_almost_equal(flux[2, 0], 0.0)  # rho*w
    
    # Check momentum flux
    np.testing.assert_almost_equal(flux[0, 1], 1.0 + 0.4)  # rho*u^2 + p
    np.testing.assert_almost_equal(flux[0, 2], 0.0)  # rho*u*v
    np.testing.assert_almost_equal(flux[0, 3], 0.0)  # rho*u*w

def test_viscous_flux():
    """Test viscous flux computation."""
    config = SolverConfig()
    solver = CompressibleSolver(config)
    
    # Test state
    q = np.array([1.0, 1.0, 0.0, 0.0, 2.5])  # [rho, rho*u, rho*v, rho*w, rho*E]
    
    # Test gradients
    dN = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ])
    
    J = np.eye(3)
    
    flux = solver._compute_viscous_flux(q, dN, J)
    assert flux.shape == (3, 5)
    
    # Check momentum flux
    np.testing.assert_almost_equal(flux[0, 1], 2.0 * config.mu)  # tau_xx
    np.testing.assert_almost_equal(flux[0, 2], 0.0)  # tau_xy
    np.testing.assert_almost_equal(flux[0, 3], 0.0)  # tau_xz

def test_solver_run():
    """Test running the solver."""
    config = SolverConfig(
        max_steps=10,  # Run for a few steps
        dt=0.001,
        cfl=0.5
    )
    solver = CompressibleSolver(config)
    
    mesh = create_test_mesh()
    solver.set_mesh(mesh)
    
    ic = create_test_initial_conditions()
    solver.set_initial_conditions(ic)
    
    # Run solver
    solver.run()
    
    # Check final state
    assert solver.time_step == 10
    assert solver.time > 0
    assert solver.solution is not None
    assert not np.allclose(solver.solution, 0)  # Solution should have evolved

if __name__ == '__main__':
    pytest.main([__file__]) 
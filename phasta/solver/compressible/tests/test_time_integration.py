"""
Unit tests for time integration schemes.
"""

import numpy as np
import pytest
from ..time_integration import (
    TimeIntegrationConfig,
    ExplicitRK,
    SSPRK3,
    TVDRK3,
    create_time_integrator
)

@pytest.fixture
def config():
    """Create default time integration configuration."""
    return TimeIntegrationConfig(
        scheme="explicit_rk",
        order=3,
        cfl=0.5
    )

@pytest.fixture
def solution():
    """Create test solution array."""
    n_nodes = 100
    solution = np.zeros((n_nodes, 7))  # [rho, u, v, w, E, k, omega]
    
    # Set some initial values
    solution[:, 0] = 1.0  # density
    solution[:, 1:4] = 1.0  # velocity
    solution[:, 4] = 2.0  # energy
    solution[:, 5] = 0.1  # k
    solution[:, 6] = 1.0  # omega
    
    return solution

@pytest.fixture
def mesh():
    """Create test mesh."""
    return {
        "nodes": np.random.rand(100, 3),
        "elements": np.random.randint(0, 100, (50, 8)),
        "boundary_faces": {
            "wall": np.array([0, 1, 2]),
            "inlet": np.array([3, 4]),
            "outlet": np.array([5, 6])
        }
    }

@pytest.fixture
def residual():
    """Create test residual function."""
    def compute_residual(solution, mesh):
        # Simple residual that returns a scaled version of the solution
        return -0.1 * solution
    return compute_residual

def test_explicit_rk2(config, solution, mesh, residual):
    """Test second-order explicit Runge-Kutta method."""
    config.order = 2
    integrator = ExplicitRK(config)
    
    # Compute timestep
    dt = integrator.compute_timestep(solution, mesh)
    assert dt > 0
    assert np.isfinite(dt)
    
    # Integrate solution
    new_solution = integrator.integrate(solution.copy(), mesh, residual)
    
    # Check dimensions
    assert new_solution.shape == solution.shape
    
    # Check that solution has changed
    assert not np.allclose(new_solution, solution)
    
    # Check physical constraints
    assert np.all(new_solution[:, 0] > 0)  # density
    assert np.all(new_solution[:, 4] > 0)  # energy
    assert np.all(new_solution[:, 5] >= 0)  # k
    assert np.all(new_solution[:, 6] >= 0)  # omega

def test_explicit_rk3(config, solution, mesh, residual):
    """Test third-order explicit Runge-Kutta method."""
    config.order = 3
    integrator = ExplicitRK(config)
    
    # Compute timestep
    dt = integrator.compute_timestep(solution, mesh)
    assert dt > 0
    assert np.isfinite(dt)
    
    # Integrate solution
    new_solution = integrator.integrate(solution.copy(), mesh, residual)
    
    # Check dimensions
    assert new_solution.shape == solution.shape
    
    # Check that solution has changed
    assert not np.allclose(new_solution, solution)
    
    # Check physical constraints
    assert np.all(new_solution[:, 0] > 0)  # density
    assert np.all(new_solution[:, 4] > 0)  # energy
    assert np.all(new_solution[:, 5] >= 0)  # k
    assert np.all(new_solution[:, 6] >= 0)  # omega

def test_explicit_rk4(config, solution, mesh, residual):
    """Test fourth-order explicit Runge-Kutta method."""
    config.order = 4
    integrator = ExplicitRK(config)
    
    # Compute timestep
    dt = integrator.compute_timestep(solution, mesh)
    assert dt > 0
    assert np.isfinite(dt)
    
    # Integrate solution
    new_solution = integrator.integrate(solution.copy(), mesh, residual)
    
    # Check dimensions
    assert new_solution.shape == solution.shape
    
    # Check that solution has changed
    assert not np.allclose(new_solution, solution)
    
    # Check physical constraints
    assert np.all(new_solution[:, 0] > 0)  # density
    assert np.all(new_solution[:, 4] > 0)  # energy
    assert np.all(new_solution[:, 5] >= 0)  # k
    assert np.all(new_solution[:, 6] >= 0)  # omega

def test_ssp_rk3(config, solution, mesh, residual):
    """Test Strong Stability Preserving Runge-Kutta 3rd order method."""
    config.scheme = "ssp_rk3"
    integrator = SSPRK3(config)
    
    # Compute timestep
    dt = integrator.compute_timestep(solution, mesh)
    assert dt > 0
    assert np.isfinite(dt)
    
    # Integrate solution
    new_solution = integrator.integrate(solution.copy(), mesh, residual)
    
    # Check dimensions
    assert new_solution.shape == solution.shape
    
    # Check that solution has changed
    assert not np.allclose(new_solution, solution)
    
    # Check physical constraints
    assert np.all(new_solution[:, 0] > 0)  # density
    assert np.all(new_solution[:, 4] > 0)  # energy
    assert np.all(new_solution[:, 5] >= 0)  # k
    assert np.all(new_solution[:, 6] >= 0)  # omega

def test_tvd_rk3(config, solution, mesh, residual):
    """Test Total Variation Diminishing Runge-Kutta 3rd order method."""
    config.scheme = "tvd_rk3"
    integrator = TVDRK3(config)
    
    # Compute timestep
    dt = integrator.compute_timestep(solution, mesh)
    assert dt > 0
    assert np.isfinite(dt)
    
    # Integrate solution
    new_solution = integrator.integrate(solution.copy(), mesh, residual)
    
    # Check dimensions
    assert new_solution.shape == solution.shape
    
    # Check that solution has changed
    assert not np.allclose(new_solution, solution)
    
    # Check physical constraints
    assert np.all(new_solution[:, 0] > 0)  # density
    assert np.all(new_solution[:, 4] > 0)  # energy
    assert np.all(new_solution[:, 5] >= 0)  # k
    assert np.all(new_solution[:, 6] >= 0)  # omega

def test_factory_function(config):
    """Test time integrator factory function."""
    # Test explicit RK
    config.scheme = "explicit_rk"
    integrator = create_time_integrator(config)
    assert isinstance(integrator, ExplicitRK)
    
    # Test SSP-RK3
    config.scheme = "ssp_rk3"
    integrator = create_time_integrator(config)
    assert isinstance(integrator, SSPRK3)
    
    # Test TVD-RK3
    config.scheme = "tvd_rk3"
    integrator = create_time_integrator(config)
    assert isinstance(integrator, TVDRK3)
    
    # Test invalid scheme
    config.scheme = "invalid"
    with pytest.raises(ValueError):
        create_time_integrator(config)

def test_invalid_config():
    """Test invalid configuration parameters."""
    # Test negative CFL
    with pytest.raises(ValueError):
        TimeIntegrationConfig(scheme="explicit_rk", order=2, cfl=-0.5)
    
    # Test zero order
    with pytest.raises(ValueError):
        TimeIntegrationConfig(scheme="explicit_rk", order=0, cfl=0.5)
    
    # Test invalid order for SSP-RK3
    config = TimeIntegrationConfig(scheme="ssp_rk3", order=2, cfl=0.5)
    with pytest.raises(ValueError):
        SSPRK3(config)
    
    # Test invalid order for TVD-RK3
    config = TimeIntegrationConfig(scheme="tvd_rk3", order=2, cfl=0.5)
    with pytest.raises(ValueError):
        TVDRK3(config) 
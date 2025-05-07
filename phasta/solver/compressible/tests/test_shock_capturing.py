"""
Unit tests for shock capturing schemes.
"""

import numpy as np
import pytest
from ..shock_capturing import (
    ShockCapturingConfig,
    TVDScheme,
    WENOScheme,
    ArtificialViscosity,
    create_shock_capturing
)

@pytest.fixture
def config():
    """Create default shock capturing configuration."""
    return ShockCapturingConfig(
        scheme="tvd",
        kappa=1.0,
        epsilon=1e-6,
        c_vis=0.1
    )

@pytest.fixture
def solution():
    """Create test solution array with a shock."""
    n = 100
    x = np.linspace(-1, 1, n)
    u = np.zeros(n)
    
    # Create a shock at x = 0
    u[x < 0] = 1.0
    u[x >= 0] = 0.0
    
    return u

@pytest.fixture
def dx():
    """Create test grid spacing."""
    return 0.02

def test_tvd_limiter(config, solution):
    """Test TVD limiter function."""
    scheme = TVDScheme(config)
    
    # Test limiter for smooth solution
    r = np.ones(100)
    phi = scheme.compute_limiter(r)
    assert np.allclose(phi, 1.0)
    
    # Test limiter for shock
    r = np.zeros(100)
    r[50] = 2.0
    phi = scheme.compute_limiter(r)
    assert phi[50] == 1.0
    assert np.all(phi >= 0)
    assert np.all(phi <= 2.0)

def test_tvd_reconstruction(config, solution, dx):
    """Test TVD reconstruction."""
    scheme = TVDScheme(config)
    
    # Compute reconstruction
    u_left, u_right = scheme.compute_reconstruction(solution, dx)
    
    # Check dimensions
    assert u_left.shape == solution.shape
    assert u_right.shape == solution.shape
    
    # Check that reconstruction preserves shock
    assert np.all(u_left[solution == 1.0] > 0.5)
    assert np.all(u_right[solution == 0.0] < 0.5)
    
    # Check that reconstruction is TVD
    assert np.sum(np.abs(u_left[1:] - u_left[:-1])) <= np.sum(np.abs(solution[1:] - solution[:-1]))
    assert np.sum(np.abs(u_right[1:] - u_right[:-1])) <= np.sum(np.abs(solution[1:] - solution[:-1]))

def test_weno_smoothness_indicators(config, solution, dx):
    """Test WENO smoothness indicators."""
    scheme = WENOScheme(config)
    
    # Compute smoothness indicators
    beta = scheme.compute_smoothness_indicators(solution, dx)
    
    # Check dimensions
    assert beta.shape == (3, len(solution))
    
    # Check that indicators are non-negative
    assert np.all(beta >= 0)
    
    # Check that indicators are larger near shock
    shock_idx = np.argmax(np.abs(solution[1:] - solution[:-1]))
    assert np.all(beta[:, shock_idx] > beta[:, shock_idx-1])
    assert np.all(beta[:, shock_idx] > beta[:, shock_idx+1])

def test_weno_weights(config, solution, dx):
    """Test WENO weights computation."""
    scheme = WENOScheme(config)
    
    # Compute smoothness indicators
    beta = scheme.compute_smoothness_indicators(solution, dx)
    
    # Compute weights
    weights = scheme.compute_weights(beta)
    
    # Check dimensions
    assert weights.shape == beta.shape
    
    # Check that weights sum to 1
    assert np.allclose(np.sum(weights, axis=0), 1.0)
    
    # Check that weights are non-negative
    assert np.all(weights >= 0)
    
    # Check that weights are smaller for less smooth stencils
    shock_idx = np.argmax(np.abs(solution[1:] - solution[:-1]))
    assert np.all(weights[:, shock_idx] < weights[:, shock_idx-1])
    assert np.all(weights[:, shock_idx] < weights[:, shock_idx+1])

def test_weno_reconstruction(config, solution, dx):
    """Test WENO reconstruction."""
    scheme = WENOScheme(config)
    
    # Compute reconstruction
    u_left, u_right = scheme.compute_reconstruction(solution, dx)
    
    # Check dimensions
    assert u_left.shape == solution.shape
    assert u_right.shape == solution.shape
    
    # Check that reconstruction preserves shock
    assert np.all(u_left[solution == 1.0] > 0.5)
    assert np.all(u_right[solution == 0.0] < 0.5)
    
    # Check that reconstruction is non-oscillatory
    assert np.all(np.diff(u_left) >= -1e-10)
    assert np.all(np.diff(u_right) >= -1e-10)

def test_artificial_viscosity(config, solution, dx):
    """Test artificial viscosity computation."""
    scheme = ArtificialViscosity(config)
    
    # Compute artificial viscosity
    nu = scheme.compute_artificial_viscosity(solution, dx)
    
    # Check dimensions
    assert nu.shape == solution.shape
    
    # Check that viscosity is non-negative
    assert np.all(nu >= 0)
    
    # Check that viscosity is larger near shock
    shock_idx = np.argmax(np.abs(solution[1:] - solution[:-1]))
    assert nu[shock_idx] > nu[shock_idx-1]
    assert nu[shock_idx] > nu[shock_idx+1]

def test_factory_function(config):
    """Test shock capturing factory function."""
    # Test TVD scheme
    config.scheme = "tvd"
    scheme = create_shock_capturing(config)
    assert isinstance(scheme, TVDScheme)
    
    # Test WENO scheme
    config.scheme = "weno"
    scheme = create_shock_capturing(config)
    assert isinstance(scheme, WENOScheme)
    
    # Test artificial viscosity
    config.scheme = "artificial_viscosity"
    scheme = create_shock_capturing(config)
    assert isinstance(scheme, ArtificialViscosity)
    
    # Test invalid scheme
    config.scheme = "invalid"
    with pytest.raises(ValueError):
        create_shock_capturing(config) 
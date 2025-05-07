"""
Unit tests for limiting strategies.
"""

import numpy as np
import pytest
from ..limiters import (
    LimiterConfig,
    SlopeLimiter,
    FluxLimiter,
    PressureLimiter,
    create_limiter
)

@pytest.fixture
def config():
    """Create default limiter configuration."""
    return LimiterConfig(
        scheme="slope",
        beta=1.0,
        epsilon=1e-6
    )

@pytest.fixture
def solution():
    """Create test solution array with discontinuities."""
    n = 100
    x = np.linspace(-1, 1, n)
    u = np.zeros(n)
    
    # Create a discontinuity at x = 0
    u[x < 0] = 1.0
    u[x >= 0] = 0.0
    
    return u

@pytest.fixture
def dx():
    """Create test grid spacing."""
    return 0.02

def test_slope_limiter(config, solution, dx):
    """Test slope limiter."""
    limiter = SlopeLimiter(config)
    
    # Compute limited slopes
    du_limited = limiter.compute_slope_limiter(solution, dx)
    
    # Check dimensions
    assert du_limited.shape == solution.shape
    
    # Check that slopes are limited
    assert np.all(np.abs(du_limited) <= np.abs(np.diff(solution, prepend=solution[0]) / dx))
    
    # Check that slopes are zero at discontinuities
    shock_idx = np.argmax(np.abs(solution[1:] - solution[:-1]))
    assert np.abs(du_limited[shock_idx]) < 1e-10

def test_flux_limiter(config, solution, dx):
    """Test flux limiter."""
    limiter = FluxLimiter(config)
    
    # Compute ratio of consecutive gradients
    r = np.zeros_like(solution)
    r[1:-1] = (solution[2:] - solution[1:-1]) / (solution[1:-1] - solution[:-2] + config.epsilon)
    
    # Compute limiter
    phi = limiter.compute_flux_limiter(r)
    
    # Check dimensions
    assert phi.shape == solution.shape
    
    # Check that limiter is between 0 and 2
    assert np.all(phi >= 0)
    assert np.all(phi <= 2)
    
    # Check that limiter is 0 for negative ratios
    assert np.all(phi[r <= 0] == 0)
    
    # Check that limiter is 1 for ratio = 1
    r_ones = np.ones_like(r)
    phi_ones = limiter.compute_flux_limiter(r_ones)
    assert np.allclose(phi_ones, 1.0)

def test_pressure_limiter(config):
    """Test pressure limiter."""
    limiter = PressureLimiter(config)
    
    # Create test pressure and density arrays
    n = 100
    p = np.ones(n)
    rho = np.ones(n)
    
    # Add a pressure spike
    p[50] = 2.0
    
    # Compute limited pressure
    p_limited = limiter.compute_pressure_limiter(p, rho)
    
    # Check dimensions
    assert p_limited.shape == p.shape
    
    # Check that pressure is positive
    assert np.all(p_limited >= 0)
    
    # Check that pressure spike is limited
    assert p_limited[50] < p[50]
    
    # Check that pressure is preserved for smooth regions
    smooth_mask = np.abs(np.diff(p, prepend=p[0])) < 0.1
    assert np.allclose(p_limited[smooth_mask], p[smooth_mask])

def test_factory_function(config):
    """Test limiter factory function."""
    # Test slope limiter
    config.scheme = "slope"
    limiter = create_limiter(config)
    assert isinstance(limiter, SlopeLimiter)
    
    # Test flux limiter
    config.scheme = "flux"
    limiter = create_limiter(config)
    assert isinstance(limiter, FluxLimiter)
    
    # Test pressure limiter
    config.scheme = "pressure"
    limiter = create_limiter(config)
    assert isinstance(limiter, PressureLimiter)
    
    # Test invalid scheme
    config.scheme = "invalid"
    with pytest.raises(ValueError):
        create_limiter(config) 
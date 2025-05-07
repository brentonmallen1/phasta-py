"""
Unit tests for wall functions.
"""

import numpy as np
import pytest
from phasta.solver.compressible.wall_functions import (
    WallFunctionConfig,
    StandardWallFunctions,
    EnhancedWallTreatment,
    AutomaticWallTreatment,
    create_wall_functions
)

@pytest.fixture
def config():
    """Create default wall function configuration."""
    return WallFunctionConfig()

@pytest.fixture
def y_plus_values():
    """Create test y+ values."""
    return np.array([5.0, 11.0, 30.0, 100.0, 300.0])

def test_standard_wall_functions(config, y_plus_values):
    """Test standard wall functions."""
    wall_functions = StandardWallFunctions(config)
    u_plus = wall_functions.compute_u_plus(y_plus_values)
    
    # Check viscous sublayer
    assert u_plus[0] == 5.0  # y+ < y_plus_switch
    
    # Check log layer
    expected_log = 1.0/config.kappa * np.log(config.E * y_plus_values[2:])
    np.testing.assert_allclose(u_plus[2:], expected_log, rtol=1e-5)
    
    # Check transition
    assert 5.0 < u_plus[1] < expected_log[0]  # y+ = y_plus_switch

def test_enhanced_wall_treatment(config, y_plus_values):
    """Test enhanced wall treatment."""
    wall_functions = EnhancedWallTreatment(config)
    u_plus = wall_functions.compute_u_plus(y_plus_values)
    
    # Check viscous sublayer
    assert u_plus[0] == 5.0  # y+ < y_plus_switch
    
    # Check log layer
    expected_log = 1.0/config.kappa * np.log(config.E * y_plus_values[-1])
    assert u_plus[-1] > expected_log * 0.9  # Blended solution should be close to log law
    
    # Check blending
    assert np.all(np.diff(u_plus) > 0)  # Monotonic increase

def test_automatic_wall_treatment(config, y_plus_values):
    """Test automatic wall treatment."""
    wall_functions = AutomaticWallTreatment(config)
    u_plus = wall_functions.compute_u_plus(y_plus_values)
    
    # Check viscous sublayer
    assert u_plus[0] == 5.0  # y+ < y_plus_switch
    
    # Check log layer
    expected_log = 1.0/config.kappa * np.log(config.E * y_plus_values[-1])
    assert u_plus[-1] > expected_log * 0.9  # Should approach log law
    
    # Check smooth transition
    assert np.all(np.diff(u_plus) > 0)  # Monotonic increase

def test_wall_shear_stress(config):
    """Test wall shear stress computation."""
    wall_functions = StandardWallFunctions(config)
    
    # Test data
    y = np.array([1e-5, 1e-4, 1e-3])
    u = np.array([1.0, 2.0, 3.0])
    rho = np.array([1.0, 1.0, 1.0])
    mu = np.array([1e-5, 1e-5, 1e-5])
    
    tau_wall, u_tau = wall_functions.compute_tau_wall(y, u, rho, mu)
    
    # Check dimensions
    assert tau_wall.shape == y.shape
    assert u_tau.shape == y.shape
    
    # Check physical constraints
    assert np.all(tau_wall > 0)
    assert np.all(u_tau > 0)
    
    # Check relationship between tau_wall and u_tau
    np.testing.assert_allclose(tau_wall, rho * u_tau * u_tau, rtol=1e-5)

def test_create_wall_functions():
    """Test wall functions factory function."""
    config = WallFunctionConfig()
    
    # Test standard wall functions
    wall_functions = create_wall_functions("standard", config)
    assert isinstance(wall_functions, StandardWallFunctions)
    
    # Test enhanced wall treatment
    wall_functions = create_wall_functions("enhanced", config)
    assert isinstance(wall_functions, EnhancedWallTreatment)
    
    # Test automatic wall treatment
    wall_functions = create_wall_functions("automatic", config)
    assert isinstance(wall_functions, AutomaticWallTreatment)
    
    # Test invalid type
    with pytest.raises(ValueError):
        create_wall_functions("invalid", config) 
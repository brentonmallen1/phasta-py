"""Tests for radiation models."""

import numpy as np
import pytest
from phasta.solver.radiation import (
    RadiationModel,
    DiscreteOrdinates,
    P1Model,
    MonteCarloRadiation
)


def test_discrete_ordinates():
    """Test Discrete Ordinates Method."""
    # Create model
    model = DiscreteOrdinates(n_angles=8)
    
    # Test data
    temperature = np.array([1000.0, 800.0, 600.0])
    absorption_coef = np.array([0.1, 0.2, 0.3])
    scattering_coef = np.array([0.05, 0.1, 0.15])
    emissivity = np.array([0.8, 0.7, 0.6])
    
    # Test radiation computation
    intensity, heat_source = model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    
    # Check results
    assert intensity.shape == (4, 3)  # 4 ordinates, 3 points
    assert heat_source.shape == (3,)
    assert np.all(intensity >= 0.0)
    assert np.all(heat_source >= 0.0)
    
    # Test invalid number of ordinates
    with pytest.raises(ValueError):
        model = DiscreteOrdinates(n_angles=6)


def test_p1_model():
    """Test P1 radiation model."""
    # Create model
    model = P1Model()
    
    # Test data
    temperature = np.array([1000.0, 800.0, 600.0])
    absorption_coef = np.array([0.1, 0.2, 0.3])
    scattering_coef = np.array([0.05, 0.1, 0.15])
    emissivity = np.array([0.8, 0.7, 0.6])
    
    # Test radiation computation
    intensity, heat_source = model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    
    # Check results
    assert intensity.shape == (3,)
    assert heat_source.shape == (3,)
    assert np.all(intensity >= 0.0)
    assert np.all(heat_source >= 0.0)
    
    # Test incident radiation
    assert model.g is not None
    assert model.g.shape == (3,)
    assert np.all(model.g >= 0.0)


def test_monte_carlo():
    """Test Monte Carlo radiation model."""
    # Create model
    model = MonteCarloRadiation(n_rays=1000)
    
    # Test data
    temperature = np.array([1000.0, 800.0, 600.0])
    absorption_coef = np.array([0.1, 0.2, 0.3])
    scattering_coef = np.array([0.05, 0.1, 0.15])
    emissivity = np.array([0.8, 0.7, 0.6])
    
    # Test radiation computation
    intensity, heat_source = model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    
    # Check results
    assert intensity.shape == (3,)
    assert heat_source.shape == (3,)
    assert np.all(intensity >= 0.0)
    assert np.all(heat_source >= 0.0)


def test_edge_cases():
    """Test edge cases."""
    # Create models
    dom_model = DiscreteOrdinates()
    p1_model = P1Model()
    mc_model = MonteCarloRadiation(n_rays=100)
    
    # Test data
    temperature = np.array([0.0, 0.0, 0.0])
    absorption_coef = np.array([0.0, 0.0, 0.0])
    scattering_coef = np.array([0.0, 0.0, 0.0])
    emissivity = np.array([0.0, 0.0, 0.0])
    
    # Test zero temperature
    intensity, heat_source = dom_model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    assert np.all(intensity == 0.0)
    assert np.all(heat_source == 0.0)
    
    intensity, heat_source = p1_model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    assert np.all(intensity == 0.0)
    assert np.all(heat_source == 0.0)
    
    intensity, heat_source = mc_model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    assert np.all(intensity == 0.0)
    assert np.all(heat_source == 0.0)
    
    # Test zero coefficients
    temperature = np.array([1000.0, 800.0, 600.0])
    
    intensity, heat_source = dom_model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    assert np.all(intensity >= 0.0)
    assert np.all(heat_source == 0.0)
    
    intensity, heat_source = p1_model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    assert np.all(intensity >= 0.0)
    assert np.all(heat_source == 0.0)
    
    intensity, heat_source = mc_model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    assert np.all(intensity >= 0.0)
    assert np.all(heat_source == 0.0)


def test_memory_management():
    """Test memory management."""
    # Create models
    dom_model = DiscreteOrdinates()
    p1_model = P1Model()
    mc_model = MonteCarloRadiation(n_rays=100)
    
    # Test data
    n_points = 1000
    temperature = np.ones(n_points) * 1000.0
    absorption_coef = np.ones(n_points) * 0.1
    scattering_coef = np.ones(n_points) * 0.05
    emissivity = np.ones(n_points) * 0.8
    
    # Test large arrays
    intensity, heat_source = dom_model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    assert intensity.shape == (4, n_points)
    assert heat_source.shape == (n_points,)
    
    intensity, heat_source = p1_model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    assert intensity.shape == (n_points,)
    assert heat_source.shape == (n_points,)
    
    intensity, heat_source = mc_model.compute_radiation(
        temperature, absorption_coef, scattering_coef, emissivity
    )
    assert intensity.shape == (n_points,)
    assert heat_source.shape == (n_points,)


def test_convergence():
    """Test convergence with grid refinement."""
    # Create model
    model = DiscreteOrdinates()
    
    # Test data
    n_points = [10, 20, 40, 80]
    temperature = np.array([1000.0, 800.0, 600.0])
    absorption_coef = np.array([0.1, 0.2, 0.3])
    scattering_coef = np.array([0.05, 0.1, 0.15])
    emissivity = np.array([0.8, 0.7, 0.6])
    
    # Test different grid sizes
    heat_sources = []
    
    for n in n_points:
        # Interpolate data to finer grid
        T = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 3), temperature)
        k_a = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 3), absorption_coef)
        k_s = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 3), scattering_coef)
        e = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 3), emissivity)
        
        # Compute radiation
        _, heat_source = model.compute_radiation(T, k_a, k_s, e)
        heat_sources.append(np.mean(heat_source))
    
    # Check convergence
    for i in range(len(heat_sources) - 1):
        ratio = abs(heat_sources[i] - heat_sources[i+1]) / heat_sources[i+1]
        assert ratio < 0.1  # Should converge with grid refinement 
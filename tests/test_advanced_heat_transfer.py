import pytest
import numpy as np
from phasta.models.heat_transfer import (
    ConjugateHeatTransfer,
    P1RadiationModel,
    DiscreteOrdinatesModel,
    ViewFactorCalculator
)

class TestConjugateHeatTransfer:
    @pytest.fixture
    def model(self):
        return ConjugateHeatTransfer()

    def test_interface_heat_transfer(self, model):
        """Test heat transfer at fluid-solid interface."""
        # Setup test data
        fluid_temp = np.ones((10, 10)) * 300.0  # K
        solid_temp = np.ones((10, 10)) * 350.0  # K
        interface_conductivity = 50.0  # W/m·K
        
        # Calculate heat flux
        heat_flux = model.calculate_interface_heat_flux(
            fluid_temp, solid_temp, interface_conductivity
        )
        
        # Verify heat flux properties
        assert heat_flux.shape == (10, 10)
        assert np.all(np.isfinite(heat_flux))
        assert np.all(heat_flux > 0)  # Heat should flow from solid to fluid

    def test_temperature_coupling(self, model):
        """Test temperature coupling between fluid and solid domains."""
        # Setup test data
        fluid_temp = np.ones((10, 10)) * 300.0  # K
        solid_temp = np.ones((10, 10)) * 350.0  # K
        time_step = 0.001  # s
        
        # Calculate coupled temperatures
        new_fluid_temp, new_solid_temp = model.calculate_coupled_temperatures(
            fluid_temp, solid_temp, time_step
        )
        
        # Verify temperature properties
        assert new_fluid_temp.shape == fluid_temp.shape
        assert new_solid_temp.shape == solid_temp.shape
        assert np.all(np.isfinite(new_fluid_temp))
        assert np.all(np.isfinite(new_solid_temp))
        assert np.all(new_fluid_temp > fluid_temp)
        assert np.all(new_solid_temp < solid_temp)

class TestP1RadiationModel:
    @pytest.fixture
    def model(self):
        return P1RadiationModel()

    def test_radiation_flux(self, model):
        """Test P1 radiation flux calculation."""
        # Setup test data
        temperature = np.ones((10, 10)) * 1000.0  # K
        absorption_coef = 0.1  # m^-1
        scattering_coef = 0.05  # m^-1
        
        # Calculate radiation flux
        flux = model.calculate_radiation_flux(
            temperature, absorption_coef, scattering_coef
        )
        
        # Verify flux properties
        assert flux.shape == (10, 10, 3)  # 3D flux vector
        assert np.all(np.isfinite(flux))
        assert np.all(np.abs(flux) >= 0)

    def test_radiation_source(self, model):
        """Test radiation source term calculation."""
        # Setup test data
        temperature = np.ones((10, 10)) * 1000.0  # K
        absorption_coef = 0.1  # m^-1
        scattering_coef = 0.05  # m^-1
        
        # Calculate source term
        source = model.calculate_radiation_source(
            temperature, absorption_coef, scattering_coef
        )
        
        # Verify source properties
        assert source.shape == temperature.shape
        assert np.all(np.isfinite(source))
        assert np.all(source >= 0)

class TestDiscreteOrdinatesModel:
    @pytest.fixture
    def model(self):
        return DiscreteOrdinatesModel()

    def test_angular_quadrature(self, model):
        """Test angular quadrature setup."""
        # Setup test data
        order = 4  # S4 quadrature
        
        # Get quadrature points and weights
        points, weights = model.get_angular_quadrature(order)
        
        # Verify quadrature properties
        assert len(points) == len(weights)
        assert np.all(np.isfinite(points))
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)
        assert np.isclose(np.sum(weights), 4 * np.pi)

    def test_radiation_intensity(self, model):
        """Test radiation intensity calculation."""
        # Setup test data
        temperature = np.ones((10, 10)) * 1000.0  # K
        absorption_coef = 0.1  # m^-1
        scattering_coef = 0.05  # m^-1
        order = 4  # S4 quadrature
        
        # Calculate intensity
        intensity = model.calculate_intensity(
            temperature, absorption_coef, scattering_coef, order
        )
        
        # Verify intensity properties
        assert intensity.shape == (10, 10, len(model.get_angular_quadrature(order)[0]))
        assert np.all(np.isfinite(intensity))
        assert np.all(intensity >= 0)

class TestViewFactorCalculator:
    @pytest.fixture
    def calculator(self):
        return ViewFactorCalculator()

    def test_view_factor_calculation(self, calculator):
        """Test view factor calculation between surfaces."""
        # Setup test data
        surface1 = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ])
        surface2 = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ])
        
        # Calculate view factor
        view_factor = calculator.calculate_view_factor(surface1, surface2)
        
        # Verify view factor properties
        assert isinstance(view_factor, float)
        assert 0 <= view_factor <= 1
        assert np.isfinite(view_factor)

    def test_self_view_factor(self, calculator):
        """Test view factor calculation for self-viewing surfaces."""
        # Setup test data
        surface = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ])
        
        # Calculate self view factor
        view_factor = calculator.calculate_self_view_factor(surface)
        
        # Verify view factor properties
        assert isinstance(view_factor, float)
        assert 0 <= view_factor <= 1
        assert np.isfinite(view_factor)

@pytest.mark.parametrize("model_class", [
    ConjugateHeatTransfer,
    P1RadiationModel,
    DiscreteOrdinatesModel,
    ViewFactorCalculator
])
def test_model_initialization(model_class):
    """Test proper initialization of all heat transfer models."""
    model = model_class()
    assert model is not None
    assert hasattr(model, "parameters")
    assert isinstance(model.parameters, dict)

@pytest.mark.parametrize("model_class", [
    ConjugateHeatTransfer,
    P1RadiationModel,
    DiscreteOrdinatesModel,
    ViewFactorCalculator
])
def test_model_parameter_validation(model_class):
    """Test parameter validation for all heat transfer models."""
    model = model_class()
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        model.set_parameters({"invalid_param": -1.0})
    
    # Test with valid parameters
    valid_params = model.get_default_parameters()
    model.set_parameters(valid_params)
    assert model.parameters == valid_params 
import pytest
import numpy as np
from phasta.models.multiphase import (
    PLICReconstructor,
    InterfaceSharpener,
    PhaseChangeModel,
    MultiFluidModel
)

class TestPLICReconstructor:
    @pytest.fixture
    def reconstructor(self):
        return PLICReconstructor()

    def test_interface_reconstruction(self, reconstructor):
        """Test PLIC interface reconstruction."""
        # Setup test data
        volume_fraction = np.ones((10, 10)) * 0.5
        normal = np.array([1.0, 0.0, 0.0])
        
        # Reconstruct interface
        interface = reconstructor.reconstruct_interface(
            volume_fraction, normal
        )
        
        # Verify interface properties
        assert interface.shape == (10, 10, 3)  # 3D interface points
        assert np.all(np.isfinite(interface))
        assert np.all(np.abs(interface) >= 0)

    def test_volume_conservation(self, reconstructor):
        """Test volume conservation in PLIC reconstruction."""
        # Setup test data
        volume_fraction = np.random.rand(10, 10)
        normal = np.array([1.0, 0.0, 0.0])
        
        # Reconstruct interface
        interface = reconstructor.reconstruct_interface(
            volume_fraction, normal
        )
        
        # Calculate reconstructed volume fraction
        reconstructed_volume = reconstructor.calculate_volume_fraction(interface)
        
        # Verify volume conservation
        assert np.allclose(volume_fraction, reconstructed_volume, atol=1e-6)

class TestInterfaceSharpener:
    @pytest.fixture
    def sharpener(self):
        return InterfaceSharpener()

    def test_interface_sharpening(self, sharpener):
        """Test interface sharpening operation."""
        # Setup test data
        volume_fraction = np.ones((10, 10)) * 0.5
        velocity = np.random.rand(10, 10, 3)
        
        # Sharpen interface
        sharpened = sharpener.sharpen_interface(
            volume_fraction, velocity
        )
        
        # Verify sharpened interface properties
        assert sharpened.shape == volume_fraction.shape
        assert np.all(np.isfinite(sharpened))
        assert np.all(sharpened >= 0)
        assert np.all(sharpened <= 1)

    def test_mass_conservation(self, sharpener):
        """Test mass conservation during sharpening."""
        # Setup test data
        volume_fraction = np.random.rand(10, 10)
        velocity = np.random.rand(10, 10, 3)
        
        # Calculate initial mass
        initial_mass = np.sum(volume_fraction)
        
        # Sharpen interface
        sharpened = sharpener.sharpen_interface(
            volume_fraction, velocity
        )
        
        # Calculate final mass
        final_mass = np.sum(sharpened)
        
        # Verify mass conservation
        assert np.isclose(initial_mass, final_mass, atol=1e-6)

class TestPhaseChangeModel:
    @pytest.fixture
    def model(self):
        return PhaseChangeModel()

    def test_evaporation(self, model):
        """Test evaporation model."""
        # Setup test data
        temperature = np.ones((10, 10)) * 373.15  # K
        pressure = np.ones((10, 10)) * 101325.0  # Pa
        velocity = np.random.rand(10, 10, 3)
        
        # Calculate evaporation rate
        evap_rate = model.calculate_evaporation_rate(
            temperature, pressure, velocity
        )
        
        # Verify evaporation rate properties
        assert evap_rate.shape == (10, 10)
        assert np.all(np.isfinite(evap_rate))
        assert np.all(evap_rate >= 0)

    def test_condensation(self, model):
        """Test condensation model."""
        # Setup test data
        temperature = np.ones((10, 10)) * 273.15  # K
        pressure = np.ones((10, 10)) * 101325.0  # Pa
        velocity = np.random.rand(10, 10, 3)
        
        # Calculate condensation rate
        cond_rate = model.calculate_condensation_rate(
            temperature, pressure, velocity
        )
        
        # Verify condensation rate properties
        assert cond_rate.shape == (10, 10)
        assert np.all(np.isfinite(cond_rate))
        assert np.all(cond_rate >= 0)

class TestMultiFluidModel:
    @pytest.fixture
    def model(self):
        return MultiFluidModel()

    def test_euler_euler(self, model):
        """Test Euler-Euler model."""
        # Setup test data
        phase1_volume = np.ones((10, 10)) * 0.5
        phase2_volume = np.ones((10, 10)) * 0.5
        phase1_velocity = np.random.rand(10, 10, 3)
        phase2_velocity = np.random.rand(10, 10, 3)
        
        # Calculate phase interaction
        interaction = model.calculate_phase_interaction(
            phase1_volume, phase2_volume,
            phase1_velocity, phase2_velocity
        )
        
        # Verify interaction properties
        assert interaction.shape == (10, 10, 3)
        assert np.all(np.isfinite(interaction))

    def test_euler_lagrange(self, model):
        """Test Euler-Lagrange model."""
        # Setup test data
        fluid_velocity = np.random.rand(10, 10, 3)
        particle_positions = np.random.rand(100, 3)
        particle_velocities = np.random.rand(100, 3)
        
        # Calculate particle-fluid interaction
        interaction = model.calculate_particle_fluid_interaction(
            fluid_velocity,
            particle_positions,
            particle_velocities
        )
        
        # Verify interaction properties
        assert interaction.shape == (100, 3)
        assert np.all(np.isfinite(interaction))

@pytest.mark.parametrize("model_class", [
    PLICReconstructor,
    InterfaceSharpener,
    PhaseChangeModel,
    MultiFluidModel
])
def test_model_initialization(model_class):
    """Test proper initialization of all multi-phase models."""
    model = model_class()
    assert model is not None
    assert hasattr(model, "parameters")
    assert isinstance(model.parameters, dict)

@pytest.mark.parametrize("model_class", [
    PLICReconstructor,
    InterfaceSharpener,
    PhaseChangeModel,
    MultiFluidModel
])
def test_model_parameter_validation(model_class):
    """Test parameter validation for all multi-phase models."""
    model = model_class()
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        model.set_parameters({"invalid_param": -1.0})
    
    # Test with valid parameters
    valid_params = model.get_default_parameters()
    model.set_parameters(valid_params)
    assert model.parameters == valid_params 
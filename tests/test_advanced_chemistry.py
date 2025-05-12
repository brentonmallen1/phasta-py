import pytest
import numpy as np
from phasta.models.chemistry import (
    DetailedKinetics,
    ReducedMechanism,
    SkeletalMechanism,
    MultiComponentDiffusion,
    ThermalDiffusion,
    PressureDiffusion
)

class TestDetailedKinetics:
    @pytest.fixture
    def model(self):
        return DetailedKinetics()

    def test_reaction_rates(self, model):
        """Test detailed reaction rate calculations."""
        # Setup test data
        temperature = np.ones((10, 10)) * 1000.0  # K
        pressure = np.ones((10, 10)) * 101325.0  # Pa
        species_concentrations = np.random.rand(10, 10, 5)  # 5 species
        
        # Calculate reaction rates
        rates = model.calculate_reaction_rates(
            temperature, pressure, species_concentrations
        )
        
        # Verify rate properties
        assert rates.shape == (10, 10, 5)
        assert np.all(np.isfinite(rates))

    def test_equilibrium_constants(self, model):
        """Test equilibrium constant calculations."""
        # Setup test data
        temperature = np.ones((10, 10)) * 1000.0  # K
        
        # Calculate equilibrium constants
        k_eq = model.calculate_equilibrium_constants(temperature)
        
        # Verify equilibrium constant properties
        assert k_eq.shape == (10, 10, model.n_reactions)
        assert np.all(np.isfinite(k_eq))
        assert np.all(k_eq > 0)

class TestReducedMechanism:
    @pytest.fixture
    def model(self):
        return ReducedMechanism()

    def test_species_reduction(self, model):
        """Test species reduction process."""
        # Setup test data
        full_species = np.random.rand(10, 10, 20)  # 20 species
        temperature = np.ones((10, 10)) * 1000.0  # K
        
        # Reduce species
        reduced_species = model.reduce_species(full_species, temperature)
        
        # Verify reduction properties
        assert reduced_species.shape == (10, 10, model.n_reduced_species)
        assert np.all(np.isfinite(reduced_species))
        assert model.n_reduced_species < 20

    def test_reaction_reduction(self, model):
        """Test reaction reduction process."""
        # Setup test data
        full_rates = np.random.rand(10, 10, 50)  # 50 reactions
        temperature = np.ones((10, 10)) * 1000.0  # K
        
        # Reduce reactions
        reduced_rates = model.reduce_reactions(full_rates, temperature)
        
        # Verify reduction properties
        assert reduced_rates.shape == (10, 10, model.n_reduced_reactions)
        assert np.all(np.isfinite(reduced_rates))
        assert model.n_reduced_reactions < 50

class TestSkeletalMechanism:
    @pytest.fixture
    def model(self):
        return SkeletalMechanism()

    def test_mechanism_generation(self, model):
        """Test skeletal mechanism generation."""
        # Setup test data
        full_mechanism = {
            'species': np.random.rand(20),
            'reactions': np.random.rand(50),
            'rates': np.random.rand(50)
        }
        temperature = 1000.0  # K
        
        # Generate skeletal mechanism
        skeletal = model.generate_skeletal_mechanism(
            full_mechanism, temperature
        )
        
        # Verify skeletal mechanism properties
        assert len(skeletal['species']) < 20
        assert len(skeletal['reactions']) < 50
        assert len(skeletal['rates']) < 50

    def test_mechanism_validation(self, model):
        """Test skeletal mechanism validation."""
        # Setup test data
        full_mechanism = {
            'species': np.random.rand(20),
            'reactions': np.random.rand(50),
            'rates': np.random.rand(50)
        }
        skeletal_mechanism = {
            'species': np.random.rand(10),
            'reactions': np.random.rand(20),
            'rates': np.random.rand(20)
        }
        
        # Validate mechanism
        error = model.validate_mechanism(
            full_mechanism, skeletal_mechanism
        )
        
        # Verify validation properties
        assert isinstance(error, float)
        assert error >= 0
        assert np.isfinite(error)

class TestMultiComponentDiffusion:
    @pytest.fixture
    def model(self):
        return MultiComponentDiffusion()

    def test_diffusion_coefficients(self, model):
        """Test multi-component diffusion coefficient calculations."""
        # Setup test data
        temperature = np.ones((10, 10)) * 1000.0  # K
        pressure = np.ones((10, 10)) * 101325.0  # Pa
        species_concentrations = np.random.rand(10, 10, 5)  # 5 species
        
        # Calculate diffusion coefficients
        diff_coeffs = model.calculate_diffusion_coefficients(
            temperature, pressure, species_concentrations
        )
        
        # Verify coefficient properties
        assert diff_coeffs.shape == (10, 10, 5, 5)
        assert np.all(np.isfinite(diff_coeffs))
        assert np.all(diff_coeffs >= 0)

class TestThermalDiffusion:
    @pytest.fixture
    def model(self):
        return ThermalDiffusion()

    def test_thermal_diffusion_coefficients(self, model):
        """Test thermal diffusion coefficient calculations."""
        # Setup test data
        temperature = np.ones((10, 10)) * 1000.0  # K
        species_concentrations = np.random.rand(10, 10, 5)  # 5 species
        
        # Calculate thermal diffusion coefficients
        therm_coeffs = model.calculate_thermal_diffusion_coefficients(
            temperature, species_concentrations
        )
        
        # Verify coefficient properties
        assert therm_coeffs.shape == (10, 10, 5)
        assert np.all(np.isfinite(therm_coeffs))

class TestPressureDiffusion:
    @pytest.fixture
    def model(self):
        return PressureDiffusion()

    def test_pressure_diffusion_coefficients(self, model):
        """Test pressure diffusion coefficient calculations."""
        # Setup test data
        temperature = np.ones((10, 10)) * 1000.0  # K
        pressure = np.ones((10, 10)) * 101325.0  # Pa
        species_concentrations = np.random.rand(10, 10, 5)  # 5 species
        
        # Calculate pressure diffusion coefficients
        press_coeffs = model.calculate_pressure_diffusion_coefficients(
            temperature, pressure, species_concentrations
        )
        
        # Verify coefficient properties
        assert press_coeffs.shape == (10, 10, 5)
        assert np.all(np.isfinite(press_coeffs))

@pytest.mark.parametrize("model_class", [
    DetailedKinetics,
    ReducedMechanism,
    SkeletalMechanism,
    MultiComponentDiffusion,
    ThermalDiffusion,
    PressureDiffusion
])
def test_model_initialization(model_class):
    """Test proper initialization of all chemical reaction models."""
    model = model_class()
    assert model is not None
    assert hasattr(model, "parameters")
    assert isinstance(model.parameters, dict)

@pytest.mark.parametrize("model_class", [
    DetailedKinetics,
    ReducedMechanism,
    SkeletalMechanism,
    MultiComponentDiffusion,
    ThermalDiffusion,
    PressureDiffusion
])
def test_model_parameter_validation(model_class):
    """Test parameter validation for all chemical reaction models."""
    model = model_class()
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        model.set_parameters({"invalid_param": -1.0})
    
    # Test with valid parameters
    valid_params = model.get_default_parameters()
    model.set_parameters(valid_params)
    assert model.parameters == valid_params 
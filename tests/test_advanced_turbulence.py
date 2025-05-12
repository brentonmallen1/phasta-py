import pytest
import numpy as np
from phasta.models.turbulence import (
    DynamicSubgridModel,
    HybridRANSLES,
    WallRoughnessModel
)

class TestDynamicSubgridModel:
    @pytest.fixture
    def model(self):
        return DynamicSubgridModel()

    def test_dynamic_coefficient_calculation(self, model):
        """Test dynamic coefficient calculation for LES models."""
        # Setup test data
        velocity_gradient = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        filter_width = 0.1
        
        # Calculate coefficient
        coefficient = model.calculate_dynamic_coefficient(
            velocity_gradient, filter_width
        )
        
        # Verify coefficient properties
        assert isinstance(coefficient, float)
        assert coefficient > 0
        assert coefficient < 1.0  # Typical range for Smagorinsky coefficient

    def test_test_filtering(self, model):
        """Test the test filtering operation."""
        # Setup test data
        field = np.random.rand(10, 10, 10)
        filter_width = 0.1
        
        # Apply test filter
        filtered_field = model.apply_test_filter(field, filter_width)
        
        # Verify filtering properties
        assert filtered_field.shape == field.shape
        assert np.all(np.isfinite(filtered_field))
        assert np.max(np.abs(filtered_field)) <= np.max(np.abs(field))

    def test_mixed_dynamic_model(self, model):
        """Test the mixed dynamic model implementation."""
        # Setup test data
        velocity = np.random.rand(10, 10, 10, 3)
        filter_width = 0.1
        
        # Calculate subgrid stress
        subgrid_stress = model.calculate_subgrid_stress(velocity, filter_width)
        
        # Verify subgrid stress properties
        assert subgrid_stress.shape == (10, 10, 10, 3, 3)
        assert np.all(np.isfinite(subgrid_stress))
        assert np.all(np.abs(subgrid_stress) >= 0)

class TestHybridRANSLES:
    @pytest.fixture
    def model(self):
        return HybridRANSLES()

    def test_rans_les_transition(self, model):
        """Test the RANS-LES transition mechanism."""
        # Setup test data
        wall_distance = np.linspace(0, 1, 100)
        velocity = np.random.rand(100, 3)
        
        # Calculate blending function
        blending = model.calculate_blending_function(wall_distance, velocity)
        
        # Verify blending properties
        assert blending.shape == (100,)
        assert np.all(blending >= 0)
        assert np.all(blending <= 1)
        assert np.all(np.isfinite(blending))

    def test_wall_modeled_les(self, model):
        """Test wall-modeled LES implementation."""
        # Setup test data
        wall_distance = np.linspace(0, 1, 100)
        velocity = np.random.rand(100, 3)
        wall_shear = np.array([1.0, 0.0, 0.0])
        
        # Calculate wall model
        wall_model = model.calculate_wall_model(
            wall_distance, velocity, wall_shear
        )
        
        # Verify wall model properties
        assert wall_model.shape == (100, 3)
        assert np.all(np.isfinite(wall_model))
        assert np.all(np.abs(wall_model) >= 0)

class TestWallRoughnessModel:
    @pytest.fixture
    def model(self):
        return WallRoughnessModel()

    def test_roughness_effects(self, model):
        """Test wall roughness effects on flow."""
        # Setup test data
        wall_distance = np.linspace(0, 1, 100)
        velocity = np.random.rand(100, 3)
        roughness_height = 0.001
        
        # Calculate roughness effects
        modified_velocity = model.apply_roughness_effects(
            wall_distance, velocity, roughness_height
        )
        
        # Verify modified velocity properties
        assert modified_velocity.shape == velocity.shape
        assert np.all(np.isfinite(modified_velocity))
        assert np.all(np.abs(modified_velocity) >= 0)

    def test_roughness_length_scale(self, model):
        """Test roughness length scale calculation."""
        # Setup test data
        roughness_height = 0.001
        roughness_type = "sand"
        
        # Calculate length scale
        length_scale = model.calculate_roughness_length_scale(
            roughness_height, roughness_type
        )
        
        # Verify length scale properties
        assert isinstance(length_scale, float)
        assert length_scale > 0
        assert length_scale < roughness_height

@pytest.mark.parametrize("model_class", [
    DynamicSubgridModel,
    HybridRANSLES,
    WallRoughnessModel
])
def test_model_initialization(model_class):
    """Test proper initialization of all turbulence models."""
    model = model_class()
    assert model is not None
    assert hasattr(model, "parameters")
    assert isinstance(model.parameters, dict)

@pytest.mark.parametrize("model_class", [
    DynamicSubgridModel,
    HybridRANSLES,
    WallRoughnessModel
])
def test_model_parameter_validation(model_class):
    """Test parameter validation for all turbulence models."""
    model = model_class()
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        model.set_parameters({"invalid_param": -1.0})
    
    # Test with valid parameters
    valid_params = model.get_default_parameters()
    model.set_parameters(valid_params)
    assert model.parameters == valid_params 
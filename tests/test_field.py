"""Tests for the Field class."""

import numpy as np
import pytest
from phasta.core.field import Field


def test_field_initialization():
    """Test basic field initialization."""
    # Create a simple scalar field
    mesh_size = 10
    data = np.random.rand(mesh_size, 1)
    
    field = Field("temperature", data, mesh_size)
    
    assert field.name == "temperature"
    assert field.n_components == 1
    assert field.data.shape == (mesh_size, 1)
    assert np.allclose(field.data, data)
    
    # Create a vector field
    data = np.random.rand(mesh_size, 3)
    field = Field("velocity", data, mesh_size, n_components=3)
    
    assert field.name == "velocity"
    assert field.n_components == 3
    assert field.data.shape == (mesh_size, 3)
    assert np.allclose(field.data, data)


def test_field_validation():
    """Test field validation."""
    mesh_size = 10
    
    # Test invalid data type
    with pytest.raises(TypeError):
        Field("temperature", "invalid", mesh_size)
    
    # Test invalid data shape
    with pytest.raises(ValueError):
        Field("temperature", np.random.rand(mesh_size, 2), mesh_size, n_components=3)
    
    # Test invalid mesh size
    with pytest.raises(ValueError):
        Field("temperature", np.random.rand(5, 1), mesh_size)


def test_field_operations():
    """Test field operations."""
    mesh_size = 10
    data = np.random.rand(mesh_size, 1)
    field = Field("temperature", data, mesh_size)
    
    # Test indexing
    assert np.allclose(field[0], data[0])
    assert np.allclose(field[1:3], data[1:3])
    
    # Test setting values
    new_value = np.array([[1.0]])
    field[0] = new_value
    assert np.allclose(field[0], new_value)
    
    # Test copy
    field_copy = field.copy()
    assert field_copy.name == field.name
    assert field_copy.n_components == field.n_components
    assert np.allclose(field_copy.data, field.data)
    assert field_copy.data is not field.data  # Ensure deep copy


def test_field_statistics():
    """Test field statistics methods."""
    mesh_size = 10
    data = np.random.rand(mesh_size, 1)
    field = Field("temperature", data, mesh_size)
    
    assert np.isclose(field.min(), np.min(data))
    assert np.isclose(field.max(), np.max(data))
    assert np.isclose(field.mean(), np.mean(data))
    assert np.isclose(field.norm(), np.linalg.norm(data))


def test_field_not_implemented():
    """Test that certain operations are not yet implemented."""
    mesh_size = 10
    data = np.random.rand(mesh_size, 1)
    field = Field("temperature", data, mesh_size)
    
    # Test unimplemented methods
    with pytest.raises(NotImplementedError):
        field.interpolate(np.random.rand(5, 3))
    
    with pytest.raises(NotImplementedError):
        field.gradient()
    
    with pytest.raises(NotImplementedError):
        field.divergence()
    
    with pytest.raises(NotImplementedError):
        field.curl()
    
    with pytest.raises(NotImplementedError):
        field.laplacian()
    
    with pytest.raises(NotImplementedError):
        field.boundary_average() 
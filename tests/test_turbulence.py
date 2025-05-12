"""Tests for turbulence models."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from phasta.solver.turbulence import (
    TurbulenceModel, RANSModel, LESModel,
    HybridRANSLES, DynamicSubgridModel
)
from phasta.mesh.base import Mesh


class MockMesh:
    """Mock mesh for testing."""
    
    def __init__(self, num_nodes: int = 100, num_elements: int = 200):
        """Initialize mock mesh.
        
        Args:
            num_nodes: Number of nodes
            num_elements: Number of elements
        """
        self.nodes = np.random.rand(num_nodes, 3)
        self.elements = np.random.randint(0, num_nodes, (num_elements, 4))


def test_turbulence_model_base():
    """Test base turbulence model class."""
    mesh = MockMesh()
    model = TurbulenceModel(mesh)
    
    with pytest.raises(NotImplementedError):
        model.compute_eddy_viscosity(
            np.zeros((100, 3)),
            np.zeros((100, 3, 3))
        )
    
    with pytest.raises(NotImplementedError):
        model.compute_source_terms(
            np.zeros((100, 3)),
            np.zeros((100, 3, 3))
        )


def test_rans_model():
    """Test RANS turbulence model."""
    # Create a simple mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ])
    elements = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ])
    mesh = Mesh(nodes, elements)
    
    # Create model
    model = RANSModel(mesh, dt=0.001)
    
    # Test eddy viscosity computation
    velocity = np.zeros((4, 3))
    velocity_gradient = np.zeros((4, 3, 3))
    eddy_viscosity = model.compute_eddy_viscosity(velocity, velocity_gradient)
    assert eddy_viscosity.shape == (4,)
    assert np.all(eddy_viscosity >= 0)
    
    # Test source terms computation
    source_terms = model.compute_source_terms(velocity, velocity_gradient)
    assert 'k' in source_terms
    assert 'epsilon' in source_terms
    assert source_terms['k'].shape == (4,)
    assert source_terms['epsilon'].shape == (4,)


def test_les_model():
    """Test LES model."""
    # Create model
    model = LESModel(model_type="smagorinsky")
    
    # Test data
    velocity = np.array([1.0, 0.0, 0.0])
    velocity_grad = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1]
    ])
    wall_distance = np.array([0.1])
    delta = 0.1
    
    # Test eddy viscosity computation
    nu_t = model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)
    
    # Test dynamic model
    model = LESModel(model_type="dynamic")
    nu_t = model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)
    
    # Test wall-adapted model
    model = LESModel(model_type="wall_adapted")
    nu_t = model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)
    
    # Test invalid model type
    with pytest.raises(ValueError):
        model = LESModel(model_type="invalid")


def test_hybrid_rans_les():
    """Test hybrid RANS/LES model."""
    # Create model
    model = HybridRANSLES(model_type="detached_eddy")
    
    # Test data
    velocity = np.array([1.0, 0.0, 0.0])
    velocity_grad = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1]
    ])
    wall_distance = np.array([0.1])
    delta = 0.1
    
    # Test eddy viscosity computation
    nu_t = model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)
    
    # Test wall-modeled LES
    model = HybridRANSLES(model_type="wall_modeled")
    nu_t = model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)
    
    # Test invalid model type
    with pytest.raises(ValueError):
        model = HybridRANSLES(model_type="invalid")


def test_dynamic_subgrid_model():
    """Test dynamic subgrid model."""
    # Create model
    model = DynamicSubgridModel(model_type="smagorinsky")
    
    # Test data
    velocity = np.array([1.0, 0.0, 0.0])
    velocity_grad = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1]
    ])
    wall_distance = np.array([0.1])
    delta = 0.1
    
    # Test eddy viscosity computation
    nu_t = model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)
    
    # Test mixed model
    model = DynamicSubgridModel(model_type="mixed")
    nu_t = model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)
    
    # Test invalid model type
    with pytest.raises(ValueError):
        model = DynamicSubgridModel(model_type="invalid")


def test_edge_cases():
    """Test edge cases."""
    # Create models
    les_model = LESModel()
    hybrid_model = HybridRANSLES()
    subgrid_model = DynamicSubgridModel()
    
    # Test data
    velocity = np.array([0.0, 0.0, 0.0])
    velocity_grad = np.zeros((3, 3))
    wall_distance = np.array([0.0])
    delta = 0.0
    
    # Test zero velocity
    nu_t = les_model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t == 0.0)
    
    nu_t = hybrid_model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t == 0.0)
    
    nu_t = subgrid_model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t == 0.0)
    
    # Test zero wall distance
    wall_distance = np.array([0.0])
    delta = 0.1
    
    nu_t = les_model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)
    
    nu_t = hybrid_model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)
    
    nu_t = subgrid_model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)


def test_memory_management():
    """Test memory management."""
    # Create models
    les_model = LESModel()
    hybrid_model = HybridRANSLES()
    subgrid_model = DynamicSubgridModel()
    
    # Test data
    n_points = 1000
    velocity = np.zeros((n_points, 3))
    velocity_grad = np.zeros((n_points, 3, 3))
    wall_distance = np.ones(n_points)
    delta = 0.1
    
    # Test large arrays
    nu_t = les_model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)
    
    nu_t = hybrid_model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)
    
    nu_t = subgrid_model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
    assert nu_t.shape == wall_distance.shape
    assert np.all(nu_t >= 0.0)


def test_convergence():
    """Test convergence with grid refinement."""
    # Create model
    model = LESModel(model_type="dynamic")
    
    # Test data
    velocity = np.array([1.0, 0.0, 0.0])
    velocity_grad = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1]
    ])
    wall_distance = np.array([0.1])
    
    # Test different filter widths
    deltas = [0.1, 0.05, 0.025, 0.0125]
    nu_t_values = []
    
    for delta in deltas:
        nu_t = model.compute_eddy_viscosity(velocity, velocity_grad, wall_distance, delta)
        nu_t_values.append(nu_t[0])
    
    # Check convergence
    for i in range(len(nu_t_values) - 1):
        ratio = nu_t_values[i] / nu_t_values[i+1]
        assert 1.5 <= ratio <= 4.0  # Should scale roughly with delta^2 
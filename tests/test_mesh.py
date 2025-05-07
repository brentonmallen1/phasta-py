"""Tests for the Mesh class."""

import numpy as np
import pytest
from phasta.core.mesh import Mesh


def test_mesh_initialization():
    """Test basic mesh initialization."""
    # Create a simple 2D mesh
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    cells = {
        "quad": np.array([[0, 1, 2, 3]])
    }
    
    # Create mesh
    mesh = Mesh(points, cells)
    
    # Check basic properties
    assert mesh.points.shape == (4, 3)
    assert "quad" in mesh.cells
    assert mesh.cells["quad"].shape == (1, 4)
    
    # Check boundary conditions
    assert isinstance(mesh.boundary_conditions, dict)
    assert len(mesh.boundary_conditions) == 0


def test_mesh_validation():
    """Test mesh validation."""
    # Test invalid points
    with pytest.raises(TypeError):
        Mesh("invalid", {"quad": np.array([[0, 1, 2, 3]])})
    
    with pytest.raises(ValueError):
        Mesh(np.array([1, 2, 3]), {"quad": np.array([[0, 1, 2, 3]])})
    
    # Test invalid cells
    with pytest.raises(TypeError):
        Mesh(np.zeros((4, 3)), "invalid")
    
    with pytest.raises(TypeError):
        Mesh(np.zeros((4, 3)), {"quad": "invalid"})
    
    with pytest.raises(ValueError):
        Mesh(np.zeros((4, 3)), {"quad": np.array([0, 1, 2, 3])})


def test_mesh_boundary_conditions():
    """Test mesh boundary conditions."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    cells = {
        "quad": np.array([[0, 1, 2, 3]])
    }
    
    boundary_conditions = {
        "wall": [0, 1],
        "inlet": [1, 2],
        "outlet": [2, 3],
        "symmetry": [3, 0]
    }
    
    mesh = Mesh(points, cells, boundary_conditions)
    
    assert len(mesh.boundary_conditions) == 4
    assert "wall" in mesh.boundary_conditions
    assert "inlet" in mesh.boundary_conditions
    assert "outlet" in mesh.boundary_conditions
    assert "symmetry" in mesh.boundary_conditions
    assert len(mesh.boundary_conditions["wall"]) == 2 
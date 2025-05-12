"""Tests for mesh quality improvement module."""

import numpy as np
import pytest
from phasta.mesh.quality import (
    MeshSmoother,
    LaplacianSmoother,
    OptimizedSmoother,
    MeshValidator
)


class MockMesh:
    """Mock mesh class for testing."""
    
    def __init__(self, nodes, elements):
        """Initialize mock mesh.
        
        Args:
            nodes: Node coordinates
            elements: Element connectivity
        """
        self.nodes = nodes
        self.elements = elements
        self.dim = nodes.shape[1]


def test_mesh_smoother():
    """Test mesh smoother base class."""
    # Create mock mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1]
    ])
    elements = np.array([
        [0, 1, 2], [1, 3, 2]
    ])
    mesh = MockMesh(nodes, elements)
    
    # Test base class
    smoother = MeshSmoother(mesh)
    with pytest.raises(NotImplementedError):
        smoother.smooth()


def test_laplacian_smoother():
    """Test Laplacian smoothing."""
    # Create mock mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0.5, 0.5]  # Interior node
    ])
    elements = np.array([
        [0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]
    ])
    mesh = MockMesh(nodes, elements)
    
    # Create smoother
    smoother = LaplacianSmoother(mesh, max_iterations=10)
    
    # Smooth mesh
    converged = smoother.smooth()
    
    # Check results
    assert converged
    assert not np.array_equal(mesh.nodes, nodes)  # Nodes should move
    assert np.all(np.isfinite(mesh.nodes))  # No NaN or inf values


def test_optimized_smoother():
    """Test optimization-based smoothing."""
    # Create mock mesh with poor quality elements
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0.9, 0.1]  # Interior node close to edge
    ])
    elements = np.array([
        [0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]
    ])
    mesh = MockMesh(nodes, elements)
    
    # Create smoother
    smoother = OptimizedSmoother(mesh, quality_threshold=0.5)
    
    # Smooth mesh
    converged = smoother.smooth()
    
    # Check results
    assert converged
    assert not np.array_equal(mesh.nodes, nodes)  # Nodes should move
    assert np.all(np.isfinite(mesh.nodes))  # No NaN or inf values


def test_mesh_validator():
    """Test mesh validation."""
    # Create mock mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0.5, 0.5]  # Interior node
    ])
    elements = np.array([
        [0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]
    ])
    mesh = MockMesh(nodes, elements)
    
    # Create validator
    validator = MeshValidator(mesh)
    
    # Validate mesh
    is_valid, issues = validator.validate()
    
    # Check results
    assert is_valid
    assert not any(issues.values())
    
    # Test with invalid mesh
    nodes_invalid = nodes.copy()
    nodes_invalid[4] = [0.9, 0.1]  # Move interior node to create poor quality
    mesh_invalid = MockMesh(nodes_invalid, elements)
    validator_invalid = MeshValidator(mesh_invalid)
    is_valid, issues = validator_invalid.validate()
    
    # Check results
    assert not is_valid
    assert any(issues.values())


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    # Create mock mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1]
    ])
    elements = np.array([
        [0, 1, 2], [1, 3, 2]
    ])
    mesh = MockMesh(nodes, elements)
    
    # Test invalid max iterations
    with pytest.raises(ValueError):
        LaplacianSmoother(mesh, max_iterations=0)
    
    # Test invalid tolerance
    with pytest.raises(ValueError):
        LaplacianSmoother(mesh, tolerance=-1.0)
    
    # Test invalid quality threshold
    with pytest.raises(ValueError):
        OptimizedSmoother(mesh, quality_threshold=0.0)
    
    # Test invalid angle threshold
    with pytest.raises(ValueError):
        MeshValidator(mesh, max_angle=0.0) 
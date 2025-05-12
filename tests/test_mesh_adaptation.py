"""Tests for mesh adaptation module."""

import numpy as np
import pytest
from phasta.mesh.adaptation import (
    MeshAdapter,
    ErrorBasedAdapter,
    FeatureBasedAdapter,
    SolutionBasedAdapter
)


# TODO: Additional test cases to be added:
# - Test parallel adaptation with MPI
# - Test adaptation with multiple solution fields
# - Test adaptation with different element types (quad, hex)
# - Test adaptation with hanging nodes
# - Test adaptation with curved elements
# - Test adaptation with periodic boundaries
# - Test adaptation with moving boundaries
# - Test adaptation with different refinement strategies
# - Test adaptation with different error estimators
# - Test adaptation with different feature detection methods
# - Test adaptation with different solution analysis methods
# - Test adaptation with different convergence criteria
# - Test adaptation with different quality metrics
# - Test adaptation with different optimization strategies
# - Test adaptation with different parallelization strategies


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


def test_mesh_adapter():
    """Test mesh adapter base class."""
    # Create mock mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1]
    ])
    elements = np.array([
        [0, 1, 2], [1, 3, 2]
    ])
    mesh = MockMesh(nodes, elements)
    
    # Test base class
    adapter = MeshAdapter(mesh)
    with pytest.raises(NotImplementedError):
        adapter.adapt()


def test_error_based_adapter():
    """Test error-based adaptation."""
    # Create mock mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0.5, 0.5]  # Interior node
    ])
    elements = np.array([
        [0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]
    ])
    mesh = MockMesh(nodes, elements)
    
    # Create solution with high gradient
    solution = np.array([0, 1, 0, 1, 0.5])
    
    # Create adapter
    adapter = ErrorBasedAdapter(mesh, solution, max_iterations=5)
    
    # Adapt mesh
    converged = adapter.adapt()
    
    # Check results
    assert converged
    assert len(mesh.nodes) > len(nodes)  # Should add new nodes
    assert len(mesh.elements) > len(elements)  # Should add new elements


def test_feature_based_adapter():
    """Test feature-based adaptation."""
    # Create mock mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0.5, 0.5]  # Interior node
    ])
    elements = np.array([
        [0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]
    ])
    mesh = MockMesh(nodes, elements)
    
    # Create solution with feature
    solution = np.array([0, 1, 0, 1, 0.5])
    
    # Create adapter
    adapter = FeatureBasedAdapter(mesh, solution, feature_threshold=0.5)
    
    # Adapt mesh
    converged = adapter.adapt()
    
    # Check results
    assert converged
    assert len(mesh.nodes) > len(nodes)  # Should add new nodes
    assert len(mesh.elements) > len(elements)  # Should add new elements


def test_solution_based_adapter():
    """Test solution-based adaptation."""
    # Create mock mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0.5, 0.5]  # Interior node
    ])
    elements = np.array([
        [0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]
    ])
    mesh = MockMesh(nodes, elements)
    
    # Create solution with variation
    solution = np.array([0, 1, 0, 1, 0.5])
    
    # Create adapter
    adapter = SolutionBasedAdapter(mesh, solution, solution_threshold=0.5)
    
    # Adapt mesh
    converged = adapter.adapt()
    
    # Check results
    assert converged
    assert len(mesh.nodes) > len(nodes)  # Should add new nodes
    assert len(mesh.elements) > len(elements)  # Should add new elements


def test_3d_adaptation():
    """Test adaptation in 3D."""
    # Create mock 3D mesh
    nodes = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        [0.5, 0.5, 0.5]  # Interior node
    ])
    elements = np.array([
        [0, 1, 2, 8], [1, 3, 2, 8], [3, 7, 2, 8],
        [7, 6, 2, 8], [6, 0, 2, 8], [0, 4, 1, 8],
        [1, 5, 3, 8], [3, 7, 5, 8], [7, 6, 4, 8],
        [6, 0, 4, 8], [4, 5, 1, 8], [5, 7, 3, 8]
    ])
    mesh = MockMesh(nodes, elements)
    
    # Create solution with variation
    solution = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0.5])
    
    # Test all adapters
    adapters = [
        ErrorBasedAdapter(mesh, solution),
        FeatureBasedAdapter(mesh, solution),
        SolutionBasedAdapter(mesh, solution)
    ]
    
    for adapter in adapters:
        # Adapt mesh
        converged = adapter.adapt()
        
        # Check results
        assert converged
        assert len(mesh.nodes) > len(nodes)  # Should add new nodes
        assert len(mesh.elements) > len(elements)  # Should add new elements


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
    
    # Create solution
    solution = np.array([0, 1, 0, 1])
    
    # Test invalid max iterations
    with pytest.raises(ValueError):
        ErrorBasedAdapter(mesh, solution, max_iterations=0)
    
    # Test invalid target error
    with pytest.raises(ValueError):
        ErrorBasedAdapter(mesh, solution, target_error=-0.1)
    
    # Test invalid feature threshold
    with pytest.raises(ValueError):
        FeatureBasedAdapter(mesh, solution, feature_threshold=0.0)
    
    # Test invalid solution threshold
    with pytest.raises(ValueError):
        SolutionBasedAdapter(mesh, solution, solution_threshold=0.0)
    
    # Test invalid solution size
    with pytest.raises(ValueError):
        ErrorBasedAdapter(mesh, solution[:2]) 
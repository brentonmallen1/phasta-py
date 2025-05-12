"""Tests for boundary layer meshing module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from phasta.mesh.boundary_layer import (
    BoundaryLayerGenerator,
    StructuredBoundaryLayer,
    UnstructuredBoundaryLayer,
    BoundaryLayerQuality,
    WallDistanceCalculator, FastMarchingWallDistance,
    LayerGenerator, StructuredLayerGenerator,
    QualityController, BoundaryLayerQualityController,
    BoundaryLayerMeshGenerator
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


def test_boundary_layer_generator():
    """Test base boundary layer generator."""
    # Test invalid growth rate
    with pytest.raises(ValueError):
        BoundaryLayerGenerator(first_height=0.1, growth_rate=0.5, n_layers=5)
    
    # Test invalid number of layers
    with pytest.raises(ValueError):
        BoundaryLayerGenerator(first_height=0.1, growth_rate=1.2, n_layers=0)
    
    # Test layer height computation
    generator = BoundaryLayerGenerator(first_height=0.1, growth_rate=1.2, n_layers=3)
    heights = generator._compute_layer_heights()
    assert len(heights) == 3
    assert np.allclose(heights[0], 0.1)
    assert np.allclose(heights[1], 0.12)
    assert np.allclose(heights[2], 0.144)


def test_structured_boundary_layer():
    """Test structured boundary layer generation."""
    # Create surface nodes and normals
    surface_nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ])
    surface_normals = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ])
    
    # Create generator
    generator = StructuredBoundaryLayer(
        first_height=0.1,
        growth_rate=1.2,
        n_layers=3,
        surface_nodes=surface_nodes,
        surface_normals=surface_normals
    )
    
    # Generate mesh
    nodes, elements = generator.generate()
    
    # Check nodes
    assert nodes.shape[0] == 16  # 4 surface nodes * 4 layers
    assert nodes.shape[1] == 3
    assert np.allclose(nodes[:4], surface_nodes)  # First layer matches surface
    
    # Check elements
    assert elements.shape[1] == 3  # Triangular elements
    assert len(elements) == 18  # 3 layers * 2 triangles per cell * 3 cells


def test_unstructured_boundary_layer():
    """Test unstructured boundary layer generation."""
    # Create surface nodes and normals
    surface_nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ])
    surface_normals = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ])
    
    # Create generator
    generator = UnstructuredBoundaryLayer(
        first_height=0.1,
        growth_rate=1.2,
        n_layers=3,
        surface_nodes=surface_nodes,
        surface_normals=surface_normals,
        max_angle=30.0
    )
    
    # Generate mesh
    nodes, elements = generator.generate()
    
    # Check nodes
    assert nodes.shape[1] == 3
    assert np.allclose(nodes[:4], surface_nodes)  # First layer matches surface
    
    # Check elements
    assert elements.shape[1] == 3  # Triangular elements
    assert len(elements) > 0  # Should have some elements


def test_boundary_layer_quality():
    """Test boundary layer quality metrics."""
    # Create a simple mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0.1],
        [1, 0, 0.1],
        [0, 1, 0.1]
    ])
    elements = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [0, 1, 3],
        [1, 4, 3],
        [1, 2, 4],
        [2, 5, 4],
        [0, 2, 3],
        [2, 5, 3]
    ])
    surface_nodes = nodes[:3]
    surface_normals = np.array([[0, 0, 1]] * 3)
    
    # Create quality checker
    quality = BoundaryLayerQuality(nodes, elements, surface_nodes, surface_normals)
    
    # Check quality metrics
    metrics = quality.check_quality()
    assert 'aspect_ratio' in metrics
    assert 'orthogonality' in metrics
    assert 'smoothness' in metrics
    
    # Check metric ranges
    assert 0 <= metrics['aspect_ratio'] <= 10
    assert 0 <= metrics['orthogonality'] <= 1
    assert 0 <= metrics['smoothness'] <= 1


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    # Test mismatched nodes and normals
    surface_nodes = np.array([[0, 0, 0], [1, 0, 0]])
    surface_normals = np.array([[0, 0, 1]])
    
    with pytest.raises(ValueError):
        StructuredBoundaryLayer(
            first_height=0.1,
            growth_rate=1.2,
            n_layers=3,
            surface_nodes=surface_nodes,
            surface_normals=surface_normals
        )
    
    # Test invalid max_angle
    with pytest.raises(ValueError):
        UnstructuredBoundaryLayer(
            first_height=0.1,
            growth_rate=1.2,
            n_layers=3,
            surface_nodes=surface_nodes,
            surface_normals=surface_normals,
            max_angle=-1.0
        )


def test_wall_distance_calculator_base():
    """Test base wall distance calculator class."""
    calculator = WallDistanceCalculator()
    
    with pytest.raises(NotImplementedError):
        calculator.calculate_distances(MockMesh(), [0, 1, 2])


def test_fast_marching_wall_distance():
    """Test fast marching wall distance calculation."""
    calculator = FastMarchingWallDistance()
    mesh = MockMesh()
    wall_faces = [0, 1, 2]
    
    # Test distance calculation
    distances = calculator.calculate_distances(mesh, wall_faces)
    
    assert distances.shape == (len(mesh.nodes),)
    assert np.all(distances >= 0)
    assert np.all(distances[wall_faces] == 0)


def test_layer_generator_base():
    """Test base layer generator class."""
    generator = LayerGenerator()
    
    with pytest.raises(NotImplementedError):
        generator.generate_layers(MockMesh(), np.zeros(100), 5, 1.2)


def test_structured_layer_generator():
    """Test structured layer generator."""
    generator = StructuredLayerGenerator()
    mesh = MockMesh()
    distances = np.zeros(len(mesh.nodes))
    
    # Test layer generation
    layered_mesh = generator.generate_layers(mesh, distances, 5, 1.2)
    
    assert isinstance(layered_mesh, Mesh)
    assert layered_mesh.nodes is not None
    assert layered_mesh.elements is not None


def test_quality_controller_base():
    """Test base quality controller class."""
    controller = QualityController()
    
    with pytest.raises(NotImplementedError):
        controller.check_quality(MockMesh())


def test_boundary_layer_quality_controller():
    """Test boundary layer quality controller."""
    controller = BoundaryLayerQualityController()
    mesh = MockMesh()
    
    # Test quality check
    quality_ok, metrics = controller.check_quality(mesh)
    
    assert isinstance(quality_ok, bool)
    assert isinstance(metrics, dict)
    assert 'min_angle' in metrics
    assert 'max_skewness' in metrics
    assert 'min_orthogonality' in metrics


def test_boundary_layer_mesh_generator():
    """Test boundary layer mesh generator."""
    # Create components
    distance_calculator = FastMarchingWallDistance()
    layer_generator = StructuredLayerGenerator()
    quality_controller = BoundaryLayerQualityController()
    
    # Create generator
    generator = BoundaryLayerMeshGenerator(
        distance_calculator,
        layer_generator,
        quality_controller
    )
    
    # Test mesh generation
    mesh = MockMesh()
    wall_faces = [0, 1, 2]
    layered_mesh = generator.generate_mesh(
        mesh, wall_faces, num_layers=5, growth_ratio=1.2)
    
    assert isinstance(layered_mesh, Mesh)
    assert layered_mesh.nodes is not None
    assert layered_mesh.elements is not None


def test_gpu_accelerated_generation():
    """Test GPU-accelerated boundary layer generation."""
    # Create components
    distance_calculator = FastMarchingWallDistance()
    layer_generator = StructuredLayerGenerator()
    quality_controller = BoundaryLayerQualityController()
    
    # Mock GPU device
    gpu_device = Mock()
    gpu_device.allocate_memory.return_value = 1
    gpu_device.copy_to_device.return_value = None
    gpu_device.copy_from_device.return_value = np.random.rand(100, 3)
    gpu_device.free_memory.return_value = None
    
    # Create generator with GPU device
    generator = BoundaryLayerMeshGenerator(
        distance_calculator,
        layer_generator,
        quality_controller,
        gpu_device=gpu_device
    )
    
    # Test GPU-accelerated generation
    mesh = MockMesh()
    wall_faces = [0, 1, 2]
    layered_mesh = generator.generate_mesh(
        mesh, wall_faces, num_layers=5, growth_ratio=1.2)
    
    assert isinstance(layered_mesh, Mesh)
    assert layered_mesh.nodes is not None
    assert layered_mesh.elements is not None
    
    # Verify GPU operations
    assert gpu_device.allocate_memory.called
    assert gpu_device.copy_to_device.called
    assert gpu_device.copy_from_device.called
    assert gpu_device.free_memory.called


def test_layer_heights():
    """Test layer height calculation."""
    generator = StructuredLayerGenerator()
    mesh = MockMesh()
    distances = np.zeros(len(mesh.nodes))
    
    # Test with different growth ratios
    layered_mesh = generator.generate_layers(mesh, distances, 5, 1.2)
    assert isinstance(layered_mesh, Mesh)
    
    layered_mesh = generator.generate_layers(mesh, distances, 5, 1.5)
    assert isinstance(layered_mesh, Mesh)


def test_quality_improvement():
    """Test quality improvement iterations."""
    # Create components
    distance_calculator = FastMarchingWallDistance()
    layer_generator = StructuredLayerGenerator()
    quality_controller = BoundaryLayerQualityController()
    
    # Create generator
    generator = BoundaryLayerMeshGenerator(
        distance_calculator,
        layer_generator,
        quality_controller
    )
    
    # Test with different iteration limits
    mesh = MockMesh()
    wall_faces = [0, 1, 2]
    
    # Test with max_iterations=1
    layered_mesh = generator.generate_mesh(
        mesh, wall_faces, max_iterations=1)
    assert isinstance(layered_mesh, Mesh)
    
    # Test with max_iterations=5
    layered_mesh = generator.generate_mesh(
        mesh, wall_faces, max_iterations=5)
    assert isinstance(layered_mesh, Mesh)


def test_memory_management():
    """Test memory management during generation."""
    # Create components
    distance_calculator = FastMarchingWallDistance()
    layer_generator = StructuredLayerGenerator()
    quality_controller = BoundaryLayerQualityController()
    
    # Create generator
    generator = BoundaryLayerMeshGenerator(
        distance_calculator,
        layer_generator,
        quality_controller
    )
    
    # Test with large mesh
    large_mesh = MockMesh(num_nodes=10000, num_elements=20000)
    wall_faces = list(range(100))  # More wall faces for large mesh
    layered_mesh = generator.generate_mesh(large_mesh, wall_faces)
    
    assert isinstance(layered_mesh, Mesh)
    assert layered_mesh.nodes is not None
    assert layered_mesh.elements is not None


def test_quality_metrics():
    """Test quality metric calculations."""
    controller = BoundaryLayerQualityController()
    mesh = MockMesh()
    
    # Test quality metrics
    quality_ok, metrics = controller.check_quality(mesh)
    
    # Verify metric properties
    assert metrics['min_angle'] >= 0
    assert metrics['max_skewness'] >= 0
    assert metrics['min_orthogonality'] >= 0
    assert metrics['max_skewness'] <= 1
    assert metrics['min_orthogonality'] <= 1


def test_wall_distance_accuracy():
    """Test accuracy of wall distance calculation."""
    calculator = FastMarchingWallDistance()
    mesh = MockMesh()
    wall_faces = [0, 1, 2]
    
    # Test distance calculation
    distances = calculator.calculate_distances(mesh, wall_faces)
    
    # Verify distance properties
    assert np.all(distances >= 0)  # Distances should be non-negative
    assert np.all(np.isfinite(distances))  # Distances should be finite
    assert np.all(distances[wall_faces] == 0)  # Wall faces should have zero distance 
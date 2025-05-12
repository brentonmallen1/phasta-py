"""Tests for mesh generation and optimization module."""

import numpy as np
import pytest
from phasta.mesh.generator import (
    MeshGenerator,
    StructuredMeshGenerator,
    UnstructuredMeshGenerator,
    MeshOptimizer
)


def test_structured_mesh_2d():
    """Test 2D structured mesh generation."""
    # Create mesh generator
    bounds = ((0, 1), (0, 1))
    n_points = (3, 3)
    generator = StructuredMeshGenerator(bounds, n_points, dim=2)
    
    # Generate mesh
    nodes, elements = generator.generate()
    
    # Check nodes
    assert nodes.shape == (9, 2)  # 3x3 grid
    assert np.all(nodes >= 0) and np.all(nodes <= 1)
    
    # Check elements
    assert elements.shape == (8, 3)  # 4 cells, 2 triangles each
    assert np.all(elements >= 0) and np.all(elements < len(nodes))


def test_structured_mesh_3d():
    """Test 3D structured mesh generation."""
    # Create mesh generator
    bounds = ((0, 1), (0, 1), (0, 1))
    n_points = (2, 2, 2)
    generator = StructuredMeshGenerator(bounds, n_points, dim=3)
    
    # Generate mesh
    nodes, elements = generator.generate()
    
    # Check nodes
    assert nodes.shape == (8, 3)  # 2x2x2 grid
    assert np.all(nodes >= 0) and np.all(nodes <= 1)
    
    # Check elements
    assert elements.shape == (6, 4)  # 1 cell, 6 tetrahedra
    assert np.all(elements >= 0) and np.all(elements < len(nodes))


def test_unstructured_mesh():
    """Test unstructured mesh generation."""
    # Create point cloud
    points = np.random.rand(10, 2)
    
    # Create mesh generator
    generator = UnstructuredMeshGenerator(points, dim=2)
    
    # Generate mesh
    nodes, elements = generator.generate()
    
    # Check nodes
    assert nodes.shape == points.shape
    assert np.allclose(nodes, points)
    
    # Check elements
    assert elements.shape[1] == 3  # Triangles
    assert np.all(elements >= 0) and np.all(elements < len(nodes))


def test_mesh_optimization():
    """Test mesh optimization."""
    # Create a simple mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1],  # First triangle
        [1, 1], [2, 1], [1, 2]   # Second triangle
    ])
    elements = np.array([
        [0, 1, 2],  # First triangle
        [3, 4, 5]   # Second triangle
    ])
    
    # Create optimizer
    optimizer = MeshOptimizer(nodes, elements)
    
    # Optimize mesh
    optimized_nodes = optimizer.optimize(max_iter=10, quality_threshold=0.1)
    
    # Check optimization
    assert optimized_nodes.shape == nodes.shape
    assert not np.allclose(optimized_nodes, nodes)  # Should have moved nodes
    
    # Check quality improvement
    original_quality = optimizer._compute_mesh_quality(nodes)
    optimized_quality = optimizer._compute_mesh_quality(optimized_nodes)
    assert np.mean(optimized_quality) >= np.mean(original_quality)


def test_triangle_quality():
    """Test triangle quality metric."""
    # Create equilateral triangle
    nodes = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ])
    elements = np.array([[0, 1, 2]])
    
    optimizer = MeshOptimizer(nodes, elements)
    quality = optimizer._compute_element_quality(nodes, elements[0])
    
    # Equilateral triangle should have quality close to 1
    assert np.isclose(quality, 1.0, atol=1e-6)


def test_tetrahedron_quality():
    """Test tetrahedron quality metric."""
    # Create regular tetrahedron
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3)/2, 0],
        [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
    ])
    elements = np.array([[0, 1, 2, 3]])
    
    optimizer = MeshOptimizer(nodes, elements)
    quality = optimizer._compute_element_quality(nodes, elements[0])
    
    # Regular tetrahedron should have quality close to 1
    assert np.isclose(quality, 1.0, atol=1e-6)


def test_invalid_dimension():
    """Test invalid dimension handling."""
    with pytest.raises(ValueError):
        MeshGenerator(dim=4)


def test_invalid_bounds():
    """Test invalid bounds handling."""
    with pytest.raises(ValueError):
        StructuredMeshGenerator(((0, 1),), (3, 3), dim=2)


def test_invalid_points():
    """Test invalid points handling."""
    points = np.random.rand(10, 3)  # 3D points
    with pytest.raises(ValueError):
        UnstructuredMeshGenerator(points, dim=2) 
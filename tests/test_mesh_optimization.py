"""Tests for mesh optimization."""

import numpy as np
import pytest
from phasta.solver.mesh_optimization import (
    MeshQualityMetric,
    AspectRatioMetric,
    SkewnessMetric,
    MeshOptimizer
)


def test_aspect_ratio_metric():
    """Test aspect ratio metric."""
    # Create metric
    metric = AspectRatioMetric()
    
    # Test data
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    elements = np.array([
        [0, 1, 2, 3]  # Tetrahedron
    ])
    
    # Test quality computation
    quality = metric.compute_quality(vertices, elements)
    
    # Check results
    assert quality.shape == (1,)
    assert np.all(quality >= 0.0)
    assert np.all(quality <= 1.0)
    
    # Test gradient computation
    gradients = metric.compute_gradient(vertices, elements)
    
    # Check results
    assert gradients.shape == (1, 4, 3)  # 1 element, 4 vertices, 3 coordinates
    assert np.all(np.isfinite(gradients))


def test_skewness_metric():
    """Test skewness metric."""
    # Create metric
    metric = SkewnessMetric()
    
    # Test data
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    elements = np.array([
        [0, 1, 2, 3]  # Tetrahedron
    ])
    
    # Test quality computation
    quality = metric.compute_quality(vertices, elements)
    
    # Check results
    assert quality.shape == (1,)
    assert np.all(quality >= 0.0)
    assert np.all(quality <= 1.0)
    
    # Test gradient computation
    gradients = metric.compute_gradient(vertices, elements)
    
    # Check results
    assert gradients.shape == (1, 4, 3)  # 1 element, 4 vertices, 3 coordinates
    assert np.all(np.isfinite(gradients))


def test_mesh_optimizer():
    """Test mesh optimizer."""
    # Create metric and optimizer
    metric = AspectRatioMetric()
    optimizer = MeshOptimizer(metric)
    
    # Test data
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    elements = np.array([
        [0, 1, 2, 3]  # Tetrahedron
    ])
    
    # Test optimization
    optimized_vertices, final_quality = optimizer.optimize(
        vertices, elements
    )
    
    # Check results
    assert optimized_vertices.shape == vertices.shape
    assert np.all(np.isfinite(optimized_vertices))
    assert final_quality >= 0.0
    assert final_quality <= 1.0
    
    # Test fixed vertices
    fixed_vertices = [0]  # Fix first vertex
    optimized_vertices, final_quality = optimizer.optimize(
        vertices, elements, fixed_vertices
    )
    
    # Check results
    assert np.allclose(optimized_vertices[0], vertices[0])  # Fixed vertex unchanged


def test_edge_cases():
    """Test edge cases."""
    # Create metric and optimizer
    metric = AspectRatioMetric()
    optimizer = MeshOptimizer(metric)
    
    # Test zero vertices
    vertices = np.zeros((0, 3))
    elements = np.zeros((0, 4), dtype=int)
    
    with pytest.raises(ValueError):
        optimizer.optimize(vertices, elements)
    
    # Test zero elements
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    elements = np.zeros((0, 4), dtype=int)
    
    with pytest.raises(ValueError):
        optimizer.optimize(vertices, elements)
    
    # Test invalid element indices
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    elements = np.array([
        [0, 1, 2, 4]  # Invalid vertex index
    ])
    
    with pytest.raises(IndexError):
        optimizer.optimize(vertices, elements)


def test_memory_management():
    """Test memory management."""
    # Create metric and optimizer
    metric = AspectRatioMetric()
    optimizer = MeshOptimizer(metric)
    
    # Test large mesh
    n_vertices = 1000
    n_elements = 2000
    
    # Generate random mesh
    vertices = np.random.rand(n_vertices, 3)
    elements = np.random.randint(0, n_vertices, (n_elements, 4))
    
    # Test optimization
    optimized_vertices, final_quality = optimizer.optimize(
        vertices, elements
    )
    
    # Check results
    assert optimized_vertices.shape == vertices.shape
    assert np.all(np.isfinite(optimized_vertices))
    assert final_quality >= 0.0
    assert final_quality <= 1.0


def test_convergence():
    """Test convergence with mesh refinement."""
    # Create metric and optimizer
    metric = AspectRatioMetric()
    optimizer = MeshOptimizer(metric)
    
    # Test data
    n_vertices = [10, 20, 40, 80]
    n_elements = [20, 40, 80, 160]
    
    # Test different mesh sizes
    qualities = []
    
    for nv, ne in zip(n_vertices, n_elements):
        # Generate random mesh
        vertices = np.random.rand(nv, 3)
        elements = np.random.randint(0, nv, (ne, 4))
        
        # Optimize mesh
        _, final_quality = optimizer.optimize(vertices, elements)
        qualities.append(final_quality)
    
    # Check convergence
    for i in range(len(qualities) - 1):
        ratio = abs(qualities[i] - qualities[i+1]) / qualities[i+1]
        assert ratio < 0.1  # Should converge with mesh refinement 
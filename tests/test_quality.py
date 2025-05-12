"""Tests for mesh quality and refinement module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from phasta.mesh.quality import (
    QualityMetric, AspectRatioMetric, SkewnessMetric,
    MeshSmoother, MeshCoarsener, AdaptiveRefiner
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


def test_quality_metric_base():
    """Test base quality metric class."""
    metric = QualityMetric()
    
    with pytest.raises(NotImplementedError):
        metric.calculate(MockMesh())


def test_aspect_ratio_metric():
    """Test aspect ratio metric calculation."""
    # Create a simple mesh with known aspect ratios
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [2, 0, 0],
        [0, 2, 0]
    ])
    elements = np.array([
        [0, 1, 2],  # Equilateral triangle
        [0, 3, 4]   # Right triangle
    ])
    mesh = Mesh(nodes, elements)
    
    # Calculate aspect ratio
    metric = AspectRatioMetric()
    aspect_ratio = metric.calculate(mesh)
    
    # Equilateral triangle should have aspect ratio close to 1
    # Right triangle should have larger aspect ratio
    assert aspect_ratio > 1.0
    assert aspect_ratio < 3.0


def test_skewness_metric():
    """Test skewness metric calculation."""
    # Create a simple mesh with known angles
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [2, 0, 0],
        [0, 2, 0]
    ])
    elements = np.array([
        [0, 1, 2],  # Equilateral triangle
        [0, 3, 4]   # Right triangle
    ])
    mesh = Mesh(nodes, elements)
    
    # Calculate skewness
    metric = SkewnessMetric()
    skewness = metric.calculate(mesh)
    
    # Equilateral triangle should have low skewness
    # Right triangle should have higher skewness
    assert skewness > 0.0
    assert skewness < 1.0


def test_mesh_smoother():
    """Test mesh smoothing operations."""
    # Create a simple mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0.5, 0.5, 0.1]  # Node to be smoothed
    ])
    elements = np.array([
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3]
    ])
    mesh = Mesh(nodes, elements)
    
    # Create smoother
    smoother = MeshSmoother(max_iterations=10)
    
    # Smooth mesh
    smoothed_mesh = smoother.smooth(mesh, fixed_nodes=[0, 1, 2])
    
    # Check that smoothed node moved towards center
    assert smoothed_mesh.nodes[3, 2] < 0.1
    assert np.allclose(smoothed_mesh.nodes[3, :2], [0.5, 0.5], atol=0.1)


def test_mesh_coarsener():
    """Test mesh coarsening operations."""
    # Create a simple mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0.5, 0.5, 0]
    ])
    elements = np.array([
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3]
    ])
    mesh = Mesh(nodes, elements)
    
    # Create coarsener
    coarsener = MeshCoarsener(quality_threshold=0.8)
    
    # Coarsen mesh
    coarsened_mesh = coarsener.coarsen(mesh)
    
    # Check that mesh was coarsened
    assert len(coarsened_mesh.nodes) < len(mesh.nodes)
    assert len(coarsened_mesh.elements) < len(mesh.elements)


def test_adaptive_refiner():
    """Test adaptive mesh refinement."""
    # Create a simple mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])
    elements = np.array([
        [0, 1, 2]
    ])
    mesh = Mesh(nodes, elements)
    
    # Create refiner
    refiner = AdaptiveRefiner(error_threshold=0.5)
    
    # Create error indicator
    error_indicator = np.array([0.8])  # High error in the element
    
    # Refine mesh
    refined_mesh = refiner.refine(mesh, error_indicator)
    
    # Check that mesh was refined
    assert len(refined_mesh.nodes) > len(mesh.nodes)
    assert len(refined_mesh.elements) > len(mesh.elements)


def test_quality_improvement():
    """Test quality improvement through smoothing."""
    # Create a mesh with poor quality
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0.9, 0.9, 0.1]  # Node causing poor quality
    ])
    elements = np.array([
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3]
    ])
    mesh = Mesh(nodes, elements)
    
    # Calculate initial quality
    metric = AspectRatioMetric()
    initial_quality = metric.calculate(mesh)
    
    # Smooth mesh
    smoother = MeshSmoother(max_iterations=20)
    smoothed_mesh = smoother.smooth(mesh, fixed_nodes=[0, 1, 2])
    
    # Calculate final quality
    final_quality = metric.calculate(smoothed_mesh)
    
    # Check that quality improved
    assert final_quality < initial_quality


def test_coarsening_quality():
    """Test quality preservation during coarsening."""
    # Create a mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0],
        [0, 0.5, 0]
    ])
    elements = np.array([
        [0, 4, 5],
        [4, 1, 3],
        [5, 3, 2],
        [4, 3, 5]
    ])
    mesh = Mesh(nodes, elements)
    
    # Calculate initial quality
    metric = AspectRatioMetric()
    initial_quality = metric.calculate(mesh)
    
    # Coarsen mesh
    coarsener = MeshCoarsener(quality_threshold=0.5)
    coarsened_mesh = coarsener.coarsen(mesh)
    
    # Calculate final quality
    final_quality = metric.calculate(coarsened_mesh)
    
    # Check that quality is preserved
    assert final_quality >= initial_quality * 0.8


def test_refinement_quality():
    """Test quality preservation during refinement."""
    # Create a simple mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])
    elements = np.array([
        [0, 1, 2]
    ])
    mesh = Mesh(nodes, elements)
    
    # Calculate initial quality
    metric = AspectRatioMetric()
    initial_quality = metric.calculate(mesh)
    
    # Refine mesh
    refiner = AdaptiveRefiner(error_threshold=0.5)
    error_indicator = np.array([0.8])
    refined_mesh = refiner.refine(mesh, error_indicator)
    
    # Calculate final quality
    final_quality = metric.calculate(refined_mesh)
    
    # Check that quality is preserved
    assert final_quality >= initial_quality * 0.8


def test_memory_management():
    """Test memory management during operations."""
    # Create a large mesh
    large_mesh = MockMesh(num_nodes=10000, num_elements=20000)
    
    # Test smoothing
    smoother = MeshSmoother(max_iterations=5)
    smoothed_mesh = smoother.smooth(large_mesh)
    assert isinstance(smoothed_mesh, Mesh)
    
    # Test coarsening
    coarsener = MeshCoarsener(quality_threshold=0.5)
    coarsened_mesh = coarsener.coarsen(large_mesh)
    assert isinstance(coarsened_mesh, Mesh)
    
    # Test refinement
    refiner = AdaptiveRefiner(error_threshold=0.5)
    error_indicator = np.random.rand(len(large_mesh.elements))
    refined_mesh = refiner.refine(large_mesh, error_indicator)
    assert isinstance(refined_mesh, Mesh)


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test empty mesh
    empty_mesh = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    
    # Test smoothing
    smoother = MeshSmoother()
    smoothed_mesh = smoother.smooth(empty_mesh)
    assert len(smoothed_mesh.nodes) == 0
    assert len(smoothed_mesh.elements) == 0
    
    # Test coarsening
    coarsener = MeshCoarsener()
    coarsened_mesh = coarsener.coarsen(empty_mesh)
    assert len(coarsened_mesh.nodes) == 0
    assert len(coarsened_mesh.elements) == 0
    
    # Test refinement
    refiner = AdaptiveRefiner()
    error_indicator = np.zeros(0)
    refined_mesh = refiner.refine(empty_mesh, error_indicator)
    assert len(refined_mesh.nodes) == 0
    assert len(refined_mesh.elements) == 0
    
    # Test invalid error indicator
    mesh = MockMesh()
    with pytest.raises(ValueError):
        refiner.refine(mesh, np.zeros(10))  # Wrong size 
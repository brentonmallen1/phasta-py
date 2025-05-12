"""Tests for adaptive mesh refinement module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from phasta.mesh.adaptive import (
    ErrorEstimator, GradientBasedErrorEstimator,
    RefinementCriterion, ThresholdRefinementCriterion,
    AdaptiveMeshRefiner
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


def test_error_estimator_base():
    """Test base error estimator class."""
    estimator = ErrorEstimator()
    
    with pytest.raises(NotImplementedError):
        estimator.estimate_error(MockMesh())


def test_gradient_based_error_estimator():
    """Test gradient-based error estimator."""
    estimator = GradientBasedErrorEstimator()
    mesh = MockMesh()
    
    # Test error estimation
    errors = estimator.estimate_error(mesh)
    assert errors.shape == (len(mesh.elements),)
    assert np.all(errors >= 0)


def test_refinement_criterion_base():
    """Test base refinement criterion class."""
    criterion = RefinementCriterion()
    
    with pytest.raises(NotImplementedError):
        criterion.should_refine(0.1, np.array([0, 1, 2, 3]))
    
    with pytest.raises(NotImplementedError):
        criterion.should_coarsen(0.1, np.array([0, 1, 2, 3]))


def test_threshold_refinement_criterion():
    """Test threshold-based refinement criterion."""
    criterion = ThresholdRefinementCriterion(
        refine_threshold=0.1,
        coarsen_threshold=0.01
    )
    
    # Test refinement decision
    assert criterion.should_refine(0.2, np.array([0, 1, 2, 3]))
    assert not criterion.should_refine(0.05, np.array([0, 1, 2, 3]))
    
    # Test coarsening decision
    assert criterion.should_coarsen(0.005, np.array([0, 1, 2, 3]))
    assert not criterion.should_coarsen(0.05, np.array([0, 1, 2, 3]))


def test_adaptive_mesh_refiner():
    """Test adaptive mesh refiner."""
    error_estimator = GradientBasedErrorEstimator()
    refinement_criterion = ThresholdRefinementCriterion()
    refiner = AdaptiveMeshRefiner(error_estimator, refinement_criterion)
    
    # Test mesh refinement
    mesh = MockMesh()
    refined_mesh = refiner.refine_mesh(mesh)
    
    assert isinstance(refined_mesh, Mesh)
    assert refined_mesh.nodes is not None
    assert refined_mesh.elements is not None


def test_gpu_accelerated_refinement():
    """Test GPU-accelerated mesh refinement."""
    error_estimator = GradientBasedErrorEstimator()
    refinement_criterion = ThresholdRefinementCriterion()
    
    # Mock GPU device
    gpu_device = Mock()
    gpu_device.allocate_memory.return_value = 1
    gpu_device.copy_to_device.return_value = None
    gpu_device.copy_from_device.return_value = np.random.rand(100, 3)
    gpu_device.free_memory.return_value = None
    
    refiner = AdaptiveMeshRefiner(
        error_estimator,
        refinement_criterion,
        gpu_device=gpu_device
    )
    
    # Test GPU-accelerated refinement
    mesh = MockMesh()
    refined_mesh = refiner.refine_mesh(mesh)
    
    assert isinstance(refined_mesh, Mesh)
    assert refined_mesh.nodes is not None
    assert refined_mesh.elements is not None
    
    # Verify GPU operations
    assert gpu_device.allocate_memory.called
    assert gpu_device.copy_to_device.called
    assert gpu_device.copy_from_device.called
    assert gpu_device.free_memory.called


def test_refinement_iterations():
    """Test refinement iterations."""
    error_estimator = GradientBasedErrorEstimator()
    refinement_criterion = ThresholdRefinementCriterion()
    refiner = AdaptiveMeshRefiner(error_estimator, refinement_criterion)
    
    # Test with different iteration limits
    mesh = MockMesh()
    
    # Test with max_iterations=1
    refined_mesh = refiner.refine_mesh(mesh, max_iterations=1)
    assert isinstance(refined_mesh, Mesh)
    
    # Test with max_iterations=5
    refined_mesh = refiner.refine_mesh(mesh, max_iterations=5)
    assert isinstance(refined_mesh, Mesh)


def test_target_error():
    """Test target error threshold."""
    error_estimator = GradientBasedErrorEstimator()
    refinement_criterion = ThresholdRefinementCriterion()
    refiner = AdaptiveMeshRefiner(error_estimator, refinement_criterion)
    
    # Test with different target errors
    mesh = MockMesh()
    
    # Test with high target error
    refined_mesh = refiner.refine_mesh(mesh, target_error=0.1)
    assert isinstance(refined_mesh, Mesh)
    
    # Test with low target error
    refined_mesh = refiner.refine_mesh(mesh, target_error=0.001)
    assert isinstance(refined_mesh, Mesh)


def test_memory_management():
    """Test memory management during refinement."""
    error_estimator = GradientBasedErrorEstimator()
    refinement_criterion = ThresholdRefinementCriterion()
    refiner = AdaptiveMeshRefiner(error_estimator, refinement_criterion)
    
    # Test with large mesh
    large_mesh = MockMesh(num_nodes=10000, num_elements=20000)
    refined_mesh = refiner.refine_mesh(large_mesh)
    
    assert isinstance(refined_mesh, Mesh)
    assert refined_mesh.nodes is not None
    assert refined_mesh.elements is not None


def test_error_estimation_accuracy():
    """Test accuracy of error estimation."""
    error_estimator = GradientBasedErrorEstimator()
    mesh = MockMesh()
    
    # Test error estimation
    errors = error_estimator.estimate_error(mesh)
    
    # Verify error properties
    assert np.all(errors >= 0)  # Errors should be non-negative
    assert np.all(np.isfinite(errors))  # Errors should be finite
    assert errors.shape == (len(mesh.elements),)  # Correct shape


def test_refinement_criterion_consistency():
    """Test consistency of refinement criteria."""
    criterion = ThresholdRefinementCriterion(
        refine_threshold=0.1,
        coarsen_threshold=0.01
    )
    
    # Test consistency of decisions
    for error in np.linspace(0, 0.2, 100):
        element = np.array([0, 1, 2, 3])
        refine = criterion.should_refine(error, element)
        coarsen = criterion.should_coarsen(error, element)
        
        # An element should not be both refined and coarsened
        assert not (refine and coarsen)
        
        # If error is above refine threshold, it should be refined
        if error > 0.1:
            assert refine
        
        # If error is below coarsen threshold, it should be coarsened
        if error < 0.01:
            assert coarsen 
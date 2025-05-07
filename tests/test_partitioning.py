"""Tests for mesh partitioning module."""

import numpy as np
import pytest
from phasta.fem.partitioning import MeshPartitioner, LoadBalancer
from phasta.fem.mesh import Mesh


def test_mesh_partitioner():
    """Test mesh partitioner."""
    # Create test mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],  # Square corners
        [0.5, 0.5]  # Center point
    ])
    elements = np.array([
        [0, 1, 4],  # Bottom triangle
        [1, 3, 4],  # Right triangle
        [3, 2, 4],  # Top triangle
        [2, 0, 4]   # Left triangle
    ])
    mesh = Mesh(nodes, elements)
    
    # Create partitioner
    partitioner = MeshPartitioner(n_parts=2)
    
    # Partition mesh
    partition_array, ghost_nodes = partitioner.partition(mesh)
    
    # Check partition array
    assert len(partition_array) == len(nodes)
    assert set(partition_array) == {0, 1}  # Two partitions
    
    # Check ghost nodes
    assert len(ghost_nodes) == 2  # Two partitions
    assert all(isinstance(nodes, list) for nodes in ghost_nodes.values())
    
    # Check that center node is a ghost node
    center_node = 4
    assert any(center_node in nodes for nodes in ghost_nodes.values())


def test_load_balancer():
    """Test load balancer."""
    # Create test mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],  # Square corners
        [0.5, 0.5]  # Center point
    ])
    elements = np.array([
        [0, 1, 4],  # Bottom triangle
        [1, 3, 4],  # Right triangle
        [3, 2, 4],  # Top triangle
        [2, 0, 4]   # Left triangle
    ])
    mesh = Mesh(nodes, elements)
    
    # Create partitioner and load balancer
    partitioner = MeshPartitioner(n_parts=2)
    balancer = LoadBalancer(partitioner)
    
    # Balance mesh
    partition_array, ghost_nodes = balancer.balance(mesh)
    
    # Check partition array
    assert len(partition_array) == len(nodes)
    assert set(partition_array) == {0, 1}  # Two partitions
    
    # Check ghost nodes
    assert len(ghost_nodes) == 2  # Two partitions
    assert all(isinstance(nodes, list) for nodes in ghost_nodes.values())
    
    # Check that center node is a ghost node
    center_node = 4
    assert any(center_node in nodes for nodes in ghost_nodes.values())


def test_element_sizes():
    """Test element size computation."""
    # Create test mesh with triangles
    nodes = np.array([
        [0, 0], [1, 0], [0, 1],  # First triangle
        [1, 1], [2, 1], [1, 2]   # Second triangle
    ])
    elements = np.array([
        [0, 1, 2],  # First triangle
        [3, 4, 5]   # Second triangle
    ])
    mesh = Mesh(nodes, elements)
    
    # Create load balancer
    partitioner = MeshPartitioner(n_parts=2)
    balancer = LoadBalancer(partitioner)
    
    # Compute element sizes
    sizes = balancer._compute_element_sizes(mesh)
    
    # Check sizes
    assert len(sizes) == len(elements)
    assert np.all(sizes > 0)  # All sizes should be positive
    assert np.allclose(sizes[0], 0.5)  # First triangle area
    assert np.allclose(sizes[1], 0.5)  # Second triangle area


def test_weight_adjustment():
    """Test weight adjustment for load balancing."""
    # Create test mesh
    nodes = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],  # Square corners
        [0.5, 0.5]  # Center point
    ])
    elements = np.array([
        [0, 1, 4],  # Bottom triangle
        [1, 3, 4],  # Right triangle
        [3, 2, 4],  # Top triangle
        [2, 0, 4]   # Left triangle
    ])
    mesh = Mesh(nodes, elements)
    
    # Create load balancer
    partitioner = MeshPartitioner(n_parts=2)
    balancer = LoadBalancer(partitioner)
    
    # Create initial weights
    weights = np.ones(len(elements))
    weights[0] = 2.0  # Make first element heavier
    
    # Create initial partition
    partition_array = np.array([0, 0, 1, 1, 0])  # Initial partition
    partition_weights = np.array([3.0, 2.0])  # Initial weights
    
    # Adjust weights
    adjusted_weights = balancer._adjust_weights(weights, partition_array, partition_weights)
    
    # Check adjusted weights
    assert len(adjusted_weights) == len(weights)
    assert np.all(adjusted_weights > 0)  # All weights should be positive
    assert adjusted_weights[0] < weights[0]  # Heavy element should be reduced 
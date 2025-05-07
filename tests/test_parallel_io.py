"""Tests for parallel I/O module."""

import os
import numpy as np
import pytest
from mpi4py import MPI
from phasta.fem.parallel_io import ParallelIO
from phasta.fem.mesh import Mesh
from phasta.fem.partitioning import MeshPartitioner


@pytest.mark.mpi
def test_write_read_mesh():
    """Test writing and reading distributed mesh."""
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
    
    # Create partitioner and parallel I/O
    partitioner = MeshPartitioner(n_parts=2)
    pio = ParallelIO()
    
    # Write mesh
    filename = 'test_mesh.h5'
    pio.write_mesh(filename, mesh, partitioner)
    
    # Read mesh
    read_mesh, partition_array, ghost_nodes = pio.read_mesh(filename)
    
    # Check mesh data
    assert np.allclose(read_mesh.nodes, mesh.nodes)
    assert np.allclose(read_mesh.elements, mesh.elements)
    
    # Check partition data
    assert len(partition_array) == len(nodes)
    assert set(partition_array) == {0, 1}  # Two partitions
    
    # Check ghost nodes
    assert len(ghost_nodes) == 2  # Two partitions
    assert all(isinstance(nodes, list) for nodes in ghost_nodes.values())
    
    # Clean up
    if pio.rank == 0:
        os.remove(filename)


@pytest.mark.mpi
def test_write_read_solution():
    """Test writing and reading distributed solution."""
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
    
    # Create partitioner and parallel I/O
    partitioner = MeshPartitioner(n_parts=2)
    pio = ParallelIO()
    
    # Create test solution
    solution = np.arange(len(nodes), dtype=float)
    
    # Get partition information
    partition_array, ghost_nodes = partitioner.partition(mesh)
    
    # Write solution
    filename = 'test_solution.h5'
    pio.write_solution(filename, solution, partition_array, ghost_nodes)
    
    # Read solution
    read_solution = pio.read_solution(filename, partition_array)
    
    # Check solution data
    assert np.allclose(read_solution, solution)
    
    # Clean up
    if pio.rank == 0:
        os.remove(filename)


@pytest.mark.mpi
def test_write_read_checkpoint():
    """Test writing and reading checkpoint files."""
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
    
    # Create partitioner and parallel I/O
    partitioner = MeshPartitioner(n_parts=2)
    pio = ParallelIO()
    
    # Create test solution
    solution = np.arange(len(nodes), dtype=float)
    
    # Write checkpoint
    filename = 'test_checkpoint.h5'
    step = 42
    time = 1.23
    pio.write_checkpoint(filename, mesh, solution, partitioner, step, time)
    
    # Read checkpoint
    read_mesh, read_solution, read_step, read_time = pio.read_checkpoint(filename)
    
    # Check mesh data
    assert np.allclose(read_mesh.nodes, mesh.nodes)
    assert np.allclose(read_mesh.elements, mesh.elements)
    
    # Check solution data
    assert np.allclose(read_solution, solution)
    
    # Check metadata
    assert read_step == step
    assert read_time == time
    
    # Clean up
    if pio.rank == 0:
        os.remove(filename)


@pytest.mark.mpi
def test_parallel_io_consistency():
    """Test consistency of parallel I/O operations."""
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
    
    # Create partitioner and parallel I/O
    partitioner = MeshPartitioner(n_parts=2)
    pio = ParallelIO()
    
    # Create test solution
    solution = np.arange(len(nodes), dtype=float)
    
    # Write checkpoint
    filename = 'test_consistency.h5'
    step = 42
    time = 1.23
    pio.write_checkpoint(filename, mesh, solution, partitioner, step, time)
    
    # Read checkpoint
    read_mesh, read_solution, read_step, read_time = pio.read_checkpoint(filename)
    
    # Write checkpoint again
    pio.write_checkpoint(filename, read_mesh, read_solution, partitioner, read_step, read_time)
    
    # Read checkpoint again
    final_mesh, final_solution, final_step, final_time = pio.read_checkpoint(filename)
    
    # Check consistency
    assert np.allclose(final_mesh.nodes, mesh.nodes)
    assert np.allclose(final_mesh.elements, mesh.elements)
    assert np.allclose(final_solution, solution)
    assert final_step == step
    assert final_time == time
    
    # Clean up
    if pio.rank == 0:
        os.remove(filename) 
"""Tests for parallel computing module."""

import numpy as np
import pytest
from scipy import sparse
from mpi4py import MPI
from phasta.fem.parallel import ParallelMesh, ParallelAssembly, ParallelSolver
from phasta.fem.mesh import Mesh
from phasta.fem.solvers import GMRESSolver


@pytest.mark.mpi
def test_parallel_mesh():
    """Test parallel mesh partitioning."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(nodes, elements)
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh)
    
    # Check local data
    assert parallel_mesh.local_nodes is not None
    assert parallel_mesh.local_elements is not None
    assert parallel_mesh.ghost_nodes is not None
    
    # Check communication buffers
    assert isinstance(parallel_mesh.send_buffers, dict)
    assert isinstance(parallel_mesh.recv_buffers, dict)


@pytest.mark.mpi
def test_parallel_assembly():
    """Test parallel matrix assembly."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(nodes, elements)
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh)
    
    # Create parallel assembly
    assembly = ParallelAssembly(parallel_mesh)
    
    # Define element matrix function
    def element_matrix(element):
        return np.eye(3)  # Simple identity matrix for testing
    
    # Assemble matrix
    matrix = assembly.assemble_matrix(element_matrix)
    
    # Check matrix properties
    assert isinstance(matrix, sparse.spmatrix)
    assert matrix.shape[0] == matrix.shape[1]  # Square matrix
    assert matrix.nnz > 0  # Non-zero entries


@pytest.mark.mpi
def test_parallel_solver():
    """Test parallel solver."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(nodes, elements)
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh)
    
    # Create local solver
    local_solver = GMRESSolver()
    
    # Create parallel solver
    solver = ParallelSolver(parallel_mesh, local_solver)
    
    # Create test system
    n = len(parallel_mesh.local_nodes)
    A = sparse.eye(n)
    b = np.ones(n)
    
    # Solve system
    x = solver.solve(A, b)
    
    # Check solution
    assert x.shape == b.shape
    assert not np.allclose(x, 0)  # Non-zero solution


@pytest.mark.mpi
def test_parallel_communication():
    """Test parallel communication."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(nodes, elements)
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh)
    
    # Test communication buffers
    for rank, nodes in parallel_mesh.ghost_nodes.items():
        # Check send buffer
        assert rank in parallel_mesh.send_buffers
        assert parallel_mesh.send_buffers[rank].shape == (len(nodes),)
        
        # Check receive buffer
        assert rank in parallel_mesh.recv_buffers
        assert parallel_mesh.recv_buffers[rank].shape == (len(nodes),)


@pytest.mark.mpi
def test_parallel_assembly_communication():
    """Test communication during parallel assembly."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(nodes, elements)
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh)
    
    # Create parallel assembly
    assembly = ParallelAssembly(parallel_mesh)
    
    # Define element matrix function
    def element_matrix(element):
        return np.eye(3)  # Simple identity matrix for testing
    
    # Assemble matrix
    matrix = assembly.assemble_matrix(element_matrix)
    
    # Check that ghost node contributions were communicated
    for rank, nodes in parallel_mesh.ghost_nodes.items():
        for node in nodes:
            assert matrix[node, node] != 0  # Non-zero contribution 
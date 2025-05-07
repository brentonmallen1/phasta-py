"""Tests for parallel solvers."""

import numpy as np
import pytest
from scipy import sparse
from mpi4py import MPI
from phasta.fem.parallel_solvers import (
    ParallelLinearSolver, ParallelGMRES, ParallelCG, ParallelBiCGSTAB,
    ParallelSolverFactory
)
from phasta.fem.parallel import ParallelMesh
from phasta.fem.mesh import Mesh


@pytest.mark.mpi
def test_parallel_gmres():
    """Test parallel GMRES solver."""
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
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh, n_parts=2)
    
    # Create test matrix (Laplacian)
    n = len(nodes)
    A = sparse.lil_matrix((n, n))
    for i in range(n):
        A[i, i] = 4
        if i > 0:
            A[i, i-1] = -1
        if i < n-1:
            A[i, i+1] = -1
    A = A.tocsr()
    
    # Create right-hand side
    b = np.ones(n)
    
    # Create solver
    solver = ParallelGMRES(max_iter=100, tol=1e-6, restart=10)
    
    # Solve system
    x = solver.solve(A, b, parallel_mesh)
    
    # Check solution
    assert np.allclose(A @ x, b, atol=1e-6)


@pytest.mark.mpi
def test_parallel_cg():
    """Test parallel Conjugate Gradient solver."""
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
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh, n_parts=2)
    
    # Create test matrix (symmetric positive definite)
    n = len(nodes)
    A = sparse.lil_matrix((n, n))
    for i in range(n):
        A[i, i] = 4
        if i > 0:
            A[i, i-1] = -1
        if i < n-1:
            A[i, i+1] = -1
    A = A.tocsr()
    
    # Create right-hand side
    b = np.ones(n)
    
    # Create solver
    solver = ParallelCG(max_iter=100, tol=1e-6)
    
    # Solve system
    x = solver.solve(A, b, parallel_mesh)
    
    # Check solution
    assert np.allclose(A @ x, b, atol=1e-6)


@pytest.mark.mpi
def test_parallel_bicgstab():
    """Test parallel BiCGSTAB solver."""
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
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh, n_parts=2)
    
    # Create test matrix
    n = len(nodes)
    A = sparse.lil_matrix((n, n))
    for i in range(n):
        A[i, i] = 4
        if i > 0:
            A[i, i-1] = -1
        if i < n-1:
            A[i, i+1] = -1
    A = A.tocsr()
    
    # Create right-hand side
    b = np.ones(n)
    
    # Create solver
    solver = ParallelBiCGSTAB(max_iter=100, tol=1e-6)
    
    # Solve system
    x = solver.solve(A, b, parallel_mesh)
    
    # Check solution
    assert np.allclose(A @ x, b, atol=1e-6)


@pytest.mark.mpi
def test_parallel_solver_factory():
    """Test parallel solver factory."""
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
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh, n_parts=2)
    
    # Create test matrix
    n = len(nodes)
    A = sparse.lil_matrix((n, n))
    for i in range(n):
        A[i, i] = 4
        if i > 0:
            A[i, i-1] = -1
        if i < n-1:
            A[i, i+1] = -1
    A = A.tocsr()
    
    # Create right-hand side
    b = np.ones(n)
    
    # Test each solver type
    for solver_type in ['gmres', 'cg', 'bicgstab']:
        # Create solver
        solver = ParallelSolverFactory.create(solver_type, max_iter=100, tol=1e-6)
        
        # Solve system
        x = solver.solve(A, b, parallel_mesh)
        
        # Check solution
        assert np.allclose(A @ x, b, atol=1e-6)
    
    # Test invalid solver type
    with pytest.raises(ValueError):
        ParallelSolverFactory.create('invalid_solver') 
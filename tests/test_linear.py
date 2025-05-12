"""Tests for linear solvers module."""

import numpy as np
import pytest
from scipy.sparse import spdiags
from phasta.solver.linear import (
    LinearSolver, ConjugateGradient, GMRES, DirectSolver, MatrixFreeSolver
)


def test_conjugate_gradient():
    """Test Conjugate Gradient solver."""
    # Create test matrix (Laplacian)
    n = 100
    A = spdiags([-1, 2, -1], [-1, 0, 1], n, n).toarray()
    b = np.ones(n)
    
    # Create solver
    solver = ConjugateGradient(max_iter=1000, tol=1e-6)
    
    # Solve system
    x = solver.solve(A, b)
    
    # Check solution
    assert np.allclose(A @ x, b, rtol=1e-6)
    assert solver.iterations < solver.max_iter
    assert solver.residual < solver.tol
    
    # Test with different tolerance
    solver = ConjugateGradient(max_iter=1000, tol=1e-10)
    x = solver.solve(A, b)
    assert np.allclose(A @ x, b, rtol=1e-10)
    assert solver.residual < solver.tol


def test_gmres():
    """Test GMRES solver."""
    # Create test matrix (nonsymmetric)
    n = 100
    A = np.random.rand(n, n)
    A = A + A.T  # Make symmetric positive definite
    A = A + n * np.eye(n)  # Make diagonally dominant
    b = np.ones(n)
    
    # Create solver
    solver = GMRES(max_iter=1000, tol=1e-6, restart=30)
    
    # Solve system
    x = solver.solve(A, b)
    
    # Check solution
    assert np.allclose(A @ x, b, rtol=1e-6)
    assert solver.iterations < solver.max_iter
    assert solver.residual < solver.tol
    
    # Test with different restart
    solver = GMRES(max_iter=1000, tol=1e-6, restart=50)
    x = solver.solve(A, b)
    assert np.allclose(A @ x, b, rtol=1e-6)


def test_direct_solver():
    """Test Direct solver."""
    # Create test matrix
    n = 100
    A = spdiags([-1, 2, -1], [-1, 0, 1], n, n)
    b = np.ones(n)
    
    # Create solver
    solver = DirectSolver()
    
    # Solve system
    x = solver.solve(A, b)
    
    # Check solution
    assert np.allclose(A @ x, b, rtol=1e-6)
    assert solver.iterations == 1
    
    # Test with dense matrix
    A_dense = A.toarray()
    x = solver.solve(A_dense, b)
    assert np.allclose(A_dense @ x, b, rtol=1e-6)


def test_matrix_free_solver():
    """Test Matrix-free solver."""
    # Create test matrix
    n = 100
    A = spdiags([-1, 2, -1], [-1, 0, 1], n, n)
    
    # Create matrix-vector product function
    def matvec(x):
        return A @ x
    
    # Create solver
    solver = MatrixFreeSolver(matvec=matvec)
    
    # Solve system
    b = np.ones(n)
    x = solver.solve(None, b)  # A is ignored
    
    # Check solution
    assert np.allclose(A @ x, b, rtol=1e-6)
    assert solver.iterations < solver.max_iter
    assert solver.residual < solver.tol


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test zero right-hand side
    n = 100
    A = spdiags([-1, 2, -1], [-1, 0, 1], n, n)
    b = np.zeros(n)
    
    # Test CG
    solver = ConjugateGradient()
    x = solver.solve(A, b)
    assert np.allclose(x, 0)
    
    # Test GMRES
    solver = GMRES()
    x = solver.solve(A, b)
    assert np.allclose(x, 0)
    
    # Test Direct
    solver = DirectSolver()
    x = solver.solve(A, b)
    assert np.allclose(x, 0)
    
    # Test Matrix-free
    def matvec(x):
        return A @ x
    solver = MatrixFreeSolver(matvec=matvec)
    x = solver.solve(None, b)
    assert np.allclose(x, 0)
    
    # Test singular matrix
    A_singular = np.ones((n, n))
    b = np.ones(n)
    
    with pytest.raises(np.linalg.LinAlgError):
        solver = DirectSolver()
        solver.solve(A_singular, b)


def test_memory_management():
    """Test memory management during computations."""
    # Create large system
    n = 1000
    A = spdiags([-1, 2, -1], [-1, 0, 1], n, n)
    b = np.ones(n)
    
    # Test all solvers
    solvers = [
        ConjugateGradient(),
        GMRES(),
        DirectSolver(),
        MatrixFreeSolver(matvec=lambda x: A @ x)
    ]
    
    for solver in solvers:
        x = solver.solve(A, b)
        assert np.allclose(A @ x, b, rtol=1e-6)
        assert solver.residual < solver.tol 
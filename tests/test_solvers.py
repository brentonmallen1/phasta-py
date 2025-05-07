"""Tests for linear solvers module."""

import numpy as np
import pytest
from scipy import sparse
from typing import Tuple
from phasta.fem.solvers import (
    LinearSolver, DirectSolver, GMRESSolver,
    ConjugateGradientSolver, BiCGSTABSolver, SolverFactory
)


def create_test_system(n: int = 100) -> Tuple[sparse.spmatrix, np.ndarray]:
    """Create a test linear system.
    
    Args:
        n: System size
        
    Returns:
        Tuple of (system matrix, right-hand side)
    """
    # Create a symmetric positive definite matrix
    A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    
    # Create right-hand side
    b = np.ones(n)
    
    return A, b


def test_direct_solver():
    """Test direct solver."""
    # Create test system
    A, b = create_test_system()
    
    # Create solver
    solver = DirectSolver()
    
    # Solve system
    x = solver.solve(A, b)
    
    # Check solution
    assert np.allclose(A @ x, b, rtol=1e-10)
    assert solver.iterations == 0  # Direct solver doesn't track iterations


def test_gmres_solver():
    """Test GMRES solver."""
    # Create test system
    A, b = create_test_system()
    
    # Create solver
    solver = GMRESSolver(max_iter=100, tol=1e-8, restart=20)
    
    # Solve system
    x = solver.solve(A, b)
    
    # Check solution
    assert np.allclose(A @ x, b, rtol=1e-8)
    assert solver.iterations > 0
    assert solver.residual < 1e-8


def test_conjugate_gradient_solver():
    """Test conjugate gradient solver."""
    # Create test system
    A, b = create_test_system()
    
    # Create solver
    solver = ConjugateGradientSolver(max_iter=100, tol=1e-8)
    
    # Solve system
    x = solver.solve(A, b)
    
    # Check solution
    assert np.allclose(A @ x, b, rtol=1e-8)
    assert solver.iterations > 0
    assert solver.residual < 1e-8


def test_bicgstab_solver():
    """Test BiCGSTAB solver."""
    # Create test system
    A, b = create_test_system()
    
    # Create solver
    solver = BiCGSTABSolver(max_iter=100, tol=1e-8)
    
    # Solve system
    x = solver.solve(A, b)
    
    # Check solution
    assert np.allclose(A @ x, b, rtol=1e-8)
    assert solver.iterations > 0
    assert solver.residual < 1e-8


def test_solver_factory():
    """Test solver factory."""
    # Create test system
    A, b = create_test_system()
    
    # Test creating different solvers
    solver_types = ['direct', 'gmres', 'cg', 'bicgstab']
    for solver_type in solver_types:
        solver = SolverFactory.create_solver(solver_type)
        x = solver.solve(A, b)
        assert np.allclose(A @ x, b, rtol=1e-8)
    
    # Test invalid solver type
    with pytest.raises(ValueError):
        SolverFactory.create_solver('invalid')


def test_preconditioner():
    """Test solver with preconditioner."""
    # Create test system
    A, b = create_test_system()
    
    # Create simple diagonal preconditioner
    def preconditioner(x):
        return x / A.diagonal()
    
    # Test GMRES with preconditioner
    solver = GMRESSolver(max_iter=100, tol=1e-8, preconditioner=preconditioner)
    x = solver.solve(A, b)
    assert np.allclose(A @ x, b, rtol=1e-8)
    
    # Test CG with preconditioner
    solver = ConjugateGradientSolver(max_iter=100, tol=1e-8,
                                   preconditioner=preconditioner)
    x = solver.solve(A, b)
    assert np.allclose(A @ x, b, rtol=1e-8)
    
    # Test BiCGSTAB with preconditioner
    solver = BiCGSTABSolver(max_iter=100, tol=1e-8,
                           preconditioner=preconditioner)
    x = solver.solve(A, b)
    assert np.allclose(A @ x, b, rtol=1e-8)


def test_non_symmetric_system():
    """Test solvers with non-symmetric system."""
    # Create non-symmetric system
    n = 100
    A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    A[0, -1] = 1  # Make matrix non-symmetric
    b = np.ones(n)
    
    # Test GMRES (should work)
    solver = GMRESSolver(max_iter=100, tol=1e-8)
    x = solver.solve(A, b)
    assert np.allclose(A @ x, b, rtol=1e-8)
    
    # Test BiCGSTAB (should work)
    solver = BiCGSTABSolver(max_iter=100, tol=1e-8)
    x = solver.solve(A, b)
    assert np.allclose(A @ x, b, rtol=1e-8)
    
    # Test CG (should fail for non-symmetric system)
    solver = ConjugateGradientSolver(max_iter=100, tol=1e-8)
    with pytest.raises(np.linalg.LinAlgError):
        solver.solve(A, b) 
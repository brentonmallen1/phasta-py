"""Tests for linear solvers module."""

import numpy as np
import pytest
from scipy import sparse
from typing import Tuple
from phasta.fem.solvers import (
    LinearSolver, DirectSolver, GMRESSolver,
    ConjugateGradientSolver, BiCGSTABSolver, SolverFactory
)
from phasta.solvers.direct import (
    LUSolver,
    CholeskySolver,
    SparseDirectSolver,
    BlockDirectSolver,
    solve_direct
)
from phasta.solvers.multigrid import (
    MultiGridSolver,
    GeometricMultiGrid,
    AlgebraicMultiGrid
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


def test_lu_solver():
    """Test LU decomposition solver."""
    # Create test matrix and vector
    A = np.array([
        [4, 1, 1],
        [1, 3, 2],
        [1, 2, 5]
    ])
    b = np.array([1, 2, 3])
    
    # Create solver
    solver = LUSolver(A)
    
    # Solve system
    x = solver.solve(b)
    
    # Check solution
    assert np.allclose(A @ x, b)
    
    # Test with sparse matrix
    A_sparse = sparse.csr_matrix(A)
    solver_sparse = LUSolver(A_sparse)
    x_sparse = solver_sparse.solve(b)
    assert np.allclose(x, x_sparse)


def test_cholesky_solver():
    """Test Cholesky decomposition solver."""
    # Create symmetric positive definite matrix
    A = np.array([
        [4, 1, 1],
        [1, 3, 2],
        [1, 2, 5]
    ])
    A = A.T @ A  # Make it positive definite
    b = np.array([1, 2, 3])
    
    # Create solver
    solver = CholeskySolver(A)
    
    # Solve system
    x = solver.solve(b)
    
    # Check solution
    assert np.allclose(A @ x, b)
    
    # Test with sparse matrix
    A_sparse = sparse.csr_matrix(A)
    solver_sparse = CholeskySolver(A_sparse)
    x_sparse = solver_sparse.solve(b)
    assert np.allclose(x, x_sparse)


def test_sparse_direct_solver():
    """Test sparse direct solver."""
    # Create sparse matrix
    A = sparse.csr_matrix([
        [4, 1, 0],
        [1, 3, 2],
        [0, 2, 5]
    ])
    b = np.array([1, 2, 3])
    
    # Test different methods
    methods = ['umfpack', 'superlu']
    for method in methods:
        solver = SparseDirectSolver(A, method=method)
        x = solver.solve(b)
        assert np.allclose(A @ x, b)
    
    # Test fallback for pardiso
    solver = SparseDirectSolver(A, method='pardiso')
    x = solver.solve(b)
    assert np.allclose(A @ x, b)


def test_block_direct_solver():
    """Test block direct solver."""
    # Create block matrices
    blocks = [
        np.array([[4, 1], [1, 3]]),
        np.array([[5, 2], [2, 6]])
    ]
    b = np.array([1, 2, 3, 4])
    
    # Create solver
    solver = BlockDirectSolver(blocks)
    
    # Solve system
    x = solver.solve(b)
    
    # Check solution
    A = sparse.block_diag(blocks)
    assert np.allclose(A @ x, b)


def test_solve_direct():
    """Test convenience function for direct solution."""
    # Create test matrices
    A_dense = np.array([
        [4, 1, 1],
        [1, 3, 2],
        [1, 2, 5]
    ])
    A_sparse = sparse.csr_matrix(A_dense)
    b = np.array([1, 2, 3])
    
    # Test different methods
    methods = ['auto', 'lu', 'cholesky', 'sparse']
    for method in methods:
        x = solve_direct(A_dense, b, method=method)
        assert np.allclose(A_dense @ x, b)
    
    # Test with sparse matrix
    x = solve_direct(A_sparse, b)
    assert np.allclose(A_sparse @ x, b)


def test_geometric_multigrid():
    """Test geometric multi-grid solver."""
    # Create grid hierarchy
    nodes_fine = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0.5, 0], [0, 0.5], [0.5, 1], [1, 0.5], [0.5, 0.5]
    ])
    nodes_coarse = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1]
    ])
    elements_fine = np.array([
        [0, 4, 5], [4, 1, 8], [5, 8, 2], [8, 7, 2],
        [1, 7, 8], [7, 3, 8], [8, 3, 6], [6, 2, 8]
    ])
    elements_coarse = np.array([
        [0, 1, 2], [1, 3, 2]
    ])
    
    # Create system matrix
    A = sparse.csr_matrix((9, 9))
    A[0, 0] = 4
    A[1, 1] = 4
    A[2, 2] = 4
    A[3, 3] = 4
    A[4, 4] = 4
    A[5, 5] = 4
    A[6, 6] = 4
    A[7, 7] = 4
    A[8, 8] = 4
    
    # Create solver
    solver = GeometricMultiGrid(
        A,
        grid_hierarchy=[(nodes_fine, elements_fine), (nodes_coarse, elements_coarse)],
        max_levels=2,
        smoother='jacobi',
        n_smooth=2
    )
    
    # Solve system
    b = np.ones(9)
    x = solver.solve(b)
    
    # Check solution
    assert np.allclose(A @ x, b, atol=1e-6)


def test_algebraic_multigrid():
    """Test algebraic multi-grid solver."""
    # Create sparse matrix
    A = sparse.csr_matrix([
        [4, 1, 0, 1],
        [1, 3, 2, 0],
        [0, 2, 5, 1],
        [1, 0, 1, 4]
    ])
    b = np.array([1, 2, 3, 4])
    
    # Create solver
    solver = AlgebraicMultiGrid(
        A,
        max_levels=2,
        smoother='jacobi',
        n_smooth=2,
        strength_threshold=0.25
    )
    
    # Solve system
    x = solver.solve(b)
    
    # Check solution
    assert np.allclose(A @ x, b, atol=1e-6)


def test_multigrid_smoothers():
    """Test multi-grid smoothers."""
    # Create test matrix
    A = sparse.csr_matrix([
        [4, 1, 0],
        [1, 3, 2],
        [0, 2, 5]
    ])
    b = np.array([1, 2, 3])
    x0 = np.zeros_like(b)
    
    # Test different smoothers
    smoothers = ['jacobi', 'gauss_seidel', 'sor']
    for smoother in smoothers:
        solver = GeometricMultiGrid(
            A,
            grid_hierarchy=[(np.array([[0, 0], [1, 0], [0, 1]]),
                           np.array([[0, 1, 2]]))],
            smoother=smoother,
            n_smooth=2
        )
        x = solver.solve(b, x0=x0)
        assert np.allclose(A @ x, b, atol=1e-6)


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    # Test invalid smoother
    A = sparse.csr_matrix([[4, 1], [1, 3]])
    with pytest.raises(ValueError):
        MultiGridSolver(A, smoother='invalid')
    
    # Test invalid method
    with pytest.raises(ValueError):
        SparseDirectSolver(A, method='invalid')
    
    # Test invalid solve_direct method
    with pytest.raises(ValueError):
        solve_direct(A, np.array([1, 2]), method='invalid') 
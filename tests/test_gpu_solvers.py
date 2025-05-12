"""Tests for GPU-enabled solvers."""

import numpy as np
import pytest
import scipy.sparse as sp
from phasta.solvers.gpu import (
    GPUSolver,
    CUDASolver,
    OpenCLSolver,
    solve_gpu
)


def test_gpu_solver():
    """Test GPU solver base class."""
    # Create test matrix
    A = np.array([
        [4, 1, 1],
        [1, 3, 2],
        [1, 2, 5]
    ])
    
    # Test base class
    solver = GPUSolver(A)
    with pytest.raises(NotImplementedError):
        solver._setup_cuda()
    with pytest.raises(NotImplementedError):
        solver._setup_opencl()
    with pytest.raises(NotImplementedError):
        solver.solve(np.array([1, 2, 3]))


@pytest.mark.skipif(not CUDASolver.CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_solver():
    """Test CUDA solver."""
    # Create test matrix
    A = np.array([
        [4, 1, 1],
        [1, 3, 2],
        [1, 2, 5]
    ])
    b = np.array([1, 2, 3])
    
    # Create solver
    solver = CUDASolver(A)
    
    # Solve system
    x = solver.solve(b)
    
    # Check solution
    assert np.allclose(A @ x, b)
    
    # Test with sparse matrix
    A_sparse = sp.csr_matrix(A)
    solver_sparse = CUDASolver(A_sparse)
    x_sparse = solver_sparse.solve(b)
    assert np.allclose(x, x_sparse)


@pytest.mark.skipif(not OpenCLSolver.OPENCL_AVAILABLE, reason="OpenCL not available")
def test_opencl_solver():
    """Test OpenCL solver."""
    # Create test matrix
    A = np.array([
        [4, 1, 1],
        [1, 3, 2],
        [1, 2, 5]
    ])
    b = np.array([1, 2, 3])
    
    # Create solver
    solver = OpenCLSolver(A)
    
    # Solve system
    x = solver.solve(b)
    
    # Check solution
    assert np.allclose(A @ x, b)
    
    # Test with sparse matrix
    A_sparse = sp.csr_matrix(A)
    solver_sparse = OpenCLSolver(A_sparse)
    x_sparse = solver_sparse.solve(b)
    assert np.allclose(x, x_sparse)


@pytest.mark.skipif(not CUDASolver.CUDA_AVAILABLE and not OpenCLSolver.OPENCL_AVAILABLE,
                    reason="No GPU backend available")
def test_solve_gpu():
    """Test convenience function for GPU solution."""
    # Create test matrix
    A = np.array([
        [4, 1, 1],
        [1, 3, 2],
        [1, 2, 5]
    ])
    b = np.array([1, 2, 3])
    
    # Test CUDA backend if available
    if CUDASolver.CUDA_AVAILABLE:
        x_cuda = solve_gpu(A, b, backend='cuda')
        assert np.allclose(A @ x_cuda, b)
    
    # Test OpenCL backend if available
    if OpenCLSolver.OPENCL_AVAILABLE:
        x_opencl = solve_gpu(A, b, backend='opencl')
        assert np.allclose(A @ x_opencl, b)
    
    # Test invalid backend
    with pytest.raises(ValueError):
        solve_gpu(A, b, backend='invalid')


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    # Create test matrix
    A = np.array([
        [4, 1, 1],
        [1, 3, 2],
        [1, 2, 5]
    ])
    
    # Test invalid backend
    with pytest.raises(ValueError):
        GPUSolver(A, backend='invalid')
    
    # Test invalid matrix shape
    with pytest.raises(ValueError):
        CUDASolver(A[:2, :])
    
    # Test invalid right-hand side
    with pytest.raises(ValueError):
        solver = CUDASolver(A)
        solver.solve(np.array([1, 2])) 
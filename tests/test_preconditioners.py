"""Tests for advanced preconditioners."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, spdiags
from phasta.solver.preconditioners import (
    Preconditioner, AMGPreconditioner, ILUPreconditioner, BlockPreconditioner
)


def test_amg_preconditioner():
    """Test algebraic multigrid preconditioner."""
    # Create test matrix (Laplacian)
    n = 100
    A = spdiags([-1, 2, -1], [-1, 0, 1], n, n).tocsr()
    
    # Create preconditioner
    preconditioner = AMGPreconditioner(
        max_levels=5,
        coarsening_type='RS',
        interpolation_type='classical',
        smoother_type='jacobi',
        smoothing_steps=2
    )
    
    # Setup preconditioner
    preconditioner.setup(A)
    
    # Test preconditioner
    b = np.ones(n)
    x = preconditioner.apply(b)
    
    # Check solution
    assert len(x) == n
    assert np.all(np.isfinite(x))
    
    # Test different coarsening types
    for coarsening_type in ['RS', 'PMIS', 'HMIS']:
        preconditioner = AMGPreconditioner(coarsening_type=coarsening_type)
        preconditioner.setup(A)
        x = preconditioner.apply(b)
        assert len(x) == n
        assert np.all(np.isfinite(x))
    
    # Test different smoothers
    for smoother_type in ['jacobi', 'gauss_seidel', 'ilu']:
        preconditioner = AMGPreconditioner(smoother_type=smoother_type)
        preconditioner.setup(A)
        x = preconditioner.apply(b)
        assert len(x) == n
        assert np.all(np.isfinite(x))


def test_ilu_preconditioner():
    """Test incomplete LU preconditioner."""
    # Create test matrix
    n = 100
    A = spdiags([-1, 2, -1], [-1, 0, 1], n, n).tocsr()
    
    # Create preconditioner
    preconditioner = ILUPreconditioner(
        fill_factor=10,
        drop_tol=1e-4
    )
    
    # Setup preconditioner
    preconditioner.setup(A)
    
    # Test preconditioner
    b = np.ones(n)
    x = preconditioner.apply(b)
    
    # Check solution
    assert len(x) == n
    assert np.all(np.isfinite(x))
    
    # Test different fill factors
    for fill_factor in [5, 10, 20]:
        preconditioner = ILUPreconditioner(fill_factor=fill_factor)
        preconditioner.setup(A)
        x = preconditioner.apply(b)
        assert len(x) == n
        assert np.all(np.isfinite(x))
    
    # Test different drop tolerances
    for drop_tol in [1e-3, 1e-4, 1e-5]:
        preconditioner = ILUPreconditioner(drop_tol=drop_tol)
        preconditioner.setup(A)
        x = preconditioner.apply(b)
        assert len(x) == n
        assert np.all(np.isfinite(x))


def test_block_preconditioner():
    """Test block preconditioner."""
    # Create test matrix
    n = 100
    block_size = 10
    A = spdiags([-1, 2, -1], [-1, 0, 1], n, n).tocsr()
    
    # Create preconditioner
    preconditioner = BlockPreconditioner(
        block_size=block_size,
        preconditioner_type='ilu'
    )
    
    # Setup preconditioner
    preconditioner.setup(A)
    
    # Test preconditioner
    b = np.ones(n)
    x = preconditioner.apply(b)
    
    # Check solution
    assert len(x) == n
    assert np.all(np.isfinite(x))
    
    # Test different block sizes
    for block_size in [5, 10, 20]:
        preconditioner = BlockPreconditioner(block_size=block_size)
        preconditioner.setup(A)
        x = preconditioner.apply(b)
        assert len(x) == n
        assert np.all(np.isfinite(x))
    
    # Test different preconditioner types
    for preconditioner_type in ['ilu', 'amg']:
        preconditioner = BlockPreconditioner(
            block_size=block_size,
            preconditioner_type=preconditioner_type
        )
        preconditioner.setup(A)
        x = preconditioner.apply(b)
        assert len(x) == n
        assert np.all(np.isfinite(x))


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test zero matrix
    n = 10
    A = csr_matrix((n, n))
    
    # AMG preconditioner
    preconditioner = AMGPreconditioner()
    with pytest.raises(ValueError):
        preconditioner.setup(A)
    
    # ILU preconditioner
    preconditioner = ILUPreconditioner()
    with pytest.raises(ValueError):
        preconditioner.setup(A)
    
    # Block preconditioner
    preconditioner = BlockPreconditioner(block_size=2)
    with pytest.raises(ValueError):
        preconditioner.setup(A)
    
    # Test singular matrix
    A = csr_matrix((n, n))
    A[0, 0] = 1.0
    
    # AMG preconditioner
    preconditioner = AMGPreconditioner()
    preconditioner.setup(A)
    b = np.ones(n)
    x = preconditioner.apply(b)
    assert np.all(np.isfinite(x))
    
    # ILU preconditioner
    preconditioner = ILUPreconditioner()
    preconditioner.setup(A)
    x = preconditioner.apply(b)
    assert np.all(np.isfinite(x))
    
    # Block preconditioner
    preconditioner = BlockPreconditioner(block_size=2)
    preconditioner.setup(A)
    x = preconditioner.apply(b)
    assert np.all(np.isfinite(x))


def test_memory_management():
    """Test memory management during preconditioner setup."""
    # Create large matrix
    n = 1000
    A = spdiags([-1, 2, -1], [-1, 0, 1], n, n).tocsr()
    b = np.ones(n)
    
    # Test AMG preconditioner
    preconditioner = AMGPreconditioner()
    preconditioner.setup(A)
    x = preconditioner.apply(b)
    assert len(x) == n
    assert np.all(np.isfinite(x))
    
    # Test ILU preconditioner
    preconditioner = ILUPreconditioner()
    preconditioner.setup(A)
    x = preconditioner.apply(b)
    assert len(x) == n
    assert np.all(np.isfinite(x))
    
    # Test block preconditioner
    preconditioner = BlockPreconditioner(block_size=100)
    preconditioner.setup(A)
    x = preconditioner.apply(b)
    assert len(x) == n
    assert np.all(np.isfinite(x))


def test_convergence():
    """Test preconditioner convergence."""
    # Create test matrix
    n = 100
    A = spdiags([-1, 2, -1], [-1, 0, 1], n, n).tocsr()
    b = np.ones(n)
    
    # Test AMG preconditioner
    preconditioner = AMGPreconditioner()
    preconditioner.setup(A)
    x = preconditioner.apply(b)
    residual = np.linalg.norm(b - A @ x)
    assert residual < 1e-6
    
    # Test ILU preconditioner
    preconditioner = ILUPreconditioner()
    preconditioner.setup(A)
    x = preconditioner.apply(b)
    residual = np.linalg.norm(b - A @ x)
    assert residual < 1e-6
    
    # Test block preconditioner
    preconditioner = BlockPreconditioner(block_size=10)
    preconditioner.setup(A)
    x = preconditioner.apply(b)
    residual = np.linalg.norm(b - A @ x)
    assert residual < 1e-6 
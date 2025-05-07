"""Tests for preconditioners module."""

import numpy as np
import pytest
from scipy import sparse
from phasta.fem.preconditioners import (
    Preconditioner, DiagonalPreconditioner, ILUPreconditioner,
    BlockJacobiPreconditioner, AMGPreconditioner, PreconditionerFactory
)


def create_test_system(n: int = 100) -> sparse.spmatrix:
    """Create a test system matrix.
    
    Args:
        n: System size
        
    Returns:
        System matrix
    """
    # Create a symmetric positive definite matrix
    A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    return A


def test_diagonal_preconditioner():
    """Test diagonal preconditioner."""
    # Create test system
    A = create_test_system()
    
    # Create preconditioner
    precond = DiagonalPreconditioner()
    precond.setup(A)
    
    # Test preconditioner
    x = np.ones(A.shape[0])
    y = precond.apply(x)
    
    # Check result
    assert y.shape == x.shape
    assert not np.allclose(y, x)  # Should modify the vector
    assert np.all(np.isfinite(y))  # Should be finite


def test_ilu_preconditioner():
    """Test ILU preconditioner."""
    # Create test system
    A = create_test_system()
    
    # Create preconditioner
    precond = ILUPreconditioner(fill_factor=5.0, drop_tol=1e-4)
    precond.setup(A)
    
    # Test preconditioner
    x = np.ones(A.shape[0])
    y = precond.apply(x)
    
    # Check result
    assert y.shape == x.shape
    assert not np.allclose(y, x)  # Should modify the vector
    assert np.all(np.isfinite(y))  # Should be finite


def test_block_jacobi_preconditioner():
    """Test block Jacobi preconditioner."""
    # Create test system
    A = create_test_system()
    
    # Create preconditioner
    precond = BlockJacobiPreconditioner(block_size=4)
    precond.setup(A)
    
    # Test preconditioner
    x = np.ones(A.shape[0])
    y = precond.apply(x)
    
    # Check result
    assert y.shape == x.shape
    assert not np.allclose(y, x)  # Should modify the vector
    assert np.all(np.isfinite(y))  # Should be finite


def test_amg_preconditioner():
    """Test AMG preconditioner."""
    # Create test system
    A = create_test_system()
    
    # Create preconditioner
    precond = AMGPreconditioner(strength=0.25, max_levels=2)
    precond.setup(A)
    
    # Test preconditioner
    x = np.ones(A.shape[0])
    y = precond.apply(x)
    
    # Check result
    assert y.shape == x.shape
    assert not np.allclose(y, x)  # Should modify the vector
    assert np.all(np.isfinite(y))  # Should be finite


def test_preconditioner_factory():
    """Test preconditioner factory."""
    # Create test system
    A = create_test_system()
    
    # Test creating different preconditioners
    precond_types = ['diagonal', 'ilu', 'block_jacobi', 'amg']
    for precond_type in precond_types:
        precond = PreconditionerFactory.create_preconditioner(precond_type)
        precond.setup(A)
        x = np.ones(A.shape[0])
        y = precond.apply(x)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
    
    # Test invalid preconditioner type
    with pytest.raises(ValueError):
        PreconditionerFactory.create_preconditioner('invalid')


def test_preconditioner_effectiveness():
    """Test effectiveness of preconditioners."""
    # Create test system
    A = create_test_system(n=200)
    b = np.ones(A.shape[0])
    
    # Solve without preconditioner
    x0, info0 = sparse.linalg.gmres(A, b, maxiter=100, tol=1e-8)
    iterations0 = info0
    
    # Test each preconditioner
    precond_types = ['diagonal', 'ilu', 'block_jacobi', 'amg']
    for precond_type in precond_types:
        precond = PreconditionerFactory.create_preconditioner(precond_type)
        precond.setup(A)
        
        # Solve with preconditioner
        x1, info1 = sparse.linalg.gmres(A, b, M=precond.apply,
                                      maxiter=100, tol=1e-8)
        iterations1 = info1
        
        # Check that preconditioner helps
        assert iterations1 <= iterations0
        assert np.allclose(A @ x1, b, rtol=1e-8) 
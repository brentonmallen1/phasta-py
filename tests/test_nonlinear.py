"""Tests for nonlinear solvers module."""

import numpy as np
import pytest
from phasta.solver.nonlinear import NonlinearSolver, NewtonKrylov, TrustRegion


def test_newton_krylov():
    """Test Newton-Krylov solver."""
    # Test problem: x^2 - 4 = 0
    def residual(x):
        return np.array([x[0]**2 - 4])
    
    def jacobian(x):
        return np.array([[2 * x[0]]])
    
    # Create solver
    solver = NewtonKrylov(max_iter=100, tol=1e-6)
    
    # Solve system
    x0 = np.array([1.0])
    x = solver.solve(residual, jacobian, x0)
    
    # Check solution
    assert np.allclose(residual(x), 0, rtol=1e-6)
    assert solver.iterations < solver.max_iter
    assert solver.residual < solver.tol
    
    # Test with different initial guess
    x0 = np.array([-1.0])
    x = solver.solve(residual, jacobian, x0)
    assert np.allclose(residual(x), 0, rtol=1e-6)
    assert abs(x[0] + 2) < 1e-6


def test_trust_region():
    """Test Trust-region solver."""
    # Test problem: x^2 + y^2 - 1 = 0, x - y = 0
    def residual(x):
        return np.array([x[0]**2 + x[1]**2 - 1, x[0] - x[1]])
    
    def jacobian(x):
        return np.array([[2 * x[0], 2 * x[1]], [1, -1]])
    
    # Create solver
    solver = TrustRegion(max_iter=100, tol=1e-6)
    
    # Solve system
    x0 = np.array([0.5, 0.5])
    x = solver.solve(residual, jacobian, x0)
    
    # Check solution
    assert np.allclose(residual(x), 0, rtol=1e-6)
    assert solver.iterations < solver.max_iter
    assert solver.residual < solver.tol
    
    # Test with different initial guess
    x0 = np.array([-0.5, -0.5])
    x = solver.solve(residual, jacobian, x0)
    assert np.allclose(residual(x), 0, rtol=1e-6)
    assert np.allclose(x, [-1/np.sqrt(2), -1/np.sqrt(2)], rtol=1e-6)


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test zero residual
    def residual(x):
        return np.zeros_like(x)
    
    def jacobian(x):
        return np.eye(len(x))
    
    # Test Newton-Krylov
    solver = NewtonKrylov()
    x0 = np.array([1.0, 2.0])
    x = solver.solve(residual, jacobian, x0)
    assert np.allclose(x, x0)
    
    # Test Trust-region
    solver = TrustRegion()
    x = solver.solve(residual, jacobian, x0)
    assert np.allclose(x, x0)
    
    # Test singular Jacobian
    def singular_jacobian(x):
        return np.zeros((2, 2))
    
    with pytest.raises(np.linalg.LinAlgError):
        solver = NewtonKrylov()
        solver.solve(residual, singular_jacobian, x0)


def test_memory_management():
    """Test memory management during computations."""
    # Create large system
    n = 100
    def residual(x):
        return x**2 - 1
    
    def jacobian(x):
        return np.diag(2 * x)
    
    # Test Newton-Krylov
    solver = NewtonKrylov()
    x0 = np.ones(n)
    x = solver.solve(residual, jacobian, x0)
    assert np.allclose(residual(x), 0, rtol=1e-6)
    
    # Test Trust-region
    solver = TrustRegion()
    x = solver.solve(residual, jacobian, x0)
    assert np.allclose(residual(x), 0, rtol=1e-6)


def test_convergence_rates():
    """Test convergence rates of solvers."""
    # Test problem with known solution
    def residual(x):
        return np.array([np.exp(x[0]) - 2, x[0]**2 + x[1]**2 - 1])
    
    def jacobian(x):
        return np.array([[np.exp(x[0]), 0], [2 * x[0], 2 * x[1]]])
    
    # Test Newton-Krylov
    solver = NewtonKrylov(max_iter=100, tol=1e-10)
    x0 = np.array([0.5, 0.5])
    x = solver.solve(residual, jacobian, x0)
    assert np.allclose(residual(x), 0, rtol=1e-10)
    assert solver.iterations < 10  # Should converge quickly
    
    # Test Trust-region
    solver = TrustRegion(max_iter=100, tol=1e-10)
    x = solver.solve(residual, jacobian, x0)
    assert np.allclose(residual(x), 0, rtol=1e-10)
    assert solver.iterations < 15  # May take more iterations 
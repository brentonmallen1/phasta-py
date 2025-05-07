"""Tests for solver module."""

import numpy as np
import pytest
from phasta.fem.global_assembly import GlobalAssembly
from phasta.fem.solver import LinearSolver, TimeDependentSolver, NonlinearSolver


def test_linear_solver():
    """Test linear solver."""
    # Create a simple 1D mesh with 3 elements
    nodes = np.array([[0.0], [0.5], [1.0], [1.5]])
    elements = np.array([[0, 1], [1, 2], [2, 3]])
    
    # Create assembly and solver
    assembly = GlobalAssembly('line', order=1)
    solver = LinearSolver(assembly)
    
    # Test with constant source term
    u = solver.solve(nodes, elements, f=1.0)
    expected = np.array([0, 0.125, 0.25, 0.375])
    np.testing.assert_allclose(u, expected)
    
    # Test with function source term
    def source_func(x):
        return x[:, 0]**2
    
    u = solver.solve(nodes, elements, f=source_func)
    expected = np.array([0, 0.03125, 0.125, 0.28125])
    np.testing.assert_allclose(u, expected)
    
    # Test with Dirichlet BC
    dirichlet_nodes = np.array([0, 3])
    dirichlet_values = np.array([0.0, 1.0])
    u = solver.solve(nodes, elements, f=1.0,
                    dirichlet_nodes=dirichlet_nodes,
                    dirichlet_values=dirichlet_values)
    expected = np.array([0.0, 0.375, 0.75, 1.0])
    np.testing.assert_allclose(u, expected)


def test_time_dependent_solver():
    """Test time-dependent solver."""
    # Create a simple 1D mesh with 3 elements
    nodes = np.array([[0.0], [0.5], [1.0], [1.5]])
    elements = np.array([[0, 1], [1, 2], [2, 3]])
    
    # Create assembly and solver
    assembly = GlobalAssembly('line', order=1)
    solver = TimeDependentSolver(assembly, dt=0.1)
    
    # Initial condition
    u0 = np.zeros(4)
    
    # Test with constant source term
    times, u = solver.solve(nodes, elements, u0, t_end=0.5, f=1.0)
    
    # Check time points
    expected_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(times, expected_times)
    
    # Check final solution
    expected_final = np.array([0, 0.125, 0.25, 0.375])
    np.testing.assert_allclose(u[-1], expected_final, rtol=1e-2)
    
    # Test with time-dependent source term
    def source_func(x, t):
        return x[:, 0]**2 * t
    
    times, u = solver.solve(nodes, elements, u0, t_end=0.5, f=source_func)
    
    # Check that solution increases with time
    assert np.all(np.diff(u, axis=0) > 0)


def test_nonlinear_solver():
    """Test nonlinear solver."""
    # Create a simple 1D mesh with 3 elements
    nodes = np.array([[0.0], [0.5], [1.0], [1.5]])
    elements = np.array([[0, 1], [1, 2], [2, 3]])
    
    # Create assembly and solver
    assembly = GlobalAssembly('line', order=1)
    solver = NonlinearSolver(assembly)
    
    # Initial guess
    u0 = np.zeros(4)
    
    # Test with constant source term
    u = solver.solve(nodes, elements, u0, f=1.0)
    expected = np.array([0, 0.125, 0.25, 0.375])
    np.testing.assert_allclose(u, expected)
    
    # Test with function source term
    def source_func(x):
        return x[:, 0]**2
    
    u = solver.solve(nodes, elements, u0, f=source_func)
    expected = np.array([0, 0.03125, 0.125, 0.28125])
    np.testing.assert_allclose(u, expected)
    
    # Test with Dirichlet BC
    dirichlet_nodes = np.array([0, 3])
    dirichlet_values = np.array([0.0, 1.0])
    u = solver.solve(nodes, elements, u0, f=1.0,
                    dirichlet_nodes=dirichlet_nodes,
                    dirichlet_values=dirichlet_values)
    expected = np.array([0.0, 0.375, 0.75, 1.0])
    np.testing.assert_allclose(u, expected)


def test_solver_convergence():
    """Test solver convergence."""
    # Create a simple 1D mesh with 3 elements
    nodes = np.array([[0.0], [0.5], [1.0], [1.5]])
    elements = np.array([[0, 1], [1, 2], [2, 3]])
    
    # Create assembly and solver
    assembly = GlobalAssembly('line', order=1)
    solver = NonlinearSolver(assembly, max_iter=10, tol=1e-8)
    
    # Test that solver converges
    u0 = np.zeros(4)
    u = solver.solve(nodes, elements, u0, f=1.0)
    assert np.all(np.isfinite(u))
    
    # Test that solver raises error for non-convergence
    with pytest.raises(RuntimeError):
        solver = NonlinearSolver(assembly, max_iter=1, tol=1e-20)
        solver.solve(nodes, elements, u0, f=1.0) 
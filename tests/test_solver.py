"""Tests for solver module."""

import numpy as np
import pytest
from phasta.fem.global_assembly import GlobalAssembly
from phasta.fem.solver import LinearSolver, TimeDependentSolver, NonlinearSolver
from unittest.mock import Mock, patch
from pathlib import Path

from phasta.solver.base import (
    FlowSolver, IncompressibleSolver, CompressibleSolver
)
from phasta.mesh.base import Mesh


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


class MockMesh:
    """Mock mesh for testing."""
    
    def __init__(self, num_nodes: int = 100, num_elements: int = 200):
        """Initialize mock mesh.
        
        Args:
            num_nodes: Number of nodes
            num_elements: Number of elements
        """
        self.nodes = np.random.rand(num_nodes, 3)
        self.elements = np.random.randint(0, num_nodes, (num_elements, 4))


def test_flow_solver_base():
    """Test base flow solver class."""
    mesh = MockMesh()
    solver = FlowSolver(mesh)
    
    with pytest.raises(NotImplementedError):
        solver.solve()
    
    with pytest.raises(NotImplementedError):
        solver.update_boundary_conditions()
    
    with pytest.raises(NotImplementedError):
        solver.compute_residuals()


def test_incompressible_solver():
    """Test incompressible flow solver."""
    # Create a simple mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ])
    elements = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ])
    mesh = Mesh(nodes, elements)
    
    # Create solver
    solver = IncompressibleSolver(mesh, dt=0.001, max_iterations=10)
    
    # Set boundary conditions
    solver.set_boundary_condition('inlet', 'velocity', np.array([1.0, 0.0, 0.0]))
    solver.set_boundary_condition('outlet', 'pressure', 0.0)
    
    # Solve
    solver.solve()
    
    # Check solution
    solution = solver.get_solution()
    assert solution['velocity'].shape == (4, 3)
    assert solution['pressure'].shape == (4,)
    assert solution['temperature'].shape == (4,)
    assert solution['density'].shape == (4,)


def test_compressible_solver():
    """Test compressible flow solver."""
    # Create a simple mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ])
    elements = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ])
    mesh = Mesh(nodes, elements)
    
    # Create solver
    solver = CompressibleSolver(mesh, dt=0.001, max_iterations=10)
    
    # Set boundary conditions
    solver.set_boundary_condition('inlet', 'velocity', np.array([1.0, 0.0, 0.0]))
    solver.set_boundary_condition('inlet', 'temperature', 300.0)
    solver.set_boundary_condition('outlet', 'pressure', 101325.0)
    
    # Solve
    solver.solve()
    
    # Check solution
    solution = solver.get_solution()
    assert solution['velocity'].shape == (4, 3)
    assert solution['pressure'].shape == (4,)
    assert solution['temperature'].shape == (4,)
    assert solution['density'].shape == (4,)


def test_boundary_conditions():
    """Test boundary condition handling."""
    mesh = MockMesh()
    
    # Test incompressible solver
    incomp_solver = IncompressibleSolver(mesh)
    incomp_solver.set_boundary_condition('wall', 'no-slip', np.zeros(3))
    assert 'wall' in incomp_solver.boundary_conditions
    assert incomp_solver.boundary_conditions['wall']['type'] == 'no-slip'
    
    # Test compressible solver
    comp_solver = CompressibleSolver(mesh)
    comp_solver.set_boundary_condition('wall', 'adiabatic', None)
    assert 'wall' in comp_solver.boundary_conditions
    assert comp_solver.boundary_conditions['wall']['type'] == 'adiabatic'


def test_solution_arrays():
    """Test solution array initialization."""
    mesh = MockMesh(num_nodes=50)
    
    # Test incompressible solver
    incomp_solver = IncompressibleSolver(mesh)
    solution = incomp_solver.get_solution()
    assert solution['velocity'].shape == (50, 3)
    assert solution['pressure'].shape == (50,)
    assert solution['temperature'].shape == (50,)
    assert solution['density'].shape == (50,)
    
    # Test compressible solver
    comp_solver = CompressibleSolver(mesh)
    solution = comp_solver.get_solution()
    assert solution['velocity'].shape == (50, 3)
    assert solution['pressure'].shape == (50,)
    assert solution['temperature'].shape == (50,)
    assert solution['density'].shape == (50,)


def test_convergence():
    """Test solver convergence."""
    mesh = MockMesh()
    
    # Test incompressible solver
    incomp_solver = IncompressibleSolver(mesh, max_iterations=5, tolerance=1e-3)
    residuals = incomp_solver.compute_residuals()
    assert 'momentum' in residuals
    assert 'continuity' in residuals
    
    # Test compressible solver
    comp_solver = CompressibleSolver(mesh, max_iterations=5, tolerance=1e-3)
    residuals = comp_solver.compute_residuals()
    assert 'mass' in residuals
    assert 'momentum' in residuals
    assert 'energy' in residuals


def test_memory_management():
    """Test memory management during solving."""
    # Create a large mesh
    large_mesh = MockMesh(num_nodes=10000, num_elements=20000)
    
    # Test incompressible solver
    incomp_solver = IncompressibleSolver(large_mesh, max_iterations=2)
    incomp_solver.solve()
    solution = incomp_solver.get_solution()
    assert all(arr.shape[0] == 10000 for arr in solution.values())
    
    # Test compressible solver
    comp_solver = CompressibleSolver(large_mesh, max_iterations=2)
    comp_solver.solve()
    solution = comp_solver.get_solution()
    assert all(arr.shape[0] == 10000 for arr in solution.values())


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test empty mesh
    empty_mesh = MockMesh(num_nodes=0, num_elements=0)
    
    # Test incompressible solver
    incomp_solver = IncompressibleSolver(empty_mesh)
    solution = incomp_solver.get_solution()
    assert all(arr.shape[0] == 0 for arr in solution.values())
    
    # Test compressible solver
    comp_solver = CompressibleSolver(empty_mesh)
    solution = comp_solver.get_solution()
    assert all(arr.shape[0] == 0 for arr in solution.values())
    
    # Test invalid boundary conditions
    mesh = MockMesh()
    solver = IncompressibleSolver(mesh)
    with pytest.raises(ValueError):
        solver.set_boundary_condition('inlet', 'invalid', None) 
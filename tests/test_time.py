"""Tests for time integration module."""

import numpy as np
import pytest
from scipy import sparse
from phasta.fem.time import (
    TimeIntegrator, ExplicitEuler, ImplicitEuler,
    CrankNicolson, BDF2, TimeDependentProblem
)


def test_explicit_euler():
    """Test explicit Euler time integration."""
    # Create simple 1D problem
    n = 10
    dx = 1.0 / (n - 1)
    dt = 0.1 * dx**2  # CFL condition
    
    # Create mass and stiffness matrices
    mass_matrix = sparse.eye(n)
    stiffness_matrix = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)) / dx**2
    
    # Create initial condition (Gaussian)
    x = np.linspace(0, 1, n)
    initial_condition = np.exp(-(x - 0.5)**2 / 0.1)
    
    # Create time-dependent problem
    problem = TimeDependentProblem(
        ExplicitEuler(dt),
        mass_matrix,
        stiffness_matrix,
        initial_condition
    )
    
    # Solve for a few time steps
    time_history, solution_history = problem.solve(0.1)
    
    # Check results
    assert len(time_history) > 1
    assert len(solution_history) == len(time_history)
    assert np.allclose(solution_history[0], initial_condition)
    assert not np.allclose(solution_history[-1], initial_condition)


def test_implicit_euler():
    """Test implicit Euler time integration."""
    # Create simple 1D problem
    n = 10
    dx = 1.0 / (n - 1)
    dt = 0.1  # Can use larger time step than explicit
    
    # Create mass and stiffness matrices
    mass_matrix = sparse.eye(n)
    stiffness_matrix = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)) / dx**2
    
    # Create initial condition (Gaussian)
    x = np.linspace(0, 1, n)
    initial_condition = np.exp(-(x - 0.5)**2 / 0.1)
    
    # Create time-dependent problem
    problem = TimeDependentProblem(
        ImplicitEuler(dt),
        mass_matrix,
        stiffness_matrix,
        initial_condition
    )
    
    # Solve for a few time steps
    time_history, solution_history = problem.solve(0.1)
    
    # Check results
    assert len(time_history) > 1
    assert len(solution_history) == len(time_history)
    assert np.allclose(solution_history[0], initial_condition)
    assert not np.allclose(solution_history[-1], initial_condition)


def test_crank_nicolson():
    """Test Crank-Nicolson time integration."""
    # Create simple 1D problem
    n = 10
    dx = 1.0 / (n - 1)
    dt = 0.1  # Can use larger time step than explicit
    
    # Create mass and stiffness matrices
    mass_matrix = sparse.eye(n)
    stiffness_matrix = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)) / dx**2
    
    # Create initial condition (Gaussian)
    x = np.linspace(0, 1, n)
    initial_condition = np.exp(-(x - 0.5)**2 / 0.1)
    
    # Create time-dependent problem
    problem = TimeDependentProblem(
        CrankNicolson(dt),
        mass_matrix,
        stiffness_matrix,
        initial_condition
    )
    
    # Solve for a few time steps
    time_history, solution_history = problem.solve(0.1)
    
    # Check results
    assert len(time_history) > 1
    assert len(solution_history) == len(time_history)
    assert np.allclose(solution_history[0], initial_condition)
    assert not np.allclose(solution_history[-1], initial_condition)


def test_bdf2():
    """Test BDF2 time integration."""
    # Create simple 1D problem
    n = 10
    dx = 1.0 / (n - 1)
    dt = 0.1  # Can use larger time step than explicit
    
    # Create mass and stiffness matrices
    mass_matrix = sparse.eye(n)
    stiffness_matrix = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)) / dx**2
    
    # Create initial condition (Gaussian)
    x = np.linspace(0, 1, n)
    initial_condition = np.exp(-(x - 0.5)**2 / 0.1)
    
    # Create time-dependent problem
    problem = TimeDependentProblem(
        BDF2(dt),
        mass_matrix,
        stiffness_matrix,
        initial_condition
    )
    
    # Solve for a few time steps
    time_history, solution_history = problem.solve(0.1)
    
    # Check results
    assert len(time_history) > 1
    assert len(solution_history) == len(time_history)
    assert np.allclose(solution_history[0], initial_condition)
    assert not np.allclose(solution_history[-1], initial_condition)


def test_time_dependent_problem():
    """Test time-dependent problem with time-varying right-hand side."""
    # Create simple 1D problem
    n = 10
    dx = 1.0 / (n - 1)
    dt = 0.1
    
    # Create mass and stiffness matrices
    mass_matrix = sparse.eye(n)
    stiffness_matrix = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)) / dx**2
    
    # Create initial condition
    initial_condition = np.zeros(n)
    
    # Create time-varying right-hand side
    def rhs_function(t):
        return np.sin(2 * np.pi * t) * np.ones(n)
    
    # Create time-dependent problem
    problem = TimeDependentProblem(
        ImplicitEuler(dt),
        mass_matrix,
        stiffness_matrix,
        initial_condition,
        rhs_function
    )
    
    # Solve for a few time steps
    time_history, solution_history = problem.solve(0.5)
    
    # Check results
    assert len(time_history) > 1
    assert len(solution_history) == len(time_history)
    assert np.allclose(solution_history[0], initial_condition)
    assert not np.allclose(solution_history[-1], initial_condition)
    
    # Check that solution varies with time
    solution_variation = np.max(np.abs(np.diff(solution_history, axis=0)))
    assert solution_variation > 0 
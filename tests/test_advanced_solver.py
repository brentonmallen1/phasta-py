"""Tests for advanced solver features."""

import numpy as np
import pytest
from phasta.solver.advanced import (
    TimeIntegrator, RK4Integrator, AdaptiveTimeStepper,
    Preconditioner, ILUPreconditioner, AMGPreconditioner
)


def test_time_integrator_base():
    """Test base time integrator class."""
    integrator = TimeIntegrator(dt=0.001)
    
    with pytest.raises(NotImplementedError):
        integrator.step(np.array([1.0]), lambda x: x)


def test_rk4_integrator():
    """Test RK4 time integrator."""
    # Create integrator
    integrator = RK4Integrator(dt=0.001)
    assert integrator.order == 4
    
    # Test on simple ODE: dy/dt = y
    def rhs(y):
        return y
    
    # Initial condition
    y0 = np.array([1.0])
    
    # Take one step
    y1 = integrator.step(y0, rhs)
    
    # Compare with exact solution: y(t) = exp(t)
    exact = np.exp(0.001)
    assert abs(y1[0] - exact) < 1e-10
    
    # Test on system of ODEs
    def system_rhs(y):
        return np.array([y[1], -y[0]])  # Simple harmonic oscillator
    
    y0 = np.array([1.0, 0.0])
    y1 = integrator.step(y0, system_rhs)
    
    # Check conservation of energy
    energy0 = y0[0]**2 + y0[1]**2
    energy1 = y1[0]**2 + y1[1]**2
    assert abs(energy1 - energy0) < 1e-10


def test_adaptive_time_stepper():
    """Test adaptive time stepping."""
    # Create time stepper
    stepper = AdaptiveTimeStepper(
        min_dt=1e-6,
        max_dt=1.0,
        target_error=1e-4,
        safety_factor=0.9
    )
    
    # Test error below target
    dt = stepper.adjust_step(1e-5)
    assert dt > stepper.current_dt
    
    # Test error above target
    dt = stepper.adjust_step(1e-3)
    assert dt < stepper.current_dt
    
    # Test minimum time step
    stepper.current_dt = 1e-5
    dt = stepper.adjust_step(1e-2)
    assert dt >= stepper.min_dt
    
    # Test maximum time step
    stepper.current_dt = 0.5
    dt = stepper.adjust_step(1e-6)
    assert dt <= stepper.max_dt


def test_preconditioner_base():
    """Test base preconditioner class."""
    preconditioner = Preconditioner()
    
    with pytest.raises(NotImplementedError):
        preconditioner.setup(np.eye(2))
    
    with pytest.raises(NotImplementedError):
        preconditioner.apply(np.array([1.0, 1.0]))


def test_ilu_preconditioner():
    """Test ILU preconditioner."""
    # Create preconditioner
    preconditioner = ILUPreconditioner(fill_level=0)
    
    # Test matrix
    A = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 4.0, 1.0],
        [0.0, 1.0, 4.0]
    ])
    
    # Set up preconditioner
    preconditioner.setup(A)
    
    # Test vector
    b = np.array([1.0, 2.0, 3.0])
    
    # Apply preconditioner
    x = preconditioner.apply(b)
    
    # Check result
    assert x.shape == (3,)
    assert not np.allclose(x, b)  # Should be different from input
    
    # Check that L and U are triangular
    assert np.allclose(np.triu(preconditioner.L, 1), 0)
    assert np.allclose(np.tril(preconditioner.U, -1), 0)


def test_amg_preconditioner():
    """Test AMG preconditioner."""
    # Create preconditioner
    preconditioner = AMGPreconditioner(
        max_levels=3,
        coarsening_factor=0.5
    )
    
    # Test matrix (Laplacian)
    n = 100
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2.0
        if i > 0:
            A[i, i-1] = -1.0
        if i < n-1:
            A[i, i+1] = -1.0
    
    # Set up preconditioner
    preconditioner.setup(A)
    
    # Test vector
    b = np.ones(n)
    
    # Apply preconditioner
    x = preconditioner.apply(b)
    
    # Check result
    assert x.shape == (n,)
    assert not np.allclose(x, b)  # Should be different from input
    
    # Check that we have the right number of levels
    assert len(preconditioner.levels) <= preconditioner.max_levels
    
    # Check that coarsest level is small enough
    coarsest_size = preconditioner.levels[-1]['matrix'].shape[0]
    assert coarsest_size < n


def test_memory_management():
    """Test memory management during computations."""
    # Create large system
    n = 1000
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    
    # Test ILU preconditioner
    ilu = ILUPreconditioner()
    ilu.setup(A)
    x = ilu.apply(b)
    assert x.shape == (n,)
    
    # Test AMG preconditioner
    amg = AMGPreconditioner()
    amg.setup(A)
    x = amg.apply(b)
    assert x.shape == (n,)


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test zero matrix
    A = np.zeros((2, 2))
    b = np.array([1.0, 1.0])
    
    # Test ILU preconditioner
    ilu = ILUPreconditioner()
    with pytest.raises(np.linalg.LinAlgError):
        ilu.setup(A)
    
    # Test AMG preconditioner
    amg = AMGPreconditioner()
    with pytest.raises(np.linalg.LinAlgError):
        amg.setup(A)
    
    # Test invalid time step
    stepper = AdaptiveTimeStepper()
    with pytest.raises(ValueError):
        stepper.adjust_step(-1.0)
    
    # Test invalid matrix size
    integrator = RK4Integrator()
    with pytest.raises(ValueError):
        integrator.step(np.array([]), lambda x: x) 
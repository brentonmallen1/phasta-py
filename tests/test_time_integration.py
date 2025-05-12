"""Tests for time integration module."""

import numpy as np
import pytest
from phasta.solver.time_integration import (
    TimeIntegrator, RungeKutta, AdamsBashforth, AdamsMoulton, BDF
)


def test_runge_kutta():
    """Test Runge-Kutta time integrator."""
    # Test problem: y' = -y, y(0) = 1
    def f(t, y):
        return -y
    
    # Initial conditions
    y0 = np.array([1.0])
    t0 = 0.0
    t1 = 1.0
    dt = 0.1
    
    # Test different orders
    for order in [1, 2, 3, 4]:
        integrator = RungeKutta(order=order)
        y, times = integrator.integrate(f, y0, t0, t1, dt)
        
        # Check solution
        assert len(y) == 1
        assert np.all(np.isfinite(y))
        assert len(times) > 0
        assert times[-1] == t1
        
        # Check accuracy
        exact = np.exp(-t1)
        error = abs(y[0] - exact)
        assert error < 1e-3 * (dt ** order)


def test_adams_bashforth():
    """Test Adams-Bashforth time integrator."""
    # Test problem: y' = -y, y(0) = 1
    def f(t, y):
        return -y
    
    # Initial conditions
    y0 = np.array([1.0])
    t0 = 0.0
    t1 = 1.0
    dt = 0.1
    
    # Test different orders
    for order in [1, 2, 3, 4]:
        integrator = AdamsBashforth(order=order)
        y, times = integrator.integrate(f, y0, t0, t1, dt)
        
        # Check solution
        assert len(y) == 1
        assert np.all(np.isfinite(y))
        assert len(times) > 0
        assert times[-1] == t1
        
        # Check accuracy
        exact = np.exp(-t1)
        error = abs(y[0] - exact)
        assert error < 1e-3 * (dt ** order)


def test_adams_moulton():
    """Test Adams-Moulton time integrator."""
    # Test problem: y' = -y, y(0) = 1
    def f(t, y):
        return -y
    
    # Initial conditions
    y0 = np.array([1.0])
    t0 = 0.0
    t1 = 1.0
    dt = 0.1
    
    # Test different orders
    for order in [1, 2, 3, 4]:
        integrator = AdamsMoulton(order=order)
        y, times = integrator.integrate(f, y0, t0, t1, dt)
        
        # Check solution
        assert len(y) == 1
        assert np.all(np.isfinite(y))
        assert len(times) > 0
        assert times[-1] == t1
        
        # Check accuracy
        exact = np.exp(-t1)
        error = abs(y[0] - exact)
        assert error < 1e-3 * (dt ** order)


def test_bdf():
    """Test BDF time integrator."""
    # Test problem: y' = -y, y(0) = 1
    def f(t, y):
        return -y
    
    # Initial conditions
    y0 = np.array([1.0])
    t0 = 0.0
    t1 = 1.0
    dt = 0.1
    
    # Test different orders
    for order in [1, 2, 3, 4, 5, 6]:
        integrator = BDF(order=order)
        y, times = integrator.integrate(f, y0, t0, t1, dt)
        
        # Check solution
        assert len(y) == 1
        assert np.all(np.isfinite(y))
        assert len(times) > 0
        assert times[-1] == t1
        
        # Check accuracy
        exact = np.exp(-t1)
        error = abs(y[0] - exact)
        assert error < 1e-3 * (dt ** order)


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test problem: y' = -y, y(0) = 1
    def f(t, y):
        return -y
    
    # Initial conditions
    y0 = np.array([1.0])
    t0 = 0.0
    t1 = 1.0
    dt = 0.1
    
    # Test invalid order
    with pytest.raises(ValueError):
        RungeKutta(order=5)
    
    with pytest.raises(ValueError):
        AdamsBashforth(order=5)
    
    with pytest.raises(ValueError):
        AdamsMoulton(order=5)
    
    with pytest.raises(ValueError):
        BDF(order=7)
    
    # Test zero time step
    integrator = RungeKutta()
    with pytest.raises(ValueError):
        integrator.integrate(f, y0, t0, t1, 0.0)
    
    # Test negative time step
    with pytest.raises(ValueError):
        integrator.integrate(f, y0, t0, t1, -0.1)
    
    # Test t1 < t0
    with pytest.raises(ValueError):
        integrator.integrate(f, y0, t1, t0, dt)


def test_memory_management():
    """Test memory management during time integration."""
    # Test problem: y' = -y, y(0) = 1
    def f(t, y):
        return -y
    
    # Initial conditions
    y0 = np.array([1.0])
    t0 = 0.0
    t1 = 10.0
    dt = 0.1
    
    # Test different integrators
    integrators = [
        RungeKutta(order=4),
        AdamsBashforth(order=4),
        AdamsMoulton(order=4),
        BDF(order=4)
    ]
    
    for integrator in integrators:
        y, times = integrator.integrate(f, y0, t0, t1, dt)
        
        # Check solution
        assert len(y) == 1
        assert np.all(np.isfinite(y))
        assert len(times) > 0
        assert times[-1] == t1
        
        # Check history size
        assert len(integrator.history) <= integrator.order


def test_convergence():
    """Test convergence rates."""
    # Test problem: y' = -y, y(0) = 1
    def f(t, y):
        return -y
    
    # Initial conditions
    y0 = np.array([1.0])
    t0 = 0.0
    t1 = 1.0
    
    # Test different time steps
    dts = [0.1, 0.05, 0.025, 0.0125]
    errors = []
    
    for dt in dts:
        integrator = RungeKutta(order=4)
        y, _ = integrator.integrate(f, y0, t0, t1, dt)
        exact = np.exp(-t1)
        errors.append(abs(y[0] - exact))
    
    # Check convergence rate
    for i in range(len(errors) - 1):
        rate = np.log2(errors[i] / errors[i+1])
        assert rate >= 3.5  # Should be close to 4 for RK4 
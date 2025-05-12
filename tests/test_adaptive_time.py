"""Tests for adaptive time stepping."""

import numpy as np
import pytest
from phasta.solver.adaptive_time import (
    RelativeErrorEstimator,
    PIDController,
    AdaptiveTimeStepper
)
from phasta.solver.time_integration import RungeKutta


def test_relative_error_estimator():
    """Test relative error estimator."""
    # Create estimator
    estimator = RelativeErrorEstimator(tol=1e-6)
    
    # Test with identical solutions
    y1 = np.array([1.0, 2.0, 3.0])
    y2 = np.array([1.0, 2.0, 3.0])
    error = estimator.estimate_error(y1, y2, 0.1)
    assert error == 0.0
    
    # Test with different solutions
    y2 = np.array([1.1, 2.0, 3.0])
    error = estimator.estimate_error(y1, y2, 0.1)
    assert error == pytest.approx(0.1)
    
    # Test with zero solution
    y1 = np.array([0.0, 0.0, 0.0])
    y2 = np.array([0.1, 0.0, 0.0])
    error = estimator.estimate_error(y1, y2, 0.1)
    assert error == pytest.approx(1.0)


def test_pid_controller():
    """Test PID controller."""
    # Create controller
    controller = PIDController(
        min_dt=0.1,
        max_dt=1.0,
        safety_factor=0.9,
        kp=0.075,
        ki=0.175,
        kd=0.01
    )
    
    # Test with zero error
    dt = controller.compute_step_size(0.5, 0.0, 2)
    assert dt == pytest.approx(1.0)
    
    # Test with large error
    dt = controller.compute_step_size(0.5, 2.0, 2)
    assert dt == pytest.approx(0.1)
    
    # Test with moderate error
    dt = controller.compute_step_size(0.5, 0.5, 2)
    assert 0.1 <= dt <= 1.0
    
    # Test with different orders
    dt1 = controller.compute_step_size(0.5, 0.5, 2)
    dt2 = controller.compute_step_size(0.5, 0.5, 4)
    assert dt1 != dt2


def test_adaptive_time_stepper():
    """Test adaptive time stepper."""
    # Create components
    integrator = RungeKutta(order=4)
    estimator = RelativeErrorEstimator(tol=1e-6)
    controller = PIDController(
        min_dt=0.01,
        max_dt=0.5,
        safety_factor=0.9
    )
    
    # Create stepper
    stepper = AdaptiveTimeStepper(integrator, estimator, controller)
    
    # Test problem: y' = -y, y(0) = 1
    def f(t, y):
        return -y
    
    # Test single step
    y0 = np.array([1.0])
    y_new, new_dt = stepper.step(f, y0, 0.1)
    assert y_new.shape == y0.shape
    assert 0.01 <= new_dt <= 0.5
    
    # Test integration
    y_final, times = stepper.integrate(f, y0, 0.0, 1.0, 0.1)
    assert len(times) > 1
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(1.0)
    assert y_final[0] == pytest.approx(np.exp(-1.0), rel=1e-3)


def test_edge_cases():
    """Test edge cases."""
    # Create components
    integrator = RungeKutta(order=4)
    estimator = RelativeErrorEstimator(tol=1e-6)
    controller = PIDController(
        min_dt=0.01,
        max_dt=0.5,
        safety_factor=0.9
    )
    
    # Create stepper
    stepper = AdaptiveTimeStepper(integrator, estimator, controller)
    
    # Test problem: y' = -y, y(0) = 1
    def f(t, y):
        return -y
    
    # Test with zero time step
    y0 = np.array([1.0])
    with pytest.raises(ValueError):
        stepper.step(f, y0, 0.0)
    
    # Test with negative time step
    with pytest.raises(ValueError):
        stepper.step(f, y0, -0.1)
    
    # Test with final time less than initial time
    with pytest.raises(ValueError):
        stepper.integrate(f, y0, 1.0, 0.0, 0.1)


def test_memory_management():
    """Test memory management."""
    # Create components
    integrator = RungeKutta(order=4)
    estimator = RelativeErrorEstimator(tol=1e-6)
    controller = PIDController(
        min_dt=0.01,
        max_dt=0.5,
        safety_factor=0.9
    )
    
    # Create stepper
    stepper = AdaptiveTimeStepper(integrator, estimator, controller)
    
    # Test problem: y' = -y, y(0) = 1
    def f(t, y):
        return -y
    
    # Test with large number of steps
    y0 = np.array([1.0])
    y_final, times = stepper.integrate(f, y0, 0.0, 10.0, 0.1)
    assert len(times) > 1
    assert times[-1] == pytest.approx(10.0)
    assert y_final[0] == pytest.approx(np.exp(-10.0), rel=1e-3)


def test_convergence():
    """Test convergence with refinement."""
    # Create components
    integrator = RungeKutta(order=4)
    estimator = RelativeErrorEstimator(tol=1e-6)
    controller = PIDController(
        min_dt=0.01,
        max_dt=0.5,
        safety_factor=0.9
    )
    
    # Create stepper
    stepper = AdaptiveTimeStepper(integrator, estimator, controller)
    
    # Test problem: y' = -y, y(0) = 1
    def f(t, y):
        return -y
    
    # Test with different tolerances
    y0 = np.array([1.0])
    exact = np.exp(-1.0)
    
    # Test with loose tolerance
    estimator.tol = 1e-3
    y_final, times1 = stepper.integrate(f, y0, 0.0, 1.0, 0.1)
    error1 = abs(y_final[0] - exact)
    
    # Test with tight tolerance
    estimator.tol = 1e-6
    y_final, times2 = stepper.integrate(f, y0, 0.0, 1.0, 0.1)
    error2 = abs(y_final[0] - exact)
    
    # Check that tighter tolerance gives better accuracy
    assert error2 < error1
    assert len(times2) > len(times1) 
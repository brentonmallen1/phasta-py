"""Adaptive time stepping module.

This module provides functionality for:
- Error estimation
- Time step control
- Adaptive time stepping
"""

import numpy as np
from typing import Callable, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod
from phasta.solver.time_integration import TimeIntegrator, RungeKutta

logger = logging.getLogger(__name__)


class ErrorEstimator(ABC):
    """Base class for error estimators."""
    
    @abstractmethod
    def estimate_error(self,
                      y1: np.ndarray,
                      y2: np.ndarray,
                      dt: float) -> float:
        """Estimate error between two solutions.
        
        Args:
            y1: First solution
            y2: Second solution
            dt: Time step size
            
        Returns:
            Error estimate
        """
        pass


class RelativeErrorEstimator(ErrorEstimator):
    """Relative error estimator."""
    
    def __init__(self, tol: float = 1e-6):
        """Initialize relative error estimator.
        
        Args:
            tol: Error tolerance
        """
        self.tol = tol
    
    def estimate_error(self,
                      y1: np.ndarray,
                      y2: np.ndarray,
                      dt: float) -> float:
        """Estimate relative error between two solutions.
        
        Args:
            y1: First solution
            y2: Second solution
            dt: Time step size
            
        Returns:
            Relative error estimate
        """
        return np.max(np.abs(y1 - y2) / (np.abs(y1) + self.tol))


class TimeStepController(ABC):
    """Base class for time step controllers."""
    
    def __init__(self,
                 min_dt: float = 1e-10,
                 max_dt: float = 1.0,
                 safety_factor: float = 0.9):
        """Initialize time step controller.
        
        Args:
            min_dt: Minimum time step size
            max_dt: Maximum time step size
            safety_factor: Safety factor for time step adjustment
        """
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.safety_factor = safety_factor
    
    @abstractmethod
    def compute_step_size(self,
                         dt: float,
                         error: float,
                         order: int) -> float:
        """Compute new time step size.
        
        Args:
            dt: Current time step size
            error: Error estimate
            order: Order of accuracy
            
        Returns:
            New time step size
        """
        pass


class PIDController(TimeStepController):
    """PID controller for time step control."""
    
    def __init__(self,
                 min_dt: float = 1e-10,
                 max_dt: float = 1.0,
                 safety_factor: float = 0.9,
                 kp: float = 0.075,
                 ki: float = 0.175,
                 kd: float = 0.01):
        """Initialize PID controller.
        
        Args:
            min_dt: Minimum time step size
            max_dt: Maximum time step size
            safety_factor: Safety factor for time step adjustment
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        super().__init__(min_dt, max_dt, safety_factor)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_history = []
    
    def compute_step_size(self,
                         dt: float,
                         error: float,
                         order: int) -> float:
        """Compute new time step size using PID control.
        
        Args:
            dt: Current time step size
            error: Error estimate
            order: Order of accuracy
            
        Returns:
            New time step size
        """
        # Store error history
        self.error_history.append(error)
        if len(self.error_history) > 3:
            self.error_history.pop(0)
        
        # Compute PID terms
        if len(self.error_history) == 1:
            p_term = self.kp * (1.0 - error)
            i_term = 0.0
            d_term = 0.0
        else:
            p_term = self.kp * (1.0 - error)
            i_term = self.ki * sum(1.0 - e for e in self.error_history)
            d_term = self.kd * (error - self.error_history[-2])
        
        # Compute new time step size
        factor = self.safety_factor * (p_term + i_term + d_term)
        new_dt = dt * factor ** (1.0 / order)
        
        # Apply bounds
        new_dt = max(self.min_dt, min(self.max_dt, new_dt))
        
        return new_dt


class AdaptiveTimeStepper:
    """Adaptive time stepper."""
    
    def __init__(self,
                 integrator: TimeIntegrator,
                 error_estimator: ErrorEstimator,
                 controller: TimeStepController):
        """Initialize adaptive time stepper.
        
        Args:
            integrator: Time integrator
            error_estimator: Error estimator
            controller: Time step controller
        """
        self.integrator = integrator
        self.error_estimator = error_estimator
        self.controller = controller
        self.dt = 0.0
        self.t = 0.0
        self.history = []
    
    def step(self,
             f: Callable[[float, np.ndarray], np.ndarray],
             y: np.ndarray,
             dt: float) -> Tuple[np.ndarray, float]:
        """Take an adaptive time step.
        
        Args:
            f: Right-hand side function f(t, y)
            y: Current solution
            dt: Initial time step size
            
        Returns:
            Tuple of (updated solution, new time step size)
        """
        # Take two steps with half the time step
        y1 = self.integrator.step(f, y, dt)
        y2 = self.integrator.step(f, y1, dt)
        
        # Take one step with full time step
        y3 = self.integrator.step(f, y, 2 * dt)
        
        # Estimate error
        error = self.error_estimator.estimate_error(y2, y3, dt)
        
        # Compute new time step size
        new_dt = self.controller.compute_step_size(
            dt, error, self.integrator.order
        )
        
        # Accept or reject step
        if error <= 1.0:
            return y2, new_dt
        else:
            return y, new_dt / 2.0
    
    def integrate(self,
                 f: Callable[[float, np.ndarray], np.ndarray],
                 y0: np.ndarray,
                 t0: float,
                 t1: float,
                 dt0: float) -> Tuple[np.ndarray, List[float]]:
        """Integrate system over time interval with adaptive time stepping.
        
        Args:
            f: Right-hand side function f(t, y)
            y0: Initial solution
            t0: Initial time
            t1: Final time
            dt0: Initial time step size
            
        Returns:
            Tuple of (final solution, time points)
        """
        self.dt = dt0
        self.t = t0
        y = y0.copy()
        times = [t0]
        
        while self.t < t1:
            # Take step
            y_new, new_dt = self.step(f, y, self.dt)
            
            # Update solution if step was accepted
            if y_new is not y:
                y = y_new
                self.t += self.dt
                times.append(self.t)
            
            # Update time step size
            self.dt = min(new_dt, t1 - self.t)
        
        return y, times 
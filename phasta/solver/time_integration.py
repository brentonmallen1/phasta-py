"""High-order time integration module.

This module provides functionality for:
- High-order Runge-Kutta methods
- Adams-Bashforth methods
- Adams-Moulton methods
- Backward differentiation formulas (BDF)
"""

import numpy as np
from typing import Callable, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TimeIntegrator(ABC):
    """Base class for time integrators."""
    
    def __init__(self, order: int = 1):
        """Initialize time integrator.
        
        Args:
            order: Order of accuracy
        """
        self.order = order
        self.dt = 0.0
        self.t = 0.0
        self.history = []
    
    @abstractmethod
    def step(self,
             f: Callable[[float, np.ndarray], np.ndarray],
             y: np.ndarray,
             dt: float) -> np.ndarray:
        """Take a time step.
        
        Args:
            f: Right-hand side function f(t, y)
            y: Current solution
            dt: Time step size
            
        Returns:
            Updated solution
        """
        pass
    
    def integrate(self,
                 f: Callable[[float, np.ndarray], np.ndarray],
                 y0: np.ndarray,
                 t0: float,
                 t1: float,
                 dt: float) -> Tuple[np.ndarray, List[float]]:
        """Integrate system over time interval.
        
        Args:
            f: Right-hand side function f(t, y)
            y0: Initial solution
            t0: Initial time
            t1: Final time
            dt: Time step size
            
        Returns:
            Tuple of (final solution, time points)
        """
        self.dt = dt
        self.t = t0
        y = y0.copy()
        times = [t0]
        
        while self.t < t1:
            y = self.step(f, y, dt)
            self.t += dt
            times.append(self.t)
        
        return y, times


class RungeKutta(TimeIntegrator):
    """Runge-Kutta time integrator."""
    
    def __init__(self, order: int = 4):
        """Initialize Runge-Kutta integrator.
        
        Args:
            order: Order of accuracy (1, 2, 3, or 4)
        """
        super().__init__(order)
        self._setup_coefficients()
    
    def _setup_coefficients(self):
        """Setup Runge-Kutta coefficients."""
        if self.order == 1:
            # Forward Euler
            self.a = np.array([[0]])
            self.b = np.array([1])
            self.c = np.array([0])
        elif self.order == 2:
            # Heun's method
            self.a = np.array([[0, 0],
                             [1, 0]])
            self.b = np.array([0.5, 0.5])
            self.c = np.array([0, 1])
        elif self.order == 3:
            # Kutta's method
            self.a = np.array([[0, 0, 0],
                             [0.5, 0, 0],
                             [-1, 2, 0]])
            self.b = np.array([1/6, 2/3, 1/6])
            self.c = np.array([0, 0.5, 1])
        elif self.order == 4:
            # Classical RK4
            self.a = np.array([[0, 0, 0, 0],
                             [0.5, 0, 0, 0],
                             [0, 0.5, 0, 0],
                             [0, 0, 1, 0]])
            self.b = np.array([1/6, 1/3, 1/3, 1/6])
            self.c = np.array([0, 0.5, 0.5, 1])
        else:
            raise ValueError(f"Unsupported order: {self.order}")
    
    def step(self,
             f: Callable[[float, np.ndarray], np.ndarray],
             y: np.ndarray,
             dt: float) -> np.ndarray:
        """Take a time step using Runge-Kutta method.
        
        Args:
            f: Right-hand side function f(t, y)
            y: Current solution
            dt: Time step size
            
        Returns:
            Updated solution
        """
        n_stages = len(self.b)
        k = np.zeros((n_stages, len(y)))
        
        # Compute stages
        for i in range(n_stages):
            t_i = self.t + self.c[i] * dt
            y_i = y.copy()
            for j in range(i):
                y_i += dt * self.a[i, j] * k[j]
            k[i] = f(t_i, y_i)
        
        # Update solution
        y_new = y.copy()
        for i in range(n_stages):
            y_new += dt * self.b[i] * k[i]
        
        return y_new


class AdamsBashforth(TimeIntegrator):
    """Adams-Bashforth time integrator."""
    
    def __init__(self, order: int = 2):
        """Initialize Adams-Bashforth integrator.
        
        Args:
            order: Order of accuracy (1-4)
        """
        super().__init__(order)
        self._setup_coefficients()
    
    def _setup_coefficients(self):
        """Setup Adams-Bashforth coefficients."""
        if self.order == 1:
            # Forward Euler
            self.beta = np.array([1])
        elif self.order == 2:
            self.beta = np.array([3/2, -1/2])
        elif self.order == 3:
            self.beta = np.array([23/12, -16/12, 5/12])
        elif self.order == 4:
            self.beta = np.array([55/24, -59/24, 37/24, -9/24])
        else:
            raise ValueError(f"Unsupported order: {self.order}")
    
    def step(self,
             f: Callable[[float, np.ndarray], np.ndarray],
             y: np.ndarray,
             dt: float) -> np.ndarray:
        """Take a time step using Adams-Bashforth method.
        
        Args:
            f: Right-hand side function f(t, y)
            y: Current solution
            dt: Time step size
            
        Returns:
            Updated solution
        """
        # Store current function evaluation
        self.history.append(f(self.t, y))
        
        # Keep only necessary history
        if len(self.history) > self.order:
            self.history.pop(0)
        
        # Compute solution
        y_new = y.copy()
        for i, beta in enumerate(self.beta):
            if i < len(self.history):
                y_new += dt * beta * self.history[-(i+1)]
        
        return y_new


class AdamsMoulton(TimeIntegrator):
    """Adams-Moulton time integrator."""
    
    def __init__(self, order: int = 2):
        """Initialize Adams-Moulton integrator.
        
        Args:
            order: Order of accuracy (1-4)
        """
        super().__init__(order)
        self._setup_coefficients()
    
    def _setup_coefficients(self):
        """Setup Adams-Moulton coefficients."""
        if self.order == 1:
            # Backward Euler
            self.beta = np.array([1])
        elif self.order == 2:
            self.beta = np.array([1/2, 1/2])
        elif self.order == 3:
            self.beta = np.array([5/12, 8/12, -1/12])
        elif self.order == 4:
            self.beta = np.array([9/24, 19/24, -5/24, 1/24])
        else:
            raise ValueError(f"Unsupported order: {self.order}")
    
    def step(self,
             f: Callable[[float, np.ndarray], np.ndarray],
             y: np.ndarray,
             dt: float) -> np.ndarray:
        """Take a time step using Adams-Moulton method.
        
        Args:
            f: Right-hand side function f(t, y)
            y: Current solution
            dt: Time step size
            
        Returns:
            Updated solution
        """
        # Store current function evaluation
        self.history.append(f(self.t, y))
        
        # Keep only necessary history
        if len(self.history) > self.order - 1:
            self.history.pop(0)
        
        # Predictor step (Adams-Bashforth)
        y_pred = y.copy()
        for i in range(len(self.history)):
            y_pred += dt * self.beta[i] * self.history[-(i+1)]
        
        # Corrector step
        y_new = y.copy()
        for i in range(len(self.history)):
            y_new += dt * self.beta[i] * self.history[-(i+1)]
        y_new += dt * self.beta[-1] * f(self.t + dt, y_pred)
        
        return y_new


class BDF(TimeIntegrator):
    """Backward differentiation formula time integrator."""
    
    def __init__(self, order: int = 2):
        """Initialize BDF integrator.
        
        Args:
            order: Order of accuracy (1-6)
        """
        super().__init__(order)
        self._setup_coefficients()
    
    def _setup_coefficients(self):
        """Setup BDF coefficients."""
        if self.order == 1:
            # Backward Euler
            self.alpha = np.array([1, -1])
            self.beta = np.array([1, 0])
        elif self.order == 2:
            self.alpha = np.array([3/2, -2, 1/2])
            self.beta = np.array([1, 0, 0])
        elif self.order == 3:
            self.alpha = np.array([11/6, -3, 3/2, -1/3])
            self.beta = np.array([1, 0, 0, 0])
        elif self.order == 4:
            self.alpha = np.array([25/12, -4, 3, -4/3, 1/4])
            self.beta = np.array([1, 0, 0, 0, 0])
        elif self.order == 5:
            self.alpha = np.array([137/60, -5, 5, -10/3, 5/4, -1/5])
            self.beta = np.array([1, 0, 0, 0, 0, 0])
        elif self.order == 6:
            self.alpha = np.array([147/60, -6, 15/2, -20/3, 15/4, -6/5, 1/6])
            self.beta = np.array([1, 0, 0, 0, 0, 0, 0])
        else:
            raise ValueError(f"Unsupported order: {self.order}")
    
    def step(self,
             f: Callable[[float, np.ndarray], np.ndarray],
             y: np.ndarray,
             dt: float) -> np.ndarray:
        """Take a time step using BDF method.
        
        Args:
            f: Right-hand side function f(t, y)
            y: Current solution
            dt: Time step size
            
        Returns:
            Updated solution
        """
        # Store solution history
        self.history.append(y)
        
        # Keep only necessary history
        if len(self.history) > self.order:
            self.history.pop(0)
        
        # Predictor step
        y_pred = np.zeros_like(y)
        for i in range(len(self.history)):
            y_pred += self.alpha[i] * self.history[-(i+1)]
        
        # Corrector step
        y_new = y_pred.copy()
        for i in range(len(self.history) - 1):
            y_new += dt * self.beta[i] * f(self.t - i * dt, self.history[-(i+1)])
        y_new += dt * self.beta[-1] * f(self.t + dt, y_pred)
        
        return y_new 
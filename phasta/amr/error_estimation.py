"""
Error estimation framework for Adaptive Mesh Refinement (AMR).

This module provides the base classes and interfaces for error estimation
in AMR, including solution-based, physics-based, and user-defined indicators.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
import logging

if TYPE_CHECKING:
    from ..mesh import Mesh

class ErrorIndicator(ABC):
    """Base class for error indicators."""
    
    def __init__(self, name: str):
        """Initialize error indicator.
        
        Args:
            name: Indicator name
        """
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def compute(self, solution: Dict[str, np.ndarray],
                mesh: 'Mesh') -> np.ndarray:
        """Compute error indicator.
        
        Args:
            solution: Solution variables
            mesh: Mesh object
            
        Returns:
            Error indicator values for each element
        """
        pass
    
    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize error indicator values.
        
        Args:
            values: Raw error indicator values
            
        Returns:
            Normalized values in [0, 1]
        """
        if values.size == 0:
            return values
        vmin, vmax = values.min(), values.max()
        if vmax > vmin:
            return (values - vmin) / (vmax - vmin)
        return np.zeros_like(values)

class GradientBasedIndicator(ErrorIndicator):
    """Gradient-based error indicator."""
    
    def __init__(self, variable: str, order: int = 1):
        """Initialize gradient-based indicator.
        
        Args:
            variable: Solution variable name
            order: Gradient order (1 for first-order, 2 for second-order)
        """
        super().__init__(f"gradient_{variable}")
        self.variable = variable
        self.order = order
    
    def compute(self, solution: Dict[str, np.ndarray],
                mesh: 'Mesh') -> np.ndarray:
        """Compute gradient-based error indicator.
        
        Args:
            solution: Solution variables
            mesh: Mesh object
            
        Returns:
            Gradient-based error indicator values
        """
        if self.variable not in solution:
            self.logger.warning(f"Variable {self.variable} not found in solution")
            return np.zeros(mesh.num_elements)
        
        # Compute gradients
        if self.order == 1:
            return self._compute_first_order(solution[self.variable], mesh)
        else:
            return self._compute_second_order(solution[self.variable], mesh)
    
    def _compute_first_order(self, values: np.ndarray,
                           mesh: 'Mesh') -> np.ndarray:
        """Compute first-order gradient indicator.
        
        Args:
            values: Solution values
            mesh: Mesh object
            
        Returns:
            First-order gradient indicator values
        """
        # Compute element gradients
        gradients = mesh.compute_gradients(values)
        
        # Compute gradient magnitude
        return np.sqrt(np.sum(gradients**2, axis=1))
    
    def _compute_second_order(self, values: np.ndarray,
                            mesh: 'Mesh') -> np.ndarray:
        """Compute second-order gradient indicator.
        
        Args:
            values: Solution values
            mesh: Mesh object
            
        Returns:
            Second-order gradient indicator values
        """
        # Compute first-order gradients
        first_order = self._compute_first_order(values, mesh)
        
        # Compute gradients of first-order gradients
        second_order = mesh.compute_gradients(first_order)
        
        # Compute second-order gradient magnitude
        return np.sqrt(np.sum(second_order**2, axis=1))

class JumpBasedIndicator(ErrorIndicator):
    """Jump-based error indicator."""
    
    def __init__(self, variable: str):
        """Initialize jump-based indicator.
        
        Args:
            variable: Solution variable name
        """
        super().__init__(f"jump_{variable}")
        self.variable = variable
    
    def compute(self, solution: Dict[str, np.ndarray],
                mesh: 'Mesh') -> np.ndarray:
        """Compute jump-based error indicator.
        
        Args:
            solution: Solution variables
            mesh: Mesh object
            
        Returns:
            Jump-based error indicator values
        """
        if self.variable not in solution:
            self.logger.warning(f"Variable {self.variable} not found in solution")
            return np.zeros(mesh.num_elements)
        
        # Get element values
        element_values = mesh.get_element_values(solution[self.variable])
        
        # Compute jumps across faces
        jumps = mesh.compute_face_jumps(element_values)
        
        # Compute jump magnitude
        return np.sqrt(np.sum(jumps**2, axis=1))

class PhysicsBasedIndicator(ErrorIndicator):
    """Physics-based error indicator."""
    
    def __init__(self, variables: List[str], weights: Optional[List[float]] = None):
        """Initialize physics-based indicator.
        
        Args:
            variables: List of physics variables
            weights: Optional weights for each variable
        """
        super().__init__("physics")
        self.variables = variables
        self.weights = weights if weights is not None else [1.0] * len(variables)
    
    def compute(self, solution: Dict[str, np.ndarray],
                mesh: 'Mesh') -> np.ndarray:
        """Compute physics-based error indicator.
        
        Args:
            solution: Solution variables
            mesh: Mesh object
            
        Returns:
            Physics-based error indicator values
        """
        # Check if all variables are present
        missing = [v for v in self.variables if v not in solution]
        if missing:
            self.logger.warning(f"Variables {missing} not found in solution")
            return np.zeros(mesh.num_elements)
        
        # Compute indicators for each variable
        indicators = []
        for var, weight in zip(self.variables, self.weights):
            grad_indicator = GradientBasedIndicator(var)
            jump_indicator = JumpBasedIndicator(var)
            
            grad_values = grad_indicator.compute(solution, mesh)
            jump_values = jump_indicator.compute(solution, mesh)
            
            # Combine indicators
            combined = weight * (grad_values + jump_values)
            indicators.append(combined)
        
        # Combine all indicators
        return np.sum(indicators, axis=0)

class UserDefinedIndicator(ErrorIndicator):
    """User-defined error indicator."""
    
    def __init__(self, name: str, function: callable):
        """Initialize user-defined indicator.
        
        Args:
            name: Indicator name
            function: User-defined function that computes the indicator
        """
        super().__init__(name)
        self.function = function
    
    def compute(self, solution: Dict[str, np.ndarray],
                mesh: 'Mesh') -> np.ndarray:
        """Compute user-defined error indicator.
        
        Args:
            solution: Solution variables
            mesh: Mesh object
            
        Returns:
            User-defined error indicator values
        """
        try:
            return self.function(solution, mesh)
        except Exception as e:
            self.logger.error(f"Error in user-defined indicator: {e}")
            return np.zeros(mesh.num_elements)

class ErrorEstimator:
    """Error estimator that combines multiple indicators."""
    
    def __init__(self):
        """Initialize error estimator."""
        self.indicators: List[ErrorIndicator] = []
        self.weights: List[float] = []
        self.logger = logging.getLogger(__name__)
    
    def add_indicator(self, indicator: ErrorIndicator, weight: float = 1.0):
        """Add error indicator.
        
        Args:
            indicator: Error indicator
            weight: Indicator weight
        """
        self.indicators.append(indicator)
        self.weights.append(weight)
    
    def compute(self, solution: Dict[str, np.ndarray],
                mesh: 'Mesh') -> np.ndarray:
        """Compute combined error indicator.
        
        Args:
            solution: Solution variables
            mesh: Mesh object
            
        Returns:
            Combined error indicator values
        """
        if not self.indicators:
            self.logger.warning("No indicators added to estimator")
            return np.zeros(mesh.num_elements)
        
        # Compute individual indicators
        indicators = []
        for indicator, weight in zip(self.indicators, self.weights):
            values = indicator.compute(solution, mesh)
            normalized = indicator.normalize(values)
            indicators.append(weight * normalized)
        
        # Combine indicators
        return np.sum(indicators, axis=0)
    
    def get_refinement_flags(self, values: np.ndarray,
                           thresholds: Tuple[float, float]) -> np.ndarray:
        """Get refinement flags based on error values.
        
        Args:
            values: Error indicator values
            thresholds: (coarsen_threshold, refine_threshold)
            
        Returns:
            Array of refinement flags (-1 for coarsen, 0 for keep, 1 for refine)
        """
        coarsen_threshold, refine_threshold = thresholds
        flags = np.zeros_like(values, dtype=np.int8)
        
        # Set coarsening flags
        flags[values < coarsen_threshold] = -1
        
        # Set refinement flags
        flags[values > refine_threshold] = 1
        
        return flags 
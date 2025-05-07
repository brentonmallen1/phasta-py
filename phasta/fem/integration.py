"""Numerical integration for finite elements."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from .basis import LagrangeShapeFunction
from .metrics import ElementMetrics


class QuadratureRule:
    """Base class for quadrature rules."""
    
    def __init__(self, element_type: str, order: int):
        """Initialize quadrature rule.
        
        Args:
            element_type: Type of element ('line', 'tri', 'quad', 'tet', 'hex')
            order: Polynomial order to integrate exactly
        """
        self.element_type = element_type
        self.order = order
        self._setup_quadrature()
    
    def _setup_quadrature(self):
        """Set up quadrature points and weights."""
        if self.element_type == 'line':
            self._setup_line_quadrature()
        elif self.element_type == 'tri':
            self._setup_tri_quadrature()
        elif self.element_type == 'quad':
            self._setup_quad_quadrature()
        elif self.element_type == 'tet':
            self._setup_tet_quadrature()
        elif self.element_type == 'hex':
            self._setup_hex_quadrature()
        else:
            raise ValueError(f"Unsupported element type: {self.element_type}")
    
    def _setup_line_quadrature(self):
        """Set up quadrature for line elements."""
        if self.order <= 1:
            # 1-point rule (order 1)
            self.points = np.array([[0.0]])
            self.weights = np.array([2.0])
        elif self.order <= 3:
            # 2-point rule (order 3)
            self.points = np.array([[-0.577350269189626, 0.577350269189626]])
            self.weights = np.array([1.0, 1.0])
        else:
            # 3-point rule (order 5)
            self.points = np.array([[-0.774596669241483, 0.0, 0.774596669241483]])
            self.weights = np.array([0.555555555555556, 0.888888888888889, 0.555555555555556])
    
    def _setup_tri_quadrature(self):
        """Set up quadrature for triangular elements."""
        if self.order <= 1:
            # 1-point rule (order 1)
            self.points = np.array([[1/3, 1/3]])
            self.weights = np.array([0.5])
        elif self.order <= 2:
            # 3-point rule (order 2)
            self.points = np.array([
                [1/6, 1/6],
                [2/3, 1/6],
                [1/6, 2/3]
            ])
            self.weights = np.array([1/6, 1/6, 1/6])
        else:
            # 7-point rule (order 5)
            a = 0.101286507323456
            b = 0.470142064105115
            c = 0.797426985353087
            self.points = np.array([
                [1/3, 1/3],
                [a, a],
                [1-2*a, a],
                [a, 1-2*a],
                [b, b],
                [1-2*b, b],
                [b, 1-2*b]
            ])
            self.weights = np.array([
                0.225,
                0.125939180544827,
                0.125939180544827,
                0.125939180544827,
                0.132394152788506,
                0.132394152788506,
                0.132394152788506
            ])
    
    def _setup_quad_quadrature(self):
        """Set up quadrature for quadrilateral elements."""
        if self.order <= 1:
            # 1-point rule (order 1)
            self.points = np.array([[0.0, 0.0]])
            self.weights = np.array([4.0])
        elif self.order <= 3:
            # 2x2 rule (order 3)
            g = 0.577350269189626
            self.points = np.array([
                [-g, -g], [g, -g],
                [-g, g], [g, g]
            ])
            self.weights = np.array([1.0, 1.0, 1.0, 1.0])
        else:
            # 3x3 rule (order 5)
            g1 = 0.774596669241483
            g2 = 0.0
            w1 = 0.555555555555556
            w2 = 0.888888888888889
            self.points = np.array([
                [-g1, -g1], [g2, -g1], [g1, -g1],
                [-g1, g2], [g2, g2], [g1, g2],
                [-g1, g1], [g2, g1], [g1, g1]
            ])
            self.weights = np.array([
                w1*w1, w2*w1, w1*w1,
                w1*w2, w2*w2, w1*w2,
                w1*w1, w2*w1, w1*w1
            ])
    
    def _setup_tet_quadrature(self):
        """Set up quadrature for tetrahedral elements."""
        if self.order <= 1:
            # 1-point rule (order 1)
            self.points = np.array([[0.25, 0.25, 0.25]])
            self.weights = np.array([1/6])
        elif self.order <= 2:
            # 4-point rule (order 2)
            a = 0.585410196624969
            b = 0.138196601125011
            self.points = np.array([
                [b, b, b],
                [a, b, b],
                [b, a, b],
                [b, b, a]
            ])
            self.weights = np.array([1/24, 1/24, 1/24, 1/24])
        else:
            # 11-point rule (order 4)
            a = 0.399403576166799
            b = 0.100596423833201
            c = 0.585410196624969
            d = 0.138196601125011
            self.points = np.array([
                [0.25, 0.25, 0.25],
                [a, b, b], [b, a, b], [b, b, a],
                [b, a, a], [a, b, a], [a, a, b],
                [c, d, d], [d, c, d], [d, d, c]
            ])
            self.weights = np.array([
                0.007622222222222,
                0.024888888888889,
                0.024888888888889,
                0.024888888888889,
                0.024888888888889,
                0.024888888888889,
                0.024888888888889,
                0.03125,
                0.03125,
                0.03125
            ])
    
    def _setup_hex_quadrature(self):
        """Set up quadrature for hexahedral elements."""
        if self.order <= 1:
            # 1-point rule (order 1)
            self.points = np.array([[0.0, 0.0, 0.0]])
            self.weights = np.array([8.0])
        elif self.order <= 3:
            # 2x2x2 rule (order 3)
            g = 0.577350269189626
            self.points = np.array([
                [-g, -g, -g], [g, -g, -g],
                [-g, g, -g], [g, g, -g],
                [-g, -g, g], [g, -g, g],
                [-g, g, g], [g, g, g]
            ])
            self.weights = np.array([1.0] * 8)
        else:
            # 3x3x3 rule (order 5)
            g1 = 0.774596669241483
            g2 = 0.0
            w1 = 0.555555555555556
            w2 = 0.888888888888889
            points = []
            weights = []
            for z in [-g1, g2, g1]:
                for y in [-g1, g2, g1]:
                    for x in [-g1, g2, g1]:
                        points.append([x, y, z])
                        weights.append(w1 * w1 * w1 if x != 0 and y != 0 and z != 0
                                     else w1 * w1 * w2 if (x == 0) != (y == 0) != (z == 0)
                                     else w1 * w2 * w2 if (x == 0) + (y == 0) + (z == 0) == 2
                                     else w2 * w2 * w2)
            self.points = np.array(points)
            self.weights = np.array(weights)


class ElementIntegrator:
    """Class for performing element integrations."""
    
    def __init__(self, element_type: str, order: int = 1):
        """Initialize element integrator.
        
        Args:
            element_type: Type of element ('line', 'tri', 'quad', 'tet', 'hex')
            order: Polynomial order of the element
        """
        self.element_type = element_type
        self.order = order
        self.quadrature = QuadratureRule(element_type, order)
        self.metrics = ElementMetrics(element_type, order)
        self.shape_function = LagrangeShapeFunction(element_type, order)
    
    def integrate(self, nodes: np.ndarray, func: callable) -> float:
        """Integrate a function over an element.
        
        Args:
            nodes: Physical coordinates of element nodes, shape (n_nodes, dim)
            func: Function to integrate, should take (xi, nodes) as arguments
                  and return values at quadrature points
            
        Returns:
            Integral value
        """
        # Get quadrature points and weights
        xi = self.quadrature.points
        weights = self.quadrature.weights
        
        # Compute Jacobian determinants at quadrature points
        _, detJ = self.metrics.compute_jacobian(xi, nodes)
        
        # Evaluate function at quadrature points
        f_vals = func(xi, nodes)
        
        # Compute integral
        integral = np.sum(weights * detJ * f_vals)
        
        return integral
    
    def integrate_field(self, nodes: np.ndarray, field: np.ndarray) -> float:
        """Integrate a field over an element.
        
        Args:
            nodes: Physical coordinates of element nodes, shape (n_nodes, dim)
            field: Field values at element nodes, shape (n_nodes,)
            
        Returns:
            Integral value
        """
        def field_func(xi, nodes):
            # Evaluate shape functions at quadrature points
            N = self.shape_function.evaluate(xi)
            # Interpolate field values
            return np.sum(N * field, axis=1)
        
        return self.integrate(nodes, field_func)
    
    def integrate_gradient(self, nodes: np.ndarray, field: np.ndarray) -> np.ndarray:
        """Integrate field gradient over an element.
        
        Args:
            nodes: Physical coordinates of element nodes, shape (n_nodes, dim)
            field: Field values at element nodes, shape (n_nodes,)
            
        Returns:
            Gradient integral, shape (dim,)
        """
        def grad_func(xi, nodes):
            # Get physical derivatives at quadrature points
            dN_phys = self.metrics.compute_physical_derivatives(xi, nodes)
            # Compute gradient at quadrature points
            grad = np.zeros((xi.shape[0], nodes.shape[1]))
            for i in range(xi.shape[0]):
                grad[i] = np.sum(dN_phys[i] * field[:, np.newaxis], axis=0)
            return grad
        
        return self.integrate(nodes, grad_func)

"""Tests for numerical integration."""

import numpy as np
import pytest
from phasta.fem.integration import QuadratureRule, ElementIntegrator


def test_line_quadrature():
    """Test quadrature rules for line elements."""
    # Test order 1 rule
    quad = QuadratureRule('line', order=1)
    assert quad.points.shape == (1, 1)
    assert quad.weights.shape == (1,)
    np.testing.assert_allclose(quad.points, [[0.0]])
    np.testing.assert_allclose(quad.weights, [2.0])
    
    # Test order 3 rule
    quad = QuadratureRule('line', order=3)
    assert quad.points.shape == (2, 1)
    assert quad.weights.shape == (2,)
    g = 0.577350269189626
    np.testing.assert_allclose(quad.points, [[-g], [g]])
    np.testing.assert_allclose(quad.weights, [1.0, 1.0])
    
    # Test order 5 rule
    quad = QuadratureRule('line', order=5)
    assert quad.points.shape == (3, 1)
    assert quad.weights.shape == (3,)
    g1 = 0.774596669241483
    g2 = 0.0
    w1 = 0.555555555555556
    w2 = 0.888888888888889
    np.testing.assert_allclose(quad.points, [[-g1], [g2], [g1]])
    np.testing.assert_allclose(quad.weights, [w1, w2, w1])


def test_tri_quadrature():
    """Test quadrature rules for triangular elements."""
    # Test order 1 rule
    quad = QuadratureRule('tri', order=1)
    assert quad.points.shape == (1, 2)
    assert quad.weights.shape == (1,)
    np.testing.assert_allclose(quad.points, [[1/3, 1/3]])
    np.testing.assert_allclose(quad.weights, [0.5])
    
    # Test order 2 rule
    quad = QuadratureRule('tri', order=2)
    assert quad.points.shape == (3, 2)
    assert quad.weights.shape == (3,)
    np.testing.assert_allclose(quad.points, [[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
    np.testing.assert_allclose(quad.weights, [1/6, 1/6, 1/6])


def test_quad_quadrature():
    """Test quadrature rules for quadrilateral elements."""
    # Test order 1 rule
    quad = QuadratureRule('quad', order=1)
    assert quad.points.shape == (1, 2)
    assert quad.weights.shape == (1,)
    np.testing.assert_allclose(quad.points, [[0.0, 0.0]])
    np.testing.assert_allclose(quad.weights, [4.0])
    
    # Test order 3 rule
    quad = QuadratureRule('quad', order=3)
    assert quad.points.shape == (4, 2)
    assert quad.weights.shape == (4,)
    g = 0.577350269189626
    np.testing.assert_allclose(quad.points, [[-g, -g], [g, -g], [-g, g], [g, g]])
    np.testing.assert_allclose(quad.weights, [1.0, 1.0, 1.0, 1.0])


def test_tet_quadrature():
    """Test quadrature rules for tetrahedral elements."""
    # Test order 1 rule
    quad = QuadratureRule('tet', order=1)
    assert quad.points.shape == (1, 3)
    assert quad.weights.shape == (1,)
    np.testing.assert_allclose(quad.points, [[0.25, 0.25, 0.25]])
    np.testing.assert_allclose(quad.weights, [1/6])
    
    # Test order 2 rule
    quad = QuadratureRule('tet', order=2)
    assert quad.points.shape == (4, 3)
    assert quad.weights.shape == (4,)
    a = 0.585410196624969
    b = 0.138196601125011
    np.testing.assert_allclose(quad.points, [[b, b, b], [a, b, b], [b, a, b], [b, b, a]])
    np.testing.assert_allclose(quad.weights, [1/24, 1/24, 1/24, 1/24])


def test_hex_quadrature():
    """Test quadrature rules for hexahedral elements."""
    # Test order 1 rule
    quad = QuadratureRule('hex', order=1)
    assert quad.points.shape == (1, 3)
    assert quad.weights.shape == (1,)
    np.testing.assert_allclose(quad.points, [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(quad.weights, [8.0])
    
    # Test order 3 rule
    quad = QuadratureRule('hex', order=3)
    assert quad.points.shape == (8, 3)
    assert quad.weights.shape == (8,)
    g = 0.577350269189626
    points = np.array([
        [-g, -g, -g], [g, -g, -g],
        [-g, g, -g], [g, g, -g],
        [-g, -g, g], [g, -g, g],
        [-g, g, g], [g, g, g]
    ])
    np.testing.assert_allclose(quad.points, points)
    np.testing.assert_allclose(quad.weights, [1.0] * 8)


def test_invalid_element_type():
    """Test that invalid element types raise ValueError."""
    with pytest.raises(ValueError):
        QuadratureRule('invalid', order=1)


def test_line_integration():
    """Test integration over line elements."""
    integrator = ElementIntegrator('line', order=1)
    
    # Test constant function
    nodes = np.array([[0.0], [2.0]])
    def const_func(xi, nodes):
        return np.ones(xi.shape[0])
    integral = integrator.integrate(nodes, const_func)
    np.testing.assert_allclose(integral, 2.0)
    
    # Test linear function
    def linear_func(xi, nodes):
        return 2 * xi[:, 0] + 1
    integral = integrator.integrate(nodes, linear_func)
    np.testing.assert_allclose(integral, 4.0)
    
    # Test field integration
    field = np.array([1.0, 2.0])
    integral = integrator.integrate_field(nodes, field)
    np.testing.assert_allclose(integral, 3.0)


def test_tri_integration():
    """Test integration over triangular elements."""
    integrator = ElementIntegrator('tri', order=1)
    
    # Create a right triangle
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    
    # Test constant function
    def const_func(xi, nodes):
        return np.ones(xi.shape[0])
    integral = integrator.integrate(nodes, const_func)
    np.testing.assert_allclose(integral, 0.5)
    
    # Test field integration
    field = np.array([1.0, 2.0, 3.0])
    integral = integrator.integrate_field(nodes, field)
    np.testing.assert_allclose(integral, 1.0)
    
    # Test gradient integration
    grad = integrator.integrate_gradient(nodes, field)
    np.testing.assert_allclose(grad, [1.0, 1.0])


def test_quad_integration():
    """Test integration over quadrilateral elements."""
    integrator = ElementIntegrator('quad', order=1)
    
    # Create a unit square
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    
    # Test constant function
    def const_func(xi, nodes):
        return np.ones(xi.shape[0])
    integral = integrator.integrate(nodes, const_func)
    np.testing.assert_allclose(integral, 4.0)
    
    # Test field integration
    field = np.array([1.0, 2.0, 3.0, 4.0])
    integral = integrator.integrate_field(nodes, field)
    np.testing.assert_allclose(integral, 10.0)
    
    # Test gradient integration
    grad = integrator.integrate_gradient(nodes, field)
    np.testing.assert_allclose(grad, [1.0, 1.0]) 
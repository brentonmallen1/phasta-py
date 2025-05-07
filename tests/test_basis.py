"""Tests for finite element basis functions."""

import numpy as np
import pytest
from phasta.fem.basis import LagrangeShapeFunction


def test_line_shape_functions():
    """Test 1D line element shape functions."""
    shape = LagrangeShapeFunction('line', order=1)
    
    # Test points in natural coordinates
    xi = np.array([[-1.0], [0.0], [1.0]])
    
    # Evaluate shape functions
    N = shape.evaluate(xi)
    
    # Check shape
    assert N.shape == (3, 2)
    
    # Check values at nodes
    np.testing.assert_allclose(N[0], [1.0, 0.0])  # xi = -1
    np.testing.assert_allclose(N[1], [0.5, 0.5])  # xi = 0
    np.testing.assert_allclose(N[2], [0.0, 1.0])  # xi = 1
    
    # Test derivatives
    dN = shape.evaluate_derivatives(xi)
    
    # Check shape
    assert dN.shape == (3, 2, 1)
    
    # Check derivative values (constant for linear elements)
    np.testing.assert_allclose(dN[:, 0, 0], -0.5)  # dN1/dxi
    np.testing.assert_allclose(dN[:, 1, 0], 0.5)   # dN2/dxi


def test_tri_shape_functions():
    """Test 2D triangular element shape functions."""
    shape = LagrangeShapeFunction('tri', order=1)
    
    # Test points in natural coordinates
    xi = np.array([
        [0.0, 0.0],  # Node 1
        [1.0, 0.0],  # Node 2
        [0.0, 1.0],  # Node 3
        [1/3, 1/3]   # Interior point
    ])
    
    # Evaluate shape functions
    N = shape.evaluate(xi)
    
    # Check shape
    assert N.shape == (4, 3)
    
    # Check values at nodes
    np.testing.assert_allclose(N[0], [1.0, 0.0, 0.0])  # Node 1
    np.testing.assert_allclose(N[1], [0.0, 1.0, 0.0])  # Node 2
    np.testing.assert_allclose(N[2], [0.0, 0.0, 1.0])  # Node 3
    
    # Check partition of unity at interior point
    np.testing.assert_allclose(np.sum(N[3]), 1.0)
    
    # Test derivatives
    dN = shape.evaluate_derivatives(xi)
    
    # Check shape
    assert dN.shape == (4, 3, 2)
    
    # Check derivative values (constant for linear elements)
    np.testing.assert_allclose(dN[:, 0, 0], -1.0)  # dN1/dxi
    np.testing.assert_allclose(dN[:, 0, 1], -1.0)  # dN1/deta
    np.testing.assert_allclose(dN[:, 1, 0], 1.0)   # dN2/dxi
    np.testing.assert_allclose(dN[:, 1, 1], 0.0)   # dN2/deta
    np.testing.assert_allclose(dN[:, 2, 0], 0.0)   # dN3/dxi
    np.testing.assert_allclose(dN[:, 2, 1], 1.0)   # dN3/deta


def test_quad_shape_functions():
    """Test 2D quadrilateral element shape functions."""
    shape = LagrangeShapeFunction('quad', order=1)
    
    # Test points in natural coordinates
    xi = np.array([
        [-1.0, -1.0],  # Node 1
        [1.0, -1.0],   # Node 2
        [1.0, 1.0],    # Node 3
        [-1.0, 1.0],   # Node 4
        [0.0, 0.0]     # Center
    ])
    
    # Evaluate shape functions
    N = shape.evaluate(xi)
    
    # Check shape
    assert N.shape == (5, 4)
    
    # Check values at nodes
    np.testing.assert_allclose(N[0], [1.0, 0.0, 0.0, 0.0])  # Node 1
    np.testing.assert_allclose(N[1], [0.0, 1.0, 0.0, 0.0])  # Node 2
    np.testing.assert_allclose(N[2], [0.0, 0.0, 1.0, 0.0])  # Node 3
    np.testing.assert_allclose(N[3], [0.0, 0.0, 0.0, 1.0])  # Node 4
    
    # Check partition of unity at center
    np.testing.assert_allclose(np.sum(N[4]), 1.0)
    
    # Test derivatives
    dN = shape.evaluate_derivatives(xi)
    
    # Check shape
    assert dN.shape == (5, 4, 2)
    
    # Check derivative values at nodes
    # Node 1
    np.testing.assert_allclose(dN[0, 0, 0], -0.25)  # dN1/dxi
    np.testing.assert_allclose(dN[0, 0, 1], -0.25)  # dN1/deta
    # Node 2
    np.testing.assert_allclose(dN[1, 1, 0], 0.25)   # dN2/dxi
    np.testing.assert_allclose(dN[1, 1, 1], -0.25)  # dN2/deta
    # Node 3
    np.testing.assert_allclose(dN[2, 2, 0], 0.25)   # dN3/dxi
    np.testing.assert_allclose(dN[2, 2, 1], 0.25)   # dN3/deta
    # Node 4
    np.testing.assert_allclose(dN[3, 3, 0], -0.25)  # dN4/dxi
    np.testing.assert_allclose(dN[3, 3, 1], 0.25)   # dN4/deta


def test_tet_shape_functions():
    """Test 3D tetrahedral element shape functions."""
    shape = LagrangeShapeFunction('tet', order=1)
    
    # Test points in natural coordinates
    xi = np.array([
        [0.0, 0.0, 0.0],  # Node 1
        [1.0, 0.0, 0.0],  # Node 2
        [0.0, 1.0, 0.0],  # Node 3
        [0.0, 0.0, 1.0],  # Node 4
        [0.25, 0.25, 0.25]  # Interior point
    ])
    
    # Evaluate shape functions
    N = shape.evaluate(xi)
    
    # Check shape
    assert N.shape == (5, 4)
    
    # Check values at nodes
    np.testing.assert_allclose(N[0], [1.0, 0.0, 0.0, 0.0])  # Node 1
    np.testing.assert_allclose(N[1], [0.0, 1.0, 0.0, 0.0])  # Node 2
    np.testing.assert_allclose(N[2], [0.0, 0.0, 1.0, 0.0])  # Node 3
    np.testing.assert_allclose(N[3], [0.0, 0.0, 0.0, 1.0])  # Node 4
    
    # Check partition of unity at interior point
    np.testing.assert_allclose(np.sum(N[4]), 1.0)
    
    # Test derivatives
    dN = shape.evaluate_derivatives(xi)
    
    # Check shape
    assert dN.shape == (5, 4, 3)
    
    # Check derivative values (constant for linear elements)
    # Node 1
    np.testing.assert_allclose(dN[:, 0, 0], -1.0)  # dN1/dxi
    np.testing.assert_allclose(dN[:, 0, 1], -1.0)  # dN1/deta
    np.testing.assert_allclose(dN[:, 0, 2], -1.0)  # dN1/dzeta
    # Node 2
    np.testing.assert_allclose(dN[:, 1, 0], 1.0)   # dN2/dxi
    np.testing.assert_allclose(dN[:, 1, 1], 0.0)   # dN2/deta
    np.testing.assert_allclose(dN[:, 1, 2], 0.0)   # dN2/dzeta
    # Node 3
    np.testing.assert_allclose(dN[:, 2, 0], 0.0)   # dN3/dxi
    np.testing.assert_allclose(dN[:, 2, 1], 1.0)   # dN3/deta
    np.testing.assert_allclose(dN[:, 2, 2], 0.0)   # dN3/dzeta
    # Node 4
    np.testing.assert_allclose(dN[:, 3, 0], 0.0)   # dN4/dxi
    np.testing.assert_allclose(dN[:, 3, 1], 0.0)   # dN4/deta
    np.testing.assert_allclose(dN[:, 3, 2], 1.0)   # dN4/dzeta


def test_hex_shape_functions():
    """Test 3D hexahedral element shape functions."""
    shape = LagrangeShapeFunction('hex', order=1)
    
    # Test points in natural coordinates
    xi = np.array([
        [-1.0, -1.0, -1.0],  # Node 1
        [1.0, -1.0, -1.0],   # Node 2
        [1.0, 1.0, -1.0],    # Node 3
        [-1.0, 1.0, -1.0],   # Node 4
        [-1.0, -1.0, 1.0],   # Node 5
        [1.0, -1.0, 1.0],    # Node 6
        [1.0, 1.0, 1.0],     # Node 7
        [-1.0, 1.0, 1.0],    # Node 8
        [0.0, 0.0, 0.0]      # Center
    ])
    
    # Evaluate shape functions
    N = shape.evaluate(xi)
    
    # Check shape
    assert N.shape == (9, 8)
    
    # Check values at nodes
    np.testing.assert_allclose(N[0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Node 1
    np.testing.assert_allclose(N[1], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Node 2
    np.testing.assert_allclose(N[2], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Node 3
    np.testing.assert_allclose(N[3], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # Node 4
    np.testing.assert_allclose(N[4], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Node 5
    np.testing.assert_allclose(N[5], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # Node 6
    np.testing.assert_allclose(N[6], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # Node 7
    np.testing.assert_allclose(N[7], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Node 8
    
    # Check partition of unity at center
    np.testing.assert_allclose(np.sum(N[8]), 1.0)
    
    # Test derivatives
    dN = shape.evaluate_derivatives(xi)
    
    # Check shape
    assert dN.shape == (9, 8, 3)
    
    # Check derivative values at nodes
    # Node 1
    np.testing.assert_allclose(dN[0, 0, 0], -0.125)  # dN1/dxi
    np.testing.assert_allclose(dN[0, 0, 1], -0.125)  # dN1/deta
    np.testing.assert_allclose(dN[0, 0, 2], -0.125)  # dN1/dzeta
    # Node 2
    np.testing.assert_allclose(dN[1, 1, 0], 0.125)   # dN2/dxi
    np.testing.assert_allclose(dN[1, 1, 1], -0.125)  # dN2/deta
    np.testing.assert_allclose(dN[1, 1, 2], -0.125)  # dN2/dzeta
    # Node 3
    np.testing.assert_allclose(dN[2, 2, 0], 0.125)   # dN3/dxi
    np.testing.assert_allclose(dN[2, 2, 1], 0.125)   # dN3/deta
    np.testing.assert_allclose(dN[2, 2, 2], -0.125)  # dN3/dzeta
    # Node 4
    np.testing.assert_allclose(dN[3, 3, 0], -0.125)  # dN4/dxi
    np.testing.assert_allclose(dN[3, 3, 1], 0.125)   # dN4/deta
    np.testing.assert_allclose(dN[3, 3, 2], -0.125)  # dN4/dzeta


def test_invalid_element_type():
    """Test that invalid element types raise ValueError."""
    with pytest.raises(ValueError):
        LagrangeShapeFunction('invalid', order=1)


def test_invalid_order():
    """Test that invalid orders raise ValueError."""
    with pytest.raises(ValueError):
        LagrangeShapeFunction('line', order=0)


def test_higher_order_not_implemented():
    """Test that higher order elements raise NotImplementedError."""
    shape = LagrangeShapeFunction('line', order=2)
    xi = np.array([[-1.0], [0.0], [1.0]])
    
    with pytest.raises(NotImplementedError):
        shape.evaluate(xi)
    
    with pytest.raises(NotImplementedError):
        shape.evaluate_derivatives(xi) 
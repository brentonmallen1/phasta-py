"""Tests for element metrics and Jacobian calculations."""

import numpy as np
import pytest
from phasta.fem.metrics import ElementMetrics


def test_line_jacobian():
    """Test Jacobian computation for 1D line element."""
    metrics = ElementMetrics('line', order=1)
    
    # Create a line element from x=0 to x=2
    nodes = np.array([[0.0], [2.0]])
    
    # Test points in natural coordinates
    xi = np.array([[-1.0], [0.0], [1.0]])
    
    # Compute Jacobian
    J, detJ = metrics.compute_jacobian(xi, nodes)
    
    # Check Jacobian matrices
    expected_J = np.array([[[1.0]], [[1.0]], [[1.0]]])
    np.testing.assert_allclose(J, expected_J)
    
    # Check determinants
    expected_detJ = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(detJ, expected_detJ)
    
    # Test inverse Jacobian
    invJ = metrics.compute_inverse_jacobian(J)
    expected_invJ = np.array([[[1.0]], [[1.0]], [[1.0]]])
    np.testing.assert_allclose(invJ, expected_invJ)
    
    # Test physical derivatives
    dN_phys = metrics.compute_physical_derivatives(xi, nodes)
    expected_dN_phys = np.array([[[-0.5]], [[-0.5]], [[-0.5]]])
    np.testing.assert_allclose(dN_phys[:, 0], expected_dN_phys)
    
    # Test element length
    length = metrics.compute_element_volume(nodes)
    np.testing.assert_allclose(length, 2.0)


def test_tri_jacobian():
    """Test Jacobian computation for 2D triangular element."""
    metrics = ElementMetrics('tri', order=1)
    
    # Create a right triangle
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    
    # Test points in natural coordinates
    xi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1/3, 1/3]])
    
    # Compute Jacobian
    J, detJ = metrics.compute_jacobian(xi, nodes)
    
    # Check Jacobian matrices (should be constant for linear triangle)
    expected_J = np.array([[[1.0, 0.0], [0.0, 1.0]]] * 4)
    np.testing.assert_allclose(J, expected_J)
    
    # Check determinants
    expected_detJ = np.array([1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(detJ, expected_detJ)
    
    # Test inverse Jacobian
    invJ = metrics.compute_inverse_jacobian(J)
    expected_invJ = np.array([[[1.0, 0.0], [0.0, 1.0]]] * 4)
    np.testing.assert_allclose(invJ, expected_invJ)
    
    # Test physical derivatives
    dN_phys = metrics.compute_physical_derivatives(xi, nodes)
    expected_dN_phys = np.array([
        [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
        [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
        [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
        [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]
    ])
    np.testing.assert_allclose(dN_phys, expected_dN_phys)
    
    # Test element area
    area = metrics.compute_element_volume(nodes)
    np.testing.assert_allclose(area, 0.5)


def test_quad_jacobian():
    """Test Jacobian computation for 2D quadrilateral element."""
    metrics = ElementMetrics('quad', order=1)
    
    # Create a unit square
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    
    # Test points in natural coordinates
    xi = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]])
    
    # Compute Jacobian
    J, detJ = metrics.compute_jacobian(xi, nodes)
    
    # Check Jacobian matrices (should be constant for bilinear quad)
    expected_J = np.array([[[1.0, 0.0], [0.0, 1.0]]] * 5)
    np.testing.assert_allclose(J, expected_J)
    
    # Check determinants
    expected_detJ = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(detJ, expected_detJ)
    
    # Test inverse Jacobian
    invJ = metrics.compute_inverse_jacobian(J)
    expected_invJ = np.array([[[1.0, 0.0], [0.0, 1.0]]] * 5)
    np.testing.assert_allclose(invJ, expected_invJ)
    
    # Test element area
    area = metrics.compute_element_volume(nodes)
    np.testing.assert_allclose(area, 4.0)


def test_tet_jacobian():
    """Test Jacobian computation for 3D tetrahedral element."""
    metrics = ElementMetrics('tet', order=1)
    
    # Create a regular tetrahedron
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Test points in natural coordinates
    xi = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.25, 0.25, 0.25]
    ])
    
    # Compute Jacobian
    J, detJ = metrics.compute_jacobian(xi, nodes)
    
    # Check Jacobian matrices (should be constant for linear tet)
    expected_J = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 5)
    np.testing.assert_allclose(J, expected_J)
    
    # Check determinants
    expected_detJ = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(detJ, expected_detJ)
    
    # Test inverse Jacobian
    invJ = metrics.compute_inverse_jacobian(J)
    expected_invJ = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 5)
    np.testing.assert_allclose(invJ, expected_invJ)
    
    # Test element volume
    volume = metrics.compute_element_volume(nodes)
    np.testing.assert_allclose(volume, 1/6)


def test_hex_jacobian():
    """Test Jacobian computation for 3D hexahedral element."""
    metrics = ElementMetrics('hex', order=1)
    
    # Create a unit cube
    nodes = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0]
    ])
    
    # Test points in natural coordinates
    xi = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0]
    ])
    
    # Compute Jacobian
    J, detJ = metrics.compute_jacobian(xi, nodes)
    
    # Check Jacobian matrices (should be constant for trilinear hex)
    expected_J = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 9)
    np.testing.assert_allclose(J, expected_J)
    
    # Check determinants
    expected_detJ = np.array([1.0] * 9)
    np.testing.assert_allclose(detJ, expected_detJ)
    
    # Test inverse Jacobian
    invJ = metrics.compute_inverse_jacobian(J)
    expected_invJ = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 9)
    np.testing.assert_allclose(invJ, expected_invJ)
    
    # Test element volume
    volume = metrics.compute_element_volume(nodes)
    np.testing.assert_allclose(volume, 8.0)


def test_invalid_element_type():
    """Test that invalid element types raise ValueError."""
    with pytest.raises(ValueError):
        ElementMetrics('invalid', order=1) 
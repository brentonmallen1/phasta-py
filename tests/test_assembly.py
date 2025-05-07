"""Tests for element assembly."""

import numpy as np
import pytest
from phasta.fem.assembly import ElementAssembly


def test_mass_matrix():
    """Test mass matrix computation."""
    # Test line element
    assembly = ElementAssembly('line', order=1)
    nodes = np.array([[0.0], [1.0]])
    M = assembly.compute_mass_matrix(nodes)
    expected = np.array([[1/3, 1/6], [1/6, 1/3]])
    np.testing.assert_allclose(M, expected)
    
    # Test triangular element
    assembly = ElementAssembly('tri', order=1)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    M = assembly.compute_mass_matrix(nodes)
    expected = np.array([[1/6, 1/12, 1/12],
                        [1/12, 1/6, 1/12],
                        [1/12, 1/12, 1/6]])
    np.testing.assert_allclose(M, expected)


def test_stiffness_matrix():
    """Test stiffness matrix computation."""
    # Test line element
    assembly = ElementAssembly('line', order=1)
    nodes = np.array([[0.0], [1.0]])
    K = assembly.compute_stiffness_matrix(nodes)
    expected = np.array([[1, -1], [-1, 1]])
    np.testing.assert_allclose(K, expected)
    
    # Test triangular element
    assembly = ElementAssembly('tri', order=1)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    K = assembly.compute_stiffness_matrix(nodes)
    expected = np.array([[1, -0.5, -0.5],
                        [-0.5, 0.5, 0],
                        [-0.5, 0, 0.5]])
    np.testing.assert_allclose(K, expected)


def test_load_vector():
    """Test load vector computation."""
    # Test constant source term
    assembly = ElementAssembly('line', order=1)
    nodes = np.array([[0.0], [1.0]])
    F = assembly.compute_load_vector(nodes, f=1.0)
    expected = np.array([0.5, 0.5])
    np.testing.assert_allclose(F, expected)
    
    # Test function source term
    def source_func(x):
        return x[:, 0]**2
    
    F = assembly.compute_load_vector(nodes, f=source_func)
    expected = np.array([1/12, 1/4])
    np.testing.assert_allclose(F, expected)


def test_convection_matrix():
    """Test convection matrix computation."""
    assembly = ElementAssembly('line', order=1)
    nodes = np.array([[0.0], [1.0]])
    v = np.array([1.0])
    C = assembly.compute_convection_matrix(nodes, v)
    expected = np.array([[-0.5, 0.5], [-0.5, 0.5]])
    np.testing.assert_allclose(C, expected)
    
    # Test 2D convection
    assembly = ElementAssembly('tri', order=1)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    v = np.array([1.0, 1.0])
    C = assembly.compute_convection_matrix(nodes, v)
    expected = np.array([[0, 0.5, 0.5],
                        [-0.5, 0, 0.5],
                        [-0.5, -0.5, 0]])
    np.testing.assert_allclose(C, expected)


def test_boundary_vector():
    """Test boundary vector computation."""
    # Test constant boundary condition
    assembly = ElementAssembly('line', order=1)
    nodes = np.array([[0.0], [1.0]])
    G = assembly.compute_boundary_vector(nodes, g=1.0)
    expected = np.array([0.5, 0.5])
    np.testing.assert_allclose(G, expected)
    
    # Test function boundary condition
    def boundary_func(x):
        return x[:, 0]**2
    
    G = assembly.compute_boundary_vector(nodes, g=boundary_func)
    expected = np.array([1/12, 1/4])
    np.testing.assert_allclose(G, expected)


def test_invalid_element_type():
    """Test that invalid element types raise ValueError."""
    with pytest.raises(ValueError):
        ElementAssembly('invalid', order=1) 
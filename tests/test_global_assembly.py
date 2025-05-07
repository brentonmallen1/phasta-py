"""Tests for global assembly."""

import numpy as np
import pytest
from scipy import sparse
from phasta.fem.global_assembly import GlobalAssembly


def test_mass_matrix_assembly():
    """Test global mass matrix assembly."""
    # Create a simple 1D mesh with 3 elements
    nodes = np.array([[0.0], [0.5], [1.0], [1.5]])
    elements = np.array([[0, 1], [1, 2], [2, 3]])
    
    # Assemble global mass matrix
    assembly = GlobalAssembly('line', order=1)
    M = assembly.assemble_mass_matrix(nodes, elements)
    
    # Expected result (scaled by element length)
    expected = np.array([
        [1/6, 1/12, 0, 0],
        [1/12, 1/3, 1/12, 0],
        [0, 1/12, 1/3, 1/12],
        [0, 0, 1/12, 1/6]
    ]) * 0.5  # Scale by element length
    
    np.testing.assert_allclose(M.toarray(), expected)


def test_stiffness_matrix_assembly():
    """Test global stiffness matrix assembly."""
    # Create a simple 1D mesh with 3 elements
    nodes = np.array([[0.0], [0.5], [1.0], [1.5]])
    elements = np.array([[0, 1], [1, 2], [2, 3]])
    
    # Assemble global stiffness matrix
    assembly = GlobalAssembly('line', order=1)
    K = assembly.assemble_stiffness_matrix(nodes, elements)
    
    # Expected result (scaled by element length)
    expected = np.array([
        [2, -2, 0, 0],
        [-2, 4, -2, 0],
        [0, -2, 4, -2],
        [0, 0, -2, 2]
    ]) / 0.5  # Scale by element length
    
    np.testing.assert_allclose(K.toarray(), expected)


def test_load_vector_assembly():
    """Test global load vector assembly."""
    # Create a simple 1D mesh with 3 elements
    nodes = np.array([[0.0], [0.5], [1.0], [1.5]])
    elements = np.array([[0, 1], [1, 2], [2, 3]])
    
    # Test constant source term
    assembly = GlobalAssembly('line', order=1)
    F = assembly.assemble_load_vector(nodes, elements, f=1.0)
    
    # Expected result (scaled by element length)
    expected = np.array([0.25, 0.5, 0.5, 0.25]) * 0.5  # Scale by element length
    
    np.testing.assert_allclose(F, expected)
    
    # Test function source term
    def source_func(x):
        return x[:, 0]**2
    
    F = assembly.assemble_load_vector(nodes, elements, f=source_func)
    expected = np.array([0, 0.0625, 0.25, 0.5625]) * 0.5  # Scale by element length
    np.testing.assert_allclose(F, expected)


def test_convection_matrix_assembly():
    """Test global convection matrix assembly."""
    # Create a simple 1D mesh with 3 elements
    nodes = np.array([[0.0], [0.5], [1.0], [1.5]])
    elements = np.array([[0, 1], [1, 2], [2, 3]])
    
    # Test constant velocity
    assembly = GlobalAssembly('line', order=1)
    v = np.array([1.0])
    C = assembly.assemble_convection_matrix(nodes, elements, v)
    
    # Expected result
    expected = np.array([
        [-0.5, 0.5, 0, 0],
        [-0.5, 0, 0.5, 0],
        [0, -0.5, 0, 0.5],
        [0, 0, -0.5, 0.5]
    ])
    
    np.testing.assert_allclose(C.toarray(), expected)
    
    # Test function velocity
    def velocity_func(x):
        return np.array([x[0]])
    
    C = assembly.assemble_convection_matrix(nodes, elements, v=velocity_func)
    # Note: The expected result will be different due to velocity variation
    # We just check that the matrix has the right shape and properties
    assert C.shape == (4, 4)
    assert np.allclose(C.sum(axis=1), 0)  # Conservation property


def test_dirichlet_bc():
    """Test application of Dirichlet boundary conditions."""
    # Create a simple 1D mesh with 3 elements
    nodes = np.array([[0.0], [0.5], [1.0], [1.5]])
    elements = np.array([[0, 1], [1, 2], [2, 3]])
    
    # Assemble global stiffness matrix and load vector
    assembly = GlobalAssembly('line', order=1)
    K = assembly.assemble_stiffness_matrix(nodes, elements)
    F = assembly.assemble_load_vector(nodes, elements, f=1.0)
    
    # Apply Dirichlet BC at first and last nodes
    dirichlet_nodes = np.array([0, 3])
    dirichlet_values = np.array([0.0, 1.0])
    K_mod, F_mod = assembly.apply_dirichlet_bc(K, F, dirichlet_nodes, dirichlet_values)
    
    # Check that rows and columns are zeroed out
    assert np.all(K_mod[0, :].toarray() == 0)
    assert np.all(K_mod[3, :].toarray() == 0)
    assert np.all(K_mod[:, 0].toarray() == 0)
    assert np.all(K_mod[:, 3].toarray() == 0)
    
    # Check that diagonal entries are 1
    assert K_mod[0, 0] == 1
    assert K_mod[3, 3] == 1
    
    # Check that load vector values are set correctly
    assert F_mod[0] == 0.0
    assert F_mod[3] == 1.0


def test_invalid_element_type():
    """Test that invalid element types raise ValueError."""
    with pytest.raises(ValueError):
        GlobalAssembly('invalid', order=1) 
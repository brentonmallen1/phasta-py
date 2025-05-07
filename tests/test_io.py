"""Tests for I/O module."""

import os
import numpy as np
import pytest
from phasta.fem.io import MeshIO, SolutionIO, PHASTAIO


def test_mesh_io():
    """Test mesh I/O functionality."""
    # Create test mesh
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    elements = {
        'triangle': np.array([
            [0, 1, 2],
            [1, 3, 2]
        ])
    }
    metadata = {
        'point_data': {
            'temperature': np.array([0.0, 1.0, 0.0, 1.0])
        }
    }
    
    # Create I/O handler
    mesh_io = MeshIO()
    
    # Write mesh
    mesh_io.write_mesh('test_mesh.vtk', nodes, elements, metadata)
    
    # Read mesh
    nodes_read, elements_read, metadata_read = mesh_io.read_mesh('test_mesh.vtk')
    
    # Check results
    np.testing.assert_allclose(nodes_read, nodes)
    assert 'triangle' in elements_read
    np.testing.assert_allclose(elements_read['triangle'], elements['triangle'])
    assert 'temperature' in metadata_read['point_data']
    np.testing.assert_allclose(metadata_read['point_data']['temperature'],
                             metadata['point_data']['temperature'])
    
    # Clean up
    os.remove('test_mesh.vtk')


def test_solution_io():
    """Test solution I/O functionality."""
    # Create test solution
    solution = {
        'velocity': np.array([1.0, 2.0, 3.0]),
        'pressure': np.array([4.0, 5.0, 6.0])
    }
    
    # Create I/O handler
    solution_io = SolutionIO()
    
    # Write solution
    solution_io.write_solution('test_solution.npz', solution)
    
    # Read solution
    solution_read = solution_io.read_solution('test_solution.npz')
    
    # Check results
    assert 'velocity' in solution_read
    assert 'pressure' in solution_read
    np.testing.assert_allclose(solution_read['velocity'], solution['velocity'])
    np.testing.assert_allclose(solution_read['pressure'], solution['pressure'])
    
    # Clean up
    os.remove('test_solution.npz')


def test_phasta_io():
    """Test PHASTA I/O functionality."""
    # Create test mesh
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    elements = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ])
    
    # Create test solution
    solution = {
        'velocity': np.array([1.0, 2.0, 3.0, 4.0]),
        'pressure': np.array([5.0, 6.0, 7.0, 8.0])
    }
    
    # Create I/O handler
    phasta_io = PHASTAIO()
    
    # Write mesh and solution
    os.makedirs('test_phasta', exist_ok=True)
    phasta_io.write_phasta_mesh('test_phasta', nodes, elements)
    phasta_io.write_phasta_solution('test_phasta', solution)
    
    # Read mesh and solution
    nodes_read, elements_read, _ = phasta_io.read_phasta_mesh('test_phasta')
    solution_read = phasta_io.read_phasta_solution('test_phasta')
    
    # Check results
    np.testing.assert_allclose(nodes_read, nodes)
    np.testing.assert_allclose(elements_read, elements)
    assert 'velocity' in solution_read
    assert 'pressure' in solution_read
    np.testing.assert_allclose(solution_read['velocity'], solution['velocity'])
    np.testing.assert_allclose(solution_read['pressure'], solution['pressure'])
    
    # Test parallel I/O
    phasta_io.write_phasta_mesh('test_phasta_parallel', nodes, elements, n_procs=2)
    phasta_io.write_phasta_solution('test_phasta_parallel', solution, n_procs=2)
    
    # Read parallel mesh and solution
    nodes_read, elements_read, _ = phasta_io.read_phasta_mesh('test_phasta_parallel')
    solution_read = phasta_io.read_phasta_solution('test_phasta_parallel')
    
    # Check results
    np.testing.assert_allclose(nodes_read, nodes)
    np.testing.assert_allclose(elements_read, elements)
    assert 'velocity' in solution_read
    assert 'pressure' in solution_read
    np.testing.assert_allclose(solution_read['velocity'], solution['velocity'])
    np.testing.assert_allclose(solution_read['pressure'], solution['pressure'])
    
    # Clean up
    import shutil
    shutil.rmtree('test_phasta')
    shutil.rmtree('test_phasta_parallel') 
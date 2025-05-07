"""
Performance benchmarks for the compressible flow solver.

This module contains benchmarks to measure the performance of the solver
for different test cases and mesh sizes.
"""

import time
import numpy as np
import pytest
from ..solver import CompressibleSolver
from .test_airfoil import (
    create_airfoil_test_mesh,
    create_boundary_conditions as create_airfoil_bc,
    create_airfoil_mesh
)
from .test_cone import create_cone_test_mesh, create_boundary_conditions as create_cone_bc

def benchmark_airfoil():
    """Benchmark airfoil test case."""
    # Create mesh and boundary conditions
    mesh = create_airfoil_test_mesh()
    bc = create_airfoil_bc()
    
    # Create solver
    solver = CompressibleSolver(
        mesh=mesh,
        boundary_conditions=bc,
        time_integration="tvd_rk3",
        shock_capturing="weno",
        limiting="flux"
    )
    
    # Measure wall time
    start_time = time.time()
    
    # Run simulation
    solver.run(max_steps=5000, convergence_tol=1e-6)
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    # Get performance metrics
    n_cells = mesh.n_cells
    n_steps = solver.n_steps
    cells_per_second = n_cells * n_steps / wall_time
    
    return {
        "wall_time": wall_time,
        "n_cells": n_cells,
        "n_steps": n_steps,
        "cells_per_second": cells_per_second
    }

def benchmark_cone():
    """Benchmark cone test case."""
    # Create mesh and boundary conditions
    mesh = create_cone_test_mesh()
    bc = create_cone_bc()
    
    # Create solver
    solver = CompressibleSolver(
        mesh=mesh,
        boundary_conditions=bc,
        time_integration="tvd_rk3",
        shock_capturing="weno",
        limiting="flux"
    )
    
    # Measure wall time
    start_time = time.time()
    
    # Run simulation
    solver.run(max_steps=5000, convergence_tol=1e-6)
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    # Get performance metrics
    n_cells = mesh.n_cells
    n_steps = solver.n_steps
    cells_per_second = n_cells * n_steps / wall_time
    
    return {
        "wall_time": wall_time,
        "n_cells": n_cells,
        "n_steps": n_steps,
        "cells_per_second": cells_per_second
    }

def test_airfoil_performance():
    """Test airfoil performance."""
    results = benchmark_airfoil()
    
    # Check if performance meets minimum requirements
    assert results["cells_per_second"] > 1e5, \
        f"Performance too low: {results['cells_per_second']} cells/second"
    
    # Print results
    print("\nAirfoil Performance Results:")
    print(f"Wall time: {results['wall_time']:.2f} seconds")
    print(f"Number of cells: {results['n_cells']}")
    print(f"Number of steps: {results['n_steps']}")
    print(f"Cells per second: {results['cells_per_second']:.2e}")

def test_cone_performance():
    """Test cone performance."""
    results = benchmark_cone()
    
    # Check if performance meets minimum requirements
    assert results["cells_per_second"] > 1e5, \
        f"Performance too low: {results['cells_per_second']} cells/second"
    
    # Print results
    print("\nCone Performance Results:")
    print(f"Wall time: {results['wall_time']:.2f} seconds")
    print(f"Number of cells: {results['n_cells']}")
    print(f"Number of steps: {results['n_steps']}")
    print(f"Cells per second: {results['cells_per_second']:.2e}")

def test_scaling():
    """Test solver scaling with mesh size."""
    # Mesh sizes to test
    mesh_sizes = [(100, 50), (200, 100), (400, 200)]
    times = []
    cells = []
    
    for nx, ny in mesh_sizes:
        # Create mesh
        mesh = create_airfoil_mesh(
            airfoil_type="naca0012",
            chord=1.0,
            x_min=-10.0, x_max=10.0,
            y_min=-5.0, y_max=5.0,
            nx=nx, ny=ny,
            first_cell_height=1e-6
        )
        
        # Create solver
        solver = CompressibleSolver(
            mesh=mesh,
            boundary_conditions=create_airfoil_bc(),
            time_integration="tvd_rk3",
            shock_capturing="weno",
            limiting="flux"
        )
        
        # Measure wall time
        start_time = time.time()
        solver.run(max_steps=1000, convergence_tol=1e-6)
        end_time = time.time()
        
        times.append(end_time - start_time)
        cells.append(mesh.n_cells)
    
    # Compute scaling
    times = np.array(times)
    cells = np.array(cells)
    
    # Linear scaling would be times ~ cells
    # Ideal scaling would be times ~ cells * log(cells)
    scaling = np.polyfit(np.log(cells), np.log(times), 1)[0]
    
    # Check if scaling is reasonable
    # Should be between linear and ideal
    assert 1.0 <= scaling <= 1.2, \
        f"Scaling exponent {scaling} outside expected range"
    
    # Print results
    print("\nScaling Results:")
    print(f"Scaling exponent: {scaling:.2f}")
    for i in range(len(mesh_sizes)):
        print(f"Mesh {mesh_sizes[i]}: {times[i]:.2f} seconds, {cells[i]} cells") 
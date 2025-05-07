"""
Integration test for flow over a flat plate.

This test verifies the implementation of the compressible flow solver
for the classical problem of flow over a flat plate, comparing results
with the Blasius solution for the boundary layer.
"""

import numpy as np
import pytest
from ..solver import CompressibleSolver
from ..boundary_conditions import (
    InletBoundaryCondition,
    WallBoundaryCondition,
    OutletBoundaryCondition
)
from ..mesh import create_rectangular_mesh

def create_flat_plate_mesh():
    """Create mesh for flat plate test case."""
    # Domain dimensions
    L = 1.0  # Plate length
    H = 0.5  # Domain height
    
    # Mesh parameters
    nx = 100  # Points in x-direction
    ny = 50   # Points in y-direction
    
    # Create mesh
    mesh = create_rectangular_mesh(
        x_min=0.0, x_max=L,
        y_min=0.0, y_max=H,
        nx=nx, ny=ny
    )
    
    return mesh

def create_boundary_conditions():
    """Create boundary conditions for flat plate test case."""
    # Inlet conditions
    inlet = InletBoundaryCondition(
        type="subsonic",
        mach=0.2,
        pressure=101325.0,  # 1 atm
        temperature=300.0    # 300 K
    )
    
    # Wall conditions
    wall = WallBoundaryCondition(
        type="no_slip",
        temperature=300.0  # Isothermal wall
    )
    
    # Outlet conditions
    outlet = OutletBoundaryCondition(
        type="subsonic",
        pressure=101325.0  # 1 atm
    )
    
    return {
        "inlet": inlet,
        "wall": wall,
        "outlet": outlet
    }

def compute_blasius_solution(x, y, Re_x):
    """Compute Blasius solution for boundary layer."""
    # Similarity variable
    eta = y * np.sqrt(Re_x / x)
    
    # Approximate Blasius solution
    # Using a simple approximation for the velocity profile
    u = np.zeros_like(eta)
    mask = eta < 5.0  # Boundary layer thickness
    u[mask] = 1.0 - np.exp(-eta[mask])
    
    return u

def test_flat_plate():
    """Test flow over a flat plate."""
    # Create mesh and boundary conditions
    mesh = create_flat_plate_mesh()
    bc = create_boundary_conditions()
    
    # Create solver
    solver = CompressibleSolver(
        mesh=mesh,
        boundary_conditions=bc,
        time_integration="tvd_rk3",
        shock_capturing="weno",
        limiting="flux"
    )
    
    # Run simulation
    solver.run(max_steps=1000, convergence_tol=1e-6)
    
    # Get solution at x = 0.5 (middle of plate)
    x_test = 0.5
    y = mesh.nodes[:, 1]
    u = solver.solution[:, 1]  # x-velocity
    
    # Compute Reynolds number
    Re_x = solver.compute_reynolds_number(x_test)
    
    # Compute Blasius solution
    u_blasius = compute_blasius_solution(x_test, y, Re_x)
    
    # Compare solutions
    # Only compare within boundary layer
    boundary_layer_thickness = 5.0 * np.sqrt(x_test / Re_x)
    mask = y < boundary_layer_thickness
    
    # Compute error
    error = np.abs(u[mask] - u_blasius[mask])
    max_error = np.max(error)
    
    # Check if error is within tolerance
    assert max_error < 0.1, f"Maximum error {max_error} exceeds tolerance"
    
    # Check boundary layer thickness
    # Find where velocity reaches 99% of freestream
    u_freestream = solver.boundary_conditions["inlet"].velocity[0]
    idx = np.where(u > 0.99 * u_freestream)[0][0]
    computed_thickness = y[idx]
    
    # Compare with theoretical thickness
    theoretical_thickness = 5.0 * np.sqrt(x_test / Re_x)
    thickness_error = np.abs(computed_thickness - theoretical_thickness)
    
    assert thickness_error < 0.1 * theoretical_thickness, \
        f"Boundary layer thickness error {thickness_error} exceeds tolerance"

def test_flat_plate_convergence():
    """Test convergence of flat plate solution."""
    # Mesh sizes to test
    mesh_sizes = [(50, 25), (100, 50), (200, 100)]
    errors = []
    
    for nx, ny in mesh_sizes:
        # Create mesh
        mesh = create_rectangular_mesh(
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=0.5,
            nx=nx, ny=ny
        )
        
        # Create solver
        solver = CompressibleSolver(
            mesh=mesh,
            boundary_conditions=create_boundary_conditions(),
            time_integration="tvd_rk3",
            shock_capturing="weno",
            limiting="flux"
        )
        
        # Run simulation
        solver.run(max_steps=1000, convergence_tol=1e-6)
        
        # Compute error
        x_test = 0.5
        y = mesh.nodes[:, 1]
        u = solver.solution[:, 1]
        Re_x = solver.compute_reynolds_number(x_test)
        u_blasius = compute_blasius_solution(x_test, y, Re_x)
        
        # Compute L2 error
        dx = 1.0 / nx
        dy = 0.5 / ny
        error = np.sqrt(np.sum((u - u_blasius)**2) * dx * dy)
        errors.append(error)
    
    # Check convergence rate
    # Should be approximately second order
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 1.5), "Convergence rate is not second order" 
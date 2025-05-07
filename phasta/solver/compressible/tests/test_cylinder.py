"""
Integration test for flow over a cylinder.

This test verifies the implementation of the compressible flow solver
for the classical problem of flow over a cylinder, comparing results
with experimental data for drag coefficient and wake characteristics.
"""

import numpy as np
import pytest
from ..solver import CompressibleSolver
from ..boundary_conditions import (
    InletBoundaryCondition,
    WallBoundaryCondition,
    OutletBoundaryCondition
)
from ..mesh import create_cylinder_mesh

def create_cylinder_test_mesh():
    """Create mesh for cylinder test case."""
    # Domain dimensions
    R = 0.5      # Cylinder radius
    L = 20 * R   # Domain length
    H = 10 * R   # Domain height
    
    # Mesh parameters
    nx = 200  # Points in x-direction
    ny = 100  # Points in y-direction
    
    # Create mesh
    mesh = create_cylinder_mesh(
        center=(0.0, 0.0),
        radius=R,
        x_min=-L/2, x_max=L/2,
        y_min=-H/2, y_max=H/2,
        nx=nx, ny=ny
    )
    
    return mesh

def create_boundary_conditions():
    """Create boundary conditions for cylinder test case."""
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

def compute_drag_coefficient(solver, cylinder_radius):
    """Compute drag coefficient from solution."""
    # Get pressure and velocity on cylinder surface
    surface_nodes = solver.get_surface_nodes()
    p = solver.solution[surface_nodes, 0]  # Pressure
    u = solver.solution[surface_nodes, 1:3]  # Velocity components
    
    # Compute surface normal vectors
    normals = solver.compute_surface_normals()
    
    # Compute drag force
    drag_force = np.sum(p * normals[:, 0])
    
    # Compute drag coefficient
    rho_inf = solver.boundary_conditions["inlet"].density
    u_inf = solver.boundary_conditions["inlet"].velocity[0]
    A = 2 * cylinder_radius  # Projected area
    
    Cd = drag_force / (0.5 * rho_inf * u_inf**2 * A)
    
    return Cd

def test_cylinder_flow():
    """Test flow over a cylinder."""
    # Create mesh and boundary conditions
    mesh = create_cylinder_test_mesh()
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
    solver.run(max_steps=2000, convergence_tol=1e-6)
    
    # Compute drag coefficient
    Cd = compute_drag_coefficient(solver, cylinder_radius=0.5)
    
    # Compare with experimental data
    # For Re = 100, Cd ≈ 1.4
    Re = solver.compute_reynolds_number(1.0)  # Based on diameter
    if Re < 50:
        expected_Cd = 1.4
    else:
        expected_Cd = 1.0  # For higher Re
    
    # Check if drag coefficient is within expected range
    assert 0.8 * expected_Cd < Cd < 1.2 * expected_Cd, \
        f"Drag coefficient {Cd} outside expected range"
    
    # Check wake characteristics
    # Get velocity field behind cylinder
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    u = solver.solution[:, 1]
    v = solver.solution[:, 2]
    
    # Find wake region
    wake_mask = (x > 0.5) & (x < 2.0) & (np.abs(y) < 1.0)
    
    # Check for recirculation
    u_wake = u[wake_mask]
    assert np.any(u_wake < 0), "No recirculation zone detected in wake"
    
    # Check wake width
    wake_width = np.max(np.abs(y[wake_mask][u_wake < 0]))
    expected_width = 1.0  # Based on experimental data
    assert 0.8 * expected_width < wake_width < 1.2 * expected_width, \
        f"Wake width {wake_width} outside expected range"

def test_cylinder_convergence():
    """Test convergence of cylinder flow solution."""
    # Mesh sizes to test
    mesh_sizes = [(100, 50), (200, 100), (400, 200)]
    drag_coefficients = []
    
    for nx, ny in mesh_sizes:
        # Create mesh
        mesh = create_cylinder_mesh(
            center=(0.0, 0.0),
            radius=0.5,
            x_min=-10.0, x_max=10.0,
            y_min=-5.0, y_max=5.0,
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
        solver.run(max_steps=2000, convergence_tol=1e-6)
        
        # Compute drag coefficient
        Cd = compute_drag_coefficient(solver, cylinder_radius=0.5)
        drag_coefficients.append(Cd)
    
    # Check convergence of drag coefficient
    # Should be within 5% of each other
    max_diff = np.max(np.abs(np.diff(drag_coefficients)))
    assert max_diff < 0.05 * np.mean(drag_coefficients), \
        "Drag coefficient not converged with mesh refinement" 
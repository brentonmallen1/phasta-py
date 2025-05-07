"""
Validation test for transonic flow over a NACA 0012 airfoil.

This test verifies the implementation of the compressible flow solver
for transonic flow over a NACA 0012 airfoil, comparing results with
experimental data for pressure distribution and shock location.
"""

import numpy as np
import pytest
from ..solver import CompressibleSolver
from ..boundary_conditions import (
    InletBoundaryCondition,
    WallBoundaryCondition,
    OutletBoundaryCondition
)
from ..mesh import create_airfoil_mesh

def create_airfoil_test_mesh():
    """Create mesh for NACA 0012 airfoil test case."""
    # Domain dimensions
    chord = 1.0  # Airfoil chord length
    L = 20 * chord  # Domain length
    H = 10 * chord  # Domain height
    
    # Mesh parameters
    nx = 400  # Points in x-direction
    ny = 200  # Points in y-direction
    
    # Create mesh
    mesh = create_airfoil_mesh(
        airfoil_type="naca0012",
        chord=chord,
        x_min=-L/2, x_max=L/2,
        y_min=-H/2, y_max=H/2,
        nx=nx, ny=ny,
        first_cell_height=1e-6  # For boundary layer resolution
    )
    
    return mesh

def create_boundary_conditions():
    """Create boundary conditions for airfoil test case."""
    # Inlet conditions
    inlet = InletBoundaryCondition(
        type="subsonic",
        mach=0.8,
        alpha=1.25,  # Angle of attack in degrees
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

def compute_pressure_coefficient(p, p_inf, rho_inf, u_inf):
    """Compute pressure coefficient."""
    return (p - p_inf) / (0.5 * rho_inf * u_inf**2)

def test_airfoil_flow():
    """Test transonic flow over NACA 0012 airfoil."""
    # Create mesh and boundary conditions
    mesh = create_airfoil_test_mesh()
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
    solver.run(max_steps=5000, convergence_tol=1e-6)
    
    # Get solution on airfoil surface
    surface_nodes = solver.get_surface_nodes()
    x = mesh.nodes[surface_nodes, 0]
    p = solver.solution[surface_nodes, 4]  # Pressure
    
    # Sort points by x-coordinate
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    p = p[sort_idx]
    
    # Compute pressure coefficient
    p_inf = bc["inlet"].pressure
    rho_inf = bc["inlet"].density
    u_inf = bc["inlet"].velocity[0]
    cp = compute_pressure_coefficient(p, p_inf, rho_inf, u_inf)
    
    # Load experimental data
    # Format: x/c, Cp
    exp_data = np.loadtxt("validation_data/naca0012_cp_m0.8_alpha1.25.dat")
    x_exp = exp_data[:, 0]
    cp_exp = exp_data[:, 1]
    
    # Interpolate numerical solution to experimental points
    cp_num = np.interp(x_exp, x, cp)
    
    # Compute error
    error = np.abs(cp_num - cp_exp)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))
    
    # Check if errors are within tolerance
    assert max_error < 0.1, f"Maximum Cp error {max_error} exceeds tolerance"
    assert rms_error < 0.05, f"RMS Cp error {rms_error} exceeds tolerance"
    
    # Check shock location
    # Find maximum pressure gradient
    shock_loc_num = x[np.argmax(np.abs(np.gradient(cp)))]
    shock_loc_exp = 0.55  # From experimental data
    shock_error = np.abs(shock_loc_num - shock_loc_exp)
    
    assert shock_error < 0.05, \
        f"Shock location error {shock_error} exceeds tolerance"
    
    # Check lift coefficient
    cl = solver.compute_lift_coefficient()
    cl_exp = 0.35  # From experimental data
    cl_error = np.abs(cl - cl_exp)
    
    assert cl_error < 0.05, \
        f"Lift coefficient error {cl_error} exceeds tolerance"
    
    # Check drag coefficient
    cd = solver.compute_drag_coefficient()
    cd_exp = 0.02  # From experimental data
    cd_error = np.abs(cd - cd_exp)
    
    assert cd_error < 0.005, \
        f"Drag coefficient error {cd_error} exceeds tolerance"

def test_airfoil_convergence():
    """Test convergence of airfoil flow solution."""
    # Mesh sizes to test
    mesh_sizes = [(200, 100), (400, 200), (800, 400)]
    errors = []
    
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
            boundary_conditions=create_boundary_conditions(),
            time_integration="tvd_rk3",
            shock_capturing="weno",
            limiting="flux"
        )
        
        # Run simulation
        solver.run(max_steps=5000, convergence_tol=1e-6)
        
        # Get solution on airfoil surface
        surface_nodes = solver.get_surface_nodes()
        x = mesh.nodes[surface_nodes, 0]
        p = solver.solution[surface_nodes, 4]
        
        # Sort points by x-coordinate
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        p = p[sort_idx]
        
        # Compute pressure coefficient
        p_inf = solver.boundary_conditions["inlet"].pressure
        rho_inf = solver.boundary_conditions["inlet"].density
        u_inf = solver.boundary_conditions["inlet"].velocity[0]
        cp = compute_pressure_coefficient(p, p_inf, rho_inf, u_inf)
        
        # Load experimental data
        exp_data = np.loadtxt("validation_data/naca0012_cp_m0.8_alpha1.25.dat")
        x_exp = exp_data[:, 0]
        cp_exp = exp_data[:, 1]
        
        # Interpolate numerical solution to experimental points
        cp_num = np.interp(x_exp, x, cp)
        
        # Compute error
        error = np.sqrt(np.mean((cp_num - cp_exp)**2))
        errors.append(error)
    
    # Check convergence rate
    # Should be approximately second order
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.mean(rates) > 1.5, "Convergence rate too low" 
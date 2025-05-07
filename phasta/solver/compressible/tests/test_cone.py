"""
Validation test for supersonic flow over a cone.

This test verifies the implementation of the compressible flow solver
for supersonic flow over a cone, comparing results with experimental data
for shock angle, surface pressure, and heat transfer.
"""

import numpy as np
import pytest
from ..solver import CompressibleSolver
from ..boundary_conditions import (
    InletBoundaryCondition,
    WallBoundaryCondition,
    OutletBoundaryCondition
)
from ..mesh import create_cone_mesh

def create_cone_test_mesh():
    """Create mesh for cone test case."""
    # Cone parameters
    length = 1.0  # Cone length
    half_angle = 10.0  # Cone half-angle in degrees
    
    # Domain dimensions
    L = 5 * length  # Domain length
    H = 3 * length  # Domain height
    
    # Mesh parameters
    nx = 300  # Points in x-direction
    ny = 150  # Points in y-direction
    
    # Create mesh
    mesh = create_cone_mesh(
        length=length,
        half_angle=half_angle,
        x_min=0.0, x_max=L,
        y_min=0.0, y_max=H,
        nx=nx, ny=ny,
        first_cell_height=1e-6  # For boundary layer resolution
    )
    
    return mesh

def create_boundary_conditions():
    """Create boundary conditions for cone test case."""
    # Inlet conditions
    inlet = InletBoundaryCondition(
        type="supersonic",
        mach=2.0,
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
        type="supersonic"
    )
    
    return {
        "inlet": inlet,
        "wall": wall,
        "outlet": outlet
    }

def compute_oblique_shock_angle(mach, cone_angle):
    """Compute theoretical oblique shock angle using Taylor-Maccoll solution."""
    # This is a simplified version - in practice, you'd use the full
    # Taylor-Maccoll solution or experimental data
    gamma = 1.4
    beta = np.arcsin(1.0 / mach)  # Initial guess
    theta = np.radians(cone_angle)
    
    # Iterative solution
    for _ in range(10):
        tan_theta = 2.0 * (1.0 / np.tan(beta)) * \
            ((mach**2 * np.sin(beta)**2 - 1.0) / \
             (mach**2 * (gamma + np.cos(2.0 * beta)) + 2.0))
        beta = np.arctan(1.0 / tan_theta)
    
    return np.degrees(beta)

def test_cone_flow():
    """Test supersonic flow over cone."""
    # Create mesh and boundary conditions
    mesh = create_cone_test_mesh()
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
    
    # Get solution on cone surface
    surface_nodes = solver.get_surface_nodes()
    x = mesh.nodes[surface_nodes, 0]
    p = solver.solution[surface_nodes, 4]  # Pressure
    T = solver.solution[surface_nodes, 5]  # Temperature
    
    # Sort points by x-coordinate
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    p = p[sort_idx]
    T = T[sort_idx]
    
    # Compute pressure coefficient
    p_inf = bc["inlet"].pressure
    rho_inf = bc["inlet"].density
    u_inf = bc["inlet"].velocity[0]
    cp = (p - p_inf) / (0.5 * rho_inf * u_inf**2)
    
    # Load experimental data
    # Format: x/L, Cp, T/T_inf
    exp_data = np.loadtxt("validation_data/cone_cp_m2.0_theta10.dat")
    x_exp = exp_data[:, 0]
    cp_exp = exp_data[:, 1]
    T_exp = exp_data[:, 2] * bc["inlet"].temperature
    
    # Interpolate numerical solution to experimental points
    cp_num = np.interp(x_exp, x, cp)
    T_num = np.interp(x_exp, x, T)
    
    # Compute errors
    cp_error = np.abs(cp_num - cp_exp)
    T_error = np.abs(T_num - T_exp) / T_exp
    
    max_cp_error = np.max(cp_error)
    rms_cp_error = np.sqrt(np.mean(cp_error**2))
    max_T_error = np.max(T_error)
    rms_T_error = np.sqrt(np.mean(T_error**2))
    
    # Check if errors are within tolerance
    assert max_cp_error < 0.05, f"Maximum Cp error {max_cp_error} exceeds tolerance"
    assert rms_cp_error < 0.02, f"RMS Cp error {rms_cp_error} exceeds tolerance"
    assert max_T_error < 0.10, f"Maximum temperature error {max_T_error} exceeds tolerance"
    assert rms_T_error < 0.05, f"RMS temperature error {rms_T_error} exceeds tolerance"
    
    # Check shock angle
    # Find shock location by looking for maximum pressure gradient
    shock_loc = x[np.argmax(np.abs(np.gradient(p)))]
    shock_angle = np.degrees(np.arctan2(shock_loc, x[-1]))
    shock_angle_theory = compute_oblique_shock_angle(2.0, 10.0)
    shock_error = np.abs(shock_angle - shock_angle_theory)
    
    assert shock_error < 1.0, \
        f"Shock angle error {shock_error} degrees exceeds tolerance"
    
    # Check boundary layer thickness
    # Compute boundary layer thickness at x/L = 0.5
    x_ref = 0.5 * mesh.nodes[surface_nodes[-1], 0]
    idx = np.argmin(np.abs(x - x_ref))
    u = solver.solution[surface_nodes, 1]  # x-velocity
    u_inf = bc["inlet"].velocity[0]
    y = mesh.nodes[surface_nodes, 1]
    
    # Find where u = 0.99 * u_inf
    delta = y[np.argmin(np.abs(u - 0.99 * u_inf))]
    delta_exp = 0.02  # From experimental data
    delta_error = np.abs(delta - delta_exp) / delta_exp
    
    assert delta_error < 0.05, \
        f"Boundary layer thickness error {delta_error} exceeds tolerance"

def test_cone_convergence():
    """Test convergence of cone flow solution."""
    # Mesh sizes to test
    mesh_sizes = [(150, 75), (300, 150), (600, 300)]
    errors = []
    
    for nx, ny in mesh_sizes:
        # Create mesh
        mesh = create_cone_mesh(
            length=1.0,
            half_angle=10.0,
            x_min=0.0, x_max=5.0,
            y_min=0.0, y_max=3.0,
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
        
        # Get solution on cone surface
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
        cp = (p - p_inf) / (0.5 * rho_inf * u_inf**2)
        
        # Load experimental data
        exp_data = np.loadtxt("validation_data/cone_cp_m2.0_theta10.dat")
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
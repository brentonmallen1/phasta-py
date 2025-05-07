"""
Integration test for supersonic flow over a wedge.

This test verifies the implementation of the compressible flow solver
for supersonic flow over a wedge, comparing results with the exact
oblique shock solution.
"""

import numpy as np
import pytest
from ..solver import CompressibleSolver
from ..boundary_conditions import (
    InletBoundaryCondition,
    WallBoundaryCondition,
    OutletBoundaryCondition
)
from ..mesh import create_wedge_mesh

def create_wedge_test_mesh():
    """Create mesh for wedge test case."""
    # Domain dimensions
    L = 2.0  # Domain length
    H = 1.0  # Domain height
    
    # Wedge parameters
    wedge_angle = 15.0  # degrees
    wedge_length = 0.5  # length of wedge
    
    # Mesh parameters
    nx = 200  # Points in x-direction
    ny = 100  # Points in y-direction
    
    # Create mesh
    mesh = create_wedge_mesh(
        wedge_angle=wedge_angle,
        wedge_length=wedge_length,
        x_min=0.0, x_max=L,
        y_min=0.0, y_max=H,
        nx=nx, ny=ny
    )
    
    return mesh

def create_boundary_conditions():
    """Create boundary conditions for wedge test case."""
    # Inlet conditions
    inlet = InletBoundaryCondition(
        type="supersonic",
        mach=2.0,
        pressure=101325.0,  # 1 atm
        temperature=300.0    # 300 K
    )
    
    # Wall conditions
    wall = WallBoundaryCondition(
        type="slip"  # Inviscid flow
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

def compute_oblique_shock_solution(M1, theta, gamma=1.4):
    """Compute exact oblique shock solution."""
    # Convert angle to radians
    theta_rad = np.radians(theta)
    
    # Initial guess for shock angle
    beta = np.arcsin(1.0 / M1) + theta_rad
    
    # Iterate to find shock angle
    for _ in range(10):
        # Compute tan(beta - theta)
        tan_beta_theta = np.tan(beta - theta_rad)
        
        # Compute tan(beta)
        tan_beta = np.tan(beta)
        
        # Compute M1^2 * sin^2(beta)
        M1_sin_beta_sq = M1**2 * np.sin(beta)**2
        
        # Compute new beta
        beta_new = np.arctan(
            (2.0 / tan_beta_theta) * (
                (M1_sin_beta_sq - 1.0) /
                (M1**2 * (gamma + np.cos(2.0 * beta)) + 2.0)
            )
        )
        
        # Check convergence
        if np.abs(beta_new - beta) < 1e-6:
            break
        
        beta = beta_new
    
    # Compute post-shock conditions
    M1n = M1 * np.sin(beta)
    M2n = np.sqrt(
        (M1n**2 + 2.0 / (gamma - 1.0)) /
        (2.0 * gamma * M1n**2 / (gamma - 1.0) - 1.0)
    )
    
    M2 = M2n / np.sin(beta - theta_rad)
    
    # Compute pressure ratio
    p2_p1 = 1.0 + 2.0 * gamma * (M1n**2 - 1.0) / (gamma + 1.0)
    
    # Compute density ratio
    rho2_rho1 = (gamma + 1.0) * M1n**2 / (
        (gamma - 1.0) * M1n**2 + 2.0
    )
    
    return {
        "shock_angle": np.degrees(beta),
        "M2": M2,
        "p2_p1": p2_p1,
        "rho2_rho1": rho2_rho1
    }

def test_wedge_flow():
    """Test supersonic flow over a wedge."""
    # Create mesh and boundary conditions
    mesh = create_wedge_test_mesh()
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
    
    # Get numerical solution
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    rho = solver.solution[:, 0]
    u = solver.solution[:, 1]
    v = solver.solution[:, 2]
    p = solver.solution[:, 4]
    
    # Compute Mach number
    c = np.sqrt(1.4 * p / rho)  # Speed of sound
    M = np.sqrt(u**2 + v**2) / c
    
    # Find shock location
    # Look for maximum gradient in density
    shock_mask = (x > 0.5) & (x < 1.5)  # Region where shock should be
    shock_pos = x[shock_mask][np.argmax(np.abs(np.gradient(rho[shock_mask])))]
    
    # Compute shock angle
    shock_angle = np.degrees(np.arctan2(
        y[shock_mask][np.argmax(np.abs(np.gradient(rho[shock_mask])))],
        shock_pos
    ))
    
    # Compute exact solution
    exact = compute_oblique_shock_solution(
        M1=2.0,  # Inlet Mach number
        theta=15.0  # Wedge angle
    )
    
    # Check shock angle
    assert np.abs(shock_angle - exact["shock_angle"]) < 2.0, \
        f"Shock angle error too large: {shock_angle} vs {exact['shock_angle']}"
    
    # Check post-shock Mach number
    # Average Mach number behind shock
    post_shock_mask = (x > shock_pos) & (x < shock_pos + 0.2)
    M2_num = np.mean(M[post_shock_mask])
    assert np.abs(M2_num - exact["M2"]) < 0.1, \
        f"Post-shock Mach number error too large: {M2_num} vs {exact['M2']}"
    
    # Check pressure ratio
    p2_p1_num = np.mean(p[post_shock_mask]) / bc["inlet"].pressure
    assert np.abs(p2_p1_num - exact["p2_p1"]) < 0.1, \
        f"Pressure ratio error too large: {p2_p1_num} vs {exact['p2_p1']}"
    
    # Check density ratio
    rho2_rho1_num = np.mean(rho[post_shock_mask]) / bc["inlet"].density
    assert np.abs(rho2_rho1_num - exact["rho2_rho1"]) < 0.1, \
        f"Density ratio error too large: {rho2_rho1_num} vs {exact['rho2_rho1']}"

def test_wedge_convergence():
    """Test convergence of wedge flow solution."""
    # Mesh sizes to test
    mesh_sizes = [(100, 50), (200, 100), (400, 200)]
    errors = []
    
    for nx, ny in mesh_sizes:
        # Create mesh
        mesh = create_wedge_mesh(
            wedge_angle=15.0,
            wedge_length=0.5,
            x_min=0.0, x_max=2.0,
            y_min=0.0, y_max=1.0,
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
        
        # Get numerical solution
        x = mesh.nodes[:, 0]
        y = mesh.nodes[:, 1]
        rho = solver.solution[:, 0]
        u = solver.solution[:, 1]
        v = solver.solution[:, 2]
        p = solver.solution[:, 4]
        
        # Compute Mach number
        c = np.sqrt(1.4 * p / rho)
        M = np.sqrt(u**2 + v**2) / c
        
        # Find shock location
        shock_mask = (x > 0.5) & (x < 1.5)
        shock_pos = x[shock_mask][np.argmax(np.abs(np.gradient(rho[shock_mask])))]
        
        # Compute shock angle
        shock_angle = np.degrees(np.arctan2(
            y[shock_mask][np.argmax(np.abs(np.gradient(rho[shock_mask])))],
            shock_pos
        ))
        
        # Compute error
        exact = compute_oblique_shock_solution(M1=2.0, theta=15.0)
        error = np.abs(shock_angle - exact["shock_angle"])
        errors.append(error)
    
    # Check convergence rate
    # Should be approximately second order
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.mean(rates) > 1.5, "Convergence rate too low" 
"""
Integration test for the Sod shock tube problem.

This test verifies the implementation of the compressible flow solver
for the classical Sod shock tube problem, comparing results with the
exact solution for shock, contact discontinuity, and expansion fan.
"""

import numpy as np
import pytest
from ..solver import CompressibleSolver
from ..boundary_conditions import (
    InletBoundaryCondition,
    OutletBoundaryCondition
)
from ..mesh import create_1d_mesh

# Shock tube parameters
S = 1.75216  # Shock speed
u_m = 0.92745  # Velocity behind shock

def create_shock_tube_mesh():
    """Create mesh for shock tube test case."""
    # Domain dimensions
    L = 1.0  # Domain length
    
    # Mesh parameters
    nx = 1000  # Points in x-direction
    
    # Create mesh
    mesh = create_1d_mesh(
        x_min=0.0,
        x_max=L,
        nx=nx
    )
    
    return mesh

def create_boundary_conditions():
    """Create boundary conditions for shock tube test case."""
    # Inlet conditions (left state)
    inlet = InletBoundaryCondition(
        type="supersonic",
        density=1.0,
        velocity=[0.0, 0.0, 0.0],
        pressure=1.0
    )
    
    # Outlet conditions (right state)
    outlet = OutletBoundaryCondition(
        type="supersonic",
        density=0.125,
        velocity=[0.0, 0.0, 0.0],
        pressure=0.1
    )
    
    return {
        "inlet": inlet,
        "outlet": outlet
    }

def compute_exact_solution(x, t):
    """Compute exact solution for Sod shock tube problem."""
    # Initial conditions
    rho_l = 1.0    # Left density
    u_l = 0.0      # Left velocity
    p_l = 1.0      # Left pressure
    rho_r = 0.125  # Right density
    u_r = 0.0      # Right velocity
    p_r = 0.1      # Right pressure
    gamma = 1.4    # Specific heat ratio
    
    # Compute wave speeds
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)
    
    # Compute shock speed
    p_m = 0.30313  # Pressure behind shock
    rho_m = 0.42632  # Density behind shock
    u_m = 0.92745  # Velocity behind shock
    S = 1.75216  # Shock speed
    
    # Initialize solution arrays
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)
    
    # Compute solution for each point
    for i, xi in enumerate(x):
        # Position relative to diaphragm
        x_pos = xi - 0.5
        
        # Expansion fan
        if x_pos < -c_l * t:
            rho[i] = rho_l
            u[i] = u_l
            p[i] = p_l
        
        # Left of contact discontinuity
        elif x_pos < u_m * t:
            # Expansion fan
            if x_pos < (u_m - c_l) * t:
                c = (x_pos / t + 2 * c_l) / (gamma + 1)
                rho[i] = rho_l * (c / c_l)**(2 / (gamma - 1))
                u[i] = x_pos / t + c
                p[i] = p_l * (c / c_l)**(2 * gamma / (gamma - 1))
            else:
                rho[i] = rho_m
                u[i] = u_m
                p[i] = p_m
        
        # Right of contact discontinuity
        elif x_pos < S * t:
            rho[i] = rho_m
            u[i] = u_m
            p[i] = p_m
        
        # Right state
        else:
            rho[i] = rho_r
            u[i] = u_r
            p[i] = p_r
    
    return rho, u, p

def test_shock_tube():
    """Test Sod shock tube problem."""
    # Create mesh and boundary conditions
    mesh = create_shock_tube_mesh()
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
    t = solver.current_time
    
    rho_num = solver.solution[:, 0]
    u_num = solver.solution[:, 1]
    p_num = solver.solution[:, 4]
    
    # Compute exact solution
    rho_exact, u_exact, p_exact = compute_exact_solution(x, t)
    
    # Compute errors
    rho_error = np.abs(rho_num - rho_exact)
    u_error = np.abs(u_num - u_exact)
    p_error = np.abs(p_num - p_exact)
    
    # Check maximum errors
    assert np.max(rho_error) < 0.1, "Density error too large"
    assert np.max(u_error) < 0.1, "Velocity error too large"
    assert np.max(p_error) < 0.1, "Pressure error too large"
    
    # Check shock position
    shock_pos_num = x[np.argmax(np.abs(np.gradient(rho_num)))]
    shock_pos_exact = 0.5 + S * t
    assert np.abs(shock_pos_num - shock_pos_exact) < 0.05, \
        "Shock position error too large"
    
    # Check contact discontinuity position
    contact_pos_num = x[np.argmax(np.abs(np.gradient(u_num)))]
    contact_pos_exact = 0.5 + u_m * t
    assert np.abs(contact_pos_num - contact_pos_exact) < 0.05, \
        "Contact discontinuity position error too large"

def test_shock_tube_convergence():
    """Test convergence of shock tube solution."""
    # Mesh sizes to test
    mesh_sizes = [100, 200, 400, 800]
    errors = []
    
    for nx in mesh_sizes:
        # Create mesh
        mesh = create_1d_mesh(
            x_min=0.0,
            x_max=1.0,
            nx=nx
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
        t = solver.current_time
        
        rho_num = solver.solution[:, 0]
        u_num = solver.solution[:, 1]
        p_num = solver.solution[:, 4]
        
        # Compute exact solution
        rho_exact, u_exact, p_exact = compute_exact_solution(x, t)
        
        # Compute L2 error
        dx = 1.0 / nx
        error = np.sqrt(dx * np.sum(
            (rho_num - rho_exact)**2 +
            (u_num - u_exact)**2 +
            (p_num - p_exact)**2
        ))
        errors.append(error)
    
    # Check convergence rate
    # Should be approximately second order
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.mean(rates) > 1.5, "Convergence rate too low" 
"""
Sod shock tube example.

This script demonstrates the use of the compressible flow solver by simulating
the Sod shock tube problem, which is a standard test case for compressible flow solvers.
"""

import numpy as np
import matplotlib.pyplot as plt
from ...solver.compressible.solver import CompressibleSolver, SolverConfig
from ...solver.compressible.shape_functions import ElementType

def create_mesh(n_elements: int = 100):
    """
    Create a 1D mesh for the shock tube.
    
    Args:
        n_elements: Number of elements
        
    Returns:
        dict: Mesh data
    """
    # Create nodes
    nodes = np.zeros((n_elements + 1, 3))
    nodes[:, 0] = np.linspace(0.0, 1.0, n_elements + 1)
    
    # Create elements (1D elements in 3D space)
    elements = np.zeros((n_elements, 2), dtype=int)
    for i in range(n_elements):
        elements[i] = [i, i + 1]
        
    return {
        'nodes': nodes,
        'elements': elements,
        'element_type': 'line'
    }

def create_initial_conditions(n_nodes: int):
    """
    Create initial conditions for the Sod shock tube.
    
    Args:
        n_nodes: Number of nodes
        
    Returns:
        dict: Initial conditions
    """
    # Initialize arrays
    density = np.ones(n_nodes)
    velocity = np.zeros((n_nodes, 3))
    pressure = np.ones(n_nodes)
    
    # Set left state (high pressure)
    left_idx = n_nodes // 2
    density[:left_idx] = 1.0
    pressure[:left_idx] = 1.0
    
    # Set right state (low pressure)
    density[left_idx:] = 0.125
    pressure[left_idx:] = 0.1
    
    return {
        'density': density,
        'velocity': velocity,
        'pressure': pressure
    }

def plot_solution(solver: CompressibleSolver, time: float):
    """
    Plot the solution at a given time.
    
    Args:
        solver: Compressible flow solver
        time: Current time
    """
    # Extract solution
    x = solver.mesh['nodes'][:, 0]
    rho = solver.solution[:, 0]
    u = solver.solution[:, 1] / rho
    p = solver._compute_pressure()
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
    
    # Plot density
    ax1.plot(x, rho, 'b-', label='Density')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.grid(True)
    ax1.legend()
    
    # Plot velocity
    ax2.plot(x, u, 'r-', label='Velocity')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Velocity')
    ax2.grid(True)
    ax2.legend()
    
    # Plot pressure
    ax3.plot(x, p, 'g-', label='Pressure')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Pressure')
    ax3.grid(True)
    ax3.legend()
    
    # Add title
    fig.suptitle(f'Sod Shock Tube Solution at t = {time:.3f}')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()

def main():
    """Run the Sod shock tube simulation."""
    # Create solver configuration
    config = SolverConfig(
        dt=0.001,  # Time step
        max_steps=1000,  # Maximum number of time steps
        cfl=0.5,  # CFL number
        gamma=1.4,  # Ratio of specific heats
        output_dir='output/sod_shock_tube',  # Output directory
        save_frequency=100  # Save frequency
    )
    
    # Create solver
    solver = CompressibleSolver(config)
    
    # Create mesh
    mesh = create_mesh(n_elements=100)
    solver.set_mesh(mesh)
    
    # Set initial conditions
    ic = create_initial_conditions(len(mesh['nodes']))
    solver.set_initial_conditions(ic)
    
    # Plot initial conditions
    plot_solution(solver, 0.0)
    
    # Run simulation
    solver.run()
    
    # Plot final solution
    plot_solution(solver, solver.time)

if __name__ == '__main__':
    main() 
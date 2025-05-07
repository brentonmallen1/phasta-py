"""Lid-driven cavity flow example for PHASTA-Py.

This script demonstrates a simple lid-driven cavity flow simulation.
"""

import numpy as np
from pathlib import Path
import time

from phasta.mesh.generator import generate_cavity_mesh
from phasta.core.field import Field
from phasta.visualization.plotter import (
    plot_mesh_3d,
    plot_velocity_field_3d,
    plot_streamlines_3d
)


def create_initial_conditions(mesh):
    """Create initial conditions for the simulation.
    
    Args:
        mesh: Mesh to create initial conditions for
        
    Returns:
        Dictionary of initial fields
    """
    points = mesh.points
    n_points = len(points)
    
    # Initialize velocity field (zero everywhere except top wall)
    velocity = np.zeros((n_points, 3))
    # Set top wall velocity (y = 1)
    top_wall = np.abs(points[:, 1] - 1.0) < 1e-6
    velocity[top_wall, 0] = 1.0  # Moving wall velocity
    
    # Initialize pressure field (zero everywhere)
    pressure = np.zeros(n_points)
    
    return {
        'velocity': Field(name='velocity', data=velocity),
        'pressure': Field(name='pressure', data=pressure)
    }


def apply_boundary_conditions(fields, mesh):
    """Apply boundary conditions to the fields.
    
    Args:
        fields: Dictionary of fields
        mesh: Mesh
    """
    points = mesh.points
    
    # No-slip walls (zero velocity)
    walls = (
        (np.abs(points[:, 0]) < 1e-6) |  # x = 0
        (np.abs(points[:, 0] - 1.0) < 1e-6) |  # x = 1
        (np.abs(points[:, 1]) < 1e-6) |  # y = 0
        (np.abs(points[:, 2]) < 1e-6) |  # z = 0
        (np.abs(points[:, 2] - 1.0) < 1e-6)  # z = 1
    )
    fields['velocity'].data[walls] = 0.0
    
    # Moving top wall
    top_wall = np.abs(points[:, 1] - 1.0) < 1e-6
    fields['velocity'].data[top_wall, 0] = 1.0


def solve_timestep(fields, mesh, dt, nu):
    """Solve one timestep of the simulation.
    
    Args:
        fields: Dictionary of fields
        mesh: Mesh
        dt: Timestep size
        nu: Kinematic viscosity
        
    Returns:
        Updated fields
    """
    # This is a simplified timestep solver for demonstration
    # In a real implementation, this would use proper numerical methods
    
    velocity = fields['velocity'].data
    pressure = fields['pressure'].data
    
    # Simple explicit Euler timestep
    # Note: This is not a proper CFD solver, just for demonstration
    velocity_new = velocity.copy()
    for i in range(len(mesh.points)):
        if not np.any(np.abs(mesh.points[i] - 1.0) < 1e-6):  # Skip top wall
            # Simple diffusion
            velocity_new[i] += nu * dt * np.random.randn(3) * 0.1
    
    fields['velocity'].data = velocity_new
    return fields


def main():
    """Run lid-driven cavity simulation."""
    # Create output directory
    output_dir = Path("output/simulations/lid_driven_cavity")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulation parameters
    n_points = (20, 20, 20)
    dt = 0.001
    nu = 0.01  # Kinematic viscosity
    n_steps = 100
    save_interval = 10
    
    # Generate mesh
    print("Generating mesh...")
    mesh = generate_cavity_mesh(
        size=1.0,
        n_points=n_points
    )
    
    # Create initial conditions
    print("Setting up initial conditions...")
    fields = create_initial_conditions(mesh)
    
    # Visualize initial state
    print("Visualizing initial state...")
    plot_mesh_3d(
        mesh,
        title="Cavity Mesh",
        output_file=output_dir / "mesh.png"
    )
    
    plot_velocity_field_3d(
        mesh,
        fields['velocity'].data,
        title="Initial Velocity Field",
        output_file=output_dir / "velocity_initial.png"
    )
    
    # Run simulation
    print(f"\nRunning simulation for {n_steps} steps...")
    start_time = time.time()
    
    for step in range(n_steps):
        # Apply boundary conditions
        apply_boundary_conditions(fields, mesh)
        
        # Solve timestep
        fields = solve_timestep(fields, mesh, dt, nu)
        
        # Save results periodically
        if (step + 1) % save_interval == 0:
            print(f"Step {step + 1}/{n_steps}")
            
            # Visualize current state
            plot_velocity_field_3d(
                mesh,
                fields['velocity'].data,
                title=f"Velocity Field (Step {step + 1})",
                output_file=output_dir / f"velocity_step_{step + 1}.png"
            )
            
            plot_streamlines_3d(
                mesh,
                fields['velocity'].data,
                n_points=100,
                title=f"Streamlines (Step {step + 1})",
                output_file=output_dir / f"streamlines_step_{step + 1}.png"
            )
    
    end_time = time.time()
    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main() 
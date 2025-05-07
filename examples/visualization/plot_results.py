"""Visualization examples for PHASTA-Py.

This script demonstrates various visualization capabilities of PHASTA-Py.
"""

import numpy as np
from pathlib import Path

from phasta.mesh.generator import generate_pipe_mesh
from phasta.visualization.plotter import (
    plot_mesh_2d,
    plot_mesh_3d,
    plot_solution_2d,
    plot_solution_3d,
    plot_velocity_field_2d,
    plot_velocity_field_3d,
    plot_streamlines_2d,
    plot_streamlines_3d,
    plot_isosurface_3d,
    plot_slice_3d
)


def create_sample_solution(mesh):
    """Create a sample solution field for visualization.
    
    Args:
        mesh: Mesh to create solution for
        
    Returns:
        Dictionary of solution fields
    """
    points = mesh.points
    
    # Create velocity field (parabolic profile in pipe)
    r = np.sqrt(points[:, 1]**2 + points[:, 2]**2)
    max_vel = 1.0
    velocity = np.zeros((len(points), 3))
    velocity[:, 0] = max_vel * (1 - (r/1.0)**2)  # Parabolic profile
    
    # Create pressure field
    pressure = -points[:, 0]  # Linear pressure drop
    
    # Create temperature field
    temperature = 300 + 50 * np.exp(-r**2/0.5)  # Gaussian temperature profile
    
    return {
        'velocity': velocity,
        'pressure': pressure,
        'temperature': temperature
    }


def main():
    """Run visualization examples."""
    # Create output directory
    output_dir = Path("output/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a test mesh
    print("Generating test mesh...")
    mesh = generate_pipe_mesh(
        length=10.0,
        radius=1.0,
        n_points=(20, 10, 10)
    )
    
    # Create sample solution
    print("Creating sample solution...")
    solution = create_sample_solution(mesh)
    
    # 1. Basic mesh visualization
    print("\nVisualizing mesh...")
    plot_mesh_3d(
        mesh,
        title="3D Pipe Mesh",
        output_file=output_dir / "mesh_3d.png"
    )
    
    # 2. Solution field visualization
    print("Visualizing solution fields...")
    plot_solution_3d(
        mesh,
        solution['pressure'],
        field_name="pressure",
        title="Pressure Field",
        output_file=output_dir / "pressure_field.png"
    )
    
    plot_solution_3d(
        mesh,
        solution['temperature'],
        field_name="temperature",
        title="Temperature Field",
        output_file=output_dir / "temperature_field.png"
    )
    
    # 3. Velocity field visualization
    print("Visualizing velocity field...")
    plot_velocity_field_3d(
        mesh,
        solution['velocity'],
        title="Velocity Field",
        output_file=output_dir / "velocity_field.png"
    )
    
    # 4. Streamlines
    print("Visualizing streamlines...")
    plot_streamlines_3d(
        mesh,
        solution['velocity'],
        n_points=100,
        title="Streamlines",
        output_file=output_dir / "streamlines.png"
    )
    
    # 5. Isosurfaces
    print("Visualizing isosurfaces...")
    plot_isosurface_3d(
        mesh,
        solution['temperature'],
        field_name="temperature",
        levels=[320, 330, 340],
        title="Temperature Isosurfaces",
        output_file=output_dir / "isosurfaces.png"
    )
    
    # 6. Slices
    print("Visualizing slices...")
    plot_slice_3d(
        mesh,
        solution['pressure'],
        field_name="pressure",
        normal=[1, 0, 0],
        origin=[5, 0, 0],
        title="Pressure Slice",
        output_file=output_dir / "slice.png"
    )
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main() 
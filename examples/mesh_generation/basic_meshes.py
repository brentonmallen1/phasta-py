"""Basic mesh generation examples for PHASTA-Py.

This script demonstrates how to generate various types of meshes using PHASTA-Py.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from phasta.mesh.generator import (
    generate_pipe_mesh,
    generate_channel_mesh,
    generate_cavity_mesh,
    generate_cylinder_mesh
)
from phasta.visualization.plotter import plot_mesh_2d, plot_mesh_3d


def main():
    """Generate and visualize various basic meshes."""
    # Create output directory
    output_dir = Path("output/meshes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate and visualize a pipe mesh
    print("Generating pipe mesh...")
    pipe_mesh = generate_pipe_mesh(
        length=10.0,
        radius=1.0,
        n_points=(20, 10, 10)
    )
    plot_mesh_3d(
        pipe_mesh,
        title="Pipe Mesh",
        output_file=output_dir / "pipe_mesh.png"
    )
    
    # 2. Generate and visualize a channel mesh
    print("Generating channel mesh...")
    channel_mesh = generate_channel_mesh(
        length=10.0,
        height=2.0,
        width=2.0,
        n_points=(20, 10, 10)
    )
    plot_mesh_3d(
        channel_mesh,
        title="Channel Mesh",
        output_file=output_dir / "channel_mesh.png"
    )
    
    # 3. Generate and visualize a cavity mesh
    print("Generating cavity mesh...")
    cavity_mesh = generate_cavity_mesh(
        size=1.0,
        n_points=(20, 20, 20)
    )
    plot_mesh_3d(
        cavity_mesh,
        title="Cavity Mesh",
        output_file=output_dir / "cavity_mesh.png"
    )
    
    # 4. Generate and visualize a cylinder mesh
    print("Generating cylinder mesh...")
    cylinder_mesh = generate_cylinder_mesh(
        radius=1.0,
        length=10.0,
        n_points=(20, 10, 10)
    )
    plot_mesh_3d(
        cylinder_mesh,
        title="Cylinder Mesh",
        output_file=output_dir / "cylinder_mesh.png"
    )
    
    print(f"All meshes generated and saved to {output_dir}")


if __name__ == "__main__":
    main() 
"""GPU acceleration example for PHASTA-Py.

This script demonstrates how to use GPU acceleration for mesh operations and simulations.
"""

import numpy as np
import torch
from pathlib import Path
import time

from phasta.mesh.generator import generate_pipe_mesh
from phasta.acceleration import get_best_available_backend
from phasta.visualization.plotter import plot_mesh_3d


def benchmark_mesh_operations(mesh, backend):
    """Benchmark mesh operations with different backends.
    
    Args:
        mesh: Mesh to operate on
        backend: Acceleration backend to use
    """
    # Convert mesh data to backend format
    points = backend.from_numpy(mesh.points)
    cells = backend.from_numpy(mesh.cells)
    
    # Benchmark element volume calculation
    start_time = time.time()
    
    # Calculate element volumes (simplified example)
    element_volumes = []
    for cell in cells:
        cell_points = points[cell]
        # Simple volume calculation (for demonstration)
        volume = torch.det(cell_points[1:] - cell_points[0]).abs()
        element_volumes.append(volume)
    
    end_time = time.time()
    print(f"Time for element volume calculation: {end_time - start_time:.4f} seconds")
    
    return element_volumes


def main():
    """Run GPU acceleration example."""
    # Create output directory
    output_dir = Path("output/acceleration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a test mesh
    print("Generating test mesh...")
    mesh = generate_pipe_mesh(
        length=10.0,
        radius=1.0,
        n_points=(50, 25, 25)  # Larger mesh for better benchmarking
    )
    
    # Get available acceleration backend
    backend = get_best_available_backend()
    print(f"Using acceleration backend: {backend.name}")
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    element_volumes = benchmark_mesh_operations(mesh, backend)
    
    # Visualize mesh
    print("\nVisualizing mesh...")
    plot_mesh_3d(
        mesh,
        title=f"Pipe Mesh (Accelerated with {backend.name})",
        output_file=output_dir / "accelerated_mesh.png"
    )
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main() 
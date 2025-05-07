"""Parallel mesh operations example for PHASTA-Py.

This script demonstrates how to perform parallel mesh operations using MPI.
"""

import numpy as np
from pathlib import Path
import time

from phasta.mesh.generator import generate_pipe_mesh
from phasta.core.parallel import get_parallel_manager
from phasta.core.io import write_parallel_mesh
from phasta.visualization.plotter import plot_mesh_3d


def partition_mesh(mesh, num_parts):
    """Partition a mesh into multiple parts.
    
    Args:
        mesh: Mesh to partition
        num_parts: Number of parts to create
        
    Returns:
        List of partitioned meshes
    """
    # Simple partitioning by splitting along x-axis
    points = mesh.points
    cells = mesh.cells
    
    # Sort cells by x-coordinate of their centroids
    cell_centers = np.mean(points[cells], axis=1)
    sorted_indices = np.argsort(cell_centers[:, 0])
    
    # Split cells into parts
    cells_per_part = len(cells) // num_parts
    partitioned_cells = []
    for i in range(num_parts):
        start_idx = i * cells_per_part
        end_idx = start_idx + cells_per_part if i < num_parts - 1 else len(cells)
        part_cells = cells[sorted_indices[start_idx:end_idx]]
        partitioned_cells.append(part_cells)
    
    # Create partitioned meshes
    partitioned_meshes = []
    for part_cells in partitioned_cells:
        # Get unique points used by these cells
        part_points = np.unique(part_cells)
        # Create new point numbering
        point_map = {old: new for new, old in enumerate(part_points)}
        # Remap cell indices
        remapped_cells = np.array([[point_map[p] for p in cell] for cell in part_cells])
        # Create new mesh
        part_mesh = type(mesh)(
            points=points[part_points],
            cells=remapped_cells,
            point_data={k: v[part_points] for k, v in mesh.point_data.items()},
            cell_data={k: v[sorted_indices[start_idx:end_idx]] 
                      for k, v in mesh.cell_data.items()}
        )
        partitioned_meshes.append(part_mesh)
    
    return partitioned_meshes


def main():
    """Run parallel mesh example."""
    # Get parallel manager
    manager = get_parallel_manager()
    
    # Create output directory
    output_dir = Path("output/parallel")
    if manager.is_root():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate mesh on root process
    if manager.is_root():
        print("Generating mesh...")
        mesh = generate_pipe_mesh(
            length=10.0,
            radius=1.0,
            n_points=(50, 25, 25)
        )
        
        # Partition mesh
        print(f"Partitioning mesh into {manager.size} parts...")
        partitioned_meshes = partition_mesh(mesh, manager.size)
    else:
        partitioned_meshes = None
    
    # Broadcast partitioned meshes to all processes
    partitioned_meshes = manager.broadcast(partitioned_meshes)
    
    # Each process works with its part
    local_mesh = partitioned_meshes[manager.rank]
    print(f"Process {manager.rank}: Working with {len(local_mesh.cells)} cells")
    
    # Write parallel mesh
    print(f"Process {manager.rank}: Writing mesh part...")
    write_parallel_mesh(partitioned_meshes, output_dir / "parallel_mesh")
    
    # Visualize local mesh part
    if manager.is_root():
        print("\nVisualizing mesh parts...")
        for i, part_mesh in enumerate(partitioned_meshes):
            plot_mesh_3d(
                part_mesh,
                title=f"Mesh Part {i}",
                output_file=output_dir / f"mesh_part_{i}.png"
            )
    
    # Synchronize
    manager.barrier()
    print(f"Process {manager.rank}: Done")


if __name__ == "__main__":
    main() 
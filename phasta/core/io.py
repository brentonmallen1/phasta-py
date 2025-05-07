"""I/O utilities for PHASTA-Py.

This module provides functionality for reading and writing mesh and solution data.
"""

import numpy as np
import meshio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from phasta.core.mesh import Mesh
from phasta.core.field import Field
from phasta.core.parallel import get_parallel_manager


def read_mesh(file_path: Union[str, Path]) -> Mesh:
    """Read a mesh from a file.
    
    Args:
        file_path: Path to mesh file
        
    Returns:
        Mesh object
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {file_path}")
    
    # Read mesh using meshio
    mesh_data = meshio.read(file_path)
    
    # Convert to internal mesh format
    return Mesh(
        points=mesh_data.points,
        cells=mesh_data.cells,
        point_data=mesh_data.point_data,
        cell_data=mesh_data.cell_data
    )


def write_mesh(mesh: Mesh, file_path: Union[str, Path]) -> None:
    """Write a mesh to a file.
    
    Args:
        mesh: Mesh to write
        file_path: Path to output file
    """
    file_path = Path(file_path)
    
    # Convert to meshio format
    mesh_data = meshio.Mesh(
        points=mesh.points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data
    )
    
    # Write mesh
    meshio.write(file_path, mesh_data)


def read_solution(file_path: Union[str, Path]) -> Dict[str, Field]:
    """Read solution data from a file.
    
    Args:
        file_path: Path to solution file
        
    Returns:
        Dictionary of field names to Field objects
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Solution file not found: {file_path}")
    
    # Read solution using meshio
    solution_data = meshio.read(file_path)
    
    # Convert to internal field format
    fields = {}
    for name, data in solution_data.point_data.items():
        fields[name] = Field(name=name, data=data)
    
    return fields


def write_solution(fields: Dict[str, Field], file_path: Union[str, Path]) -> None:
    """Write solution data to a file.
    
    Args:
        fields: Dictionary of field names to Field objects
        file_path: Path to output file
    """
    file_path = Path(file_path)
    
    # Convert to meshio format
    point_data = {name: field.data for name, field in fields.items()}
    
    # Create meshio mesh with solution data
    solution_data = meshio.Mesh(
        points=np.zeros((len(next(iter(fields.values())).data), 3)),  # Dummy points
        cells=[("vertex", np.arange(len(next(iter(fields.values())).data)).reshape(-1, 1))],
        point_data=point_data
    )
    
    # Write solution
    meshio.write(file_path, solution_data)


def read_checkpoint(file_path: Union[str, Path]) -> Tuple[Mesh, Dict[str, Field]]:
    """Read a checkpoint file containing both mesh and solution data.
    
    Args:
        file_path: Path to checkpoint file
        
    Returns:
        Tuple of (mesh, fields)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
    
    # Read checkpoint using meshio
    checkpoint_data = meshio.read(file_path)
    
    # Convert to internal formats
    mesh = Mesh(
        points=checkpoint_data.points,
        cells=checkpoint_data.cells,
        point_data=checkpoint_data.point_data,
        cell_data=checkpoint_data.cell_data
    )
    
    fields = {}
    for name, data in checkpoint_data.point_data.items():
        fields[name] = Field(name=name, data=data)
    
    return mesh, fields


def write_checkpoint(mesh: Mesh, fields: Dict[str, Field],
                    file_path: Union[str, Path]) -> None:
    """Write a checkpoint file containing both mesh and solution data.
    
    Args:
        mesh: Mesh to write
        fields: Dictionary of field names to Field objects
        file_path: Path to output file
    """
    file_path = Path(file_path)
    
    # Convert to meshio format
    point_data = {name: field.data for name, field in fields.items()}
    point_data.update(mesh.point_data)
    
    checkpoint_data = meshio.Mesh(
        points=mesh.points,
        cells=mesh.cells,
        point_data=point_data,
        cell_data=mesh.cell_data
    )
    
    # Write checkpoint
    meshio.write(file_path, checkpoint_data)


def read_parallel_mesh(base_path: Union[str, Path], num_parts: int) -> List[Mesh]:
    """Read a parallel mesh split into multiple parts.
    
    Args:
        base_path: Base path for mesh files
        num_parts: Number of mesh parts
        
    Returns:
        List of mesh parts
    """
    base_path = Path(base_path)
    manager = get_parallel_manager()
    
    # Read local part
    local_part = read_mesh(base_path / f"part_{manager.rank}.vtk")
    
    # Gather all parts on root
    all_parts = manager.gather(local_part)
    
    if manager.is_root():
        return all_parts
    return []


def write_parallel_mesh(meshes: List[Mesh], base_path: Union[str, Path]) -> None:
    """Write a parallel mesh split into multiple parts.
    
    Args:
        meshes: List of mesh parts
        base_path: Base path for output files
    """
    base_path = Path(base_path)
    manager = get_parallel_manager()
    
    # Write local part
    write_mesh(meshes[manager.rank], base_path / f"part_{manager.rank}.vtk")
    
    # Synchronize
    manager.barrier()

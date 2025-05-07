"""
I/O module for the compressible flow solver.

This module implements the file I/O functionality from the original PHASTA codebase.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import h5py
import json
import os

@dataclass
class IOConfig:
    """Configuration for I/O operations."""
    output_dir: str = "output"  # Output directory
    file_format: str = "h5"  # File format (h5, vtk, etc.)
    save_frequency: int = 100  # Save frequency in time steps
    save_fields: List[str] = None  # Fields to save
    compression: bool = True  # Enable compression
    compression_level: int = 6  # Compression level

class IOHandler:
    """
    Implements the I/O functionality for compressible flow.
    
    This class implements the I/O functionality from the original PHASTA codebase.
    """
    
    def __init__(self, config: IOConfig):
        """
        Initialize the I/O handler.
        
        Args:
            config: I/O configuration parameters
        """
        self.config = config
        if self.config.save_fields is None:
            self.config.save_fields = [
                "density", "velocity", "pressure", "temperature"
            ]
            
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def save_solution(self,
                    solution: np.ndarray,
                    mesh: Dict[str, np.ndarray],
                    time_step: int,
                    time: float):
        """
        Save solution to file.
        
        Args:
            solution: Solution vector
            mesh: Mesh data
            time_step: Current time step
            time: Current time
        """
        if time_step % self.config.save_frequency != 0:
            return
            
        if self.config.file_format == "h5":
            self._save_h5(solution, mesh, time_step, time)
        elif self.config.file_format == "vtk":
            self._save_vtk(solution, mesh, time_step, time)
        else:
            raise ValueError(f"Unsupported file format: {self.config.file_format}")
            
    def load_solution(self,
                    file_path: str
                    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, float]:
        """
        Load solution from file.
        
        Args:
            file_path: Path to solution file
            
        Returns:
            tuple: (solution, mesh, time_step, time)
        """
        if self.config.file_format == "h5":
            return self._load_h5(file_path)
        elif self.config.file_format == "vtk":
            return self._load_vtk(file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.config.file_format}")
            
    def _save_h5(self,
                solution: np.ndarray,
                mesh: Dict[str, np.ndarray],
                time_step: int,
                time: float):
        """
        Save solution to HDF5 file.
        
        Args:
            solution: Solution vector
            mesh: Mesh data
            time_step: Current time step
            time: Current time
        """
        file_path = os.path.join(
            self.config.output_dir,
            f"solution_{time_step:06d}.h5"
        )
        
        with h5py.File(file_path, 'w') as f:
            # Save metadata
            f.attrs['time_step'] = time_step
            f.attrs['time'] = time
            
            # Save mesh data
            mesh_group = f.create_group('mesh')
            for key, value in mesh.items():
                mesh_group.create_dataset(
                    key,
                    data=value,
                    compression='gzip' if self.config.compression else None,
                    compression_opts=self.config.compression_level
                )
                
            # Save solution data
            sol_group = f.create_group('solution')
            for i, field in enumerate(self.config.save_fields):
                sol_group.create_dataset(
                    field,
                    data=solution[:, i],
                    compression='gzip' if self.config.compression else None,
                    compression_opts=self.config.compression_level
                )
                
    def _load_h5(self,
                file_path: str
                ) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, float]:
        """
        Load solution from HDF5 file.
        
        Args:
            file_path: Path to solution file
            
        Returns:
            tuple: (solution, mesh, time_step, time)
        """
        with h5py.File(file_path, 'r') as f:
            # Load metadata
            time_step = f.attrs['time_step']
            time = f.attrs['time']
            
            # Load mesh data
            mesh = {}
            mesh_group = f['mesh']
            for key in mesh_group.keys():
                mesh[key] = mesh_group[key][:]
                
            # Load solution data
            solution = np.zeros((
                len(mesh['nodes']),
                len(self.config.save_fields)
            ))
            sol_group = f['solution']
            for i, field in enumerate(self.config.save_fields):
                solution[:, i] = sol_group[field][:]
                
            return solution, mesh, time_step, time
            
    def _save_vtk(self,
                 solution: np.ndarray,
                 mesh: Dict[str, np.ndarray],
                 time_step: int,
                 time: float):
        """
        Save solution to VTK file.
        
        Args:
            solution: Solution vector
            mesh: Mesh data
            time_step: Current time step
            time: Current time
        """
        file_path = os.path.join(
            self.config.output_dir,
            f"solution_{time_step:06d}.vtk"
        )
        
        with open(file_path, 'w') as f:
            # Write header
            f.write("# vtk DataFile Version 3.0\n")
            f.write("PHASTA Solution\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Write points
            n_points = len(mesh['nodes'])
            f.write(f"POINTS {n_points} double\n")
            for i in range(n_points):
                f.write(f"{mesh['nodes'][i, 0]} {mesh['nodes'][i, 1]} {mesh['nodes'][i, 2]}\n")
                
            # Write cells
            n_cells = len(mesh['elements'])
            n_cell_points = mesh['elements'].shape[1]
            f.write(f"CELLS {n_cells} {n_cells * (n_cell_points + 1)}\n")
            for i in range(n_cells):
                f.write(f"{n_cell_points}")
                for j in range(n_cell_points):
                    f.write(f" {mesh['elements'][i, j]}")
                f.write("\n")
                
            # Write cell types
            f.write(f"CELL_TYPES {n_cells}\n")
            cell_type = self._get_vtk_cell_type(mesh['element_type'])
            for i in range(n_cells):
                f.write(f"{cell_type}\n")
                
            # Write point data
            f.write(f"POINT_DATA {n_points}\n")
            for i, field in enumerate(self.config.save_fields):
                f.write(f"SCALARS {field} double 1\n")
                f.write("LOOKUP_TABLE default\n")
                for j in range(n_points):
                    f.write(f"{solution[j, i]}\n")
                    
    def _load_vtk(self,
                 file_path: str
                 ) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, float]:
        """
        Load solution from VTK file.
        
        Args:
            file_path: Path to solution file
            
        Returns:
            tuple: (solution, mesh, time_step, time)
        """
        with open(file_path, 'r') as f:
            # Skip header
            for _ in range(4):
                f.readline()
                
            # Read points
            line = f.readline().split()
            n_points = int(line[1])
            nodes = np.zeros((n_points, 3))
            for i in range(n_points):
                line = f.readline().split()
                nodes[i] = [float(x) for x in line]
                
            # Skip cells
            line = f.readline().split()
            n_cells = int(line[1])
            for _ in range(n_cells):
                f.readline()
                
            # Skip cell types
            f.readline()
            for _ in range(n_cells):
                f.readline()
                
            # Read point data
            f.readline()
            solution = np.zeros((n_points, len(self.config.save_fields)))
            for i, field in enumerate(self.config.save_fields):
                f.readline()  # Skip field name
                f.readline()  # Skip lookup table
                for j in range(n_points):
                    solution[j, i] = float(f.readline())
                    
            # Create mesh dictionary
            mesh = {
                'nodes': nodes,
                'elements': None,  # Not stored in VTK format
                'element_type': None  # Not stored in VTK format
            }
            
            # Extract time step from filename
            time_step = int(os.path.basename(file_path).split('_')[1].split('.')[0])
            time = 0.0  # Not stored in VTK format
            
            return solution, mesh, time_step, time
            
    def _get_vtk_cell_type(self, element_type: str) -> int:
        """
        Get VTK cell type code.
        
        Args:
            element_type: Element type
            
        Returns:
            int: VTK cell type code
        """
        if element_type == "hex":
            return 12  # VTK_HEXAHEDRON
        elif element_type == "tet":
            return 10  # VTK_TETRA
        elif element_type == "prism":
            return 13  # VTK_WEDGE
        elif element_type == "quad":
            return 9  # VTK_QUAD
        elif element_type == "tri":
            return 5  # VTK_TRIANGLE
        else:
            raise ValueError(f"Unknown element type: {element_type}")
            
    def save_checkpoint(self,
                      solution: np.ndarray,
                      mesh: Dict[str, np.ndarray],
                      time_step: int,
                      time: float,
                      solver_state: Dict[str, Any]):
        """
        Save checkpoint.
        
        Args:
            solution: Solution vector
            mesh: Mesh data
            time_step: Current time step
            time: Current time
            solver_state: Solver state
        """
        file_path = os.path.join(
            self.config.output_dir,
            f"checkpoint_{time_step:06d}.h5"
        )
        
        with h5py.File(file_path, 'w') as f:
            # Save metadata
            f.attrs['time_step'] = time_step
            f.attrs['time'] = time
            
            # Save mesh data
            mesh_group = f.create_group('mesh')
            for key, value in mesh.items():
                mesh_group.create_dataset(
                    key,
                    data=value,
                    compression='gzip' if self.config.compression else None,
                    compression_opts=self.config.compression_level
                )
                
            # Save solution data
            sol_group = f.create_group('solution')
            for i, field in enumerate(self.config.save_fields):
                sol_group.create_dataset(
                    field,
                    data=solution[:, i],
                    compression='gzip' if self.config.compression else None,
                    compression_opts=self.config.compression_level
                )
                
            # Save solver state
            state_group = f.create_group('solver_state')
            for key, value in solver_state.items():
                if isinstance(value, np.ndarray):
                    state_group.create_dataset(
                        key,
                        data=value,
                        compression='gzip' if self.config.compression else None,
                        compression_opts=self.config.compression_level
                    )
                else:
                    state_group.attrs[key] = value
                    
    def load_checkpoint(self,
                      file_path: str
                      ) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, float, Dict[str, Any]]:
        """
        Load checkpoint.
        
        Args:
            file_path: Path to checkpoint file
            
        Returns:
            tuple: (solution, mesh, time_step, time, solver_state)
        """
        with h5py.File(file_path, 'r') as f:
            # Load metadata
            time_step = f.attrs['time_step']
            time = f.attrs['time']
            
            # Load mesh data
            mesh = {}
            mesh_group = f['mesh']
            for key in mesh_group.keys():
                mesh[key] = mesh_group[key][:]
                
            # Load solution data
            solution = np.zeros((
                len(mesh['nodes']),
                len(self.config.save_fields)
            ))
            sol_group = f['solution']
            for i, field in enumerate(self.config.save_fields):
                solution[:, i] = sol_group[field][:]
                
            # Load solver state
            solver_state = {}
            state_group = f['solver_state']
            for key in state_group.keys():
                solver_state[key] = state_group[key][:]
            for key in state_group.attrs:
                solver_state[key] = state_group.attrs[key]
                
            return solution, mesh, time_step, time, solver_state 
"""
Parallel I/O operations for distributed computing.

This module provides functionality for parallel file I/O operations,
including reading/writing mesh data, solution data, and checkpoint/restart files.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .mpi_wrapper import MPIWrapper
import h5py
import logging
import os

class ParallelIO:
    """Parallel I/O operations."""
    
    def __init__(self, mpi: MPIWrapper):
        """Initialize parallel I/O.
        
        Args:
            mpi: MPI wrapper instance
        """
        self.mpi = mpi
        self.logger = logging.getLogger(__name__)
    
    def write_mesh(self, filename: str, mesh, domain_decomp):
        """Write mesh data in parallel.
        
        Args:
            filename: Output filename
            mesh: Mesh to write
            domain_decomp: Domain decomposition instance
        """
        # Create HDF5 file
        with h5py.File(filename, 'w', driver='mpio', comm=self.mpi.comm) as f:
            # Write global mesh data
            if self.mpi.rank == 0:
                f.create_dataset('n_cells', data=mesh.n_cells)
                f.create_dataset('n_nodes', data=mesh.n_nodes)
                f.create_dataset('n_faces', data=mesh.n_faces)
            
            # Write local mesh data
            local_cells = domain_decomp.get_local_cells()
            ghost_cells = []
            for partition in range(self.mpi.size):
                if partition != self.mpi.rank:
                    ghost_cells.extend(domain_decomp.get_ghost_cells(partition))
            
            # Write cell data
            cell_data = mesh.get_cell_data()
            f.create_dataset('cell_data', data=cell_data[local_cells])
            if ghost_cells:
                f.create_dataset('ghost_cell_data', data=cell_data[ghost_cells])
            
            # Write node data
            node_data = mesh.get_node_data()
            f.create_dataset('node_data', data=node_data)
            
            # Write face data
            face_data = mesh.get_face_data()
            f.create_dataset('face_data', data=face_data)
            
            # Write partition information
            f.create_dataset('partition', data=domain_decomp.partition)
    
    def read_mesh(self, filename: str, domain_decomp) -> Tuple:
        """Read mesh data in parallel.
        
        Args:
            filename: Input filename
            domain_decomp: Domain decomposition instance
            
        Returns:
            Tuple of (mesh_data, cell_data, node_data, face_data)
        """
        # Open HDF5 file
        with h5py.File(filename, 'r', driver='mpio', comm=self.mpi.comm) as f:
            # Read global mesh data
            n_cells = f['n_cells'][()]
            n_nodes = f['n_nodes'][()]
            n_faces = f['n_faces'][()]
            
            # Read local mesh data
            local_cells = domain_decomp.get_local_cells()
            ghost_cells = []
            for partition in range(self.mpi.size):
                if partition != self.mpi.rank:
                    ghost_cells.extend(domain_decomp.get_ghost_cells(partition))
            
            # Read cell data
            cell_data = np.empty((n_cells,), dtype=f['cell_data'].dtype)
            cell_data[local_cells] = f['cell_data'][local_cells]
            if 'ghost_cell_data' in f:
                cell_data[ghost_cells] = f['ghost_cell_data'][()]
            
            # Read node data
            node_data = f['node_data'][()]
            
            # Read face data
            face_data = f['face_data'][()]
            
            # Read partition information
            partition = f['partition'][()]
            
            return (n_cells, n_nodes, n_faces), cell_data, node_data, face_data
    
    def write_solution(self, filename: str, solution: np.ndarray, domain_decomp):
        """Write solution data in parallel.
        
        Args:
            filename: Output filename
            solution: Solution data to write
            domain_decomp: Domain decomposition instance
        """
        # Create HDF5 file
        with h5py.File(filename, 'w', driver='mpio', comm=self.mpi.comm) as f:
            # Write global solution data
            if self.mpi.rank == 0:
                f.create_dataset('n_vars', data=solution.shape[1])
            
            # Write local solution data
            local_cells = domain_decomp.get_local_cells()
            f.create_dataset('solution', data=solution[local_cells])
            
            # Write ghost cell data
            ghost_cells = []
            for partition in range(self.mpi.size):
                if partition != self.mpi.rank:
                    ghost_cells.extend(domain_decomp.get_ghost_cells(partition))
            if ghost_cells:
                f.create_dataset('ghost_solution', data=solution[ghost_cells])
    
    def read_solution(self, filename: str, domain_decomp) -> np.ndarray:
        """Read solution data in parallel.
        
        Args:
            filename: Input filename
            domain_decomp: Domain decomposition instance
            
        Returns:
            Solution data array
        """
        # Open HDF5 file
        with h5py.File(filename, 'r', driver='mpio', comm=self.mpi.comm) as f:
            # Read global solution data
            n_vars = f['n_vars'][()]
            
            # Read local solution data
            local_cells = domain_decomp.get_local_cells()
            solution = np.empty((domain_decomp.partition.size, n_vars),
                              dtype=f['solution'].dtype)
            solution[local_cells] = f['solution'][()]
            
            # Read ghost cell data
            ghost_cells = []
            for partition in range(self.mpi.size):
                if partition != self.mpi.rank:
                    ghost_cells.extend(domain_decomp.get_ghost_cells(partition))
            if 'ghost_solution' in f:
                solution[ghost_cells] = f['ghost_solution'][()]
            
            return solution
    
    def write_checkpoint(self, filename: str, mesh, solution: np.ndarray,
                        domain_decomp, iteration: int, time: float):
        """Write checkpoint file in parallel.
        
        Args:
            filename: Output filename
            mesh: Mesh to write
            solution: Solution data to write
            domain_decomp: Domain decomposition instance
            iteration: Current iteration number
            time: Current simulation time
        """
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.dirname(filename)
        if self.mpi.rank == 0 and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.mpi.barrier()
        
        # Create HDF5 file
        with h5py.File(filename, 'w', driver='mpio', comm=self.mpi.comm) as f:
            # Write simulation state
            if self.mpi.rank == 0:
                f.create_dataset('iteration', data=iteration)
                f.create_dataset('time', data=time)
            
            # Write mesh and solution data
            self.write_mesh(filename + '_mesh', mesh, domain_decomp)
            self.write_solution(filename + '_solution', solution, domain_decomp)
    
    def read_checkpoint(self, filename: str, domain_decomp) -> Tuple:
        """Read checkpoint file in parallel.
        
        Args:
            filename: Input filename
            domain_decomp: Domain decomposition instance
            
        Returns:
            Tuple of (mesh_data, solution, iteration, time)
        """
        # Open HDF5 file
        with h5py.File(filename, 'r', driver='mpio', comm=self.mpi.comm) as f:
            # Read simulation state
            iteration = f['iteration'][()]
            time = f['time'][()]
        
        # Read mesh and solution data
        mesh_data, cell_data, node_data, face_data = \
            self.read_mesh(filename + '_mesh', domain_decomp)
        solution = self.read_solution(filename + '_solution', domain_decomp)
        
        return (mesh_data, cell_data, node_data, face_data), solution, iteration, time 
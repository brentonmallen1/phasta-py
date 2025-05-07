"""Parallel I/O module for distributed mesh and solution data."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from mpi4py import MPI
import h5py
from .mesh import Mesh
from .partitioning import MeshPartitioner


class ParallelIO:
    """Parallel I/O operations for distributed data."""
    
    def __init__(self, comm: Optional[MPI.Comm] = None):
        """Initialize parallel I/O.
        
        Args:
            comm: MPI communicator (default: MPI.COMM_WORLD)
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
    def write_mesh(self, filename: str, mesh: Mesh, partitioner: MeshPartitioner) -> None:
        """Write distributed mesh to HDF5 file.
        
        Args:
            filename: Output filename
            mesh: Mesh to write
            partitioner: Mesh partitioner used for distribution
        """
        # Get partition information
        partition_array, ghost_nodes = partitioner.partition(mesh)
        
        # Create HDF5 file
        with h5py.File(filename, 'w', driver='mpio', comm=self.comm) as f:
            # Write global mesh data
            if self.rank == 0:
                f.create_dataset('nodes', data=mesh.nodes)
                f.create_dataset('elements', data=mesh.elements)
            
            # Write partition data
            partition_group = f.create_group('partitions')
            local_group = partition_group.create_group(f'rank_{self.rank}')
            
            # Write local node indices
            local_nodes = np.where(partition_array == self.rank)[0]
            local_group.create_dataset('local_nodes', data=local_nodes)
            
            # Write ghost node information
            ghost_group = local_group.create_group('ghost_nodes')
            for rank, nodes in ghost_nodes.items():
                if nodes:  # Only write if there are ghost nodes
                    ghost_group.create_dataset(f'from_rank_{rank}', data=nodes)
    
    def read_mesh(self, filename: str) -> Tuple[Mesh, np.ndarray, Dict[int, List[int]]]:
        """Read distributed mesh from HDF5 file.
        
        Args:
            filename: Input filename
            
        Returns:
            Tuple of (mesh, partition array, ghost nodes dictionary)
        """
        with h5py.File(filename, 'r', driver='mpio', comm=self.comm) as f:
            # Read global mesh data
            nodes = f['nodes'][:]
            elements = f['elements'][:]
            mesh = Mesh(nodes, elements)
            
            # Read partition data
            partition_group = f['partitions']
            local_group = partition_group[f'rank_{self.rank}']
            
            # Read local node indices
            local_nodes = local_group['local_nodes'][:]
            
            # Create partition array
            partition_array = np.zeros(len(nodes), dtype=int)
            partition_array[local_nodes] = self.rank
            
            # Read ghost node information
            ghost_nodes = {}
            ghost_group = local_group['ghost_nodes']
            for key in ghost_group.keys():
                if key.startswith('from_rank_'):
                    rank = int(key.split('_')[-1])
                    ghost_nodes[rank] = ghost_group[key][:].tolist()
            
            return mesh, partition_array, ghost_nodes
    
    def write_solution(self, filename: str, solution: np.ndarray, 
                      partition_array: np.ndarray, ghost_nodes: Dict[int, List[int]]) -> None:
        """Write distributed solution to HDF5 file.
        
        Args:
            filename: Output filename
            solution: Solution vector
            partition_array: Array of partition assignments
            ghost_nodes: Dictionary of ghost nodes
        """
        with h5py.File(filename, 'w', driver='mpio', comm=self.comm) as f:
            # Create solution group
            solution_group = f.create_group('solution')
            
            # Write local solution data
            local_group = solution_group.create_group(f'rank_{self.rank}')
            local_nodes = np.where(partition_array == self.rank)[0]
            local_group.create_dataset('values', data=solution[local_nodes])
            
            # Write ghost node values
            ghost_group = local_group.create_group('ghost_values')
            for rank, nodes in ghost_nodes.items():
                if nodes:  # Only write if there are ghost nodes
                    ghost_group.create_dataset(f'from_rank_{rank}', data=solution[nodes])
    
    def read_solution(self, filename: str, partition_array: np.ndarray) -> np.ndarray:
        """Read distributed solution from HDF5 file.
        
        Args:
            filename: Input filename
            partition_array: Array of partition assignments
            
        Returns:
            Solution vector
        """
        with h5py.File(filename, 'r', driver='mpio', comm=self.comm) as f:
            # Create solution array
            solution = np.zeros(len(partition_array))
            
            # Read local solution data
            solution_group = f['solution']
            local_group = solution_group[f'rank_{self.rank}']
            local_nodes = np.where(partition_array == self.rank)[0]
            solution[local_nodes] = local_group['values'][:]
            
            # Read ghost node values
            ghost_group = local_group['ghost_values']
            for key in ghost_group.keys():
                if key.startswith('from_rank_'):
                    rank = int(key.split('_')[-1])
                    nodes = ghost_group[key][:]
                    solution[nodes] = ghost_group[key][:]
            
            return solution
    
    def write_checkpoint(self, filename: str, mesh: Mesh, solution: np.ndarray,
                        partitioner: MeshPartitioner, step: int, time: float) -> None:
        """Write checkpoint file with mesh and solution data.
        
        Args:
            filename: Output filename
            mesh: Mesh
            solution: Solution vector
            partitioner: Mesh partitioner
            step: Current time step
            time: Current simulation time
        """
        with h5py.File(filename, 'w', driver='mpio', comm=self.comm) as f:
            # Write simulation metadata
            if self.rank == 0:
                f.attrs['step'] = step
                f.attrs['time'] = time
            
            # Write mesh and solution data
            self.write_mesh(filename, mesh, partitioner)
            partition_array, ghost_nodes = partitioner.partition(mesh)
            self.write_solution(filename, solution, partition_array, ghost_nodes)
    
    def read_checkpoint(self, filename: str) -> Tuple[Mesh, np.ndarray, int, float]:
        """Read checkpoint file.
        
        Args:
            filename: Input filename
            
        Returns:
            Tuple of (mesh, solution, step, time)
        """
        with h5py.File(filename, 'r', driver='mpio', comm=self.comm) as f:
            # Read simulation metadata
            step = f.attrs['step']
            time = f.attrs['time']
            
            # Read mesh and solution data
            mesh, partition_array, ghost_nodes = self.read_mesh(filename)
            solution = self.read_solution(filename, partition_array)
            
            return mesh, solution, step, time 
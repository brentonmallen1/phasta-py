"""
Domain decomposition for parallel computing.

This module handles mesh partitioning, ghost cell management, and load balancing
for parallel computing in PHASTA-Py.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .mpi_wrapper import MPIWrapper
import logging

class DomainDecomposition:
    """Domain decomposition for parallel computing."""
    
    def __init__(self, mpi: MPIWrapper):
        """Initialize domain decomposition.
        
        Args:
            mpi: MPI wrapper instance
        """
        self.mpi = mpi
        self.logger = logging.getLogger(__name__)
        self.partition = None
        self.ghost_cells = {}
        self.interface_cells = {}
    
    def partition_mesh(self, mesh, method: str = "metis"):
        """Partition mesh into subdomains.
        
        Args:
            mesh: Mesh to partition
            method: Partitioning method (default: "metis")
                Options: "metis", "geometric", "recursive_bisection"
        """
        if method == "metis":
            self._partition_metis(mesh)
        elif method == "geometric":
            self._partition_geometric(mesh)
        elif method == "recursive_bisection":
            self._partition_recursive_bisection(mesh)
        else:
            raise ValueError(f"Unknown partitioning method: {method}")
        
        # Identify ghost cells and interfaces
        self._identify_ghost_cells(mesh)
        self._identify_interfaces(mesh)
        
        if self.mpi.rank == 0:
            self.logger.info(f"Partitioned mesh using {method} method")
    
    def _partition_metis(self, mesh):
        """Partition mesh using METIS library.
        
        Args:
            mesh: Mesh to partition
        """
        try:
            import metis
        except ImportError:
            raise ImportError("METIS library not found. Please install pymetis.")
        
        # Create graph representation of mesh
        adjacency = mesh.get_adjacency_matrix()
        
        # Partition graph
        n_parts = self.mpi.size
        _, self.partition = metis.part_graph(adjacency, n_parts)
    
    def _partition_geometric(self, mesh):
        """Partition mesh using geometric partitioning.
        
        Args:
            mesh: Mesh to partition
        """
        # Get mesh bounds
        x_min, x_max = mesh.get_bounds()
        
        # Simple geometric partitioning along x-axis
        n_parts = self.mpi.size
        dx = (x_max - x_min) / n_parts
        
        # Assign cells to partitions based on x-coordinate
        cell_centers = mesh.get_cell_centers()
        self.partition = np.floor((cell_centers[:, 0] - x_min) / dx).astype(int)
    
    def _partition_recursive_bisection(self, mesh):
        """Partition mesh using recursive bisection.
        
        Args:
            mesh: Mesh to partition
        """
        # Get mesh bounds
        x_min, x_max = mesh.get_bounds()
        y_min, y_max = mesh.get_bounds()
        
        # Initialize partition
        n_cells = mesh.n_cells
        self.partition = np.zeros(n_cells, dtype=int)
        
        # Recursive bisection
        self._recursive_bisection(mesh, 0, n_cells, 0, self.mpi.size,
                                x_min, x_max, y_min, y_max)
    
    def _recursive_bisection(self, mesh, start, end, part_start, part_end,
                           x_min, x_max, y_min, y_max):
        """Helper function for recursive bisection.
        
        Args:
            mesh: Mesh to partition
            start: Start index
            end: End index
            part_start: Start partition
            part_end: End partition
            x_min, x_max, y_min, y_max: Domain bounds
        """
        if part_end - part_start <= 1:
            return
        
        # Determine split direction
        dx = x_max - x_min
        dy = y_max - y_min
        split_x = dx > dy
        
        # Split domain
        mid_part = (part_start + part_end) // 2
        if split_x:
            mid_coord = (x_min + x_max) / 2
            mask = mesh.get_cell_centers()[start:end, 0] < mid_coord
            self.partition[start:end][mask] = part_start
            self.partition[start:end][~mask] = mid_part
            self._recursive_bisection(mesh, start, end, part_start, mid_part,
                                    x_min, mid_coord, y_min, y_max)
            self._recursive_bisection(mesh, start, end, mid_part, part_end,
                                    mid_coord, x_max, y_min, y_max)
        else:
            mid_coord = (y_min + y_max) / 2
            mask = mesh.get_cell_centers()[start:end, 1] < mid_coord
            self.partition[start:end][mask] = part_start
            self.partition[start:end][~mask] = mid_part
            self._recursive_bisection(mesh, start, end, part_start, mid_part,
                                    x_min, x_max, y_min, mid_coord)
            self._recursive_bisection(mesh, start, end, mid_part, part_end,
                                    x_min, x_max, mid_coord, y_max)
    
    def _identify_ghost_cells(self, mesh):
        """Identify ghost cells for each partition.
        
        Args:
            mesh: Mesh to analyze
        """
        # Get cell neighbors
        neighbors = mesh.get_cell_neighbors()
        
        # For each cell in this partition
        local_cells = np.where(self.partition == self.mpi.rank)[0]
        for cell in local_cells:
            # Check neighbors
            for neighbor in neighbors[cell]:
                if neighbor >= 0:  # Valid neighbor
                    neighbor_part = self.partition[neighbor]
                    if neighbor_part != self.mpi.rank:
                        # This is a ghost cell
                        if neighbor_part not in self.ghost_cells:
                            self.ghost_cells[neighbor_part] = []
                        self.ghost_cells[neighbor_part].append(neighbor)
    
    def _identify_interfaces(self, mesh):
        """Identify interface cells between partitions.
        
        Args:
            mesh: Mesh to analyze
        """
        # Get cell neighbors
        neighbors = mesh.get_cell_neighbors()
        
        # For each cell in this partition
        local_cells = np.where(self.partition == self.mpi.rank)[0]
        for cell in local_cells:
            # Check neighbors
            for neighbor in neighbors[cell]:
                if neighbor >= 0:  # Valid neighbor
                    neighbor_part = self.partition[neighbor]
                    if neighbor_part != self.mpi.rank:
                        # This is an interface cell
                        if neighbor_part not in self.interface_cells:
                            self.interface_cells[neighbor_part] = []
                        self.interface_cells[neighbor_part].append(cell)
    
    def get_local_cells(self) -> np.ndarray:
        """Get local cells for current partition.
        
        Returns:
            Array of local cell indices
        """
        return np.where(self.partition == self.mpi.rank)[0]
    
    def get_ghost_cells(self, partition: int) -> np.ndarray:
        """Get ghost cells from specified partition.
        
        Args:
            partition: Partition rank
            
        Returns:
            Array of ghost cell indices
        """
        return np.array(self.ghost_cells.get(partition, []))
    
    def get_interface_cells(self, partition: int) -> np.ndarray:
        """Get interface cells with specified partition.
        
        Args:
            partition: Partition rank
            
        Returns:
            Array of interface cell indices
        """
        return np.array(self.interface_cells.get(partition, []))
    
    def exchange_ghost_data(self, data: np.ndarray) -> np.ndarray:
        """Exchange ghost cell data between partitions.
        
        Args:
            data: Data to exchange
            
        Returns:
            Updated data with ghost cell values
        """
        # Create requests for non-blocking communication
        requests = []
        
        # Send data to other partitions
        for partition, interface_cells in self.interface_cells.items():
            interface_data = data[interface_cells]
            req = self.mpi.isend(interface_data, dest=partition)
            requests.append(req)
        
        # Receive data from other partitions
        for partition, ghost_cells in self.ghost_cells.items():
            ghost_data = np.empty(len(ghost_cells), dtype=data.dtype)
            req = self.mpi.irecv(ghost_data, source=partition)
            requests.append((req, ghost_cells, ghost_data))
        
        # Wait for all communications to complete
        for req in requests:
            if isinstance(req, tuple):
                req[0].Wait()
                data[req[1]] = req[2]
            else:
                req.Wait()
        
        return data 
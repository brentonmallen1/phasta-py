"""
Load balancing for parallel computing.

This module provides functionality for dynamic load balancing in parallel
computations, including workload estimation and partition redistribution.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .mpi_wrapper import MPIWrapper
from .domain_decomposition import DomainDecomposition
import logging
import time

class LoadBalancer:
    """Load balancer for parallel computing."""
    
    def __init__(self, mpi: MPIWrapper, domain_decomp: DomainDecomposition,
                 balance_threshold: float = 0.1):
        """Initialize load balancer.
        
        Args:
            mpi: MPI wrapper instance
            domain_decomp: Domain decomposition instance
            balance_threshold: Threshold for load imbalance (default: 0.1)
        """
        self.mpi = mpi
        self.domain_decomp = domain_decomp
        self.balance_threshold = balance_threshold
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.workload_history = []
        self.balance_history = []
    
    def estimate_workload(self, mesh, solution: np.ndarray) -> np.ndarray:
        """Estimate computational workload for each cell.
        
        Args:
            mesh: Computational mesh
            solution: Current solution
            
        Returns:
            Array of workload estimates
        """
        # Get local cells
        local_cells = self.domain_decomp.get_local_cells()
        
        # Initialize workload array
        workload = np.zeros(mesh.n_cells)
        
        # Estimate workload based on:
        # 1. Solution gradient
        # 2. Cell size
        # 3. Boundary proximity
        for cell in local_cells:
            # Get cell neighbors
            neighbors = mesh.get_cell_neighbors(cell)
            
            # Compute solution gradient
            grad = self._compute_gradient(solution, cell, neighbors)
            
            # Get cell size
            cell_size = mesh.get_cell_size(cell)
            
            # Check if cell is near boundary
            is_boundary = self._is_boundary_cell(mesh, cell)
            
            # Combine factors
            workload[cell] = (
                np.sum(np.abs(grad)) *  # Gradient factor
                (1.0 / cell_size) *     # Size factor
                (2.0 if is_boundary else 1.0)  # Boundary factor
            )
        
        return workload
    
    def check_balance(self, workload: np.ndarray) -> bool:
        """Check if load balancing is needed.
        
        Args:
            workload: Workload estimates
            
        Returns:
            True if balancing is needed
        """
        # Get local workload
        local_cells = self.domain_decomp.get_local_cells()
        local_workload = np.sum(workload[local_cells])
        
        # Gather all workloads
        all_workloads = self.mpi.gather(local_workload)
        
        if self.mpi.rank == 0:
            # Compute load imbalance
            avg_workload = np.mean(all_workloads)
            max_workload = np.max(all_workloads)
            imbalance = (max_workload - avg_workload) / avg_workload
            
            # Store history
            self.workload_history.append(all_workloads)
            self.balance_history.append(imbalance)
            
            # Check if balancing is needed
            needs_balance = imbalance > self.balance_threshold
            
            if needs_balance:
                self.logger.info(f"Load imbalance detected: {imbalance:.2f}")
            
            return needs_balance
        
        return False
    
    def rebalance(self, mesh, workload: np.ndarray):
        """Rebalance computational load.
        
        Args:
            mesh: Computational mesh
            workload: Workload estimates
        """
        # Get current partition
        old_partition = self.domain_decomp.partition.copy()
        
        # Compute new partition using weighted partitioning
        new_partition = self._compute_weighted_partition(mesh, workload)
        
        # Update domain decomposition
        self.domain_decomp.partition = new_partition
        
        # Identify cells to migrate
        cells_to_migrate = self._identify_migration_cells(old_partition, new_partition)
        
        # Migrate cells
        self._migrate_cells(mesh, cells_to_migrate)
        
        if self.mpi.rank == 0:
            self.logger.info("Load balancing completed")
    
    def _compute_gradient(self, solution: np.ndarray, cell: int,
                         neighbors: List[int]) -> np.ndarray:
        """Compute solution gradient at cell.
        
        Args:
            solution: Current solution
            cell: Cell index
            neighbors: List of neighbor cell indices
            
        Returns:
            Gradient array
        """
        # Initialize gradient
        n_vars = solution.shape[1]
        grad = np.zeros(n_vars)
        
        # Compute gradient using central differences
        for neighbor in neighbors:
            if neighbor >= 0:  # Valid neighbor
                grad += solution[neighbor] - solution[cell]
        
        return grad
    
    def _is_boundary_cell(self, mesh, cell: int) -> bool:
        """Check if cell is near domain boundary.
        
        Args:
            mesh: Computational mesh
            cell: Cell index
            
        Returns:
            True if cell is near boundary
        """
        # Get cell neighbors
        neighbors = mesh.get_cell_neighbors(cell)
        
        # Check if any neighbor is invalid (boundary)
        return -1 in neighbors
    
    def _compute_weighted_partition(self, mesh, workload: np.ndarray) -> np.ndarray:
        """Compute new partition based on workload.
        
        Args:
            mesh: Computational mesh
            workload: Workload estimates
            
        Returns:
            New partition array
        """
        # Get total workload
        total_workload = self.mpi.allreduce(np.sum(workload))
        
        # Compute target workload per process
        target_workload = total_workload / self.mpi.size
        
        # Initialize new partition
        new_partition = np.zeros(mesh.n_cells, dtype=int)
        
        # Sort cells by workload
        sorted_cells = np.argsort(workload)[::-1]  # Descending order
        
        # Assign cells to processes
        current_process = 0
        current_workload = 0
        
        for cell in sorted_cells:
            # Check if we need to move to next process
            if current_workload + workload[cell] > target_workload and \
               current_process < self.mpi.size - 1:
                current_process += 1
                current_workload = 0
            
            # Assign cell to current process
            new_partition[cell] = current_process
            current_workload += workload[cell]
        
        return new_partition
    
    def _identify_migration_cells(self, old_partition: np.ndarray,
                                new_partition: np.ndarray) -> Dict[int, List[int]]:
        """Identify cells that need to be migrated.
        
        Args:
            old_partition: Old partition array
            new_partition: New partition array
            
        Returns:
            Dictionary mapping process ranks to lists of cells to migrate
        """
        # Initialize migration dictionary
        migration = {}
        
        # Find cells that changed partition
        for cell in range(len(old_partition)):
            if old_partition[cell] != new_partition[cell]:
                # Cell needs to be migrated
                if new_partition[cell] not in migration:
                    migration[new_partition[cell]] = []
                migration[new_partition[cell]].append(cell)
        
        return migration
    
    def _migrate_cells(self, mesh, migration: Dict[int, List[int]]):
        """Migrate cells between processes.
        
        Args:
            mesh: Computational mesh
            migration: Dictionary of cells to migrate
        """
        # Send cells to new processes
        for dest_rank, cells in migration.items():
            # Prepare cell data
            cell_data = {
                'indices': cells,
                'solution': mesh.get_cell_data(cells),
                'neighbors': [mesh.get_cell_neighbors(cell) for cell in cells]
            }
            
            # Send data
            self.mpi.send(cell_data, dest=dest_rank)
        
        # Receive cells from other processes
        for source_rank in range(self.mpi.size):
            if source_rank != self.mpi.rank:
                # Receive data
                cell_data = self.mpi.recv(source=source_rank)
                
                # Update mesh
                mesh.update_cells(cell_data['indices'],
                                cell_data['solution'],
                                cell_data['neighbors'])
    
    def get_performance_metrics(self) -> Dict:
        """Get load balancing performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.mpi.rank == 0:
            # Compute metrics from history
            workload_history = np.array(self.workload_history)
            balance_history = np.array(self.balance_history)
            
            metrics = {
                'avg_imbalance': np.mean(balance_history),
                'max_imbalance': np.max(balance_history),
                'min_imbalance': np.min(balance_history),
                'workload_std': np.std(workload_history, axis=1),
                'balance_frequency': len(balance_history)
            }
            
            return metrics
        
        return {} 
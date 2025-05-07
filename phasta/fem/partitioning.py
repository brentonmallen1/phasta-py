"""Mesh partitioning module using METIS."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import sparse
import pymetis
from .mesh import Mesh


class MeshPartitioner:
    """Mesh partitioner using METIS."""
    
    def __init__(self, n_parts: int, n_common_nodes: int = 1):
        """Initialize mesh partitioner.
        
        Args:
            n_parts: Number of partitions
            n_common_nodes: Number of common nodes between partitions
        """
        self.n_parts = n_parts
        self.n_common_nodes = n_common_nodes
    
    def partition(self, mesh: Mesh) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """Partition mesh using METIS.
        
        Args:
            mesh: Mesh to partition
            
        Returns:
            Tuple of (partition array, ghost nodes dictionary)
        """
        # Create adjacency list for METIS
        adjacency_list = self._create_adjacency_list(mesh)
        
        # Partition mesh using METIS
        edge_cut, partitions = pymetis.part_graph(
            self.n_parts,
            adjacency=adjacency_list,
            ncommon=self.n_common_nodes
        )
        
        # Convert partitions to numpy array
        partition_array = np.array(partitions)
        
        # Identify ghost nodes
        ghost_nodes = self._identify_ghost_nodes(mesh, partition_array)
        
        return partition_array, ghost_nodes
    
    def _create_adjacency_list(self, mesh: Mesh) -> List[List[int]]:
        """Create adjacency list for METIS.
        
        Args:
            mesh: Mesh to create adjacency list for
            
        Returns:
            List of lists containing node adjacencies
        """
        # Create sparse adjacency matrix
        n_nodes = len(mesh.nodes)
        adjacency = sparse.lil_matrix((n_nodes, n_nodes))
        
        # Add edges from elements
        for element in mesh.elements:
            for i in range(len(element)):
                for j in range(i + 1, len(element)):
                    adjacency[element[i], element[j]] = 1
                    adjacency[element[j], element[i]] = 1
        
        # Convert to adjacency list
        adjacency_list = []
        for i in range(n_nodes):
            neighbors = adjacency[i].nonzero()[1].tolist()
            adjacency_list.append(neighbors)
        
        return adjacency_list
    
    def _identify_ghost_nodes(self, mesh: Mesh, partition_array: np.ndarray) -> Dict[int, List[int]]:
        """Identify ghost nodes for each partition.
        
        Args:
            mesh: Mesh
            partition_array: Array of partition assignments
            
        Returns:
            Dictionary mapping partition number to list of ghost node indices
        """
        ghost_nodes = {i: [] for i in range(self.n_parts)}
        
        # For each element
        for element in mesh.elements:
            # Get partitions of element nodes
            element_partitions = partition_array[element]
            
            # If element spans multiple partitions
            if len(set(element_partitions)) > 1:
                # Add nodes to ghost lists of other partitions
                for i, node in enumerate(element):
                    node_partition = element_partitions[i]
                    for other_partition in set(element_partitions):
                        if other_partition != node_partition:
                            if node not in ghost_nodes[other_partition]:
                                ghost_nodes[other_partition].append(node)
        
        # Sort ghost node lists
        for partition in ghost_nodes:
            ghost_nodes[partition].sort()
        
        return ghost_nodes


class LoadBalancer:
    """Load balancer for parallel mesh computations."""
    
    def __init__(self, partitioner: MeshPartitioner):
        """Initialize load balancer.
        
        Args:
            partitioner: Mesh partitioner to use
        """
        self.partitioner = partitioner
    
    def balance(self, mesh: Mesh, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """Balance mesh partitions based on weights.
        
        Args:
            mesh: Mesh to balance
            weights: Optional array of weights for each element
            
        Returns:
            Tuple of (partition array, ghost nodes dictionary)
        """
        # If no weights provided, use element sizes
        if weights is None:
            weights = self._compute_element_sizes(mesh)
        
        # Partition mesh with weights
        partition_array, ghost_nodes = self.partitioner.partition(mesh)
        
        # Compute partition weights
        partition_weights = np.zeros(self.partitioner.n_parts)
        for i, element in enumerate(mesh.elements):
            partition = partition_array[element[0]]  # Use first node's partition
            partition_weights[partition] += weights[i]
        
        # Check load balance
        max_weight = np.max(partition_weights)
        min_weight = np.min(partition_weights)
        imbalance = (max_weight - min_weight) / max_weight
        
        # If imbalance is too high, repartition with adjusted weights
        if imbalance > 0.1:  # 10% threshold
            adjusted_weights = self._adjust_weights(weights, partition_array, partition_weights, mesh)
            partition_array, ghost_nodes = self.partitioner.partition(mesh)
        
        return partition_array, ghost_nodes
    
    def _compute_element_sizes(self, mesh: Mesh) -> np.ndarray:
        """Compute size of each element.
        
        Args:
            mesh: Mesh to compute element sizes for
            
        Returns:
            Array of element sizes
        """
        sizes = np.zeros(len(mesh.elements))
        
        for i, element in enumerate(mesh.elements):
            # Compute element volume/area
            element_nodes = mesh.nodes[element]
            if len(element) == 3:  # Triangle
                v1 = element_nodes[1] - element_nodes[0]
                v2 = element_nodes[2] - element_nodes[0]
                sizes[i] = 0.5 * np.abs(np.cross(v1, v2))
            elif len(element) == 4:  # Tetrahedron
                v1 = element_nodes[1] - element_nodes[0]
                v2 = element_nodes[2] - element_nodes[0]
                v3 = element_nodes[3] - element_nodes[0]
                sizes[i] = np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        
        return sizes
    
    def _adjust_weights(self, weights: np.ndarray, partition_array: np.ndarray,
                       partition_weights: np.ndarray, mesh: Mesh) -> np.ndarray:
        """Adjust element weights to improve load balance.
        
        Args:
            weights: Original element weights
            partition_array: Current partition assignments
            partition_weights: Current partition weights
            mesh: Mesh being partitioned
            
        Returns:
            Adjusted element weights
        """
        adjusted_weights = weights.copy()
        target_weight = np.mean(partition_weights)
        
        # Adjust weights of elements in overloaded partitions
        for i, element in enumerate(mesh.elements):
            partition = partition_array[element[0]]
            if partition_weights[partition] > target_weight:
                adjusted_weights[i] *= target_weight / partition_weights[partition]
        
        return adjusted_weights 
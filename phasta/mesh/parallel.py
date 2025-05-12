"""Parallel mesh generation module.

This module provides tools for parallel mesh generation, including domain
decomposition, load balancing, and parallel mesh operations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import logging
from pathlib import Path
from mpi4py import MPI

if TYPE_CHECKING:
    from phasta.mesh.base import Mesh
    from phasta.mesh.cad import CADMeshGenerator

logger = logging.getLogger(__name__)


class DomainDecomposer:
    """Domain decomposition for parallel mesh generation."""
    
    def __init__(self, num_domains: int, method: str = "metis"):
        """Initialize domain decomposer.
        
        Args:
            num_domains: Number of domains to decompose into
            method: Decomposition method ("metis", "rcb", "kdtree")
        """
        self.num_domains = num_domains
        self.method = method
    
    def decompose(self, mesh: 'Mesh') -> Dict[int, List[int]]:
        """Decompose mesh into domains.
        
        Args:
            mesh: Mesh to decompose
            
        Returns:
            Dictionary mapping domain IDs to element lists
        """
        if self.method == "metis":
            return self._decompose_metis(mesh)
        elif self.method == "rcb":
            return self._decompose_rcb(mesh)
        elif self.method == "kdtree":
            return self._decompose_kdtree(mesh)
        else:
            raise ValueError(f"Unknown decomposition method: {self.method}")
    
    def _decompose_metis(self, mesh: 'Mesh') -> Dict[int, List[int]]:
        """Decompose mesh using METIS.
        
        Args:
            mesh: Mesh to decompose
            
        Returns:
            Dictionary mapping domain IDs to element lists
        """
        try:
            import metis
        except ImportError:
            raise ImportError("METIS is required for this decomposition method")
        
        # Create adjacency list
        adj_list = mesh.get_element_adjacency()
        
        # Perform decomposition
        _, parts = metis.part_graph(
            adj_list,
            nparts=self.num_domains,
            recursive=True
        )
        
        # Group elements by domain
        domains = {}
        for i, part in enumerate(parts):
            if part not in domains:
                domains[part] = []
            domains[part].append(i)
        
        return domains
    
    def _decompose_rcb(self, mesh: 'Mesh') -> Dict[int, List[int]]:
        """Decompose mesh using recursive coordinate bisection.
        
        Args:
            mesh: Mesh to decompose
            
        Returns:
            Dictionary mapping domain IDs to element lists
        """
        # Get element centroids
        centroids = mesh.get_element_centroids()
        
        # Initialize domains
        domains = {i: [] for i in range(self.num_domains)}
        
        # Perform recursive bisection
        self._rcb_bisect(centroids, list(range(len(centroids))), domains, 0)
        
        return domains
    
    def _rcb_bisect(self, centroids: np.ndarray, elements: List[int],
                    domains: Dict[int, List[int]], depth: int):
        """Perform recursive coordinate bisection.
        
        Args:
            centroids: Element centroids
            elements: List of element indices
            domains: Dictionary of domains
            depth: Current recursion depth
        """
        if depth >= np.log2(self.num_domains):
            return
        
        # Find longest axis
        axis = np.argmax(np.ptp(centroids[elements], axis=0))
        
        # Sort elements by coordinate
        sorted_elements = sorted(elements, key=lambda i: centroids[i, axis])
        
        # Split elements
        mid = len(sorted_elements) // 2
        left = sorted_elements[:mid]
        right = sorted_elements[mid:]
        
        # Assign to domains
        domain_id = 2**depth
        for i in left:
            domains[domain_id].append(i)
        for i in right:
            domains[domain_id + 1].append(i)
        
        # Recurse
        self._rcb_bisect(centroids, left, domains, depth + 1)
        self._rcb_bisect(centroids, right, domains, depth + 1)
    
    def _decompose_kdtree(self, mesh: 'Mesh') -> Dict[int, List[int]]:
        """Decompose mesh using k-d tree.
        
        Args:
            mesh: Mesh to decompose
            
        Returns:
            Dictionary mapping domain IDs to element lists
        """
        from scipy.spatial import cKDTree
        
        # Get element centroids
        centroids = mesh.get_element_centroids()
        
        # Build k-d tree
        tree = cKDTree(centroids)
        
        # Query points to find nearest neighbors
        _, indices = tree.query(centroids, k=1)
        
        # Group elements by domain
        domains = {}
        for i, idx in enumerate(indices):
            if idx not in domains:
                domains[idx] = []
            domains[idx].append(i)
        
        return domains


class LoadBalancer:
    """Load balancing for parallel mesh generation."""
    
    def __init__(self, method: str = "diffusion"):
        """Initialize load balancer.
        
        Args:
            method: Load balancing method ("diffusion", "recursive")
        """
        self.method = method
    
    def balance(self, mesh: 'Mesh', domains: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """Balance load across domains.
        
        Args:
            mesh: Mesh to balance
            domains: Current domain decomposition
            
        Returns:
            Balanced domain decomposition
        """
        if self.method == "diffusion":
            return self._balance_diffusion(mesh, domains)
        elif self.method == "recursive":
            return self._balance_recursive(mesh, domains)
        else:
            raise ValueError(f"Unknown load balancing method: {self.method}")
    
    def _balance_diffusion(self, mesh: 'Mesh',
                          domains: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """Balance load using diffusion method.
        
        Args:
            mesh: Mesh to balance
            domains: Current domain decomposition
            
        Returns:
            Balanced domain decomposition
        """
        # Get element weights
        weights = mesh.get_element_weights()
        
        # Initialize domain loads
        loads = {i: sum(weights[j] for j in elements)
                for i, elements in domains.items()}
        
        # Perform diffusion
        max_iter = 100
        tolerance = 0.1
        
        for _ in range(max_iter):
            # Check convergence
            if max(loads.values()) - min(loads.values()) < tolerance:
                break
            
            # Compute load differences
            diffs = {}
            for i in domains:
                for j in domains:
                    if i != j:
                        diff = loads[i] - loads[j]
                        if abs(diff) > tolerance:
                            diffs[(i, j)] = diff
            
            # Move elements
            for (i, j), diff in diffs.items():
                if diff > 0:
                    # Move elements from i to j
                    elements_i = domains[i]
                    elements_j = domains[j]
                    
                    # Sort elements by weight
                    sorted_elements = sorted(elements_i,
                                          key=lambda k: weights[k],
                                          reverse=True)
                    
                    # Move elements until load is balanced
                    for k in sorted_elements:
                        if loads[i] - loads[j] < tolerance:
                            break
                        elements_j.append(k)
                        elements_i.remove(k)
                        loads[i] -= weights[k]
                        loads[j] += weights[k]
        
        return domains
    
    def _balance_recursive(self, mesh: 'Mesh',
                          domains: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """Balance load using recursive bisection.
        
        Args:
            mesh: Mesh to balance
            domains: Current domain decomposition
            
        Returns:
            Balanced domain decomposition
        """
        # Get element weights
        weights = mesh.get_element_weights()
        
        # Initialize domain loads
        loads = {i: sum(weights[j] for j in elements)
                for i, elements in domains.items()}
        
        # Sort domains by load
        sorted_domains = sorted(domains.items(),
                              key=lambda x: loads[x[0]],
                              reverse=True)
        
        # Balance recursively
        self._recursive_balance(domains, loads, weights, sorted_domains)
        
        return domains
    
    def _recursive_balance(self, domains: Dict[int, List[int]],
                          loads: Dict[int, float],
                          weights: np.ndarray,
                          sorted_domains: List[Tuple[int, List[int]]]):
        """Perform recursive load balancing.
        
        Args:
            domains: Domain decomposition
            loads: Domain loads
            weights: Element weights
            sorted_domains: Sorted domains by load
        """
        if len(sorted_domains) <= 1:
            return
        
        # Split domains
        mid = len(sorted_domains) // 2
        heavy = sorted_domains[:mid]
        light = sorted_domains[mid:]
        
        # Compute target load
        total_load = sum(loads[i] for i, _ in sorted_domains)
        target_load = total_load / len(sorted_domains)
        
        # Balance between heavy and light domains
        for i, elements_i in heavy:
            for j, elements_j in light:
                if loads[i] - loads[j] > target_load:
                    # Move elements from i to j
                    sorted_elements = sorted(elements_i,
                                          key=lambda k: weights[k],
                                          reverse=True)
                    
                    for k in sorted_elements:
                        if loads[i] - loads[j] <= target_load:
                            break
                        elements_j.append(k)
                        elements_i.remove(k)
                        loads[i] -= weights[k]
                        loads[j] += weights[k]
        
        # Recurse
        self._recursive_balance(domains, loads, weights, heavy)
        self._recursive_balance(domains, loads, weights, light)


class ParallelMeshGenerator:
    """Parallel mesh generator."""
    
    def __init__(self, mesh_generator: 'CADMeshGenerator',
                 num_domains: Optional[int] = None,
                 decomposition_method: str = "metis",
                 load_balancing_method: str = "diffusion"):
        """Initialize parallel mesh generator.
        
        Args:
            mesh_generator: Base mesh generator
            num_domains: Number of domains (default: number of MPI processes)
            decomposition_method: Domain decomposition method
            load_balancing_method: Load balancing method
        """
        self.mesh_generator = mesh_generator
        self.num_domains = num_domains or MPI.COMM_WORLD.Get_size()
        self.decomposer = DomainDecomposer(self.num_domains,
                                         decomposition_method)
        self.balancer = LoadBalancer(load_balancing_method)
    
    def generate_mesh(self, cad_file: Union[str, Path]) -> 'Mesh':
        """Generate mesh in parallel.
        
        Args:
            cad_file: Path to CAD file
            
        Returns:
            Generated mesh
        """
        # Get MPI communicator
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            # Generate initial mesh on root process
            mesh = self.mesh_generator.generate_mesh(cad_file)
            
            # Decompose mesh
            domains = self.decomposer.decompose(mesh)
            
            # Balance load
            domains = self.balancer.balance(mesh, domains)
            
            # Send domains to other processes
            for i in range(1, comm.Get_size()):
                domain_elements = domains.get(i, [])
                comm.send(domain_elements, dest=i)
            
            # Keep root domain
            domain_elements = domains.get(0, [])
        else:
            # Receive domain from root process
            domain_elements = comm.recv(source=0)
        
        # Create local mesh
        local_mesh = self._create_local_mesh(mesh, domain_elements)
        
        # Exchange ghost elements
        ghost_elements = self._exchange_ghost_elements(local_mesh, comm)
        
        # Add ghost elements to local mesh
        local_mesh.add_ghost_elements(ghost_elements)
        
        return local_mesh
    
    def _create_local_mesh(self, mesh: 'Mesh',
                          domain_elements: List[int]) -> 'Mesh':
        """Create local mesh for a domain.
        
        Args:
            mesh: Global mesh
            domain_elements: Elements in this domain
            
        Returns:
            Local mesh
        """
        # Extract local nodes and elements
        local_nodes = []
        local_elements = []
        node_map = {}  # Map from global to local node indices
        
        for element in domain_elements:
            element_nodes = mesh.get_element_nodes(element)
            local_element = []
            
            for node in element_nodes:
                if node not in node_map:
                    node_map[node] = len(local_nodes)
                    local_nodes.append(mesh.get_node_coordinates(node))
                local_element.append(node_map[node])
            
            local_elements.append(local_element)
        
        # Create local mesh
        from phasta.mesh.base import Mesh
        return Mesh(np.array(local_nodes), np.array(local_elements))
    
    def _exchange_ghost_elements(self, local_mesh: 'Mesh',
                                comm: MPI.Comm) -> Dict[int, List[int]]:
        """Exchange ghost elements between domains.
        
        Args:
            local_mesh: Local mesh
            comm: MPI communicator
            
        Returns:
            Dictionary mapping process ranks to ghost elements
        """
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Get element adjacency
        adj_list = local_mesh.get_element_adjacency()
        
        # Find ghost elements
        ghost_elements = {}
        for i, adj in enumerate(adj_list):
            for j in adj:
                if j >= len(local_mesh.elements):
                    # This is a ghost element
                    owner = j // len(local_mesh.elements)
                    if owner not in ghost_elements:
                        ghost_elements[owner] = []
                    ghost_elements[owner].append(i)
        
        # Exchange ghost elements
        for i in range(size):
            if i != rank:
                # Send ghost elements
                if i in ghost_elements:
                    comm.send(ghost_elements[i], dest=i)
                
                # Receive ghost elements
                if rank in ghost_elements.get(i, []):
                    received = comm.recv(source=i)
                    ghost_elements[i] = received
        
        return ghost_elements 
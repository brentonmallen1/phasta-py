"""Parallel computing support for finite element method."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import sparse
from mpi4py import MPI
from .mesh import Mesh
from .assembly import GlobalAssembly
from .solvers import LinearSolver


class ParallelMesh:
    """Parallel mesh for distributed computing."""
    
    def __init__(self, mesh: Mesh, comm: Optional[MPI.Comm] = None):
        """Initialize parallel mesh.
        
        Args:
            mesh: Mesh to partition
            comm: MPI communicator (default: MPI.COMM_WORLD)
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Partition mesh
        self.local_nodes, self.local_elements, self.ghost_nodes = self._partition_mesh(mesh)
        
        # Create communication buffers
        self.send_buffers = {}
        self.recv_buffers = {}
        self._setup_communication()
    
    def _partition_mesh(self, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]:
        """Partition mesh across processes.
        
        Args:
            mesh: Mesh to partition
            
        Returns:
            Tuple of (local nodes, local elements, ghost nodes)
        """
        # Simple partitioning: divide elements evenly
        n_elements = len(mesh.elements)
        elements_per_proc = n_elements // self.size
        start_idx = self.rank * elements_per_proc
        end_idx = start_idx + elements_per_proc if self.rank < self.size - 1 else n_elements
        
        # Get local elements
        local_elements = mesh.elements[start_idx:end_idx]
        
        # Get local nodes
        local_node_indices = set()
        for element in local_elements:
            local_node_indices.update(element)
        local_nodes = mesh.nodes[list(local_node_indices)]
        
        # Identify ghost nodes
        ghost_nodes = self._identify_ghost_nodes(mesh, local_node_indices)
        
        return local_nodes, local_elements, ghost_nodes
    
    def _identify_ghost_nodes(self, mesh: Mesh, local_node_indices: set) -> Dict[int, List[int]]:
        """Identify ghost nodes for each process.
        
        Args:
            mesh: Full mesh
            local_node_indices: Set of local node indices
            
        Returns:
            Dictionary mapping process rank to list of ghost node indices
        """
        ghost_nodes = {}
        
        # For each process
        for rank in range(self.size):
            if rank == self.rank:
                continue
            
            # Get elements owned by this process
            n_elements = len(mesh.elements)
            elements_per_proc = n_elements // self.size
            start_idx = rank * elements_per_proc
            end_idx = start_idx + elements_per_proc if rank < self.size - 1 else n_elements
            remote_elements = mesh.elements[start_idx:end_idx]
            
            # Find nodes shared with this process
            shared_nodes = set()
            for element in remote_elements:
                shared_nodes.update(element)
            shared_nodes &= local_node_indices
            
            if shared_nodes:
                ghost_nodes[rank] = list(shared_nodes)
        
        return ghost_nodes
    
    def _setup_communication(self) -> None:
        """Set up communication buffers for ghost nodes."""
        # Create send buffers
        for rank, nodes in self.ghost_nodes.items():
            self.send_buffers[rank] = np.zeros(len(nodes))
        
        # Create receive buffers
        for rank, nodes in self.ghost_nodes.items():
            self.recv_buffers[rank] = np.zeros(len(nodes))


class ParallelAssembly(GlobalAssembly):
    """Parallel assembly of finite element matrices."""
    
    def __init__(self, parallel_mesh: ParallelMesh):
        """Initialize parallel assembly.
        
        Args:
            parallel_mesh: Parallel mesh
        """
        super().__init__()
        self.parallel_mesh = parallel_mesh
        self.comm = parallel_mesh.comm
    
    def assemble_matrix(self, element_matrix_func: callable) -> sparse.spmatrix:
        """Assemble global matrix in parallel.
        
        Args:
            element_matrix_func: Function to compute element matrices
            
        Returns:
            Global matrix
        """
        # Assemble local matrix
        local_matrix = self._assemble_local_matrix(element_matrix_func)
        
        # Communicate ghost node contributions
        self._communicate_ghost_contributions(local_matrix)
        
        return local_matrix
    
    def _assemble_local_matrix(self, element_matrix_func: callable) -> sparse.spmatrix:
        """Assemble local matrix.
        
        Args:
            element_matrix_func: Function to compute element matrices
            
        Returns:
            Local matrix
        """
        # Get local elements and nodes
        local_elements = self.parallel_mesh.local_elements
        local_nodes = self.parallel_mesh.local_nodes
        
        # Create local matrix
        n_local = len(local_nodes)
        local_matrix = sparse.lil_matrix((n_local, n_local))
        
        # Assemble element contributions
        for element in local_elements:
            ke = element_matrix_func(element)
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    local_matrix[node_i, node_j] += ke[i, j]
        
        return local_matrix.tocsr()
    
    def _communicate_ghost_contributions(self, local_matrix: sparse.spmatrix) -> None:
        """Communicate ghost node contributions between processes.
        
        Args:
            local_matrix: Local matrix
        """
        # Prepare send buffers
        for rank, nodes in self.parallel_mesh.ghost_nodes.items():
            for i, node in enumerate(nodes):
                self.parallel_mesh.send_buffers[rank][i] = local_matrix[node, node]
        
        # Send and receive contributions
        requests = []
        for rank in self.parallel_mesh.ghost_nodes:
            req = self.comm.Isend(self.parallel_mesh.send_buffers[rank], dest=rank)
            requests.append(req)
            req = self.comm.Irecv(self.parallel_mesh.recv_buffers[rank], source=rank)
            requests.append(req)
        
        # Wait for communication to complete
        MPI.Request.Waitall(requests)
        
        # Add received contributions
        for rank, nodes in self.parallel_mesh.ghost_nodes.items():
            for i, node in enumerate(nodes):
                local_matrix[node, node] += self.parallel_mesh.recv_buffers[rank][i]


class ParallelSolver(LinearSolver):
    """Parallel solver for distributed systems."""
    
    def __init__(self, parallel_mesh: ParallelMesh, solver: LinearSolver):
        """Initialize parallel solver.
        
        Args:
            parallel_mesh: Parallel mesh
            solver: Local solver to use
        """
        super().__init__()
        self.parallel_mesh = parallel_mesh
        self.solver = solver
        self.comm = parallel_mesh.comm
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
        """Solve linear system in parallel.
        
        Args:
            A: System matrix
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        # Solve local system
        x_local = self.solver.solve(A, b)
        
        # Communicate solution at ghost nodes
        self._communicate_solution(x_local)
        
        return x_local
    
    def _communicate_solution(self, x: np.ndarray) -> None:
        """Communicate solution values at ghost nodes.
        
        Args:
            x: Local solution vector
        """
        # Prepare send buffers
        for rank, nodes in self.parallel_mesh.ghost_nodes.items():
            for i, node in enumerate(nodes):
                self.parallel_mesh.send_buffers[rank][i] = x[node]
        
        # Send and receive solution values
        requests = []
        for rank in self.parallel_mesh.ghost_nodes:
            req = self.comm.Isend(self.parallel_mesh.send_buffers[rank], dest=rank)
            requests.append(req)
            req = self.comm.Irecv(self.parallel_mesh.recv_buffers[rank], source=rank)
            requests.append(req)
        
        # Wait for communication to complete
        MPI.Request.Waitall(requests)
        
        # Update ghost node values
        for rank, nodes in self.parallel_mesh.ghost_nodes.items():
            for i, node in enumerate(nodes):
                x[node] = self.parallel_mesh.recv_buffers[rank][i] 
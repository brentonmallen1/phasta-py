"""Tests for parallel computing module."""

import numpy as np
import pytest
from scipy import sparse
from mpi4py import MPI
from phasta.fem.parallel import ParallelMesh, ParallelAssembly, ParallelSolver
from phasta.fem.mesh import Mesh
from phasta.fem.solvers import GMRESSolver


@pytest.mark.mpi
def test_parallel_mesh():
    """Test parallel mesh partitioning."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(nodes, elements)
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh)
    
    # Check local data
    assert parallel_mesh.local_nodes is not None
    assert parallel_mesh.local_elements is not None
    assert parallel_mesh.ghost_nodes is not None
    
    # Check communication buffers
    assert isinstance(parallel_mesh.send_buffers, dict)
    assert isinstance(parallel_mesh.recv_buffers, dict)


@pytest.mark.mpi
def test_parallel_assembly():
    """Test parallel matrix assembly."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(nodes, elements)
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh)
    
    # Create parallel assembly
    assembly = ParallelAssembly(parallel_mesh)
    
    # Define element matrix function
    def element_matrix(element):
        return np.eye(3)  # Simple identity matrix for testing
    
    # Assemble matrix
    matrix = assembly.assemble_matrix(element_matrix)
    
    # Check matrix properties
    assert isinstance(matrix, sparse.spmatrix)
    assert matrix.shape[0] == matrix.shape[1]  # Square matrix
    assert matrix.nnz > 0  # Non-zero entries


@pytest.mark.mpi
def test_parallel_solver():
    """Test parallel solver."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(nodes, elements)
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh)
    
    # Create local solver
    local_solver = GMRESSolver()
    
    # Create parallel solver
    solver = ParallelSolver(parallel_mesh, local_solver)
    
    # Create test system
    n = len(parallel_mesh.local_nodes)
    A = sparse.eye(n)
    b = np.ones(n)
    
    # Solve system
    x = solver.solve(A, b)
    
    # Check solution
    assert x.shape == b.shape
    assert not np.allclose(x, 0)  # Non-zero solution


@pytest.mark.mpi
def test_parallel_communication():
    """Test parallel communication."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(nodes, elements)
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh)
    
    # Test communication buffers
    for rank, nodes in parallel_mesh.ghost_nodes.items():
        # Check send buffer
        assert rank in parallel_mesh.send_buffers
        assert parallel_mesh.send_buffers[rank].shape == (len(nodes),)
        
        # Check receive buffer
        assert rank in parallel_mesh.recv_buffers
        assert parallel_mesh.recv_buffers[rank].shape == (len(nodes),)


@pytest.mark.mpi
def test_parallel_assembly_communication():
    """Test communication during parallel assembly."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(nodes, elements)
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh)
    
    # Create parallel assembly
    assembly = ParallelAssembly(parallel_mesh)
    
    # Define element matrix function
    def element_matrix(element):
        return np.eye(3)  # Simple identity matrix for testing
    
    # Assemble matrix
    matrix = assembly.assemble_matrix(element_matrix)
    
    # Check that ghost node contributions were communicated
    for rank, nodes in parallel_mesh.ghost_nodes.items():
        for node in nodes:
            assert matrix[node, node] != 0  # Non-zero contribution 


"""Tests for parallel mesh generation module."""

import numpy as np
import pytest
from mpi4py import MPI
from pathlib import Path
from unittest.mock import Mock, patch

from phasta.mesh.parallel import (
    DomainDecomposer,
    LoadBalancer,
    ParallelMeshGenerator
)
from phasta.mesh.base import Mesh


class MockMesh:
    """Mock mesh class for testing."""
    
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
    
    def get_element_adjacency(self):
        """Get element adjacency list."""
        # Create simple adjacency for testing
        adj = []
        for i in range(len(self.elements)):
            adj.append([j for j in range(len(self.elements))
                       if i != j and np.any(np.isin(self.elements[i],
                                                   self.elements[j]))])
        return adj
    
    def get_element_centroids(self):
        """Get element centroids."""
        return np.array([np.mean(self.nodes[e], axis=0)
                        for e in self.elements])
    
    def get_element_weights(self):
        """Get element weights."""
        # Use element volume as weight
        return np.array([np.linalg.det(self.nodes[e[1:]] - self.nodes[e[0]])
                        for e in self.elements])
    
    def get_element_nodes(self, element):
        """Get nodes of an element."""
        return self.elements[element]
    
    def get_node_coordinates(self, node):
        """Get coordinates of a node."""
        return self.nodes[node]


def test_domain_decomposer():
    """Test domain decomposition."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = MockMesh(nodes, elements)
    
    # Test METIS decomposition
    decomposer = DomainDecomposer(2, method="metis")
    with patch('metis.part_graph') as mock_part:
        mock_part.return_value = (None, [0, 1])
        domains = decomposer.decompose(mesh)
        assert len(domains) == 2
        assert len(domains[0]) == 1
        assert len(domains[1]) == 1
    
    # Test RCB decomposition
    decomposer = DomainDecomposer(2, method="rcb")
    domains = decomposer.decompose(mesh)
    assert len(domains) == 2
    assert len(domains[0]) + len(domains[1]) == 2
    
    # Test k-d tree decomposition
    decomposer = DomainDecomposer(2, method="kdtree")
    domains = decomposer.decompose(mesh)
    assert len(domains) == 2
    assert len(domains[0]) + len(domains[1]) == 2
    
    # Test invalid method
    with pytest.raises(ValueError):
        decomposer = DomainDecomposer(2, method="invalid")
        decomposer.decompose(mesh)


def test_load_balancer():
    """Test load balancing."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = MockMesh(nodes, elements)
    
    # Create initial domains
    domains = {0: [0], 1: [1]}
    
    # Test diffusion balancing
    balancer = LoadBalancer(method="diffusion")
    balanced = balancer.balance(mesh, domains)
    assert len(balanced) == 2
    assert len(balanced[0]) + len(balanced[1]) == 2
    
    # Test recursive balancing
    balancer = LoadBalancer(method="recursive")
    balanced = balancer.balance(mesh, domains)
    assert len(balanced) == 2
    assert len(balanced[0]) + len(balanced[1]) == 2
    
    # Test invalid method
    with pytest.raises(ValueError):
        balancer = LoadBalancer(method="invalid")
        balancer.balance(mesh, domains)


def test_parallel_mesh_generator():
    """Test parallel mesh generation."""
    # Create mock mesh generator
    mock_generator = Mock()
    mock_generator.generate_mesh.return_value = MockMesh(
        np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
        np.array([[0, 1, 2], [1, 3, 2]])
    )
    
    # Create parallel mesh generator
    generator = ParallelMeshGenerator(
        mock_generator,
        num_domains=2,
        decomposition_method="metis",
        load_balancing_method="diffusion"
    )
    
    # Test mesh generation
    with patch('metis.part_graph') as mock_part:
        mock_part.return_value = (None, [0, 1])
        mesh = generator.generate_mesh("test.step")
        assert isinstance(mesh, Mesh)
        assert len(mesh.nodes) > 0
        assert len(mesh.elements) > 0


def test_parallel_mesh_generator_invalid_inputs():
    """Test parallel mesh generator with invalid inputs."""
    # Create mock mesh generator
    mock_generator = Mock()
    
    # Test invalid CAD file
    generator = ParallelMeshGenerator(mock_generator)
    with pytest.raises(FileNotFoundError):
        generator.generate_mesh("nonexistent.step")
    
    # Test invalid number of domains
    with pytest.raises(ValueError):
        ParallelMeshGenerator(mock_generator, num_domains=0)
    
    # Test invalid decomposition method
    with pytest.raises(ValueError):
        ParallelMeshGenerator(mock_generator, decomposition_method="invalid")
    
    # Test invalid load balancing method
    with pytest.raises(ValueError):
        ParallelMeshGenerator(mock_generator, load_balancing_method="invalid")


def test_parallel_mesh_generator_ghost_elements():
    """Test ghost element handling in parallel mesh generation."""
    # Create test mesh with overlapping elements
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]])
    mesh = MockMesh(nodes, elements)
    
    # Create parallel mesh generator
    mock_generator = Mock()
    mock_generator.generate_mesh.return_value = mesh
    generator = ParallelMeshGenerator(mock_generator, num_domains=2)
    
    # Test ghost element exchange
    with patch('metis.part_graph') as mock_part:
        mock_part.return_value = (None, [0, 0, 1, 1])
        local_mesh = generator.generate_mesh("test.step")
        assert isinstance(local_mesh, Mesh)
        assert len(local_mesh.nodes) > 0
        assert len(local_mesh.elements) > 0


def test_parallel_mesh_generator_3d():
    """Test parallel mesh generation with 3D mesh."""
    # Create 3D test mesh
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                      [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    elements = np.array([[0, 1, 2, 4], [1, 3, 2, 5],
                        [2, 3, 7, 6], [4, 5, 6, 7]])
    mesh = MockMesh(nodes, elements)
    
    # Create parallel mesh generator
    mock_generator = Mock()
    mock_generator.generate_mesh.return_value = mesh
    generator = ParallelMeshGenerator(mock_generator, num_domains=2)
    
    # Test mesh generation
    with patch('metis.part_graph') as mock_part:
        mock_part.return_value = (None, [0, 0, 1, 1])
        local_mesh = generator.generate_mesh("test.step")
        assert isinstance(local_mesh, Mesh)
        assert len(local_mesh.nodes) > 0
        assert len(local_mesh.elements) > 0


def test_parallel_mesh_generator_large_mesh():
    """Test parallel mesh generation with large mesh."""
    # Create large test mesh
    n = 100
    nodes = np.random.rand(n, 3)
    elements = np.array([[i, i+1, i+2, i+3]
                        for i in range(0, n-3, 4)])
    mesh = MockMesh(nodes, elements)
    
    # Create parallel mesh generator
    mock_generator = Mock()
    mock_generator.generate_mesh.return_value = mesh
    generator = ParallelMeshGenerator(mock_generator, num_domains=4)
    
    # Test mesh generation
    with patch('metis.part_graph') as mock_part:
        mock_part.return_value = (None, [0] * (n//4) + [1] * (n//4) +
                                 [2] * (n//4) + [3] * (n//4))
        local_mesh = generator.generate_mesh("test.step")
        assert isinstance(local_mesh, Mesh)
        assert len(local_mesh.nodes) > 0
        assert len(local_mesh.elements) > 0


def test_parallel_mesh_generator_mpi():
    """Test parallel mesh generation with MPI."""
    # Skip if not running with MPI
    if MPI.COMM_WORLD.Get_size() < 2:
        pytest.skip("Test requires at least 2 MPI processes")
    
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = MockMesh(nodes, elements)
    
    # Create parallel mesh generator
    mock_generator = Mock()
    mock_generator.generate_mesh.return_value = mesh
    generator = ParallelMeshGenerator(mock_generator)
    
    # Test mesh generation
    with patch('metis.part_graph') as mock_part:
        mock_part.return_value = (None, [0, 1])
        local_mesh = generator.generate_mesh("test.step")
        assert isinstance(local_mesh, Mesh)
        assert len(local_mesh.nodes) > 0
        assert len(local_mesh.elements) > 0


def test_parallel_mesh_generator_quality():
    """Test mesh quality in parallel generation."""
    # Create test mesh
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = MockMesh(nodes, elements)
    
    # Create parallel mesh generator
    mock_generator = Mock()
    mock_generator.generate_mesh.return_value = mesh
    generator = ParallelMeshGenerator(mock_generator, num_domains=2)
    
    # Test mesh generation
    with patch('metis.part_graph') as mock_part:
        mock_part.return_value = (None, [0, 1])
        local_mesh = generator.generate_mesh("test.step")
        
        # Check element quality
        for element in local_mesh.elements:
            # Check element angles
            angles = []
            for i in range(3):
                v1 = local_mesh.nodes[element[(i+1)%3]] - local_mesh.nodes[element[i]]
                v2 = local_mesh.nodes[element[(i+2)%3]] - local_mesh.nodes[element[i]]
                angle = np.arccos(np.dot(v1, v2) /
                                (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angles.append(angle)
            
            # Check minimum angle
            assert min(angles) > 0.1
            
            # Check maximum angle
            assert max(angles) < np.pi - 0.1
            
            # Check aspect ratio
            edges = []
            for i in range(3):
                edge = local_mesh.nodes[element[(i+1)%3]] - local_mesh.nodes[element[i]]
                edges.append(np.linalg.norm(edge))
            aspect_ratio = max(edges) / min(edges)
            assert aspect_ratio < 2.0 


"""Tests for parallel processing."""

import numpy as np
import pytest
from phasta.solver.parallel import (
    DomainDecomposer,
    RecursiveBisection,
    LoadBalancer,
    CommunicationOptimizer
)


def test_recursive_bisection():
    """Test recursive bisection domain decomposition."""
    # Create decomposer
    decomposer = RecursiveBisection(num_cuts=2)
    
    # Test data
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    elements = np.array([
        [0, 1, 2, 4],  # Tetrahedron 1
        [1, 4, 5, 7],  # Tetrahedron 2
        [2, 4, 6, 7],  # Tetrahedron 3
        [0, 2, 3, 6]   # Tetrahedron 4
    ])
    
    # Test decomposition
    local_vertices, local_elements = decomposer.decompose(vertices, elements)
    
    # Check results
    assert local_vertices.shape[1] == 3  # 3D coordinates
    assert local_elements.shape[1] == 4  # 4 vertices per element
    assert np.all(np.isfinite(local_vertices))
    assert np.all(np.isfinite(local_elements))
    
    # Test ghost elements
    ghost_elements = decomposer.get_ghost_elements()
    assert isinstance(ghost_elements, np.ndarray)


def test_load_balancer():
    """Test load balancer."""
    # Create decomposer and balancer
    decomposer = RecursiveBisection()
    balancer = LoadBalancer(decomposer)
    
    # Test data
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    elements = np.array([
        [0, 1, 2, 4],  # Tetrahedron 1
        [1, 4, 5, 7],  # Tetrahedron 2
        [2, 4, 6, 7],  # Tetrahedron 3
        [0, 2, 3, 6]   # Tetrahedron 4
    ])
    
    # Test balancing
    balanced_vertices, balanced_elements = balancer.balance(vertices, elements)
    
    # Check results
    assert balanced_vertices.shape[1] == 3  # 3D coordinates
    assert balanced_elements.shape[1] == 4  # 4 vertices per element
    assert np.all(np.isfinite(balanced_vertices))
    assert np.all(np.isfinite(balanced_elements))


def test_communication_optimizer():
    """Test communication optimizer."""
    # Create decomposer and optimizer
    decomposer = RecursiveBisection()
    optimizer = CommunicationOptimizer(decomposer)
    
    # Test data
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    elements = np.array([
        [0, 1, 2, 4],  # Tetrahedron 1
        [1, 4, 5, 7],  # Tetrahedron 2
        [2, 4, 6, 7],  # Tetrahedron 3
        [0, 2, 3, 6]   # Tetrahedron 4
    ])
    
    # Test optimization
    optimized_vertices, optimized_elements = optimizer.optimize(vertices, elements)
    
    # Check results
    assert optimized_vertices.shape == vertices.shape
    assert optimized_elements.shape == elements.shape
    assert np.all(np.isfinite(optimized_vertices))
    assert np.all(np.isfinite(optimized_elements))


def test_edge_cases():
    """Test edge cases."""
    # Create decomposer
    decomposer = RecursiveBisection()
    
    # Test zero vertices
    vertices = np.zeros((0, 3))
    elements = np.zeros((0, 4), dtype=int)
    
    with pytest.raises(ValueError):
        decomposer.decompose(vertices, elements)
    
    # Test zero elements
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    elements = np.zeros((0, 4), dtype=int)
    
    with pytest.raises(ValueError):
        decomposer.decompose(vertices, elements)
    
    # Test invalid element indices
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    elements = np.array([
        [0, 1, 2, 4]  # Invalid vertex index
    ])
    
    with pytest.raises(IndexError):
        decomposer.decompose(vertices, elements)


def test_memory_management():
    """Test memory management."""
    # Create decomposer
    decomposer = RecursiveBisection()
    
    # Test large mesh
    n_vertices = 1000
    n_elements = 2000
    
    # Generate random mesh
    vertices = np.random.rand(n_vertices, 3)
    elements = np.random.randint(0, n_vertices, (n_elements, 4))
    
    # Test decomposition
    local_vertices, local_elements = decomposer.decompose(vertices, elements)
    
    # Check results
    assert local_vertices.shape[1] == 3  # 3D coordinates
    assert local_elements.shape[1] == 4  # 4 vertices per element
    assert np.all(np.isfinite(local_vertices))
    assert np.all(np.isfinite(local_elements))


def test_convergence():
    """Test convergence with mesh refinement."""
    # Create decomposer and balancer
    decomposer = RecursiveBisection()
    balancer = LoadBalancer(decomposer)
    
    # Test data
    n_vertices = [10, 20, 40, 80]
    n_elements = [20, 40, 80, 160]
    
    # Test different mesh sizes
    imbalances = []
    
    for nv, ne in zip(n_vertices, n_elements):
        # Generate random mesh
        vertices = np.random.rand(nv, 3)
        elements = np.random.randint(0, nv, (ne, 4))
        
        # Balance mesh
        balanced_vertices, balanced_elements = balancer.balance(vertices, elements)
        
        # Compute imbalance
        imbalance = balancer._compute_imbalance(balanced_elements)
        imbalances.append(imbalance)
    
    # Check convergence
    for i in range(len(imbalances) - 1):
        assert imbalances[i+1] <= imbalances[i]  # Should improve with mesh refinement 
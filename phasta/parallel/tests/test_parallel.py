"""
Test suite for parallel computing components.

This module contains tests for MPI wrapper, domain decomposition, and parallel I/O.
"""

import pytest
import numpy as np
from mpi4py import MPI
from ..mpi_wrapper import MPIWrapper
from ..domain_decomposition import DomainDecomposition
from ..parallel_io import ParallelIO
import os
import tempfile

@pytest.fixture
def mpi():
    """Create MPI wrapper instance."""
    return MPIWrapper()

@pytest.fixture
def domain_decomp(mpi):
    """Create domain decomposition instance."""
    return DomainDecomposition(mpi)

@pytest.fixture
def parallel_io(mpi):
    """Create parallel I/O instance."""
    return ParallelIO(mpi)

class TestMPIWrapper:
    """Test MPI wrapper functionality."""
    
    def test_initialization(self, mpi):
        """Test MPI initialization."""
        assert mpi.comm is not None
        assert mpi.rank >= 0
        assert mpi.size > 0
    
    def test_broadcast(self, mpi):
        """Test broadcast operation."""
        data = np.array([1, 2, 3]) if mpi.rank == 0 else None
        result = mpi.broadcast(data)
        assert np.array_equal(result, np.array([1, 2, 3]))
    
    def test_gather(self, mpi):
        """Test gather operation."""
        data = np.array([mpi.rank])
        result = mpi.gather(data)
        if mpi.rank == 0:
            expected = np.array([i for i in range(mpi.size)])
            assert np.array_equal(result, expected)
    
    def test_scatter(self, mpi):
        """Test scatter operation."""
        if mpi.rank == 0:
            data = [np.array([i]) for i in range(mpi.size)]
        else:
            data = None
        result = mpi.scatter(data)
        assert np.array_equal(result, np.array([mpi.rank]))
    
    def test_allreduce(self, mpi):
        """Test allreduce operation."""
        data = np.array([1])
        result = mpi.allreduce(data)
        assert result == mpi.size
    
    def test_send_recv(self, mpi):
        """Test send/receive operations."""
        if mpi.rank == 0:
            data = np.array([1, 2, 3])
            mpi.send(data, dest=1)
        elif mpi.rank == 1:
            result = mpi.recv(source=0)
            assert np.array_equal(result, np.array([1, 2, 3]))
    
    def test_isend_irecv(self, mpi):
        """Test non-blocking send/receive operations."""
        if mpi.rank == 0:
            data = np.array([1, 2, 3])
            req = mpi.isend(data, dest=1)
            req.Wait()
        elif mpi.rank == 1:
            req = mpi.irecv(source=0)
            result = req.Wait()
            assert np.array_equal(result, np.array([1, 2, 3]))

class TestDomainDecomposition:
    """Test domain decomposition functionality."""
    
    def test_partition_mesh(self, domain_decomp, mpi):
        """Test mesh partitioning."""
        # Create simple mesh
        n_cells = 100
        cell_centers = np.random.rand(n_cells, 2)
        neighbors = np.random.randint(-1, n_cells, (n_cells, 4))
        
        class MockMesh:
            def __init__(self):
                self.n_cells = n_cells
                self.cell_centers = cell_centers
                self.neighbors = neighbors
            
            def get_adjacency_matrix(self):
                return neighbors
            
            def get_bounds(self):
                return (0, 1), (0, 1)
            
            def get_cell_centers(self):
                return cell_centers
        
        mesh = MockMesh()
        
        # Test different partitioning methods
        for method in ["metis", "geometric", "recursive_bisection"]:
            try:
                domain_decomp.partition_mesh(mesh, method=method)
                assert domain_decomp.partition is not None
                assert len(domain_decomp.partition) == n_cells
            except ImportError:
                # Skip if METIS is not available
                if method == "metis":
                    continue
                raise
    
    def test_ghost_cells(self, domain_decomp, mpi):
        """Test ghost cell identification."""
        # Create simple mesh with known ghost cells
        n_cells = 100
        partition = np.zeros(n_cells, dtype=int)
        partition[50:] = 1
        
        class MockMesh:
            def __init__(self):
                self.n_cells = n_cells
                self.partition = partition
            
            def get_cell_neighbors(self):
                neighbors = np.full((n_cells, 4), -1)
                for i in range(n_cells-1):
                    neighbors[i, 0] = i + 1
                    neighbors[i+1, 1] = i
                return neighbors
        
        mesh = MockMesh()
        domain_decomp.partition = partition
        domain_decomp._identify_ghost_cells(mesh)
        
        if mpi.rank == 0:
            assert 1 in domain_decomp.ghost_cells
            assert len(domain_decomp.ghost_cells[1]) > 0
        elif mpi.rank == 1:
            assert 0 in domain_decomp.ghost_cells
            assert len(domain_decomp.ghost_cells[0]) > 0
    
    def test_interface_cells(self, domain_decomp, mpi):
        """Test interface cell identification."""
        # Create simple mesh with known interface cells
        n_cells = 100
        partition = np.zeros(n_cells, dtype=int)
        partition[50:] = 1
        
        class MockMesh:
            def __init__(self):
                self.n_cells = n_cells
                self.partition = partition
            
            def get_cell_neighbors(self):
                neighbors = np.full((n_cells, 4), -1)
                for i in range(n_cells-1):
                    neighbors[i, 0] = i + 1
                    neighbors[i+1, 1] = i
                return neighbors
        
        mesh = MockMesh()
        domain_decomp.partition = partition
        domain_decomp._identify_interfaces(mesh)
        
        if mpi.rank == 0:
            assert 1 in domain_decomp.interface_cells
            assert len(domain_decomp.interface_cells[1]) > 0
        elif mpi.rank == 1:
            assert 0 in domain_decomp.interface_cells
            assert len(domain_decomp.interface_cells[0]) > 0

class TestParallelIO:
    """Test parallel I/O functionality."""
    
    def test_write_read_mesh(self, parallel_io, domain_decomp, mpi):
        """Test mesh I/O operations."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
            filename = tmp.name
            
            # Create mock mesh
            class MockMesh:
                def __init__(self):
                    self.n_cells = 100
                    self.n_nodes = 200
                    self.n_faces = 300
                
                def get_cell_data(self):
                    return np.random.rand(self.n_cells)
                
                def get_node_data(self):
                    return np.random.rand(self.n_nodes)
                
                def get_face_data(self):
                    return np.random.rand(self.n_faces)
            
            mesh = MockMesh()
            
            # Write mesh
            parallel_io.write_mesh(filename, mesh, domain_decomp)
            
            # Read mesh
            mesh_data, cell_data, node_data, face_data = \
                parallel_io.read_mesh(filename, domain_decomp)
            
            # Verify data
            assert mesh_data[0] == mesh.n_cells
            assert mesh_data[1] == mesh.n_nodes
            assert mesh_data[2] == mesh.n_faces
            assert len(cell_data) == mesh.n_cells
            assert len(node_data) == mesh.n_nodes
            assert len(face_data) == mesh.n_faces
    
    def test_write_read_solution(self, parallel_io, domain_decomp, mpi):
        """Test solution I/O operations."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
            filename = tmp.name
            
            # Create solution data
            n_cells = 100
            n_vars = 5
            solution = np.random.rand(n_cells, n_vars)
            
            # Write solution
            parallel_io.write_solution(filename, solution, domain_decomp)
            
            # Read solution
            result = parallel_io.read_solution(filename, domain_decomp)
            
            # Verify data
            assert result.shape == solution.shape
            local_cells = domain_decomp.get_local_cells()
            assert np.allclose(result[local_cells], solution[local_cells])
    
    def test_checkpoint(self, parallel_io, domain_decomp, mpi):
        """Test checkpoint operations."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'checkpoint.h5')
            
            # Create mock mesh and solution
            class MockMesh:
                def __init__(self):
                    self.n_cells = 100
                    self.n_nodes = 200
                    self.n_faces = 300
                
                def get_cell_data(self):
                    return np.random.rand(self.n_cells)
                
                def get_node_data(self):
                    return np.random.rand(self.n_nodes)
                
                def get_face_data(self):
                    return np.random.rand(self.n_faces)
            
            mesh = MockMesh()
            solution = np.random.rand(mesh.n_cells, 5)
            iteration = 100
            time = 1.0
            
            # Write checkpoint
            parallel_io.write_checkpoint(filename, mesh, solution,
                                      domain_decomp, iteration, time)
            
            # Read checkpoint
            (mesh_data, cell_data, node_data, face_data), \
            result_solution, result_iteration, result_time = \
                parallel_io.read_checkpoint(filename, domain_decomp)
            
            # Verify data
            assert result_iteration == iteration
            assert result_time == time
            assert result_solution.shape == solution.shape
            local_cells = domain_decomp.get_local_cells()
            assert np.allclose(result_solution[local_cells],
                             solution[local_cells]) 
"""Tests for GPU-accelerated mesh generation module."""

import numpy as np
import pytest
import platform
from unittest.mock import Mock, patch
from pathlib import Path

from phasta.mesh.gpu import (
    GPUDevice, CUDADevice, MPSDevice, ROCmDevice,
    GPUManager, GPUMeshGenerator, MultiGPUMeshGenerator
)
from phasta.mesh.base import Mesh


class MockMesh:
    """Mock mesh for testing."""
    
    def __init__(self, num_nodes: int = 100, num_elements: int = 200):
        """Initialize mock mesh.
        
        Args:
            num_nodes: Number of nodes
            num_elements: Number of elements
        """
        self.nodes = np.random.rand(num_nodes, 3)
        self.elements = np.random.randint(0, num_nodes, (num_elements, 4))


class MockCADMeshGenerator:
    """Mock CAD mesh generator for testing."""
    
    def generate_mesh(self, cad_file):
        """Generate mock mesh.
        
        Args:
            cad_file: Path to CAD file
            
        Returns:
            Mock mesh
        """
        return MockMesh()


def test_gpu_device_base():
    """Test base GPU device class."""
    device = GPUDevice(0)
    
    with pytest.raises(NotImplementedError):
        device._initialize_device()
    
    with pytest.raises(NotImplementedError):
        device.allocate_memory(100)
    
    with pytest.raises(NotImplementedError):
        device.free_memory(0)
    
    with pytest.raises(NotImplementedError):
        device.copy_to_device(np.array([1, 2, 3]), 0)
    
    with pytest.raises(NotImplementedError):
        device.copy_from_device(0, (3,), np.float64)
    
    with pytest.raises(NotImplementedError):
        device.synchronize()


@patch('cupy.cuda.runtime')
def test_cuda_device(mock_runtime):
    """Test CUDA device implementation."""
    mock_runtime.getDeviceCount.return_value = 1
    
    with patch('cupy.cuda.Device') as mock_device:
        device = CUDADevice(0)
        
        # Test memory operations
        handle = device.allocate_memory(100)
        assert handle is not None
        
        data = np.array([1, 2, 3])
        device.copy_to_device(data, handle)
        
        result = device.copy_from_device(handle, (3,), np.float64)
        assert result.shape == (3,)
        
        device.free_memory(handle)
        device.synchronize()


def test_mps_device():
    """Test MPS device implementation."""
    if platform.system() != 'Darwin':
        pytest.skip("MPS is only available on macOS")
    
    with patch('torch.backends.mps.is_available', return_value=True):
        device = MPSDevice(0)
        
        # Test memory operations
        handle = device.allocate_memory(100)
        assert handle is not None
        
        data = np.array([1, 2, 3])
        device.copy_to_device(data, handle)
        
        result = device.copy_from_device(handle, (3,), np.float64)
        assert result.shape == (3,)
        
        device.free_memory(handle)
        device.synchronize()


@patch('torch.cuda.is_available', return_value=True)
def test_rocm_device(mock_cuda_available):
    """Test ROCm device implementation."""
    device = ROCmDevice(0)
    
    # Test memory operations
    handle = device.allocate_memory(100)
    assert handle is not None
    
    data = np.array([1, 2, 3])
    device.copy_to_device(data, handle)
    
    result = device.copy_from_device(handle, (3,), np.float64)
    assert result.shape == (3,)
    
    device.free_memory(handle)
    device.synchronize()


def test_gpu_manager():
    """Test GPU manager."""
    manager = GPUManager()
    
    # Test device initialization
    assert isinstance(manager.devices, dict)
    
    # Test device retrieval
    try:
        device = manager.get_device(0)
        assert isinstance(device, GPUDevice)
    except RuntimeError:
        pytest.skip("No GPU devices available")


def test_gpu_mesh_generator():
    """Test GPU mesh generator."""
    mesh_generator = MockCADMeshGenerator()
    gpu_generator = GPUMeshGenerator(mesh_generator)
    
    # Test mesh generation
    mesh = gpu_generator.generate_mesh("test.iges")
    assert isinstance(mesh, Mesh)
    assert mesh.nodes is not None
    assert mesh.elements is not None


def test_multi_gpu_mesh_generator():
    """Test multi-GPU mesh generator."""
    mesh_generator = MockCADMeshGenerator()
    gpu_generator = MultiGPUMeshGenerator(mesh_generator)
    
    # Test mesh generation
    mesh = gpu_generator.generate_mesh("test.iges")
    assert isinstance(mesh, Mesh)
    assert mesh.nodes is not None
    assert mesh.elements is not None


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    mesh_generator = MockCADMeshGenerator()
    
    # Test invalid device ID
    with pytest.raises(RuntimeError):
        GPUMeshGenerator(mesh_generator, device_id=999)
    
    # Test nonexistent file
    gpu_generator = GPUMeshGenerator(mesh_generator)
    with pytest.raises(FileNotFoundError):
        gpu_generator.generate_mesh("nonexistent.iges")


def test_mesh_optimization():
    """Test mesh optimization."""
    mesh_generator = MockCADMeshGenerator()
    gpu_generator = GPUMeshGenerator(mesh_generator)
    
    # Generate initial mesh
    initial_mesh = MockMesh()
    
    # Optimize mesh
    optimized_mesh = gpu_generator._optimize_mesh(initial_mesh)
    
    assert isinstance(optimized_mesh, Mesh)
    assert optimized_mesh.nodes.shape == initial_mesh.nodes.shape
    assert optimized_mesh.elements.shape == initial_mesh.elements.shape


def test_multi_gpu_optimization():
    """Test multi-GPU mesh optimization."""
    mesh_generator = MockCADMeshGenerator()
    gpu_generator = MultiGPUMeshGenerator(mesh_generator)
    
    # Generate initial mesh
    initial_mesh = MockMesh()
    
    # Split mesh
    mesh_parts = gpu_generator._split_mesh(initial_mesh)
    assert len(mesh_parts) > 0
    
    # Optimize mesh parts
    optimized_parts = []
    for i, (part, device) in enumerate(zip(mesh_parts, gpu_generator.devices)):
        optimized_part = gpu_generator._optimize_mesh_part(part, device)
        optimized_parts.append(optimized_part)
    
    # Merge mesh parts
    final_mesh = gpu_generator._merge_mesh_parts(optimized_parts)
    assert isinstance(final_mesh, Mesh)


def test_memory_management():
    """Test GPU memory management."""
    mesh_generator = MockCADMeshGenerator()
    gpu_generator = GPUMeshGenerator(mesh_generator)
    
    # Generate large mesh
    large_mesh = MockMesh(num_nodes=10000, num_elements=20000)
    
    # Optimize mesh
    optimized_mesh = gpu_generator._optimize_mesh(large_mesh)
    assert isinstance(optimized_mesh, Mesh)


def test_device_synchronization():
    """Test device synchronization."""
    mesh_generator = MockCADMeshGenerator()
    gpu_generator = GPUMeshGenerator(mesh_generator)
    
    # Generate mesh
    mesh = gpu_generator.generate_mesh("test.iges")
    
    # Ensure device is synchronized
    gpu_generator.device.synchronize() 
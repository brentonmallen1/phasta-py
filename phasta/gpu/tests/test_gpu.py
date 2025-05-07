"""
Test cases for GPU implementations.

This module provides test cases for CUDA, Metal, and OpenCL implementations
of the device abstraction layer.
"""

import pytest
import numpy as np
from typing import List, Tuple
from ..device import DeviceManager, Device, DeviceMemory, Kernel, Stream, Event
from ..cuda import CUDADevice, CUDAMemory, CUDAKernel, CUDAStream, CUDAEvent
from ..metal import MetalDevice, MetalMemory, MetalKernel, MetalStream, MetalEvent, MPS
from ..opencl import OpenCLDevice, OpenCLMemory, OpenCLKernel, OpenCLStream, OpenCLEvent

# Test data
TEST_ARRAY_SIZE = 1024
TEST_GRID = (32, 32, 1)
TEST_BLOCK = (16, 16, 1)

# Test kernels
CUDA_KERNEL = """
__global__ void add(float* a, float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < %d) {
        c[i] = a[i] + b[i];
    }
}
""" % TEST_ARRAY_SIZE

METAL_KERNEL = """
kernel void add(device float* a [[buffer(0)]],
                device float* b [[buffer(1)]],
                device float* c [[buffer(2)]]) {
    uint i = get_global_id(0);
    if (i < %d) {
        c[i] = a[i] + b[i];
    }
}
""" % TEST_ARRAY_SIZE

OPENCL_KERNEL = """
__kernel void add(__global float* a,
                  __global float* b,
                  __global float* c) {
    int i = get_global_id(0);
    if (i < %d) {
        c[i] = a[i] + b[i];
    }
}
""" % TEST_ARRAY_SIZE

@pytest.fixture
def device_manager():
    """Create device manager fixture."""
    manager = DeviceManager()
    manager.initialize()
    return manager

@pytest.fixture
def cuda_device(device_manager):
    """Create CUDA device fixture."""
    devices = [d for d in device_manager.devices.values() if isinstance(d, CUDADevice)]
    if not devices:
        pytest.skip("No CUDA device available")
    return devices[0]

@pytest.fixture
def metal_device(device_manager):
    """Create Metal device fixture."""
    devices = [d for d in device_manager.devices.values() if isinstance(d, MetalDevice)]
    if not devices:
        pytest.skip("No Metal device available")
    return devices[0]

@pytest.fixture
def opencl_device(device_manager):
    """Create OpenCL device fixture."""
    devices = [d for d in device_manager.devices.values() if isinstance(d, OpenCLDevice)]
    if not devices:
        pytest.skip("No OpenCL device available")
    return devices[0]

def test_device_info(device_manager):
    """Test device information."""
    devices = device_manager.get_available_devices()
    assert len(devices) > 0
    
    for device in devices:
        assert device.name
        assert device.vendor
        assert device.device_type in ["CUDA", "Metal", "OpenCL"]
        assert device.total_memory > 0
        assert device.max_work_group_size > 0

def test_memory_management(device_manager):
    """Test memory management."""
    for device in device_manager.devices.values():
        # Test memory allocation
        memory = device.create_memory(TEST_ARRAY_SIZE * 4)
        assert memory.size == TEST_ARRAY_SIZE * 4
        
        # Test memory operations
        host_data = np.random.rand(TEST_ARRAY_SIZE).astype(np.float32)
        memory.copy_to_device(host_data)
        device_data = memory.copy_to_host()
        np.testing.assert_array_equal(host_data, device_data)
        
        # Test memory freeing
        memory.free()
        assert memory._ptr is None

def test_kernel_execution(device_manager):
    """Test kernel execution."""
    for device in device_manager.devices.values():
        # Create test data
        a = np.random.rand(TEST_ARRAY_SIZE).astype(np.float32)
        b = np.random.rand(TEST_ARRAY_SIZE).astype(np.float32)
        c = np.zeros(TEST_ARRAY_SIZE, dtype=np.float32)
        
        # Allocate device memory
        a_mem = device.create_memory(a.nbytes)
        b_mem = device.create_memory(b.nbytes)
        c_mem = device.create_memory(c.nbytes)
        
        # Copy data to device
        a_mem.copy_to_device(a)
        b_mem.copy_to_device(b)
        
        # Compile and launch kernel
        if isinstance(device, CUDADevice):
            kernel = device.compile_kernel(CUDA_KERNEL, "add")
        elif isinstance(device, MetalDevice):
            kernel = device.compile_kernel(METAL_KERNEL, "add")
        else:  # OpenCL
            kernel = device.compile_kernel(OPENCL_KERNEL, "add")
        
        kernel.launch(TEST_GRID, TEST_BLOCK, [a_mem, b_mem, c_mem])
        
        # Get results
        result = c_mem.copy_to_host()
        expected = a + b
        np.testing.assert_array_almost_equal(result, expected)
        
        # Cleanup
        a_mem.free()
        b_mem.free()
        c_mem.free()

def test_stream_operations(device_manager):
    """Test stream operations."""
    for device in device_manager.devices.values():
        # Create stream
        stream = device.create_stream()
        assert stream.device == device
        
        # Test event recording
        event = stream.record_event()
        assert event.device == device
        
        # Test synchronization
        stream.synchronize()
        event.synchronize()

def test_metal_memory_pool(metal_device):
    """Test Metal memory pool."""
    # Create memory with pool
    memory1 = metal_device.create_memory(1024, use_pool=True)
    memory2 = metal_device.create_memory(1024, use_pool=True)
    
    # Test allocation
    assert memory1._ptr is not None
    assert memory2._ptr is not None
    
    # Test freeing
    memory1.free()
    memory2.free()
    
    # Create memory without pool
    memory3 = metal_device.create_memory(1024, use_pool=False)
    assert memory3._ptr is not None
    memory3.free()

def test_metal_mps(metal_device):
    """Test Metal Performance Shaders."""
    try:
        # Create MPS kernel
        kernel = metal_device.create_mps_kernel(MPS.MPSMatrixMultiplication)
        assert kernel is not None
        
        # Create MPS image
        image = metal_device.create_mps_image(32, 32, MPS.MPSImageFeatureChannelFormat.float32)
        assert image is not None
    except Exception as e:
        pytest.skip(f"MPS not available: {e}")

def test_opencl_platform(opencl_device):
    """Test OpenCL platform information."""
    assert opencl_device._platform is not None
    assert opencl_device._platform.vendor
    assert opencl_device._platform.version

def test_error_handling(device_manager):
    """Test error handling."""
    for device in device_manager.devices.values():
        # Test invalid memory size
        with pytest.raises(Exception):
            device.create_memory(-1)
        
        # Test invalid kernel
        with pytest.raises(Exception):
            device.compile_kernel("invalid kernel", "invalid")
        
        # Test invalid stream
        with pytest.raises(Exception):
            device.create_stream().synchronize()

def test_performance(device_manager):
    """Test performance monitoring."""
    for device in device_manager.devices.values():
        # Create test data
        a = np.random.rand(TEST_ARRAY_SIZE).astype(np.float32)
        b = np.random.rand(TEST_ARRAY_SIZE).astype(np.float32)
        c = np.zeros(TEST_ARRAY_SIZE, dtype=np.float32)
        
        # Allocate device memory
        a_mem = device.create_memory(a.nbytes)
        b_mem = device.create_memory(b.nbytes)
        c_mem = device.create_memory(c.nbytes)
        
        # Create stream and events
        stream = device.create_stream()
        start_event = stream.record_event()
        
        # Copy data to device
        a_mem.copy_to_device(a)
        b_mem.copy_to_device(b)
        
        # Compile and launch kernel
        if isinstance(device, CUDADevice):
            kernel = device.compile_kernel(CUDA_KERNEL, "add")
        elif isinstance(device, MetalDevice):
            kernel = device.compile_kernel(METAL_KERNEL, "add")
        else:  # OpenCL
            kernel = device.compile_kernel(OPENCL_KERNEL, "add")
        
        kernel.launch(TEST_GRID, TEST_BLOCK, [a_mem, b_mem, c_mem])
        
        # Record end event and get elapsed time
        end_event = stream.record_event()
        elapsed_time = end_event.elapsed_time(start_event)
        
        assert elapsed_time >= 0
        
        # Cleanup
        a_mem.free()
        b_mem.free()
        c_mem.free() 
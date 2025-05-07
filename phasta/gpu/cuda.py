"""
CUDA implementation of the device abstraction layer.

This module provides CUDA-specific implementations of the device abstraction
layer for NVIDIA GPUs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import ctypes
from .device import (Device, DeviceMemory, Kernel, Stream, Event,
                    DeviceInfo, DeviceError, DeviceMemoryError,
                    DeviceExecutionError)

try:
    import cupy as cp
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class CUDAMemory(DeviceMemory):
    """CUDA device memory."""
    
    def allocate(self):
        """Allocate CUDA device memory."""
        try:
            self._ptr = cuda.mem_alloc(self.size)
        except cuda.RuntimeError as e:
            raise DeviceMemoryError(f"Failed to allocate CUDA memory: {e}")
    
    def free(self):
        """Free CUDA device memory."""
        if self._ptr is not None:
            try:
                self._ptr.free()
                self._ptr = None
            except cuda.RuntimeError as e:
                raise DeviceMemoryError(f"Failed to free CUDA memory: {e}")
    
    def copy_to_device(self, host_data: np.ndarray):
        """Copy data from host to CUDA device.
        
        Args:
            host_data: Host data array
        """
        if self._ptr is None:
            raise DeviceMemoryError("Device memory not allocated")
        try:
            cuda.memcpy_htod(self._ptr, host_data)
        except cuda.RuntimeError as e:
            raise DeviceMemoryError(f"Failed to copy to CUDA device: {e}")
    
    def copy_to_host(self) -> np.ndarray:
        """Copy data from CUDA device to host.
        
        Returns:
            Host data array
        """
        if self._ptr is None:
            raise DeviceMemoryError("Device memory not allocated")
        try:
            host_data = np.empty(self.size // 8, dtype=np.float64)
            cuda.memcpy_dtoh(host_data, self._ptr)
            return host_data
        except cuda.RuntimeError as e:
            raise DeviceMemoryError(f"Failed to copy from CUDA device: {e}")

class CUDAKernel(Kernel):
    """CUDA kernel."""
    
    def __init__(self, device: 'CUDADevice', name: str, kernel):
        """Initialize CUDA kernel.
        
        Args:
            device: CUDA device instance
            name: Kernel name
            kernel: PyCUDA kernel
        """
        super().__init__(device, name)
        self._kernel = kernel
    
    def launch(self, grid: Tuple[int, ...], block: Tuple[int, ...],
               args: List[Union[np.ndarray, DeviceMemory, int, float]],
               stream: Optional['CUDAStream'] = None):
        """Launch CUDA kernel.
        
        Args:
            grid: Grid dimensions
            block: Block dimensions
            args: Kernel arguments
            stream: CUDA stream
        """
        if self._kernel is None:
            raise DeviceExecutionError("Kernel not compiled")
        try:
            self._kernel(*args, block=block, grid=grid,
                        stream=stream._stream if stream else None)
        except cuda.RuntimeError as e:
            raise DeviceExecutionError(f"Failed to launch CUDA kernel: {e}")

class CUDAStream(Stream):
    """CUDA stream."""
    
    def __init__(self, device: 'CUDADevice'):
        """Initialize CUDA stream.
        
        Args:
            device: CUDA device instance
        """
        super().__init__(device)
        try:
            self._stream = cuda.Stream()
        except cuda.RuntimeError as e:
            raise DeviceError(f"Failed to create CUDA stream: {e}")
    
    def synchronize(self):
        """Synchronize CUDA stream."""
        try:
            self._stream.synchronize()
        except cuda.RuntimeError as e:
            raise DeviceError(f"Failed to synchronize CUDA stream: {e}")
    
    def record_event(self) -> 'CUDAEvent':
        """Record CUDA event.
        
        Returns:
            CUDA event instance
        """
        try:
            event = cuda.Event()
            event.record(self._stream)
            return CUDAEvent(self.device, event)
        except cuda.RuntimeError as e:
            raise DeviceError(f"Failed to record CUDA event: {e}")

class CUDAEvent(Event):
    """CUDA event."""
    
    def __init__(self, device: 'CUDADevice', event):
        """Initialize CUDA event.
        
        Args:
            device: CUDA device instance
            event: PyCUDA event
        """
        super().__init__(device)
        self._event = event
    
    def synchronize(self):
        """Synchronize CUDA event."""
        try:
            self._event.synchronize()
        except cuda.RuntimeError as e:
            raise DeviceError(f"Failed to synchronize CUDA event: {e}")
    
    def elapsed_time(self, start_event: 'CUDAEvent') -> float:
        """Get elapsed time between CUDA events.
        
        Args:
            start_event: Start CUDA event
            
        Returns:
            Elapsed time in milliseconds
        """
        try:
            return self._event.time_since(start_event._event)
        except cuda.RuntimeError as e:
            raise DeviceError(f"Failed to get elapsed time: {e}")

class CUDADevice(Device):
    """CUDA device."""
    
    def __init__(self, device_id: int):
        """Initialize CUDA device.
        
        Args:
            device_id: CUDA device ID
        """
        super().__init__()
        self.device_id = device_id
        self._device = None
    
    @classmethod
    def get_available_devices(cls) -> List['CUDADevice']:
        """Get available CUDA devices.
        
        Returns:
            List of CUDA device instances
        """
        if not CUDA_AVAILABLE:
            return []
        
        devices = []
        try:
            for i in range(cuda.Device.count()):
                devices.append(cls(i))
        except cuda.RuntimeError as e:
            logging.getLogger(__name__).error(f"Failed to get CUDA devices: {e}")
        
        return devices
    
    def _get_device_info(self) -> DeviceInfo:
        """Get CUDA device information.
        
        Returns:
            Device information
        """
        try:
            device = cuda.Device(self.device_id)
            attributes = device.get_attributes()
            
            return DeviceInfo(
                name=device.name(),
                vendor="NVIDIA",
                device_type="CUDA",
                compute_capability=device.compute_capability(),
                total_memory=device.total_memory(),
                max_work_group_size=attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
            )
        except cuda.RuntimeError as e:
            raise DeviceError(f"Failed to get CUDA device info: {e}")
    
    def initialize(self):
        """Initialize CUDA device."""
        try:
            self._device = cuda.Device(self.device_id)
            self._device.make_context()
        except cuda.RuntimeError as e:
            raise DeviceError(f"Failed to initialize CUDA device: {e}")
    
    def finalize(self):
        """Finalize CUDA device."""
        if self._device is not None:
            try:
                self._device.pop()
            except cuda.RuntimeError as e:
                raise DeviceError(f"Failed to finalize CUDA device: {e}")
    
    def create_memory(self, size: int) -> CUDAMemory:
        """Create CUDA device memory.
        
        Args:
            size: Size in bytes
            
        Returns:
            CUDA device memory instance
        """
        return CUDAMemory(size, self)
    
    def create_stream(self) -> CUDAStream:
        """Create CUDA stream.
        
        Returns:
            CUDA stream instance
        """
        return CUDAStream(self)
    
    def synchronize(self):
        """Synchronize CUDA device."""
        try:
            cuda.Context.synchronize()
        except cuda.RuntimeError as e:
            raise DeviceError(f"Failed to synchronize CUDA device: {e}")
    
    def compile_kernel(self, source: str, kernel_name: str,
                      options: Optional[List[str]] = None) -> CUDAKernel:
        """Compile CUDA kernel.
        
        Args:
            source: CUDA kernel source code
            kernel_name: Kernel function name
            options: Compilation options
            
        Returns:
            Compiled CUDA kernel
        """
        try:
            module = SourceModule(source, options=options)
            kernel = module.get_function(kernel_name)
            return CUDAKernel(self, kernel_name, kernel)
        except cuda.RuntimeError as e:
            raise DeviceError(f"Failed to compile CUDA kernel: {e}") 
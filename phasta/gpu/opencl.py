"""
OpenCL implementation of the device abstraction layer.

This module provides OpenCL-specific implementations of the device abstraction
layer for cross-platform GPU acceleration.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import ctypes
from .device import (Device, DeviceMemory, Kernel, Stream, Event,
                    DeviceInfo, DeviceError, DeviceMemoryError,
                    DeviceExecutionError)

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

class OpenCLMemory(DeviceMemory):
    """OpenCL device memory."""
    
    def allocate(self):
        """Allocate OpenCL device memory."""
        try:
            self._ptr = cl.Buffer(
                self.device._context,
                cl.mem_flags.READ_WRITE,
                self.size
            )
        except cl.Error as e:
            raise DeviceMemoryError(f"Failed to allocate OpenCL memory: {e}")
    
    def free(self):
        """Free OpenCL device memory."""
        if self._ptr is not None:
            try:
                self._ptr.release()
                self._ptr = None
            except cl.Error as e:
                raise DeviceMemoryError(f"Failed to free OpenCL memory: {e}")
    
    def copy_to_device(self, host_data: np.ndarray):
        """Copy data from host to OpenCL device.
        
        Args:
            host_data: Host data array
        """
        if self._ptr is None:
            raise DeviceMemoryError("Device memory not allocated")
        try:
            cl.enqueue_copy(
                self.device._queue,
                self._ptr,
                host_data
            )
        except cl.Error as e:
            raise DeviceMemoryError(f"Failed to copy to OpenCL device: {e}")
    
    def copy_to_host(self) -> np.ndarray:
        """Copy data from OpenCL device to host.
        
        Returns:
            Host data array
        """
        if self._ptr is None:
            raise DeviceMemoryError("Device memory not allocated")
        try:
            host_data = np.empty(self.size // 8, dtype=np.float64)
            cl.enqueue_copy(
                self.device._queue,
                host_data,
                self._ptr
            )
            return host_data
        except cl.Error as e:
            raise DeviceMemoryError(f"Failed to copy from OpenCL device: {e}")

class OpenCLKernel(Kernel):
    """OpenCL kernel."""
    
    def __init__(self, device: 'OpenCLDevice', name: str, program):
        """Initialize OpenCL kernel.
        
        Args:
            device: OpenCL device instance
            name: Kernel name
            program: OpenCL program
        """
        super().__init__(device, name)
        self._program = program
        self._kernel = None
    
    def _ensure_kernel(self):
        """Ensure kernel is created."""
        if self._kernel is None:
            try:
                self._kernel = self._program.get_kernel(self.name)
            except cl.Error as e:
                raise DeviceExecutionError(f"Failed to create OpenCL kernel: {e}")
    
    def launch(self, grid: Tuple[int, ...], block: Tuple[int, ...],
               args: List[Union[np.ndarray, DeviceMemory, int, float]],
               stream: Optional['OpenCLStream'] = None):
        """Launch OpenCL kernel.
        
        Args:
            grid: Grid dimensions
            block: Block dimensions
            args: Kernel arguments
            stream: OpenCL stream
        """
        self._ensure_kernel()
        
        try:
            queue = stream._queue if stream else self.device._queue
            
            # Set arguments
            for i, arg in enumerate(args):
                if isinstance(arg, (np.ndarray, OpenCLMemory)):
                    self._kernel.set_arg(i, arg._ptr if isinstance(arg, OpenCLMemory) else arg)
                else:
                    self._kernel.set_arg(i, np.array(arg, dtype=np.float32))
            
            # Launch kernel
            self._kernel.enqueue_nd_range(
                queue,
                grid,
                block
            )
            
            if stream is None:
                queue.finish()
        
        except cl.Error as e:
            raise DeviceExecutionError(f"Failed to launch OpenCL kernel: {e}")

class OpenCLStream(Stream):
    """OpenCL stream."""
    
    def __init__(self, device: 'OpenCLDevice'):
        """Initialize OpenCL stream.
        
        Args:
            device: OpenCL device instance
        """
        super().__init__(device)
        try:
            self._queue = cl.CommandQueue(
                self.device._context,
                self.device._device,
                properties=cl.command_queue_properties.PROFILING_ENABLE
            )
        except cl.Error as e:
            raise DeviceError(f"Failed to create OpenCL command queue: {e}")
    
    def synchronize(self):
        """Synchronize OpenCL stream."""
        try:
            self._queue.finish()
        except cl.Error as e:
            raise DeviceError(f"Failed to synchronize OpenCL command queue: {e}")
    
    def record_event(self) -> 'OpenCLEvent':
        """Record OpenCL event.
        
        Returns:
            OpenCL event instance
        """
        try:
            event = cl.UserEvent(self.device._context)
            self._queue.enqueue_marker(event=event)
            return OpenCLEvent(self.device, event)
        except cl.Error as e:
            raise DeviceError(f"Failed to record OpenCL event: {e}")

class OpenCLEvent(Event):
    """OpenCL event."""
    
    def __init__(self, device: 'OpenCLDevice', event):
        """Initialize OpenCL event.
        
        Args:
            device: OpenCL device instance
            event: OpenCL event
        """
        super().__init__(device)
        self._event = event
    
    def synchronize(self):
        """Synchronize OpenCL event."""
        try:
            self._event.wait()
        except cl.Error as e:
            raise DeviceError(f"Failed to synchronize OpenCL event: {e}")
    
    def elapsed_time(self, start_event: 'OpenCLEvent') -> float:
        """Get elapsed time between OpenCL events.
        
        Args:
            start_event: Start OpenCL event
            
        Returns:
            Elapsed time in milliseconds
        """
        try:
            return (self._event.profile.end - start_event._event.profile.end) * 1e-6
        except cl.Error as e:
            raise DeviceError(f"Failed to get elapsed time: {e}")

class OpenCLDevice(Device):
    """OpenCL device."""
    
    def __init__(self, platform, device):
        """Initialize OpenCL device.
        
        Args:
            platform: OpenCL platform
            device: OpenCL device
        """
        super().__init__()
        self._platform = platform
        self._device = device
        self._context = None
        self._queue = None
    
    @classmethod
    def get_available_devices(cls) -> List['OpenCLDevice']:
        """Get available OpenCL devices.
        
        Returns:
            List of OpenCL device instances
        """
        if not OPENCL_AVAILABLE:
            return []
        
        devices = []
        try:
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    devices.append(cls(platform, device))
        except cl.Error as e:
            logging.getLogger(__name__).error(f"Failed to get OpenCL devices: {e}")
        
        return devices
    
    def _get_device_info(self) -> DeviceInfo:
        """Get OpenCL device information.
        
        Returns:
            Device information
        """
        try:
            return DeviceInfo(
                name=self._device.name,
                vendor=self._platform.vendor,
                device_type="OpenCL",
                compute_capability=None,  # OpenCL doesn't have compute capability
                total_memory=self._device.global_mem_size,
                max_work_group_size=self._device.max_work_group_size
            )
        except cl.Error as e:
            raise DeviceError(f"Failed to get OpenCL device info: {e}")
    
    def initialize(self):
        """Initialize OpenCL device."""
        try:
            self._context = cl.Context([self._device])
            self._queue = cl.CommandQueue(
                self._context,
                self._device,
                properties=cl.command_queue_properties.PROFILING_ENABLE
            )
        except cl.Error as e:
            raise DeviceError(f"Failed to initialize OpenCL device: {e}")
    
    def finalize(self):
        """Finalize OpenCL device."""
        try:
            if self._queue is not None:
                self._queue.finish()
            if self._context is not None:
                self._context.release()
        except cl.Error as e:
            raise DeviceError(f"Failed to finalize OpenCL device: {e}")
    
    def create_memory(self, size: int) -> OpenCLMemory:
        """Create OpenCL device memory.
        
        Args:
            size: Size in bytes
            
        Returns:
            OpenCL device memory instance
        """
        return OpenCLMemory(size, self)
    
    def create_stream(self) -> OpenCLStream:
        """Create OpenCL stream.
        
        Returns:
            OpenCL stream instance
        """
        return OpenCLStream(self)
    
    def synchronize(self):
        """Synchronize OpenCL device."""
        try:
            self._queue.finish()
        except cl.Error as e:
            raise DeviceError(f"Failed to synchronize OpenCL device: {e}")
    
    def compile_kernel(self, source: str, kernel_name: str,
                      options: Optional[List[str]] = None) -> OpenCLKernel:
        """Compile OpenCL kernel.
        
        Args:
            source: OpenCL kernel source code
            kernel_name: Kernel function name
            options: Compilation options
            
        Returns:
            Compiled OpenCL kernel
        """
        try:
            program = cl.Program(self._context, source).build(
                options=options or []
            )
            return OpenCLKernel(self, kernel_name, program)
        except cl.Error as e:
            raise DeviceError(f"Failed to compile OpenCL kernel: {e}") 
"""
Metal implementation of the device abstraction layer.

This module provides Metal-specific implementations of the device abstraction
layer for Apple GPUs, including Metal Performance Shaders (MPS) support.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import ctypes
from collections import deque
from .device import (Device, DeviceMemory, Kernel, Stream, Event,
                    DeviceInfo, DeviceError, DeviceMemoryError,
                    DeviceExecutionError)

try:
    import Metal
    import MetalKit
    import MetalPerformanceShaders as MPS
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

class MetalMemoryPool:
    """Memory pool for Metal device memory."""
    
    def __init__(self, device: 'MetalDevice', initial_size: int = 1024 * 1024):
        """Initialize memory pool.
        
        Args:
            device: Metal device instance
            initial_size: Initial pool size in bytes
        """
        self.device = device
        self.pool = deque()
        self.allocated = set()
        self._grow_pool(initial_size)
    
    def _grow_pool(self, size: int):
        """Grow memory pool.
        
        Args:
            size: Size to add in bytes
        """
        try:
            buffer = self.device._device.newBuffer(
                size,
                Metal.MTLResourceStorageModeShared
            )
            self.pool.append((buffer, size))
        except Exception as e:
            raise DeviceMemoryError(f"Failed to grow memory pool: {e}")
    
    def allocate(self, size: int) -> Metal.MTLBuffer:
        """Allocate memory from pool.
        
        Args:
            size: Size in bytes
            
        Returns:
            Metal buffer
        """
        # Find best fit
        best_fit = None
        best_fit_size = float('inf')
        
        for i, (buffer, buffer_size) in enumerate(self.pool):
            if buffer_size >= size and buffer_size < best_fit_size:
                best_fit = (i, buffer, buffer_size)
                best_fit_size = buffer_size
        
        if best_fit is None:
            # No fit found, grow pool
            self._grow_pool(max(size, 1024 * 1024))
            return self.allocate(size)
        
        # Remove from pool and add to allocated
        i, buffer, _ = best_fit
        self.pool.remove(best_fit[1:])
        self.allocated.add(buffer)
        return buffer
    
    def free(self, buffer: Metal.MTLBuffer):
        """Free memory back to pool.
        
        Args:
            buffer: Metal buffer to free
        """
        if buffer in self.allocated:
            self.allocated.remove(buffer)
            self.pool.append((buffer, buffer.length()))

class MetalMemory(DeviceMemory):
    """Metal device memory."""
    
    def __init__(self, size: int, device: 'MetalDevice', use_pool: bool = True):
        """Initialize Metal device memory.
        
        Args:
            size: Size in bytes
            device: Metal device instance
            use_pool: Whether to use memory pool
        """
        super().__init__(size, device)
        self.use_pool = use_pool
    
    def allocate(self):
        """Allocate Metal device memory."""
        try:
            if self.use_pool and self.device._memory_pool is not None:
                self._ptr = self.device._memory_pool.allocate(self.size)
            else:
                self._ptr = self.device._device.newBuffer(
                    self.size,
                    Metal.MTLResourceStorageModeShared
                )
        except Exception as e:
            raise DeviceMemoryError(f"Failed to allocate Metal memory: {e}")
    
    def free(self):
        """Free Metal device memory."""
        if self._ptr is not None:
            try:
                if self.use_pool and self.device._memory_pool is not None:
                    self.device._memory_pool.free(self._ptr)
                else:
                    self._ptr.release()
                self._ptr = None
            except Exception as e:
                raise DeviceMemoryError(f"Failed to free Metal memory: {e}")
    
    def copy_to_device(self, host_data: np.ndarray):
        """Copy data from host to Metal device.
        
        Args:
            host_data: Host data array
        """
        if self._ptr is None:
            raise DeviceMemoryError("Device memory not allocated")
        try:
            ctypes.memmove(self._ptr.contents(), host_data.ctypes.data, self.size)
        except Exception as e:
            raise DeviceMemoryError(f"Failed to copy to Metal device: {e}")
    
    def copy_to_host(self) -> np.ndarray:
        """Copy data from Metal device to host.
        
        Returns:
            Host data array
        """
        if self._ptr is None:
            raise DeviceMemoryError("Device memory not allocated")
        try:
            host_data = np.empty(self.size // 8, dtype=np.float64)
            ctypes.memmove(host_data.ctypes.data, self._ptr.contents(), self.size)
            return host_data
        except Exception as e:
            raise DeviceMemoryError(f"Failed to copy from Metal device: {e}")

class MetalKernel(Kernel):
    """Metal kernel."""
    
    def __init__(self, device: 'MetalDevice', name: str, function):
        """Initialize Metal kernel.
        
        Args:
            device: Metal device instance
            name: Kernel name
            function: Metal compute function
        """
        super().__init__(device, name)
        self._function = function
        self._pipeline = None
        self._max_threads_per_threadgroup = None
    
    def _ensure_pipeline(self):
        """Ensure compute pipeline is created."""
        if self._pipeline is None:
            try:
                self._pipeline = self.device._device.newComputePipelineStateWithFunction(
                    self._function
                )
                self._max_threads_per_threadgroup = self._pipeline.maxTotalThreadsPerThreadgroup
            except Exception as e:
                raise DeviceExecutionError(f"Failed to create compute pipeline: {e}")
    
    def _optimize_threadgroup_size(self, grid: Tuple[int, ...],
                                 block: Tuple[int, ...]) -> Tuple[int, ...]:
        """Optimize threadgroup size.
        
        Args:
            grid: Grid dimensions
            block: Block dimensions
            
        Returns:
            Optimized block dimensions
        """
        if self._max_threads_per_threadgroup is None:
            return block
        
        # Ensure block size doesn't exceed maximum
        total_threads = np.prod(block)
        if total_threads > self._max_threads_per_threadgroup:
            # Scale down block size proportionally
            scale = (self._max_threads_per_threadgroup / total_threads) ** (1/len(block))
            return tuple(int(dim * scale) for dim in block)
        
        return block
    
    def launch(self, grid: Tuple[int, ...], block: Tuple[int, ...],
               args: List[Union[np.ndarray, DeviceMemory, int, float]],
               stream: Optional['MetalStream'] = None):
        """Launch Metal kernel.
        
        Args:
            grid: Grid dimensions
            block: Block dimensions
            args: Kernel arguments
            stream: Metal stream
        """
        if self._function is None:
            raise DeviceExecutionError("Kernel not compiled")
        
        self._ensure_pipeline()
        block = self._optimize_threadgroup_size(grid, block)
        
        try:
            command_buffer = stream._command_buffer if stream else \
                           self.device._command_queue.commandBuffer()
            compute_encoder = command_buffer.computeCommandEncoder()
            
            # Set compute pipeline
            compute_encoder.setComputePipelineState(self._pipeline)
            
            # Set arguments
            for i, arg in enumerate(args):
                if isinstance(arg, (np.ndarray, MetalMemory)):
                    compute_encoder.setBuffer(arg._ptr if isinstance(arg, MetalMemory) else arg,
                                            offset=0, index=i)
                else:
                    compute_encoder.setBytes(ctypes.byref(ctypes.c_float(arg)),
                                           length=4, index=i)
            
            # Set threadgroup size and grid size
            compute_encoder.dispatchThreadgroups(
                Metal.MTLSizeMake(*grid),
                Metal.MTLSizeMake(*block)
            )
            
            compute_encoder.endEncoding()
            command_buffer.commit()
            
            if stream is None:
                command_buffer.waitUntilCompleted()
        
        except Exception as e:
            raise DeviceExecutionError(f"Failed to launch Metal kernel: {e}")

class MetalStream(Stream):
    """Metal stream."""
    
    def __init__(self, device: 'MetalDevice'):
        """Initialize Metal stream.
        
        Args:
            device: Metal device instance
        """
        super().__init__(device)
        try:
            self._command_buffer = device._command_queue.commandBuffer()
        except Exception as e:
            raise DeviceError(f"Failed to create Metal command buffer: {e}")
    
    def synchronize(self):
        """Synchronize Metal stream."""
        try:
            self._command_buffer.waitUntilCompleted()
        except Exception as e:
            raise DeviceError(f"Failed to synchronize Metal command buffer: {e}")
    
    def record_event(self) -> 'MetalEvent':
        """Record Metal event.
        
        Returns:
            Metal event instance
        """
        try:
            event = self.device._device.newSharedEvent()
            self._command_buffer.encodeSignalEvent(event, value=1)
            return MetalEvent(self.device, event)
        except Exception as e:
            raise DeviceError(f"Failed to record Metal event: {e}")

class MetalEvent(Event):
    """Metal event."""
    
    def __init__(self, device: 'MetalDevice', event):
        """Initialize Metal event.
        
        Args:
            device: Metal device instance
            event: Metal shared event
        """
        super().__init__(device)
        self._event = event
        self._value = 1
    
    def synchronize(self):
        """Synchronize Metal event."""
        try:
            self._event.notifyListener(self._value, block=True)
        except Exception as e:
            raise DeviceError(f"Failed to synchronize Metal event: {e}")
    
    def elapsed_time(self, start_event: 'MetalEvent') -> float:
        """Get elapsed time between Metal events.
        
        Args:
            start_event: Start Metal event
            
        Returns:
            Elapsed time in milliseconds
        """
        try:
            # Metal doesn't provide direct timing between events
            # We'll use the command buffer timing instead
            return (self._event.signaledValue - start_event._event.signaledValue) * 1000
        except Exception as e:
            raise DeviceError(f"Failed to get elapsed time: {e}")

class MetalDevice(Device):
    """Metal device."""
    
    def __init__(self, device):
        """Initialize Metal device.
        
        Args:
            device: Metal device
        """
        super().__init__()
        self._device = device
        self._command_queue = None
        self._library = None
        self._memory_pool = None
        self._mps_device = None
    
    @classmethod
    def get_available_devices(cls) -> List['MetalDevice']:
        """Get available Metal devices.
        
        Returns:
            List of Metal device instances
        """
        if not METAL_AVAILABLE:
            return []
        
        devices = []
        try:
            for device in Metal.MTLCopyAllDevices():
                devices.append(cls(device))
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to get Metal devices: {e}")
        
        return devices
    
    def _get_device_info(self) -> DeviceInfo:
        """Get Metal device information.
        
        Returns:
            Device information
        """
        try:
            return DeviceInfo(
                name=self._device.name,
                vendor="Apple",
                device_type="Metal",
                compute_capability=None,  # Metal doesn't have compute capability
                total_memory=self._device.recommendedMaxWorkingSetSize,
                max_work_group_size=self._device.maxThreadsPerThreadgroup
            )
        except Exception as e:
            raise DeviceError(f"Failed to get Metal device info: {e}")
    
    def initialize(self):
        """Initialize Metal device."""
        try:
            self._command_queue = self._device.newCommandQueue()
            self._library = self._device.newDefaultLibrary()
            self._memory_pool = MetalMemoryPool(self)
            
            # Initialize MPS device if available
            if METAL_AVAILABLE:
                self._mps_device = MPS.MPSDevice.deviceWithMTLDevice(self._device)
        except Exception as e:
            raise DeviceError(f"Failed to initialize Metal device: {e}")
    
    def finalize(self):
        """Finalize Metal device."""
        try:
            if self._memory_pool is not None:
                self._memory_pool = None
            if self._library is not None:
                self._library.release()
            if self._command_queue is not None:
                self._command_queue.release()
        except Exception as e:
            raise DeviceError(f"Failed to finalize Metal device: {e}")
    
    def create_memory(self, size: int, use_pool: bool = True) -> MetalMemory:
        """Create Metal device memory.
        
        Args:
            size: Size in bytes
            use_pool: Whether to use memory pool
            
        Returns:
            Metal device memory instance
        """
        return MetalMemory(size, self, use_pool)
    
    def create_stream(self) -> MetalStream:
        """Create Metal stream.
        
        Returns:
            Metal stream instance
        """
        return MetalStream(self)
    
    def synchronize(self):
        """Synchronize Metal device."""
        try:
            command_buffer = self._command_queue.commandBuffer()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()
        except Exception as e:
            raise DeviceError(f"Failed to synchronize Metal device: {e}")
    
    def compile_kernel(self, source: str, kernel_name: str,
                      options: Optional[List[str]] = None) -> MetalKernel:
        """Compile Metal kernel.
        
        Args:
            source: Metal kernel source code
            kernel_name: Kernel function name
            options: Compilation options
            
        Returns:
            Compiled Metal kernel
        """
        try:
            library = self._device.newLibraryWithSource(
                source,
                options=options or []
            )
            function = library.newFunctionWithName(kernel_name)
            return MetalKernel(self, kernel_name, function)
        except Exception as e:
            raise DeviceError(f"Failed to compile Metal kernel: {e}")
    
    def create_mps_kernel(self, kernel_class: type) -> MPS.MPSKernel:
        """Create MPS kernel.
        
        Args:
            kernel_class: MPS kernel class
            
        Returns:
            MPS kernel instance
        """
        if not METAL_AVAILABLE or self._mps_device is None:
            raise DeviceError("MPS not available")
        
        try:
            return kernel_class(self._mps_device)
        except Exception as e:
            raise DeviceError(f"Failed to create MPS kernel: {e}")
    
    def create_mps_image(self, width: int, height: int,
                        pixel_format: MPS.MPSImageFeatureChannelFormat) -> MPS.MPSImage:
        """Create MPS image.
        
        Args:
            width: Image width
            height: Image height
            pixel_format: Pixel format
            
        Returns:
            MPS image instance
        """
        if not METAL_AVAILABLE or self._mps_device is None:
            raise DeviceError("MPS not available")
        
        try:
            return MPS.MPSImage(self._mps_device,
                              width=width,
                              height=height,
                              featureChannels=1,
                              numberOfImages=1,
                              textureDescriptor=MPS.MPSImageDescriptor(
                                  width=width,
                                  height=height,
                                  featureChannels=1,
                                  numberOfImages=1,
                                  textureType=Metal.MTLTextureType.type2D,
                                  pixelFormat=pixel_format
                              ))
        except Exception as e:
            raise DeviceError(f"Failed to create MPS image: {e}") 
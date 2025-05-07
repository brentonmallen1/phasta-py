"""
Device abstraction layer for GPU computing.

This module provides a common interface for different GPU types (NVIDIA CUDA,
Apple Metal/MPS, and OpenCL) to enable cross-platform GPU acceleration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

class DeviceError(Exception):
    """Base class for device-related errors."""
    pass

class DeviceMemoryError(DeviceError):
    """Error related to device memory operations."""
    pass

class DeviceExecutionError(DeviceError):
    """Error related to device execution."""
    pass

class DeviceInfo:
    """Information about a GPU device."""
    
    def __init__(self, name: str, vendor: str, device_type: str,
                 compute_capability: Optional[Tuple[int, int]] = None,
                 total_memory: int = 0, max_work_group_size: int = 0):
        """Initialize device information.
        
        Args:
            name: Device name
            vendor: Device vendor
            device_type: Type of device (CUDA, Metal, OpenCL)
            compute_capability: Compute capability for CUDA devices
            total_memory: Total device memory in bytes
            max_work_group_size: Maximum work group size
        """
        self.name = name
        self.vendor = vendor
        self.device_type = device_type
        self.compute_capability = compute_capability
        self.total_memory = total_memory
        self.max_work_group_size = max_work_group_size

class DeviceMemory:
    """Base class for device memory management."""
    
    def __init__(self, size: int, device: 'Device'):
        """Initialize device memory.
        
        Args:
            size: Size in bytes
            device: Device instance
        """
        self.size = size
        self.device = device
        self._ptr = None
    
    @property
    def ptr(self):
        """Get memory pointer."""
        return self._ptr
    
    def __del__(self):
        """Free device memory."""
        self.free()
    
    @abstractmethod
    def allocate(self):
        """Allocate device memory."""
        pass
    
    @abstractmethod
    def free(self):
        """Free device memory."""
        pass
    
    @abstractmethod
    def copy_to_device(self, host_data: np.ndarray):
        """Copy data from host to device.
        
        Args:
            host_data: Host data array
        """
        pass
    
    @abstractmethod
    def copy_to_host(self) -> np.ndarray:
        """Copy data from device to host.
        
        Returns:
            Host data array
        """
        pass

class Device(ABC):
    """Base class for GPU devices."""
    
    def __init__(self):
        """Initialize device."""
        self.logger = logging.getLogger(__name__)
        self._info = None
        self._context = None
        self._stream = None
    
    @property
    def info(self) -> DeviceInfo:
        """Get device information."""
        if self._info is None:
            self._info = self._get_device_info()
        return self._info
    
    @abstractmethod
    def _get_device_info(self) -> DeviceInfo:
        """Get device information.
        
        Returns:
            Device information
        """
        pass
    
    @abstractmethod
    def initialize(self):
        """Initialize device."""
        pass
    
    @abstractmethod
    def finalize(self):
        """Finalize device."""
        pass
    
    @abstractmethod
    def create_memory(self, size: int) -> DeviceMemory:
        """Create device memory.
        
        Args:
            size: Size in bytes
            
        Returns:
            Device memory instance
        """
        pass
    
    @abstractmethod
    def create_stream(self):
        """Create device stream."""
        pass
    
    @abstractmethod
    def synchronize(self):
        """Synchronize device."""
        pass
    
    @abstractmethod
    def compile_kernel(self, source: str, kernel_name: str,
                      options: Optional[List[str]] = None) -> 'Kernel':
        """Compile kernel.
        
        Args:
            source: Kernel source code
            kernel_name: Kernel function name
            options: Compilation options
            
        Returns:
            Compiled kernel
        """
        pass

class Kernel(ABC):
    """Base class for GPU kernels."""
    
    def __init__(self, device: Device, name: str):
        """Initialize kernel.
        
        Args:
            device: Device instance
            name: Kernel name
        """
        self.device = device
        self.name = name
        self._kernel = None
    
    @abstractmethod
    def launch(self, grid: Tuple[int, ...], block: Tuple[int, ...],
               args: List[Union[np.ndarray, DeviceMemory, int, float]],
               stream: Optional['Stream'] = None):
        """Launch kernel.
        
        Args:
            grid: Grid dimensions
            block: Block dimensions
            args: Kernel arguments
            stream: Device stream
        """
        pass

class Stream(ABC):
    """Base class for device streams."""
    
    def __init__(self, device: Device):
        """Initialize stream.
        
        Args:
            device: Device instance
        """
        self.device = device
        self._stream = None
    
    @abstractmethod
    def synchronize(self):
        """Synchronize stream."""
        pass
    
    @abstractmethod
    def record_event(self) -> 'Event':
        """Record event.
        
        Returns:
            Event instance
        """
        pass

class Event(ABC):
    """Base class for device events."""
    
    def __init__(self, device: Device):
        """Initialize event.
        
        Args:
            device: Device instance
        """
        self.device = device
        self._event = None
    
    @abstractmethod
    def synchronize(self):
        """Synchronize event."""
        pass
    
    @abstractmethod
    def elapsed_time(self, start_event: 'Event') -> float:
        """Get elapsed time between events.
        
        Args:
            start_event: Start event
            
        Returns:
            Elapsed time in milliseconds
        """
        pass

class DeviceManager:
    """Manager for GPU devices."""
    
    def __init__(self):
        """Initialize device manager."""
        self.logger = logging.getLogger(__name__)
        self.devices: Dict[str, Device] = {}
        self.current_device: Optional[Device] = None
    
    def initialize(self):
        """Initialize device manager."""
        # Initialize CUDA devices
        try:
            from .cuda import CUDADevice
            cuda_devices = CUDADevice.get_available_devices()
            for device in cuda_devices:
                self.devices[f"cuda:{device.info.name}"] = device
        except ImportError:
            self.logger.warning("CUDA support not available")
        
        # Initialize Metal devices
        try:
            from .metal import MetalDevice
            metal_devices = MetalDevice.get_available_devices()
            for device in metal_devices:
                self.devices[f"metal:{device.info.name}"] = device
        except ImportError:
            self.logger.warning("Metal support not available")
        
        # Initialize OpenCL devices
        try:
            from .opencl import OpenCLDevice
            opencl_devices = OpenCLDevice.get_available_devices()
            for device in opencl_devices:
                self.devices[f"opencl:{device.info.name}"] = device
        except ImportError:
            self.logger.warning("OpenCL support not available")
        
        # Set default device
        if self.devices:
            self.current_device = next(iter(self.devices.values()))
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device by ID.
        
        Args:
            device_id: Device ID (format: "type:name")
            
        Returns:
            Device instance or None if not found
        """
        return self.devices.get(device_id)
    
    def set_current_device(self, device_id: str):
        """Set current device.
        
        Args:
            device_id: Device ID
        """
        device = self.get_device(device_id)
        if device:
            self.current_device = device
        else:
            raise DeviceError(f"Device not found: {device_id}")
    
    def get_available_devices(self) -> List[DeviceInfo]:
        """Get information about available devices.
        
        Returns:
            List of device information
        """
        return [device.info for device in self.devices.values()]
    
    def finalize(self):
        """Finalize device manager."""
        for device in self.devices.values():
            device.finalize()
        self.devices.clear()
        self.current_device = None 
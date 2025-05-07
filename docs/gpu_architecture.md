# GPU Architecture Documentation

## Overview

The GPU acceleration system in PHASTA is organized into two main components:

1. **Device Abstraction Layer** (`gpu/` directory)
2. **Acceleration Layer** (`acceleration/` directory)

This document explains their relationship, responsibilities, and how they work together.

## Directory Structure

```
phasta-py/
├── phasta/
│   ├── gpu/                    # Device abstraction layer
│   │   ├── device.py          # Base classes and interfaces
│   │   ├── cuda.py            # NVIDIA CUDA implementation
│   │   ├── metal.py           # Apple Metal implementation
│   │   └── opencl.py          # OpenCL implementation (planned)
│   │
│   └── acceleration/          # GPU-accelerated algorithms
│       ├── linear_algebra/    # Matrix/vector operations
│       ├── physics/           # Physics computations
│       └── mesh/              # Mesh operations
```

## Device Abstraction Layer (`gpu/`)

The device abstraction layer provides a unified interface for different GPU types and handles low-level device management.

### Key Components

1. **Base Classes** (`device.py`)
   - `Device`: Base class for GPU devices
   - `DeviceMemory`: Memory management
   - `Kernel`: Compute kernel management
   - `Stream`: Command stream management
   - `Event`: Synchronization and timing
   - `DeviceManager`: Device discovery and management

2. **Framework Implementations**
   - `cuda.py`: NVIDIA GPU support
   - `metal.py`: Apple GPU support
   - `opencl.py`: Cross-platform GPU support

### Responsibilities

- Device discovery and initialization
- Memory allocation and management
- Kernel compilation and execution
- Stream and event management
- Error handling and logging
- Cross-platform compatibility

## Acceleration Layer (`acceleration/`)

The acceleration layer implements GPU-accelerated algorithms using the device abstraction layer.

### Key Components

1. **Linear Algebra** (`linear_algebra/`)
   - Matrix-vector operations
   - Sparse matrix operations
   - Linear system solvers

2. **Physics** (`physics/`)
   - Flux computations
   - State updates
   - Boundary conditions

3. **Mesh** (`mesh/`)
   - Mesh operations
   - Interpolation
   - Gradient computations

### Responsibilities

- Algorithm implementation
- Performance optimization
- Integration with solver components
- Numerical accuracy
- Memory efficiency

## Interaction Between Layers

### Example: Matrix-Vector Multiplication

```python
# In acceleration/linear_algebra/matrix.py
class GPUMatrix:
    def __init__(self):
        # Use device abstraction layer
        self.device = DeviceManager().get_device("cuda:0")
        self.memory = self.device.create_memory(size)
        self.kernel = self.device.compile_kernel(source, "matrix_vector_multiply")
    
    def multiply(self, vector):
        # Use device abstraction for computation
        self.kernel.launch(grid, block, [self.memory, vector.memory])
```

### Data Flow

1. **Initialization**
   ```
   Acceleration Layer
   └── Requests device from DeviceManager
       └── DeviceManager
           └── Creates appropriate device (CUDA/Metal/OpenCL)
   ```

2. **Computation**
   ```
   Acceleration Layer
   └── Prepares data and parameters
       └── Device Abstraction Layer
           └── Manages memory and execution
               └── Framework-specific implementation
   ```

3. **Results**
   ```
   Framework-specific implementation
   └── Device Abstraction Layer
       └── Handles data transfer
           └── Acceleration Layer
               └── Processes results
   ```

## Best Practices

1. **Device Abstraction Layer**
   - Keep framework-specific code isolated
   - Implement comprehensive error handling
   - Maintain consistent interface across frameworks
   - Document device capabilities and limitations

2. **Acceleration Layer**
   - Focus on algorithm implementation
   - Optimize for performance
   - Handle numerical accuracy
   - Document performance characteristics

## Future Development

1. **Device Abstraction Layer**
   - Add OpenCL support
   - Enhance Metal features
   - Improve error handling
   - Add performance monitoring

2. **Acceleration Layer**
   - Implement more algorithms
   - Add performance benchmarks
   - Enhance numerical accuracy
   - Optimize memory usage

## Testing Strategy

1. **Device Abstraction Layer**
   - Unit tests for each framework
   - Memory management tests
   - Kernel compilation tests
   - Error handling tests

2. **Acceleration Layer**
   - Algorithm correctness tests
   - Performance benchmarks
   - Numerical accuracy tests
   - Integration tests

## Performance Considerations

1. **Memory Management**
   - Minimize host-device transfers
   - Use pinned memory when appropriate
   - Implement memory pooling
   - Handle memory fragmentation

2. **Kernel Optimization**
   - Optimize thread block size
   - Minimize register usage
   - Use shared memory effectively
   - Implement kernel fusion

3. **Stream Management**
   - Overlap computation and transfer
   - Use multiple streams
   - Implement asynchronous operations
   - Handle dependencies correctly 
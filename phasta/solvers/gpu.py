"""GPU-enabled solvers module.

This module provides GPU-accelerated solvers using CUDA and OpenCL backends.
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("CuPy not available, CUDA solvers will be disabled")

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    logger.warning("PyOpenCL not available, OpenCL solvers will be disabled")


class GPUSolver:
    """Base class for GPU solvers."""
    
    def __init__(self, matrix: Union[np.ndarray, sp.spmatrix],
                 backend: str = 'cuda'):
        """Initialize GPU solver.
        
        Args:
            matrix: System matrix
            backend: GPU backend ('cuda' or 'opencl')
        """
        self.matrix = matrix
        self.backend = backend
        self._setup_backend()
    
    def _setup_backend(self):
        """Set up GPU backend."""
        if self.backend == 'cuda':
            if not CUDA_AVAILABLE:
                raise RuntimeError("CUDA backend not available")
            self._setup_cuda()
        elif self.backend == 'opencl':
            if not OPENCL_AVAILABLE:
                raise RuntimeError("OpenCL backend not available")
            self._setup_opencl()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _setup_cuda(self):
        """Set up CUDA backend."""
        raise NotImplementedError
    
    def _setup_opencl(self):
        """Set up OpenCL backend."""
        raise NotImplementedError
    
    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve linear system on GPU.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        raise NotImplementedError


class CUDASolver(GPUSolver):
    """CUDA-accelerated solver."""
    
    def __init__(self, matrix: Union[np.ndarray, sp.spmatrix]):
        """Initialize CUDA solver.
        
        Args:
            matrix: System matrix
        """
        super().__init__(matrix, backend='cuda')
    
    def _setup_cuda(self):
        """Set up CUDA backend."""
        # Convert matrix to CuPy format
        if sp.issparse(self.matrix):
            self.gpu_matrix = cp_sparse.csr_matrix(self.matrix)
        else:
            self.gpu_matrix = cp.array(self.matrix)
    
    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve linear system using CUDA.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        # Transfer data to GPU
        gpu_b = cp.array(b)
        
        # Solve on GPU
        if sp.issparse(self.matrix):
            from cupyx.scipy.sparse.linalg import spsolve
            gpu_x = spsolve(self.gpu_matrix, gpu_b)
        else:
            gpu_x = cp.linalg.solve(self.gpu_matrix, gpu_b)
        
        # Transfer result back to CPU
        return cp.asnumpy(gpu_x)


class OpenCLSolver(GPUSolver):
    """OpenCL-accelerated solver."""
    
    def __init__(self, matrix: Union[np.ndarray, sp.spmatrix],
                 platform_name: Optional[str] = None,
                 device_name: Optional[str] = None):
        """Initialize OpenCL solver.
        
        Args:
            matrix: System matrix
            platform_name: OpenCL platform name
            device_name: OpenCL device name
        """
        self.platform_name = platform_name
        self.device_name = device_name
        super().__init__(matrix, backend='opencl')
    
    def _setup_opencl(self):
        """Set up OpenCL backend."""
        # Get platform and device
        platforms = cl.get_platforms()
        if self.platform_name:
            platform = next(p for p in platforms if p.name == self.platform_name)
        else:
            platform = platforms[0]
        
        devices = platform.get_devices()
        if self.device_name:
            device = next(d for d in devices if d.name == self.device_name)
        else:
            device = devices[0]
        
        # Create context and queue
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)
        
        # Create program
        self.program = cl.Program(self.ctx, """
        __kernel void solve_triangular(
            __global const float* L,
            __global const float* b,
            __global float* x,
            const int n
        ) {
            int i = get_global_id(0);
            if (i >= n) return;
            
            float sum = 0.0f;
            for (int j = 0; j < i; j++) {
                sum += L[i * n + j] * x[j];
            }
            x[i] = (b[i] - sum) / L[i * n + i];
        }
        """).build()
    
    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve linear system using OpenCL.
        
        Args:
            b: Right-hand side vector
            
        Returns:
            Solution vector
        """
        # Convert matrix to float32
        if sp.issparse(self.matrix):
            matrix = self.matrix.toarray().astype(np.float32)
        else:
            matrix = self.matrix.astype(np.float32)
        
        # Create buffers
        matrix_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=matrix)
        b_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                         hostbuf=b.astype(np.float32))
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                         size=b.nbytes)
        
        # Solve system
        n = len(b)
        self.program.solve_triangular(self.queue, (n,), None,
                                    matrix_buf, b_buf, x_buf, np.int32(n))
        
        # Get result
        x = np.empty_like(b, dtype=np.float32)
        cl.enqueue_copy(self.queue, x, x_buf)
        
        return x


def solve_gpu(A: Union[np.ndarray, sp.spmatrix], b: np.ndarray,
             backend: str = 'cuda') -> np.ndarray:
    """Convenience function for GPU solution.
    
    Args:
        A: System matrix
        b: Right-hand side vector
        backend: GPU backend ('cuda' or 'opencl')
        
    Returns:
        Solution vector
    """
    if backend == 'cuda':
        solver = CUDASolver(A)
    elif backend == 'opencl':
        solver = OpenCLSolver(A)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return solver.solve(b) 
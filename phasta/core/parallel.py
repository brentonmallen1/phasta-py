"""Parallel computing utilities for PHASTA-Py.

This module provides functionality for distributed computing using MPI.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from mpi4py import MPI


class ParallelManager:
    """Manager for parallel computing operations."""
    
    def __init__(self):
        """Initialize parallel manager."""
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
    def is_root(self) -> bool:
        """Check if current process is the root process.
        
        Returns:
            True if current process is root (rank 0)
        """
        return self.rank == 0
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        self.comm.Barrier()
    
    def broadcast(self, data: np.ndarray, root: int = 0) -> np.ndarray:
        """Broadcast data from root process to all processes.
        
        Args:
            data: Data to broadcast
            root: Root process rank
            
        Returns:
            Broadcasted data
        """
        return self.comm.bcast(data, root=root)
    
    def scatter(self, data: np.ndarray, root: int = 0) -> np.ndarray:
        """Scatter data from root process to all processes.
        
        Args:
            data: Data to scatter
            root: Root process rank
            
        Returns:
            Scattered data chunk
        """
        return self.comm.scatter(data, root=root)
    
    def gather(self, data: np.ndarray, root: int = 0) -> Optional[np.ndarray]:
        """Gather data from all processes to root process.
        
        Args:
            data: Data to gather
            root: Root process rank
            
        Returns:
            Gathered data (only on root process)
        """
        return self.comm.gather(data, root=root)
    
    def allgather(self, data: np.ndarray) -> np.ndarray:
        """Gather data from all processes to all processes.
        
        Args:
            data: Data to gather
            
        Returns:
            Gathered data on all processes
        """
        return self.comm.allgather(data)
    
    def reduce(self, data: np.ndarray, op: MPI.Op = MPI.SUM, root: int = 0) -> Optional[np.ndarray]:
        """Reduce data from all processes to root process.
        
        Args:
            data: Data to reduce
            op: Reduction operation
            root: Root process rank
            
        Returns:
            Reduced data (only on root process)
        """
        return self.comm.reduce(data, op=op, root=root)
    
    def allreduce(self, data: np.ndarray, op: MPI.Op = MPI.SUM) -> np.ndarray:
        """Reduce data from all processes to all processes.
        
        Args:
            data: Data to reduce
            op: Reduction operation
            
        Returns:
            Reduced data on all processes
        """
        return self.comm.allreduce(data, op=op)
    
    def send(self, data: np.ndarray, dest: int, tag: int = 0) -> None:
        """Send data to a specific process.
        
        Args:
            data: Data to send
            dest: Destination process rank
            tag: Message tag
        """
        self.comm.Send(data, dest=dest, tag=tag)
    
    def recv(self, source: int = MPI.ANY_SOURCE, tag: int = MPI.ANY_TAG) -> Tuple[np.ndarray, int, int]:
        """Receive data from a specific process.
        
        Args:
            source: Source process rank
            tag: Message tag
            
        Returns:
            Tuple of (received data, source rank, tag)
        """
        status = MPI.Status()
        data = self.comm.recv(source=source, tag=tag, status=status)
        return data, status.Get_source(), status.Get_tag()


def get_parallel_manager() -> ParallelManager:
    """Get the global parallel manager instance.
    
    Returns:
        ParallelManager instance
    """
    if not hasattr(get_parallel_manager, '_instance'):
        get_parallel_manager._instance = ParallelManager()
    return get_parallel_manager._instance


def parallel_for(func, data: List, chunk_size: Optional[int] = None) -> List:
    """Execute a function in parallel on distributed data.
    
    Args:
        func: Function to execute
        data: List of data items to process
        chunk_size: Optional chunk size for data distribution
        
    Returns:
        List of results
    """
    manager = get_parallel_manager()
    
    if chunk_size is None:
        chunk_size = max(1, len(data) // manager.size)
    
    # Distribute data
    if manager.is_root():
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        # Pad last chunk if necessary
        if len(chunks[-1]) < chunk_size:
            chunks[-1].extend([None] * (chunk_size - len(chunks[-1])))
    else:
        chunks = None
    
    # Scatter chunks to processes
    local_chunk = manager.scatter(chunks)
    
    # Process local chunk
    local_results = [func(item) for item in local_chunk if item is not None]
    
    # Gather results
    all_results = manager.gather(local_results)
    
    if manager.is_root():
        # Flatten results
        return [item for sublist in all_results for item in sublist]
    return []

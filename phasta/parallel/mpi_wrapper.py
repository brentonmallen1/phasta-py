"""
MPI wrapper for parallel computing operations.

This module provides a wrapper around MPI functionality for parallel computing
operations in PHASTA-Py. It handles process management, communication patterns,
and domain decomposition.
"""

import numpy as np
from mpi4py import MPI
from typing import Dict, List, Tuple, Optional, Union
import logging

class MPIWrapper:
    """Wrapper class for MPI operations."""
    
    def __init__(self):
        """Initialize MPI environment."""
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.logger = logging.getLogger(__name__)
        
        # Initialize logging
        if self.rank == 0:
            self.logger.info(f"Initialized MPI with {self.size} processes")
    
    def barrier(self):
        """Synchronize all processes."""
        self.comm.Barrier()
    
    def broadcast(self, data: Union[np.ndarray, float, int], root: int = 0):
        """Broadcast data from root process to all other processes.
        
        Args:
            data: Data to broadcast
            root: Root process rank (default: 0)
            
        Returns:
            Broadcasted data
        """
        return self.comm.bcast(data, root=root)
    
    def gather(self, data: Union[np.ndarray, float, int], root: int = 0):
        """Gather data from all processes to root process.
        
        Args:
            data: Data to gather
            root: Root process rank (default: 0)
            
        Returns:
            Gathered data (only on root process)
        """
        return self.comm.gather(data, root=root)
    
    def scatter(self, data: Union[np.ndarray, List], root: int = 0):
        """Scatter data from root process to all other processes.
        
        Args:
            data: Data to scatter
            root: Root process rank (default: 0)
            
        Returns:
            Scattered data
        """
        return self.comm.scatter(data, root=root)
    
    def allreduce(self, data: Union[np.ndarray, float, int], op: MPI.Op = MPI.SUM):
        """Perform reduction operation across all processes.
        
        Args:
            data: Data to reduce
            op: Reduction operation (default: MPI.SUM)
            
        Returns:
            Reduced data
        """
        return self.comm.allreduce(data, op=op)
    
    def send(self, data: Union[np.ndarray, float, int], dest: int, tag: int = 0):
        """Send data to a specific process.
        
        Args:
            data: Data to send
            dest: Destination process rank
            tag: Message tag (default: 0)
        """
        self.comm.send(data, dest=dest, tag=tag)
    
    def recv(self, source: int = MPI.ANY_SOURCE, tag: int = MPI.ANY_TAG):
        """Receive data from a specific process.
        
        Args:
            source: Source process rank (default: MPI.ANY_SOURCE)
            tag: Message tag (default: MPI.ANY_TAG)
            
        Returns:
            Received data
        """
        return self.comm.recv(source=source, tag=tag)
    
    def isend(self, data: Union[np.ndarray, float, int], dest: int, tag: int = 0):
        """Non-blocking send operation.
        
        Args:
            data: Data to send
            dest: Destination process rank
            tag: Message tag (default: 0)
            
        Returns:
            Request object
        """
        return self.comm.Isend(data, dest=dest, tag=tag)
    
    def irecv(self, source: int = MPI.ANY_SOURCE, tag: int = MPI.ANY_TAG):
        """Non-blocking receive operation.
        
        Args:
            source: Source process rank (default: MPI.ANY_SOURCE)
            tag: Message tag (default: MPI.ANY_TAG)
            
        Returns:
            Request object
        """
        return self.comm.Irecv(source=source, tag=tag)
    
    def finalize(self):
        """Finalize MPI environment."""
        MPI.Finalize()
        if self.rank == 0:
            self.logger.info("Finalized MPI environment") 
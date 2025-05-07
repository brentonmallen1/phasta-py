"""
Performance monitoring for parallel computing.

This module provides functionality for monitoring and analyzing parallel
computing performance, including timing, memory usage, and scaling metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .mpi_wrapper import MPIWrapper
import logging
import time
import psutil
import os

class PerformanceMonitor:
    """Performance monitoring for parallel computing."""
    
    def __init__(self, mpi: MPIWrapper):
        """Initialize performance monitor.
        
        Args:
            mpi: MPI wrapper instance
        """
        self.mpi = mpi
        self.logger = logging.getLogger(__name__)
        
        # Initialize timers
        self.timers = {
            'total': 0.0,
            'compute': 0.0,
            'communication': 0.0,
            'io': 0.0,
            'load_balance': 0.0
        }
        
        # Initialize memory tracking
        self.memory_history = []
        self.peak_memory = 0.0
        
        # Initialize scaling metrics
        self.scaling_history = []
    
    def start_timer(self, timer_name: str):
        """Start a timer.
        
        Args:
            timer_name: Name of timer to start
        """
        if timer_name not in self.timers:
            self.timers[timer_name] = 0.0
        
        self.timers[timer_name] -= time.time()
    
    def stop_timer(self, timer_name: str):
        """Stop a timer.
        
        Args:
            timer_name: Name of timer to stop
        """
        if timer_name in self.timers:
            self.timers[timer_name] += time.time()
    
    def track_memory(self):
        """Track memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Track memory usage
        memory_usage = memory_info.rss / 1024 / 1024  # Convert to MB
        self.memory_history.append(memory_usage)
        
        # Update peak memory
        self.peak_memory = max(self.peak_memory, memory_usage)
    
    def track_scaling(self, n_cells: int, n_processes: int):
        """Track scaling performance.
        
        Args:
            n_cells: Number of cells
            n_processes: Number of processes
        """
        # Compute cells per second
        cells_per_second = n_cells / self.timers['total']
        
        # Store scaling data
        self.scaling_history.append({
            'n_cells': n_cells,
            'n_processes': n_processes,
            'cells_per_second': cells_per_second,
            'wall_time': self.timers['total']
        })
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Gather timers from all processes
        all_timers = {}
        for key, value in self.timers.items():
            all_timers[key] = self.mpi.allreduce(value)
        
        # Gather memory usage
        all_memory = self.mpi.gather(self.peak_memory)
        
        # Compute scaling metrics
        scaling_metrics = self._compute_scaling_metrics()
        
        # Combine metrics
        metrics = {
            'timers': all_timers,
            'memory': {
                'peak': np.max(all_memory) if all_memory else 0.0,
                'avg': np.mean(all_memory) if all_memory else 0.0,
                'min': np.min(all_memory) if all_memory else 0.0
            },
            'scaling': scaling_metrics
        }
        
        return metrics
    
    def _compute_scaling_metrics(self) -> Dict:
        """Compute scaling metrics from history.
        
        Returns:
            Dictionary of scaling metrics
        """
        if not self.scaling_history:
            return {}
        
        # Convert history to arrays
        n_cells = np.array([d['n_cells'] for d in self.scaling_history])
        n_processes = np.array([d['n_processes'] for d in self.scaling_history])
        cells_per_second = np.array([d['cells_per_second'] for d in self.scaling_history])
        wall_time = np.array([d['wall_time'] for d in self.scaling_history])
        
        # Compute scaling efficiency
        if len(n_processes) > 1:
            # Strong scaling
            strong_scaling = wall_time[0] / (n_processes * wall_time)
            
            # Weak scaling
            weak_scaling = (n_cells[0] * wall_time[0]) / (n_cells * wall_time)
        else:
            strong_scaling = np.array([1.0])
            weak_scaling = np.array([1.0])
        
        return {
            'strong_scaling': strong_scaling,
            'weak_scaling': weak_scaling,
            'cells_per_second': cells_per_second,
            'wall_time': wall_time
        }
    
    def generate_report(self, filename: str):
        """Generate performance report.
        
        Args:
            filename: Output filename
        """
        if self.mpi.rank == 0:
            # Get metrics
            metrics = self.get_performance_metrics()
            
            # Create report
            with open(filename, 'w') as f:
                f.write("Performance Report\n")
                f.write("=================\n\n")
                
                # Write timing information
                f.write("Timing Information\n")
                f.write("-----------------\n")
                for key, value in metrics['timers'].items():
                    f.write(f"{key}: {value:.2f} seconds\n")
                f.write("\n")
                
                # Write memory information
                f.write("Memory Usage\n")
                f.write("------------\n")
                f.write(f"Peak Memory: {metrics['memory']['peak']:.2f} MB\n")
                f.write(f"Average Memory: {metrics['memory']['avg']:.2f} MB\n")
                f.write(f"Minimum Memory: {metrics['memory']['min']:.2f} MB\n\n")
                
                # Write scaling information
                f.write("Scaling Performance\n")
                f.write("------------------\n")
                if metrics['scaling']:
                    f.write("Strong Scaling Efficiency:\n")
                    for i, eff in enumerate(metrics['scaling']['strong_scaling']):
                        f.write(f"  {i+1} processes: {eff:.2f}\n")
                    
                    f.write("\nWeak Scaling Efficiency:\n")
                    for i, eff in enumerate(metrics['scaling']['weak_scaling']):
                        f.write(f"  {i+1} processes: {eff:.2f}\n")
                    
                    f.write("\nCells per Second:\n")
                    for i, cps in enumerate(metrics['scaling']['cells_per_second']):
                        f.write(f"  {i+1} processes: {cps:.2e}\n")
                    
                    f.write("\nWall Time:\n")
                    for i, wt in enumerate(metrics['scaling']['wall_time']):
                        f.write(f"  {i+1} processes: {wt:.2f} seconds\n")
    
    def plot_performance(self, filename: str):
        """Generate performance plots.
        
        Args:
            filename: Output filename prefix
        """
        if self.mpi.rank == 0:
            try:
                import matplotlib.pyplot as plt
                
                # Get metrics
                metrics = self.get_performance_metrics()
                
                if metrics['scaling']:
                    # Create scaling plots
                    n_processes = np.arange(1, len(metrics['scaling']['strong_scaling']) + 1)
                    
                    # Strong scaling
                    plt.figure(figsize=(10, 6))
                    plt.plot(n_processes, metrics['scaling']['strong_scaling'], 'bo-',
                            label='Measured')
                    plt.plot(n_processes, np.ones_like(n_processes), 'r--',
                            label='Ideal')
                    plt.xlabel('Number of Processes')
                    plt.ylabel('Strong Scaling Efficiency')
                    plt.title('Strong Scaling Performance')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f"{filename}_strong_scaling.png")
                    plt.close()
                    
                    # Weak scaling
                    plt.figure(figsize=(10, 6))
                    plt.plot(n_processes, metrics['scaling']['weak_scaling'], 'bo-',
                            label='Measured')
                    plt.plot(n_processes, np.ones_like(n_processes), 'r--',
                            label='Ideal')
                    plt.xlabel('Number of Processes')
                    plt.ylabel('Weak Scaling Efficiency')
                    plt.title('Weak Scaling Performance')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f"{filename}_weak_scaling.png")
                    plt.close()
                    
                    # Cells per second
                    plt.figure(figsize=(10, 6))
                    plt.plot(n_processes, metrics['scaling']['cells_per_second'], 'bo-')
                    plt.xlabel('Number of Processes')
                    plt.ylabel('Cells per Second')
                    plt.title('Computational Throughput')
                    plt.grid(True)
                    plt.savefig(f"{filename}_throughput.png")
                    plt.close()
                
                # Memory usage over time
                if self.memory_history:
                    plt.figure(figsize=(10, 6))
                    plt.plot(self.memory_history, 'b-')
                    plt.xlabel('Sample')
                    plt.ylabel('Memory Usage (MB)')
                    plt.title('Memory Usage Over Time')
                    plt.grid(True)
                    plt.savefig(f"{filename}_memory.png")
                    plt.close()
                
            except ImportError:
                self.logger.warning("Matplotlib not available for plotting") 
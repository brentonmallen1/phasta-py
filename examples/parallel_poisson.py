"""Example: Parallel Poisson equation solver.

This example demonstrates how to solve a parallel Poisson equation using different
iterative solvers. The problem is discretized using finite elements and solved
in parallel using MPI.
"""

import numpy as np
from scipy import sparse
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from phasta.fem import (
    Mesh, ParallelMesh, ParallelGMRES, ParallelCG, ParallelBiCGSTAB,
    ParallelSolverFactory
)


def create_poisson_matrix(n: int) -> sparse.spmatrix:
    """Create a Poisson matrix for a 2D grid.
    
    Args:
        n: Number of points in each direction
        
    Returns:
        Sparse matrix representing the Poisson operator
    """
    # Create 2D grid
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # Create nodes and elements
    nodes = np.column_stack([X.flatten(), Y.flatten()])
    elements = []
    for j in range(n-1):
        for i in range(n-1):
            idx = j*n + i
            elements.append([idx, idx+1, idx+n+1, idx+n])
    
    # Create mesh
    mesh = Mesh(nodes, np.array(elements))
    
    # Create parallel mesh
    parallel_mesh = ParallelMesh(mesh, n_parts=MPI.COMM_WORLD.Get_size())
    
    # Create stiffness matrix
    n_nodes = len(nodes)
    A = sparse.lil_matrix((n_nodes, n_nodes))
    
    # Add contributions from each element
    for element in elements:
        # Get element nodes
        element_nodes = nodes[element]
        
        # Compute element matrix (simplified for this example)
        element_matrix = np.array([
            [2, -1, -1, 0],
            [-1, 2, 0, -1],
            [-1, 0, 2, -1],
            [0, -1, -1, 2]
        ])
        
        # Add to global matrix
        for i, node_i in enumerate(element):
            for j, node_j in enumerate(element):
                A[node_i, node_j] += element_matrix[i, j]
    
    return A.tocsr(), parallel_mesh, X, Y


def plot_solution(X: np.ndarray, Y: np.ndarray, u: np.ndarray, title: str) -> None:
    """Plot the solution using matplotlib.
    
    Args:
        X: X-coordinates mesh
        Y: Y-coordinates mesh
        u: Solution vector
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, u.reshape(X.shape), levels=50, cmap='viridis')
    plt.colorbar(label='Solution')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()


def main():
    """Run the parallel Poisson example."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Create problem
    n = 50  # Grid points in each direction
    A, parallel_mesh, X, Y = create_poisson_matrix(n)
    
    # Create right-hand side (sin(πx)sin(πy))
    b = np.sin(np.pi * X) * np.sin(np.pi * Y)
    b = b.flatten()
    
    # Solve using different solvers
    solvers = {
        'GMRES': ParallelGMRES(max_iter=1000, tol=1e-6, restart=30),
        'CG': ParallelCG(max_iter=1000, tol=1e-6),
        'BiCGSTAB': ParallelBiCGSTAB(max_iter=1000, tol=1e-6)
    }
    
    # Solve and print results
    if rank == 0:
        print("\nSolving parallel Poisson equation:")
        print("----------------------------------")
    
    for name, solver in solvers.items():
        # Solve system
        x = solver.solve(A, b, parallel_mesh)
        
        # Compute residual
        r = b - A @ x
        residual_norm = np.linalg.norm(r)
        
        if rank == 0:
            print(f"\n{name}:")
            print(f"Residual norm: {residual_norm:.2e}")
            
            # Plot solution
            plot_solution(X, Y, x, f'Poisson Solution - {name}')
    
    if rank == 0:
        print("\nDone!")


if __name__ == '__main__':
    main() 
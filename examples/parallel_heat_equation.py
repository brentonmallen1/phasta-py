"""Example: Parallel heat equation solver.

This example demonstrates how to solve a parallel heat equation with time-dependent
boundary conditions using the finite element method and parallel solvers.
"""

import numpy as np
from scipy import sparse
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from phasta.fem import (
    Mesh, ParallelMesh, ParallelGMRES, ParallelCG, ParallelBiCGSTAB,
    ParallelSolverFactory, TimeIntegrator, ImplicitEuler
)


def create_heat_matrix(n: int, dt: float, alpha: float = 1.0) -> sparse.spmatrix:
    """Create a heat equation matrix for a 2D grid.
    
    Args:
        n: Number of points in each direction
        dt: Time step size
        alpha: Thermal diffusivity
        
    Returns:
        Sparse matrix representing the heat operator
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
    
    # Create mass and stiffness matrices
    n_nodes = len(nodes)
    M = sparse.lil_matrix((n_nodes, n_nodes))
    K = sparse.lil_matrix((n_nodes, n_nodes))
    
    # Add contributions from each element
    for element in elements:
        # Get element nodes
        element_nodes = nodes[element]
        
        # Compute element matrices (simplified for this example)
        mass_matrix = np.array([
            [2, 1, 1, 0],
            [1, 2, 0, 1],
            [1, 0, 2, 1],
            [0, 1, 1, 2]
        ]) / 6.0
        
        stiffness_matrix = np.array([
            [2, -1, -1, 0],
            [-1, 2, 0, -1],
            [-1, 0, 2, -1],
            [0, -1, -1, 2]
        ])
        
        # Add to global matrices
        for i, node_i in enumerate(element):
            for j, node_j in enumerate(element):
                M[node_i, node_j] += mass_matrix[i, j]
                K[node_i, node_j] += stiffness_matrix[i, j]
    
    # Create time-stepping matrix: M + dt*alpha*K
    A = M.tocsr() + dt * alpha * K.tocsr()
    
    return A, parallel_mesh, X, Y


def apply_boundary_conditions(A: sparse.spmatrix, b: np.ndarray, 
                            nodes: np.ndarray, t: float) -> None:
    """Apply time-dependent boundary conditions.
    
    Args:
        A: System matrix
        b: Right-hand side vector
        nodes: Node coordinates
        t: Current time
    """
    # Find boundary nodes
    boundary_nodes = np.where(
        (nodes[:, 0] == 0) | (nodes[:, 0] == 1) |
        (nodes[:, 1] == 0) | (nodes[:, 1] == 1)
    )[0]
    
    # Apply Dirichlet boundary conditions
    for node in boundary_nodes:
        x, y = nodes[node]
        # Time-dependent temperature on boundaries
        T = np.sin(2*np.pi*t) * np.sin(np.pi*x) * np.sin(np.pi*y)
        
        # Modify matrix and right-hand side
        A[node, :] = 0
        A[node, node] = 1
        b[node] = T


def create_animation(X: np.ndarray, Y: np.ndarray, solutions: list, 
                    times: list, filename: str) -> None:
    """Create an animation of the solution evolution.
    
    Args:
        X: X-coordinates mesh
        Y: Y-coordinates mesh
        solutions: List of solution vectors at each time step
        times: List of times corresponding to solutions
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        contour = ax.contourf(X, Y, solutions[frame].reshape(X.shape), 
                            levels=50, cmap='viridis')
        ax.set_title(f'Time: {times[frame]:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(contour, ax=ax, label='Temperature')
        return contour,
    
    anim = FuncAnimation(fig, update, frames=len(solutions),
                        interval=100, blit=True)
    anim.save(filename, writer='pillow', fps=10)
    plt.close()


def main():
    """Run the parallel heat equation example."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Problem parameters
    n = 50  # Grid points in each direction
    dt = 0.01  # Time step
    t_end = 1.0  # End time
    alpha = 1.0  # Thermal diffusivity
    
    # Create problem
    A, parallel_mesh, X, Y = create_heat_matrix(n, dt, alpha)
    
    # Create initial condition (sin(πx)sin(πy))
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u = u.flatten()
    
    # Create solver
    solver = ParallelGMRES(max_iter=1000, tol=1e-6, restart=30)
    
    # Time stepping
    t = 0.0
    step = 0
    
    # Store solutions for animation
    solutions = []
    times = []
    
    if rank == 0:
        print("\nSolving parallel heat equation:")
        print("-------------------------------")
    
    while t < t_end:
        # Create right-hand side
        b = u.copy()
        
        # Apply boundary conditions
        apply_boundary_conditions(A, b, parallel_mesh.nodes, t)
        
        # Solve system
        u_new = solver.solve(A, b, parallel_mesh)
        
        # Update solution
        u = u_new
        
        # Store solution for animation
        if rank == 0 and step % 5 == 0:  # Store every 5th step
            solutions.append(u.copy())
            times.append(t)
        
        # Update time
        t += dt
        step += 1
        
        if rank == 0 and step % 10 == 0:
            print(f"Step {step}, t = {t:.3f}")
    
    # Create animation
    if rank == 0:
        create_animation(X, Y, solutions, times, 'heat_equation_animation.gif')
        print("\nAnimation saved as 'heat_equation_animation.gif'")
    
    if rank == 0:
        print("\nDone!")


if __name__ == '__main__':
    main() 
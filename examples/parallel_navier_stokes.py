"""Example: Parallel Navier-Stokes solver.

This example demonstrates how to solve a parallel Navier-Stokes equation using
the finite element method and parallel solvers. The problem is a lid-driven
cavity flow.
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


def create_navier_stokes_matrices(n: int, dt: float, Re: float = 100.0) -> tuple:
    """Create Navier-Stokes matrices for a 2D grid.
    
    Args:
        n: Number of points in each direction
        dt: Time step size
        Re: Reynolds number
        
    Returns:
        Tuple of (velocity matrix, pressure matrix, parallel mesh, coordinates)
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
    
    # Create matrices for velocity and pressure
    n_nodes = len(nodes)
    A_u = sparse.lil_matrix((n_nodes, n_nodes))
    A_v = sparse.lil_matrix((n_nodes, n_nodes))
    A_p = sparse.lil_matrix((n_nodes, n_nodes))
    
    # Add contributions from each element
    for element in elements:
        # Get element nodes
        element_nodes = nodes[element]
        
        # Compute element matrices (simplified for this example)
        # Mass matrix
        mass_matrix = np.array([
            [2, 1, 1, 0],
            [1, 2, 0, 1],
            [1, 0, 2, 1],
            [0, 1, 1, 2]
        ]) / 6.0
        
        # Stiffness matrix
        stiffness_matrix = np.array([
            [2, -1, -1, 0],
            [-1, 2, 0, -1],
            [-1, 0, 2, -1],
            [0, -1, -1, 2]
        ])
        
        # Add to global matrices
        for i, node_i in enumerate(element):
            for j, node_j in enumerate(element):
                # Velocity matrices (with time stepping and diffusion)
                A_u[node_i, node_j] += mass_matrix[i, j] + dt/Re * stiffness_matrix[i, j]
                A_v[node_i, node_j] += mass_matrix[i, j] + dt/Re * stiffness_matrix[i, j]
                # Pressure matrix (Laplacian)
                A_p[node_i, node_j] += stiffness_matrix[i, j]
    
    return (A_u.tocsr(), A_v.tocsr(), A_p.tocsr(), 
            parallel_mesh, X, Y)


def apply_boundary_conditions(A_u: sparse.spmatrix, A_v: sparse.spmatrix,
                            b_u: np.ndarray, b_v: np.ndarray,
                            nodes: np.ndarray) -> None:
    """Apply boundary conditions for lid-driven cavity.
    
    Args:
        A_u: Velocity x-component matrix
        A_v: Velocity y-component matrix
        b_u: Velocity x-component right-hand side
        b_v: Velocity y-component right-hand side
        nodes: Node coordinates
    """
    # Find boundary nodes
    left = np.where(nodes[:, 0] == 0)[0]
    right = np.where(nodes[:, 0] == 1)[0]
    bottom = np.where(nodes[:, 1] == 0)[0]
    top = np.where(nodes[:, 1] == 1)[0]
    
    # Apply no-slip boundary conditions
    for node in np.concatenate([left, right, bottom]):
        A_u[node, :] = 0
        A_v[node, :] = 0
        A_u[node, node] = 1
        A_v[node, node] = 1
        b_u[node] = 0
        b_v[node] = 0
    
    # Apply lid velocity
    for node in top:
        A_u[node, :] = 0
        A_v[node, :] = 0
        A_u[node, node] = 1
        A_v[node, node] = 1
        b_u[node] = 1  # Moving lid
        b_v[node] = 0


def plot_velocity_field(X: np.ndarray, Y: np.ndarray, u: np.ndarray, v: np.ndarray,
                       title: str) -> None:
    """Plot the velocity field.
    
    Args:
        X: X-coordinates mesh
        Y: Y-coordinates mesh
        u: X-velocity component
        v: Y-velocity component
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    # Reshape velocity components
    u_plot = u.reshape(X.shape)
    v_plot = v.reshape(X.shape)
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_plot**2 + v_plot**2)
    
    # Plot velocity magnitude
    plt.contourf(X, Y, vel_mag, levels=50, cmap='viridis')
    plt.colorbar(label='Velocity magnitude')
    
    # Plot velocity vectors
    skip = 5  # Skip points for clarity
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
               u_plot[::skip, ::skip], v_plot[::skip, ::skip],
               color='white', scale=50)
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()


def main():
    """Run the parallel Navier-Stokes example."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Problem parameters
    n = 50  # Grid points in each direction
    dt = 0.01  # Time step
    t_end = 10.0  # End time
    Re = 100.0  # Reynolds number
    
    # Create problem
    A_u, A_v, A_p, parallel_mesh, X, Y = create_navier_stokes_matrices(n, dt, Re)
    
    # Initialize solution
    u = np.zeros(n*n)  # x-velocity
    v = np.zeros(n*n)  # y-velocity
    p = np.zeros(n*n)  # pressure
    
    # Create solvers
    solver_u = ParallelGMRES(max_iter=1000, tol=1e-6, restart=30)
    solver_v = ParallelGMRES(max_iter=1000, tol=1e-6, restart=30)
    solver_p = ParallelGMRES(max_iter=1000, tol=1e-6, restart=30)
    
    # Time stepping
    t = 0.0
    step = 0
    
    # Store solutions for animation
    solutions = []
    times = []
    
    if rank == 0:
        print("\nSolving parallel Navier-Stokes equation:")
        print("----------------------------------------")
    
    while t < t_end:
        # Create right-hand sides
        b_u = u.copy()
        b_v = v.copy()
        b_p = p.copy()
        
        # Apply boundary conditions
        apply_boundary_conditions(A_u, A_v, b_u, b_v, parallel_mesh.nodes)
        
        # Solve for velocities
        u_new = solver_u.solve(A_u, b_u, parallel_mesh)
        v_new = solver_v.solve(A_v, b_v, parallel_mesh)
        
        # Solve for pressure
        p_new = solver_p.solve(A_p, b_p, parallel_mesh)
        
        # Update solution
        u = u_new
        v = v_new
        p = p_new
        
        # Store solution for animation
        if rank == 0 and step % 10 == 0:  # Store every 10th step
            solutions.append((u.copy(), v.copy()))
            times.append(t)
        
        # Update time
        t += dt
        step += 1
        
        if rank == 0 and step % 10 == 0:
            print(f"Step {step}, t = {t:.3f}")
    
    # Create animation
    if rank == 0:
        def create_frame(frame):
            u, v = solutions[frame]
            plot_velocity_field(X, Y, u, v, f'Navier-Stokes Solution - t = {times[frame]:.3f}')
        
        for i in range(len(solutions)):
            create_frame(i)
        print("\nSolution plots saved as PNG files")
    
    if rank == 0:
        print("\nDone!")


if __name__ == '__main__':
    main() 
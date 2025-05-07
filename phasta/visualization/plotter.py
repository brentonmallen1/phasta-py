"""Visualization utilities for PHASTA-Py.

This module provides functions for visualizing meshes and simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshio
from typing import Optional, Tuple, List, Union, Dict
import pyvista as pv
from scipy.interpolate import griddata


def plot_mesh_2d(
    mesh: meshio.Mesh,
    show_cells: bool = True,
    show_points: bool = False,
    title: Optional[str] = None,
    output_file: Optional[str] = None
) -> None:
    """Plot a 2D mesh using matplotlib.
    
    Args:
        mesh: The mesh to plot
        show_cells: Whether to show cell boundaries
        show_points: Whether to show mesh points
        title: Optional plot title
        output_file: Optional file path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot cells
    if show_cells:
        for cell in mesh.cells[0].data:
            points = mesh.points[cell]
            ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=0.5)
    
    # Plot points
    if show_points:
        ax.plot(mesh.points[:, 0], mesh.points[:, 1], 'r.', markersize=1)
    
    # Set plot properties
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Save plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_mesh_3d(
    mesh: meshio.Mesh,
    show_cells: bool = True,
    show_points: bool = False,
    title: Optional[str] = None,
    output_file: Optional[str] = None
) -> None:
    """Plot a 3D mesh using PyVista.
    
    Args:
        mesh: The mesh to plot
        show_cells: Whether to show cell boundaries
        show_points: Whether to show mesh points
        title: Optional plot title
        output_file: Optional file path to save the plot
    """
    # Convert meshio mesh to PyVista mesh
    pv_mesh = pv.from_meshio(mesh)
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add mesh to plotter
    if show_cells:
        plotter.add_mesh(pv_mesh, style='wireframe', color='blue', line_width=1)
    
    if show_points:
        plotter.add_mesh(pv_mesh.points, color='red', point_size=5)
    
    # Set plot properties
    if title:
        plotter.add_title(title)
    plotter.add_axes()
    
    # Save plot if output file is specified
    if output_file:
        plotter.screenshot(output_file)
    
    # Show plot
    plotter.show()


def plot_solution_2d(
    mesh: meshio.Mesh,
    solution: np.ndarray,
    field_name: str,
    title: Optional[str] = None,
    output_file: Optional[str] = None
) -> None:
    """Plot a 2D solution field using matplotlib.
    
    Args:
        mesh: The mesh
        solution: Solution field values at mesh points
        field_name: Name of the field being plotted
        title: Optional plot title
        output_file: Optional file path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create triangulation for plotting
    triangulation = plt.tri.Triangulation(mesh.points[:, 0], mesh.points[:, 1])
    
    # Plot solution
    tcf = ax.tricontourf(triangulation, solution, levels=50, cmap='viridis')
    plt.colorbar(tcf, ax=ax, label=field_name)
    
    # Set plot properties
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Save plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_solution_3d(
    mesh: meshio.Mesh,
    solution: np.ndarray,
    field_name: str,
    title: Optional[str] = None,
    output_file: Optional[str] = None
) -> None:
    """Plot a 3D solution field using PyVista.
    
    Args:
        mesh: The mesh
        solution: Solution field values at mesh points
        field_name: Name of the field being plotted
        title: Optional plot title
        output_file: Optional file path to save the plot
    """
    # Convert meshio mesh to PyVista mesh
    pv_mesh = pv.from_meshio(mesh)
    
    # Add solution data to mesh
    pv_mesh.point_data[field_name] = solution
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add mesh with solution to plotter
    plotter.add_mesh(pv_mesh, scalars=field_name, cmap='viridis')
    
    # Set plot properties
    if title:
        plotter.add_title(title)
    plotter.add_axes()
    plotter.add_scalar_bar(title=field_name)
    
    # Save plot if output file is specified
    if output_file:
        plotter.screenshot(output_file)
    
    # Show plot
    plotter.show()


def plot_velocity_field_2d(
    mesh: meshio.Mesh,
    velocity: np.ndarray,
    title: Optional[str] = None,
    output_file: Optional[str] = None
) -> None:
    """Plot a 2D velocity field using matplotlib.
    
    Args:
        mesh: The mesh
        velocity: Velocity field values at mesh points (shape: (n_points, 2))
        title: Optional plot title
        output_file: Optional file path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot velocity vectors
    ax.quiver(mesh.points[:, 0], mesh.points[:, 1],
             velocity[:, 0], velocity[:, 1],
             scale=50, color='blue')
    
    # Set plot properties
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Save plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_velocity_field_3d(
    mesh: meshio.Mesh,
    velocity: np.ndarray,
    title: Optional[str] = None,
    output_file: Optional[str] = None
) -> None:
    """Plot a 3D velocity field using PyVista.
    
    Args:
        mesh: The mesh
        velocity: Velocity field values at mesh points (shape: (n_points, 3))
        title: Optional plot title
        output_file: Optional file path to save the plot
    """
    # Convert meshio mesh to PyVista mesh
    pv_mesh = pv.from_meshio(mesh)
    
    # Add velocity data to mesh
    pv_mesh.point_data['velocity'] = velocity
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add mesh with velocity vectors to plotter
    plotter.add_mesh(pv_mesh, scalars='velocity', cmap='viridis')
    plotter.add_arrows(pv_mesh.points, velocity, mag=0.1, color='blue')
    
    # Set plot properties
    if title:
        plotter.add_title(title)
    plotter.add_axes()
    plotter.add_scalar_bar(title='Velocity magnitude')
    
    # Save plot if output file is specified
    if output_file:
        plotter.screenshot(output_file)
    
    # Show plot
    plotter.show()


def plot_streamlines_2d(
    mesh: meshio.Mesh,
    velocity: np.ndarray,
    density: float = 1.0,
    title: Optional[str] = None,
    output_file: Optional[str] = None
) -> None:
    """Plot 2D streamlines of a velocity field using matplotlib.
    
    Args:
        mesh: The mesh
        velocity: Velocity field values at mesh points (shape: (n_points, 2))
        density: Controls the density of streamlines
        title: Optional plot title
        output_file: Optional file path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a regular grid for streamline plotting
    x = np.linspace(mesh.points[:, 0].min(), mesh.points[:, 0].max(), 100)
    y = np.linspace(mesh.points[:, 1].min(), mesh.points[:, 1].max(), 100)
    X, Y = np.meshgrid(x, y)
    
    # Interpolate velocity field to regular grid
    U = griddata(mesh.points[:, :2], velocity[:, 0], (X, Y), method='linear')
    V = griddata(mesh.points[:, :2], velocity[:, 1], (X, Y), method='linear')
    
    # Plot streamlines
    strm = ax.streamplot(X, Y, U, V, density=density, color='blue', linewidth=1)
    
    # Set plot properties
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Save plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_streamlines_3d(
    mesh: meshio.Mesh,
    velocity: np.ndarray,
    n_points: int = 100,
    title: Optional[str] = None,
    output_file: Optional[str] = None
) -> None:
    """Plot 3D streamlines of a velocity field using PyVista.
    
    Args:
        mesh: The mesh
        velocity: Velocity field values at mesh points (shape: (n_points, 3))
        n_points: Number of seed points for streamlines
        title: Optional plot title
        output_file: Optional file path to save the plot
    """
    # Convert meshio mesh to PyVista mesh
    pv_mesh = pv.from_meshio(mesh)
    
    # Add velocity data to mesh
    pv_mesh.point_data['velocity'] = velocity
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add mesh with velocity magnitude
    plotter.add_mesh(pv_mesh, scalars='velocity', cmap='viridis', opacity=0.3)
    
    # Generate seed points for streamlines
    bounds = pv_mesh.bounds
    seed_points = np.random.uniform(
        low=[bounds[0], bounds[2], bounds[4]],
        high=[bounds[1], bounds[3], bounds[5]],
        size=(n_points, 3)
    )
    
    # Add streamlines
    plotter.add_mesh(
        pv_mesh.streamlines(
            seed_points,
            vectors='velocity',
            max_time=100.0,
            n_points=100
        ),
        color='white',
        line_width=2
    )
    
    # Set plot properties
    if title:
        plotter.add_title(title)
    plotter.add_axes()
    plotter.add_scalar_bar(title='Velocity magnitude')
    
    # Save plot if output file is specified
    if output_file:
        plotter.screenshot(output_file)
    
    # Show plot
    plotter.show()


def plot_isosurface_3d(
    mesh: meshio.Mesh,
    field: np.ndarray,
    field_name: str,
    levels: List[float],
    title: Optional[str] = None,
    output_file: Optional[str] = None
) -> None:
    """Plot 3D isosurfaces of a scalar field using PyVista.
    
    Args:
        mesh: The mesh
        field: Scalar field values at mesh points
        field_name: Name of the field being plotted
        levels: List of isosurface values to plot
        title: Optional plot title
        output_file: Optional file path to save the plot
    """
    # Convert meshio mesh to PyVista mesh
    pv_mesh = pv.from_meshio(mesh)
    
    # Add field data to mesh
    pv_mesh.point_data[field_name] = field
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add isosurfaces
    for i, level in enumerate(levels):
        isosurface = pv_mesh.contour([level])
        plotter.add_mesh(
            isosurface,
            scalars=field_name,
            cmap='viridis',
            opacity=0.7,
            label=f'{field_name} = {level:.2f}'
        )
    
    # Set plot properties
    if title:
        plotter.add_title(title)
    plotter.add_axes()
    plotter.add_scalar_bar(title=field_name)
    plotter.add_legend()
    
    # Save plot if output file is specified
    if output_file:
        plotter.screenshot(output_file)
    
    # Show plot
    plotter.show()


def plot_slice_3d(
    mesh: meshio.Mesh,
    field: np.ndarray,
    field_name: str,
    normal: Tuple[float, float, float] = (1, 0, 0),
    origin: Tuple[float, float, float] = (0, 0, 0),
    title: Optional[str] = None,
    output_file: Optional[str] = None
) -> None:
    """Plot a slice through a 3D field using PyVista.
    
    Args:
        mesh: The mesh
        field: Field values at mesh points
        field_name: Name of the field being plotted
        normal: Normal vector of the slice plane
        origin: Origin point of the slice plane
        title: Optional plot title
        output_file: Optional file path to save the plot
    """
    # Convert meshio mesh to PyVista mesh
    pv_mesh = pv.from_meshio(mesh)
    
    # Add field data to mesh
    pv_mesh.point_data[field_name] = field
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add mesh with field
    plotter.add_mesh(pv_mesh, scalars=field_name, cmap='viridis', opacity=0.3)
    
    # Add slice
    slice_mesh = pv_mesh.slice(normal=normal, origin=origin)
    plotter.add_mesh(slice_mesh, scalars=field_name, cmap='viridis')
    
    # Set plot properties
    if title:
        plotter.add_title(title)
    plotter.add_axes()
    plotter.add_scalar_bar(title=field_name)
    
    # Save plot if output file is specified
    if output_file:
        plotter.screenshot(output_file)
    
    # Show plot
    plotter.show()


def plot_animation_3d(
    mesh: meshio.Mesh,
    fields: Dict[str, List[np.ndarray]],
    times: List[float],
    title: Optional[str] = None,
    output_file: Optional[str] = None,
    fps: int = 10
) -> None:
    """Create an animation of time-dependent 3D fields using PyVista.
    
    Args:
        mesh: The mesh
        fields: Dictionary of field names to lists of field values at each time step
        times: List of time values
        title: Optional plot title
        output_file: Optional file path to save the animation
        fps: Frames per second for the animation
    """
    # Convert meshio mesh to PyVista mesh
    pv_mesh = pv.from_meshio(mesh)
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add initial field data
    for field_name, field_values in fields.items():
        pv_mesh.point_data[field_name] = field_values[0]
    
    # Add mesh to plotter
    plotter.add_mesh(pv_mesh, scalars=list(fields.keys())[0], cmap='viridis')
    
    # Set plot properties
    if title:
        plotter.add_title(title)
    plotter.add_axes()
    plotter.add_scalar_bar(title=list(fields.keys())[0])
    
    # Create animation
    plotter.open_movie(output_file, fps=fps)
    
    # Update fields for each time step
    for i, t in enumerate(times):
        for field_name, field_values in fields.items():
            pv_mesh.point_data[field_name] = field_values[i]
        plotter.write_frame()
    
    # Close movie
    plotter.close() 
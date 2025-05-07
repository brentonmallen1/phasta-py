"""Mesh generation utilities for PHASTA-Py.

This module provides functions to generate common test geometries for CFD simulations.
"""

import numpy as np
import meshio
from typing import Tuple, List, Optional
import os


def generate_pipe_mesh(
    length: float = 10.0,
    radius: float = 0.5,
    n_axial: int = 50,
    n_radial: int = 20,
    n_circumferential: int = 40,
    output_file: Optional[str] = None
) -> meshio.Mesh:
    """Generate a 3D mesh for flow in a circular pipe.
    
    Args:
        length: Length of the pipe
        radius: Radius of the pipe
        n_axial: Number of points along the pipe length
        n_radial: Number of points in the radial direction
        n_circumferential: Number of points around the circumference
        output_file: Optional file path to save the mesh
        
    Returns:
        meshio.Mesh: The generated mesh
    """
    # Create cylindrical coordinates
    r = np.linspace(0, radius, n_radial)
    theta = np.linspace(0, 2*np.pi, n_circumferential, endpoint=False)
    z = np.linspace(0, length, n_axial)
    
    # Create 3D points
    points = []
    for k in range(n_axial):
        for j in range(n_circumferential):
            for i in range(n_radial):
                x = r[i] * np.cos(theta[j])
                y = r[i] * np.sin(theta[j])
                z_coord = z[k]
                points.append([x, y, z_coord])
    
    points = np.array(points)
    
    # Create hexahedral cells
    cells = []
    for k in range(n_axial-1):
        for j in range(n_circumferential-1):
            for i in range(n_radial-1):
                idx = k*n_radial*n_circumferential + j*n_radial + i
                p0 = idx
                p1 = idx + 1
                p2 = idx + n_radial + 1
                p3 = idx + n_radial
                p4 = idx + n_radial*n_circumferential
                p5 = idx + n_radial*n_circumferential + 1
                p6 = idx + n_radial*n_circumferential + n_radial + 1
                p7 = idx + n_radial*n_circumferential + n_radial
                cells.append([p0, p1, p2, p3, p4, p5, p6, p7])
    
    # Create mesh
    mesh = meshio.Mesh(
        points=points,
        cells=[("hexahedron", np.array(cells))]
    )
    
    # Save mesh if output file is specified
    if output_file:
        meshio.write(output_file, mesh)
    
    return mesh


def generate_channel_mesh(
    length: float = 10.0,
    height: float = 1.0,
    width: float = 1.0,
    nx: int = 100,
    ny: int = 20,
    nz: int = 20,
    output_file: Optional[str] = None
) -> meshio.Mesh:
    """Generate a 3D mesh for flow in a rectangular channel.
    
    Args:
        length: Length of the channel
        height: Height of the channel
        width: Width of the channel
        nx: Number of points in x-direction
        ny: Number of points in y-direction
        nz: Number of points in z-direction
        output_file: Optional file path to save the mesh
        
    Returns:
        meshio.Mesh: The generated mesh
    """
    # Create grid points
    x = np.linspace(0, length, nx)
    y = np.linspace(0, height, ny)
    z = np.linspace(0, width, nz)
    
    # Create 3D points
    points = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                points.append([x[i], y[j], z[k]])
    
    points = np.array(points)
    
    # Create hexahedral cells
    cells = []
    for k in range(nz-1):
        for j in range(ny-1):
            for i in range(nx-1):
                idx = k*ny*nx + j*nx + i
                p0 = idx
                p1 = idx + 1
                p2 = idx + nx + 1
                p3 = idx + nx
                p4 = idx + ny*nx
                p5 = idx + ny*nx + 1
                p6 = idx + ny*nx + nx + 1
                p7 = idx + ny*nx + nx
                cells.append([p0, p1, p2, p3, p4, p5, p6, p7])
    
    # Create mesh
    mesh = meshio.Mesh(
        points=points,
        cells=[("hexahedron", np.array(cells))]
    )
    
    # Save mesh if output file is specified
    if output_file:
        meshio.write(output_file, mesh)
    
    return mesh


def generate_cavity_mesh(
    size: float = 1.0,
    n: int = 50,
    output_file: Optional[str] = None
) -> meshio.Mesh:
    """Generate a 3D mesh for a cubic cavity.
    
    Args:
        size: Size of the cubic cavity
        n: Number of points in each direction
        output_file: Optional file path to save the mesh
        
    Returns:
        meshio.Mesh: The generated mesh
    """
    # Create grid points
    x = np.linspace(0, size, n)
    y = np.linspace(0, size, n)
    z = np.linspace(0, size, n)
    
    # Create 3D points
    points = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                points.append([x[i], y[j], z[k]])
    
    points = np.array(points)
    
    # Create hexahedral cells
    cells = []
    for k in range(n-1):
        for j in range(n-1):
            for i in range(n-1):
                idx = k*n*n + j*n + i
                p0 = idx
                p1 = idx + 1
                p2 = idx + n + 1
                p3 = idx + n
                p4 = idx + n*n
                p5 = idx + n*n + 1
                p6 = idx + n*n + n + 1
                p7 = idx + n*n + n
                cells.append([p0, p1, p2, p3, p4, p5, p6, p7])
    
    # Create mesh
    mesh = meshio.Mesh(
        points=points,
        cells=[("hexahedron", np.array(cells))]
    )
    
    # Save mesh if output file is specified
    if output_file:
        meshio.write(output_file, mesh)
    
    return mesh


def generate_cylinder_mesh(
    radius: float = 0.5,
    length: float = 10.0,
    n_radial: int = 20,
    n_circumferential: int = 40,
    n_axial: int = 50,
    output_file: Optional[str] = None
) -> meshio.Mesh:
    """Generate a 3D mesh for flow around a cylinder.
    
    Args:
        radius: Radius of the cylinder
        length: Length of the cylinder
        n_radial: Number of points in radial direction
        n_circumferential: Number of points around circumference
        n_axial: Number of points along cylinder length
        output_file: Optional file path to save the mesh
        
    Returns:
        meshio.Mesh: The generated mesh
    """
    # Create cylindrical coordinates
    r = np.linspace(radius, 5*radius, n_radial)  # Extend to 5x radius
    theta = np.linspace(0, 2*np.pi, n_circumferential, endpoint=False)
    z = np.linspace(0, length, n_axial)
    
    # Create 3D points
    points = []
    for k in range(n_axial):
        for j in range(n_circumferential):
            for i in range(n_radial):
                x = r[i] * np.cos(theta[j])
                y = r[i] * np.sin(theta[j])
                z_coord = z[k]
                points.append([x, y, z_coord])
    
    points = np.array(points)
    
    # Create hexahedral cells
    cells = []
    for k in range(n_axial-1):
        for j in range(n_circumferential-1):
            for i in range(n_radial-1):
                idx = k*n_radial*n_circumferential + j*n_radial + i
                p0 = idx
                p1 = idx + 1
                p2 = idx + n_radial + 1
                p3 = idx + n_radial
                p4 = idx + n_radial*n_circumferential
                p5 = idx + n_radial*n_circumferential + 1
                p6 = idx + n_radial*n_circumferential + n_radial + 1
                p7 = idx + n_radial*n_circumferential + n_radial
                cells.append([p0, p1, p2, p3, p4, p5, p6, p7])
    
    # Create mesh
    mesh = meshio.Mesh(
        points=points,
        cells=[("hexahedron", np.array(cells))]
    )
    
    # Save mesh if output file is specified
    if output_file:
        meshio.write(output_file, mesh)
    
    return mesh 
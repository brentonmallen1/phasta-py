"""Mesh generation and optimization module.

This module provides tools for generating and optimizing computational meshes,
including structured and unstructured mesh generation, mesh quality metrics,
and optimization algorithms.
"""

import numpy as np
import meshio
from typing import Tuple, List, Optional, Dict, Union
import os
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


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


class MeshGenerator:
    """Base class for mesh generators."""
    
    def __init__(self, dim: int = 3):
        """Initialize mesh generator.
        
        Args:
            dim: Mesh dimension (2 or 3)
        """
        self.dim = dim
        if dim not in (2, 3):
            raise ValueError("Dimension must be 2 or 3")
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh.
        
        Returns:
            Tuple of (nodes, elements)
        """
        raise NotImplementedError


class StructuredMeshGenerator(MeshGenerator):
    """Structured mesh generator."""
    
    def __init__(self, bounds: Tuple[Tuple[float, float], ...], 
                 n_points: Tuple[int, ...], dim: int = 3):
        """Initialize structured mesh generator.
        
        Args:
            bounds: Tuple of (min, max) for each dimension
            n_points: Number of points in each dimension
            dim: Mesh dimension (2 or 3)
        """
        super().__init__(dim)
        self.bounds = bounds
        self.n_points = n_points
        
        if len(bounds) != dim or len(n_points) != dim:
            raise ValueError("Bounds and n_points must match dimension")
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate structured mesh.
        
        Returns:
            Tuple of (nodes, elements)
        """
        # Generate node coordinates
        coords = [np.linspace(b[0], b[1], n) for b, n in zip(self.bounds, self.n_points)]
        nodes = np.array(np.meshgrid(*coords, indexing='ij')).reshape(self.dim, -1).T
        
        # Generate element connectivity
        if self.dim == 2:
            elements = self._generate_2d_elements()
        else:
            elements = self._generate_3d_elements()
        
        return nodes, elements
    
    def _generate_2d_elements(self) -> np.ndarray:
        """Generate 2D element connectivity.
        
        Returns:
            Array of element connectivity
        """
        nx, ny = self.n_points
        elements = []
        
        for i in range(nx - 1):
            for j in range(ny - 1):
                # Create two triangles for each cell
                n1 = i * ny + j
                n2 = n1 + 1
                n3 = (i + 1) * ny + j
                n4 = n3 + 1
                
                elements.extend([[n1, n2, n3], [n2, n4, n3]])
        
        return np.array(elements)
    
    def _generate_3d_elements(self) -> np.ndarray:
        """Generate 3D element connectivity.
        
        Returns:
            Array of element connectivity
        """
        nx, ny, nz = self.n_points
        elements = []
        
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    # Create 6 tetrahedra for each cell
                    n1 = i * ny * nz + j * nz + k
                    n2 = n1 + 1
                    n3 = n1 + nz
                    n4 = n3 + 1
                    n5 = n1 + ny * nz
                    n6 = n5 + 1
                    n7 = n5 + nz
                    n8 = n7 + 1
                    
                    # Split hex into 6 tets
                    elements.extend([
                        [n1, n2, n3, n5],
                        [n2, n4, n3, n6],
                        [n3, n4, n7, n6],
                        [n1, n3, n5, n7],
                        [n2, n6, n3, n7],
                        [n1, n2, n3, n7]
                    ])


class UnstructuredMeshGenerator(MeshGenerator):
    """Unstructured mesh generator using Delaunay triangulation."""
    
    def __init__(self, points: np.ndarray, dim: int = 3):
        """Initialize unstructured mesh generator.
        
        Args:
            points: Initial point cloud
            dim: Mesh dimension (2 or 3)
        """
        super().__init__(dim)
        self.points = points
        
        if points.shape[1] != dim:
            raise ValueError("Points dimension must match mesh dimension")
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate unstructured mesh.
        
        Returns:
            Tuple of (nodes, elements)
        """
        # Perform Delaunay triangulation
        tri = Delaunay(self.points)
        elements = tri.simplices
        
        return self.points, elements


class MeshOptimizer:
    """Mesh optimization and quality control."""
    
    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        """Initialize mesh optimizer.
        
        Args:
            nodes: Node coordinates
            elements: Element connectivity
        """
        self.nodes = nodes
        self.elements = elements
        self.dim = nodes.shape[1]
    
    def optimize(self, max_iter: int = 100, quality_threshold: float = 0.3) -> np.ndarray:
        """Optimize mesh quality.
        
        Args:
            max_iter: Maximum number of iterations
            quality_threshold: Minimum acceptable element quality
            
        Returns:
            Optimized node coordinates
        """
        # Initialize optimization
        n_nodes = len(self.nodes)
        x0 = self.nodes.flatten()
        
        # Define objective function
        def objective(x):
            nodes = x.reshape(n_nodes, self.dim)
            quality = self._compute_mesh_quality(nodes)
            return -np.mean(quality)  # Negative because we want to maximize
        
        # Define constraints
        constraints = []
        for i, element in enumerate(self.elements):
            def element_quality(x, i=i):
                nodes = x.reshape(n_nodes, self.dim)
                return self._compute_element_quality(nodes, element) - quality_threshold
            constraints.append({'type': 'ineq', 'fun': element_quality})
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': max_iter}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        return result.x.reshape(n_nodes, self.dim)
    
    def _compute_mesh_quality(self, nodes: np.ndarray) -> np.ndarray:
        """Compute quality metrics for all elements.
        
        Args:
            nodes: Node coordinates
            
        Returns:
            Array of element qualities
        """
        qualities = np.zeros(len(self.elements))
        for i, element in enumerate(self.elements):
            qualities[i] = self._compute_element_quality(nodes, element)
        return qualities
    
    def _compute_element_quality(self, nodes: np.ndarray, element: np.ndarray) -> float:
        """Compute quality metric for a single element.
        
        Args:
            nodes: Node coordinates
            element: Element connectivity
            
        Returns:
            Element quality metric
        """
        element_nodes = nodes[element]
        
        if len(element) == 3:  # Triangle
            return self._triangle_quality(element_nodes)
        elif len(element) == 4:  # Tetrahedron
            return self._tetrahedron_quality(element_nodes)
        else:
            raise ValueError(f"Unsupported element type with {len(element)} nodes")
    
    def _triangle_quality(self, nodes: np.ndarray) -> float:
        """Compute triangle quality metric.
        
        Args:
            nodes: Triangle node coordinates
            
        Returns:
            Triangle quality metric
        """
        # Compute edge lengths
        edges = np.diff(nodes, axis=0)
        edge_lengths = np.sqrt(np.sum(edges**2, axis=1))
        
        # Compute area
        area = 0.5 * np.abs(np.cross(edges[0], edges[1]))
        
        # Compute quality metric (ratio of inscribed to circumscribed circle)
        if area > 0:
            return 4 * np.sqrt(3) * area / np.sum(edge_lengths**2)
        return 0.0
    
    def _tetrahedron_quality(self, nodes: np.ndarray) -> float:
        """Compute tetrahedron quality metric.
        
        Args:
            nodes: Tetrahedron node coordinates
            
        Returns:
            Tetrahedron quality metric
        """
        # Compute edge vectors
        edges = np.diff(nodes, axis=0)
        
        # Compute volume
        volume = np.abs(np.dot(edges[0], np.cross(edges[1], edges[2]))) / 6.0
        
        # Compute surface area
        face_areas = [
            np.linalg.norm(np.cross(edges[0], edges[1])),
            np.linalg.norm(np.cross(edges[1], edges[2])),
            np.linalg.norm(np.cross(edges[2], edges[0])),
            np.linalg.norm(np.cross(edges[0] - edges[2], edges[1] - edges[2]))
        ]
        surface_area = 0.5 * sum(face_areas)
        
        # Compute quality metric (ratio of inscribed to circumscribed sphere)
        if volume > 0 and surface_area > 0:
            return 6 * np.sqrt(6) * volume / (surface_area**1.5)
        return 0.0 
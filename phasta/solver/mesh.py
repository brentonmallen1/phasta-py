"""Advanced mesh generation module.

This module provides functionality for:
- Multi-resolution mesh generation
- Point cloud mesh generation
- Mesh optimization
- Mesh quality metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod
from scipy.spatial import Delaunay, KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

logger = logging.getLogger(__name__)


class MeshQuality(ABC):
    """Base class for mesh quality metrics."""
    
    def __init__(self):
        """Initialize mesh quality metrics."""
        pass
    
    @abstractmethod
    def compute_quality(self, 
                       vertices: np.ndarray,
                       elements: np.ndarray) -> float:
        """Compute mesh quality metric.
        
        Args:
            vertices: Array of vertex coordinates
            elements: Array of element connectivity
            
        Returns:
            Quality metric value
        """
        pass


class AspectRatio(MeshQuality):
    """Aspect ratio quality metric."""
    
    def compute_quality(self, 
                       vertices: np.ndarray,
                       elements: np.ndarray) -> float:
        """Compute aspect ratio quality metric.
        
        Args:
            vertices: Array of vertex coordinates
            elements: Array of element connectivity
            
        Returns:
            Average aspect ratio
        """
        quality = 0.0
        n_elements = len(elements)
        
        for element in elements:
            # Get element vertices
            element_vertices = vertices[element]
            
            # Compute edge lengths
            edges = []
            for i in range(len(element)):
                j = (i + 1) % len(element)
                edge = element_vertices[j] - element_vertices[i]
                edges.append(np.linalg.norm(edge))
            
            # Compute aspect ratio
            max_edge = max(edges)
            min_edge = min(edges)
            if min_edge > 0:
                quality += max_edge / min_edge
        
        return quality / n_elements


class MultiResolutionMesh:
    """Multi-resolution mesh generator."""
    
    def __init__(self,
                 base_resolution: float = 1.0,
                 refinement_levels: int = 3,
                 quality_threshold: float = 2.0):
        """Initialize multi-resolution mesh generator.
        
        Args:
            base_resolution: Base mesh resolution
            refinement_levels: Number of refinement levels
            quality_threshold: Quality threshold for refinement
        """
        self.base_resolution = base_resolution
        self.refinement_levels = refinement_levels
        self.quality_threshold = quality_threshold
        self.quality_metric = AspectRatio()
    
    def generate_mesh(self,
                     domain: np.ndarray,
                     feature_points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate multi-resolution mesh.
        
        Args:
            domain: Domain bounds [xmin, ymin, zmin, xmax, ymax, zmax]
            feature_points: Optional feature points for refinement
            
        Returns:
            Tuple of (vertices, elements)
        """
        # Generate base mesh
        vertices, elements = self._generate_base_mesh(domain)
        
        # Refine mesh
        for level in range(self.refinement_levels):
            vertices, elements = self._refine_mesh(vertices, elements, feature_points)
        
        return vertices, elements
    
    def _generate_base_mesh(self,
                           domain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base mesh.
        
        Args:
            domain: Domain bounds
            
        Returns:
            Tuple of (vertices, elements)
        """
        # Create regular grid
        x = np.arange(domain[0], domain[3], self.base_resolution)
        y = np.arange(domain[1], domain[4], self.base_resolution)
        z = np.arange(domain[2], domain[5], self.base_resolution)
        
        # Create vertices
        vertices = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        
        # Create elements using Delaunay triangulation
        tri = Delaunay(vertices)
        elements = tri.simplices
        
        return vertices, elements
    
    def _refine_mesh(self,
                     vertices: np.ndarray,
                     elements: np.ndarray,
                     feature_points: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Refine mesh based on quality and features.
        
        Args:
            vertices: Current vertices
            elements: Current elements
            feature_points: Feature points for refinement
            
        Returns:
            Tuple of (refined vertices, refined elements)
        """
        # Compute element quality
        quality = self.quality_metric.compute_quality(vertices, elements)
        
        # Find elements to refine
        refine_elements = []
        for i, element in enumerate(elements):
            element_quality = self.quality_metric.compute_quality(
                vertices, np.array([element]))
            
            # Refine based on quality
            if element_quality > self.quality_threshold:
                refine_elements.append(i)
            
            # Refine based on features
            if feature_points is not None:
                element_center = np.mean(vertices[element], axis=0)
                distances = np.linalg.norm(feature_points - element_center, axis=1)
                if np.min(distances) < self.base_resolution:
                    refine_elements.append(i)
        
        # Refine elements
        new_vertices = vertices.copy()
        new_elements = elements.copy()
        
        for element_idx in refine_elements:
            element = elements[element_idx]
            
            # Add new vertex at element center
            center = np.mean(vertices[element], axis=0)
            new_vertices = np.vstack((new_vertices, center))
            center_idx = len(new_vertices) - 1
            
            # Create new elements
            new_element = np.array([
                [element[0], element[1], center_idx],
                [element[1], element[2], center_idx],
                [element[2], element[0], center_idx]
            ])
            
            # Replace old element with new elements
            new_elements = np.vstack((
                new_elements[:element_idx],
                new_elements[element_idx+1:],
                new_element
            ))
        
        return new_vertices, new_elements


class PointCloudMesh:
    """Point cloud mesh generator."""
    
    def __init__(self,
                 max_distance: float = 1.0,
                 min_angle: float = 20.0):
        """Initialize point cloud mesh generator.
        
        Args:
            max_distance: Maximum distance for edge creation
            min_angle: Minimum angle for triangle creation
        """
        self.max_distance = max_distance
        self.min_angle = min_angle
    
    def generate_mesh(self,
                     points: np.ndarray,
                     normals: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh from point cloud.
        
        Args:
            points: Point cloud coordinates
            normals: Optional point normals
            
        Returns:
            Tuple of (vertices, elements)
        """
        # Build KD-tree for nearest neighbor search
        tree = KDTree(points)
        
        # Find neighbors within max_distance
        neighbors = tree.query_ball_point(points, self.max_distance)
        
        # Create edges
        edges = []
        for i, neighbors_i in enumerate(neighbors):
            for j in neighbors_i:
                if i < j:  # Avoid duplicate edges
                    edges.append((i, j))
        
        # Create minimum spanning tree
        n_points = len(points)
        edge_matrix = csr_matrix((np.ones(len(edges)), 
                                ([e[0] for e in edges], [e[1] for e in edges])),
                               shape=(n_points, n_points))
        mst = minimum_spanning_tree(edge_matrix)
        
        # Create triangles
        elements = []
        for i in range(n_points):
            # Get neighbors in MST
            neighbors = mst[i].nonzero()[1]
            
            # Create triangles with neighbors
            for j in range(len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    # Check if triangle is valid
                    if self._is_valid_triangle(points[i], 
                                             points[neighbors[j]],
                                             points[neighbors[k]],
                                             normals):
                        elements.append([i, neighbors[j], neighbors[k]])
        
        return points, np.array(elements)
    
    def _is_valid_triangle(self,
                          p1: np.ndarray,
                          p2: np.ndarray,
                          p3: np.ndarray,
                          normals: Optional[np.ndarray]) -> bool:
        """Check if triangle is valid.
        
        Args:
            p1, p2, p3: Triangle vertices
            normals: Optional vertex normals
            
        Returns:
            True if triangle is valid
        """
        # Check angles
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p3 - p2
        
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3 / np.linalg.norm(v3)
        
        # Compute angles
        angle1 = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angle2 = np.arccos(np.clip(np.dot(-v1, v3), -1.0, 1.0))
        angle3 = np.arccos(np.clip(np.dot(-v2, -v3), -1.0, 1.0))
        
        # Check minimum angle
        min_angle = min(angle1, angle2, angle3) * 180 / np.pi
        if min_angle < self.min_angle:
            return False
        
        # Check normals if provided
        if normals is not None:
            # Compute triangle normal
            triangle_normal = np.cross(v1, v2)
            triangle_normal = triangle_normal / np.linalg.norm(triangle_normal)
            
            # Check normal consistency
            for normal in [normals[0], normals[1], normals[2]]:
                if np.dot(triangle_normal, normal) < 0:
                    return False
        
        return True 
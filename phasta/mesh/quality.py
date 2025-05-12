"""Mesh quality and refinement operations.

This module provides tools for mesh quality assessment, smoothing, and coarsening.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class QualityMetric(ABC):
    """Base class for mesh quality metrics."""
    
    @abstractmethod
    def calculate(self, mesh: 'Mesh') -> float:
        """Calculate quality metric for the mesh.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Quality metric value
        """
        pass


class AspectRatioMetric(QualityMetric):
    """Aspect ratio quality metric."""
    
    def calculate(self, mesh: 'Mesh') -> float:
        """Calculate aspect ratio for each element.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Average aspect ratio
        """
        aspect_ratios = []
        for element in mesh.elements:
            nodes = mesh.nodes[element]
            # Calculate edge lengths
            edges = np.array([
                np.linalg.norm(nodes[1] - nodes[0]),
                np.linalg.norm(nodes[2] - nodes[1]),
                np.linalg.norm(nodes[0] - nodes[2])
            ])
            # Aspect ratio is max edge / min edge
            aspect_ratio = np.max(edges) / np.min(edges)
            aspect_ratios.append(aspect_ratio)
        
        return np.mean(aspect_ratios)


class SkewnessMetric(QualityMetric):
    """Skewness quality metric."""
    
    def calculate(self, mesh: 'Mesh') -> float:
        """Calculate skewness for each element.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Average skewness
        """
        skewness = []
        for element in mesh.elements:
            nodes = mesh.nodes[element]
            # Calculate angles
            v1 = nodes[1] - nodes[0]
            v2 = nodes[2] - nodes[0]
            v3 = nodes[2] - nodes[1]
            
            angles = np.array([
                np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))),
                np.arccos(np.dot(-v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))),
                np.arccos(np.dot(-v2, -v3) / (np.linalg.norm(v2) * np.linalg.norm(v3)))
            ])
            
            # Skewness is deviation from ideal angle (60 degrees)
            ideal_angle = np.pi / 3
            element_skewness = np.max(np.abs(angles - ideal_angle)) / ideal_angle
            skewness.append(element_skewness)
        
        return np.mean(skewness)


class MeshSmoother:
    """Mesh smoothing operations."""
    
    def __init__(self, max_iterations: int = 100,
                 tolerance: float = 1e-6):
        """Initialize mesh smoother.
        
        Args:
            max_iterations: Maximum number of smoothing iterations
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def smooth(self, mesh: 'Mesh', fixed_nodes: Optional[List[int]] = None) -> 'Mesh':
        """Smooth mesh using Laplacian smoothing.
        
        Args:
            mesh: Input mesh
            fixed_nodes: List of node indices to keep fixed
            
        Returns:
            Smoothed mesh
        """
        if fixed_nodes is None:
            fixed_nodes = []
        
        # Create node-to-element connectivity
        node_to_elements = self._build_node_to_elements(mesh)
        
        # Initialize node positions
        new_nodes = mesh.nodes.copy()
        
        # Perform smoothing iterations
        for _ in range(self.max_iterations):
            old_nodes = new_nodes.copy()
            
            # Update each node position
            for i in range(len(mesh.nodes)):
                if i in fixed_nodes:
                    continue
                
                # Get connected elements
                connected = node_to_elements[i]
                if not connected:
                    continue
                
                # Calculate new position as average of connected nodes
                connected_nodes = set()
                for element in connected:
                    connected_nodes.update(mesh.elements[element])
                connected_nodes.remove(i)
                
                if connected_nodes:
                    new_pos = np.mean(mesh.nodes[list(connected_nodes)], axis=0)
                    new_nodes[i] = new_pos
            
            # Check convergence
            if np.max(np.abs(new_nodes - old_nodes)) < self.tolerance:
                break
        
        # Create new mesh
        from phasta.mesh.base import Mesh
        return Mesh(new_nodes, mesh.elements)
    
    def _build_node_to_elements(self, mesh: 'Mesh') -> Dict[int, List[int]]:
        """Build node-to-element connectivity.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Dictionary mapping node indices to connected element indices
        """
        node_to_elements = {i: [] for i in range(len(mesh.nodes))}
        for i, element in enumerate(mesh.elements):
            for node in element:
                node_to_elements[node].append(i)
        return node_to_elements


class MeshCoarsener:
    """Mesh coarsening operations."""
    
    def __init__(self, target_size: Optional[int] = None,
                 quality_threshold: float = 0.5):
        """Initialize mesh coarsener.
        
        Args:
            target_size: Target number of elements
            quality_threshold: Minimum quality threshold for coarsening
        """
        self.target_size = target_size
        self.quality_threshold = quality_threshold
    
    def coarsen(self, mesh: 'Mesh') -> 'Mesh':
        """Coarsen mesh by collapsing edges.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Coarsened mesh
        """
        # Calculate element quality
        quality_metric = AspectRatioMetric()
        element_quality = self._calculate_element_quality(mesh, quality_metric)
        
        # Find edges to collapse
        edges_to_collapse = self._find_edges_to_collapse(mesh, element_quality)
        
        # Collapse edges
        new_nodes, new_elements = self._collapse_edges(
            mesh.nodes, mesh.elements, edges_to_collapse)
        
        # Create new mesh
        from phasta.mesh.base import Mesh
        return Mesh(new_nodes, new_elements)
    
    def _calculate_element_quality(self, mesh: 'Mesh',
                                 metric: QualityMetric) -> np.ndarray:
        """Calculate quality for each element.
        
        Args:
            mesh: Input mesh
            metric: Quality metric
            
        Returns:
            Array of element qualities
        """
        qualities = []
        for element in mesh.elements:
            element_mesh = Mesh(mesh.nodes[element], np.array([[0, 1, 2]]))
            quality = metric.calculate(element_mesh)
            qualities.append(quality)
        return np.array(qualities)
    
    def _find_edges_to_collapse(self, mesh: 'Mesh',
                               element_quality: np.ndarray) -> List[Tuple[int, int]]:
        """Find edges that can be collapsed.
        
        Args:
            mesh: Input mesh
            element_quality: Element quality values
            
        Returns:
            List of edge tuples (node1, node2) to collapse
        """
        edges = []
        for i, element in enumerate(mesh.elements):
            if element_quality[i] < self.quality_threshold:
                continue
            
            # Add edges to list
            edges.extend([
                (element[0], element[1]),
                (element[1], element[2]),
                (element[2], element[0])
            ])
        
        # Remove duplicates and sort
        edges = list(set(tuple(sorted(edge)) for edge in edges))
        return edges
    
    def _collapse_edges(self, nodes: np.ndarray, elements: np.ndarray,
                       edges: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Collapse edges in the mesh.
        
        Args:
            nodes: Node coordinates
            elements: Element connectivity
            edges: List of edges to collapse
            
        Returns:
            Tuple of (new_nodes, new_elements)
        """
        # Create node mapping
        node_map = {i: i for i in range(len(nodes))}
        for edge in edges:
            node_map[edge[1]] = edge[0]
        
        # Update elements
        new_elements = []
        for element in elements:
            new_element = [node_map[i] for i in element]
            if len(set(new_element)) == 3:  # Check if element is still valid
                new_elements.append(new_element)
        
        # Remove unused nodes
        used_nodes = set()
        for element in new_elements:
            used_nodes.update(element)
        
        # Create new node array
        new_nodes = nodes[list(used_nodes)]
        
        # Update element indices
        node_index_map = {old: new for new, old in enumerate(used_nodes)}
        new_elements = [[node_index_map[i] for i in element]
                       for element in new_elements]
        
        return new_nodes, np.array(new_elements)


class AdaptiveRefiner:
    """Adaptive mesh refinement operations."""
    
    def __init__(self, error_threshold: float = 0.1,
                 max_refinements: int = 5):
        """Initialize adaptive refiner.
        
        Args:
            error_threshold: Error threshold for refinement
            max_refinements: Maximum number of refinement iterations
        """
        self.error_threshold = error_threshold
        self.max_refinements = max_refinements
    
    def refine(self, mesh: 'Mesh', error_indicator: np.ndarray) -> 'Mesh':
        """Refine mesh based on error indicator.
        
        Args:
            mesh: Input mesh
            error_indicator: Error indicator for each element
            
        Returns:
            Refined mesh
        """
        # Find elements to refine
        elements_to_refine = np.where(error_indicator > self.error_threshold)[0]
        
        if len(elements_to_refine) == 0:
            return mesh
        
        # Refine elements
        new_nodes, new_elements = self._refine_elements(
            mesh.nodes, mesh.elements, elements_to_refine)
        
        # Create new mesh
        from phasta.mesh.base import Mesh
        return Mesh(new_nodes, new_elements)
    
    def _refine_elements(self, nodes: np.ndarray, elements: np.ndarray,
                        elements_to_refine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Refine selected elements.
        
        Args:
            nodes: Node coordinates
            elements: Element connectivity
            elements_to_refine: Indices of elements to refine
            
        Returns:
            Tuple of (new_nodes, new_elements)
        """
        # Initialize new arrays
        new_nodes = list(nodes)
        new_elements = []
        
        # Refine each element
        for i, element in enumerate(elements):
            if i in elements_to_refine:
                # Add new nodes at edge midpoints
                midpoints = []
                for j in range(3):
                    n1, n2 = element[j], element[(j + 1) % 3]
                    midpoint = (nodes[n1] + nodes[n2]) / 2
                    midpoints.append(len(new_nodes))
                    new_nodes.append(midpoint)
                
                # Create new elements
                new_elements.extend([
                    [element[0], midpoints[0], midpoints[2]],
                    [midpoints[0], element[1], midpoints[1]],
                    [midpoints[2], midpoints[1], element[2]],
                    [midpoints[0], midpoints[1], midpoints[2]]
                ])
            else:
                new_elements.append(element)
        
        return np.array(new_nodes), np.array(new_elements) 
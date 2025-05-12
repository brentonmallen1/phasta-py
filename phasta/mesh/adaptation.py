"""Mesh adaptation module.

This module provides tools for mesh adaptation based on error estimates,
feature detection, and solution characteristics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from scipy.spatial import KDTree
import logging

if TYPE_CHECKING:
    from phasta.mesh.base import Mesh

logger = logging.getLogger(__name__)


class MeshAdapter:
    """Base class for mesh adaptation algorithms."""
    
    def __init__(self, mesh: 'Mesh', max_iterations: int = 10,
                 target_error: float = 0.1):
        """Initialize mesh adapter.
        
        Args:
            mesh: Mesh to adapt
            max_iterations: Maximum number of adaptation iterations
            target_error: Target error level
        """
        self.mesh = mesh
        self.max_iterations = max_iterations
        self.target_error = target_error
    
    def adapt(self) -> bool:
        """Adapt mesh.
        
        Returns:
            True if adaptation converged
        """
        raise NotImplementedError


class ErrorBasedAdapter(MeshAdapter):
    """Error-based mesh adaptation."""
    
    def __init__(self, mesh: 'Mesh', solution: np.ndarray,
                 max_iterations: int = 10, target_error: float = 0.1):
        """Initialize error-based adapter.
        
        Args:
            mesh: Mesh to adapt
            solution: Solution field
            max_iterations: Maximum number of adaptation iterations
            target_error: Target error level
        """
        super().__init__(mesh, max_iterations, target_error)
        self.solution = solution
    
    def adapt(self) -> bool:
        """Adapt mesh based on error estimates.
        
        Returns:
            True if adaptation converged
        """
        for iteration in range(self.max_iterations):
            # Compute error estimates
            errors = self._compute_error_estimates()
            
            # Check convergence
            if np.max(errors) < self.target_error:
                return True
            
            # Mark elements for refinement
            to_refine = errors > self.target_error
            
            # Refine elements
            if not self._refine_elements(to_refine):
                return False
        
        return False
    
    def _compute_error_estimates(self) -> np.ndarray:
        """Compute error estimates for each element.
        
        Returns:
            Array of error estimates
        """
        errors = np.zeros(len(self.mesh.elements))
        
        for i, element in enumerate(self.mesh.elements):
            nodes = self.mesh.nodes[element]
            values = self.solution[element]
            
            # Compute gradient
            if self.mesh.dim == 2:
                edges = np.diff(nodes, axis=0)
                edge_lengths = np.linalg.norm(edges, axis=1)
                value_diffs = np.diff(values)
                errors[i] = np.max(np.abs(value_diffs / edge_lengths))
            else:
                # For 3D, compute gradient using finite differences
                dx = np.diff(nodes[:, 0])
                dy = np.diff(nodes[:, 1])
                dz = np.diff(nodes[:, 2])
                dv = np.diff(values)
                errors[i] = np.max(np.abs(dv / np.sqrt(dx**2 + dy**2 + dz**2)))
        
        return errors
    
    def _refine_elements(self, to_refine: np.ndarray) -> bool:
        """Refine selected elements.
        
        Args:
            to_refine: Boolean array indicating elements to refine
            
        Returns:
            True if refinement was successful
        """
        # Get elements to refine
        elements = self.mesh.elements[to_refine]
        
        # Create new nodes
        new_nodes = []
        for element in elements:
            # Compute new node positions
            nodes = self.mesh.nodes[element]
            new_node = nodes.mean(axis=0)
            new_nodes.append(new_node)
        
        # Add new nodes to mesh
        n_old_nodes = len(self.mesh.nodes)
        self.mesh.nodes = np.vstack([self.mesh.nodes, new_nodes])
        
        # Create new elements
        new_elements = []
        for i, element in enumerate(elements):
            new_node = n_old_nodes + i
            # Split element into smaller elements
            for j in range(len(element)):
                new_elements.append([
                    element[j],
                    element[(j + 1) % len(element)],
                    new_node
                ])
        
        # Update mesh elements
        self.mesh.elements = np.vstack([
            self.mesh.elements[~to_refine],
            new_elements
        ])
        
        return True


class FeatureBasedAdapter(MeshAdapter):
    """Feature-based mesh adaptation."""
    
    def __init__(self, mesh: 'Mesh', solution: np.ndarray,
                 max_iterations: int = 10, target_error: float = 0.1,
                 feature_threshold: float = 0.5):
        """Initialize feature-based adapter.
        
        Args:
            mesh: Mesh to adapt
            solution: Solution field
            max_iterations: Maximum number of adaptation iterations
            target_error: Target error level
            feature_threshold: Threshold for feature detection
        """
        super().__init__(mesh, max_iterations, target_error)
        self.solution = solution
        self.feature_threshold = feature_threshold
    
    def adapt(self) -> bool:
        """Adapt mesh based on feature detection.
        
        Returns:
            True if adaptation converged
        """
        for iteration in range(self.max_iterations):
            # Detect features
            features = self._detect_features()
            
            # Check convergence
            if not np.any(features):
                return True
            
            # Mark elements for refinement
            to_refine = features
            
            # Refine elements
            if not self._refine_elements(to_refine):
                return False
        
        return False
    
    def _detect_features(self) -> np.ndarray:
        """Detect features in the solution.
        
        Returns:
            Boolean array indicating elements containing features
        """
        features = np.zeros(len(self.mesh.elements), dtype=bool)
        
        for i, element in enumerate(self.mesh.elements):
            nodes = self.mesh.nodes[element]
            values = self.solution[element]
            
            # Compute gradient magnitude
            if self.mesh.dim == 2:
                edges = np.diff(nodes, axis=0)
                edge_lengths = np.linalg.norm(edges, axis=1)
                value_diffs = np.diff(values)
                grad_mag = np.max(np.abs(value_diffs / edge_lengths))
            else:
                # For 3D, compute gradient using finite differences
                dx = np.diff(nodes[:, 0])
                dy = np.diff(nodes[:, 1])
                dz = np.diff(nodes[:, 2])
                dv = np.diff(values)
                grad_mag = np.max(np.abs(dv / np.sqrt(dx**2 + dy**2 + dz**2)))
            
            # Check if element contains a feature
            features[i] = grad_mag > self.feature_threshold
        
        return features
    
    def _refine_elements(self, to_refine: np.ndarray) -> bool:
        """Refine selected elements.
        
        Args:
            to_refine: Boolean array indicating elements to refine
            
        Returns:
            True if refinement was successful
        """
        # Implementation similar to ErrorBasedAdapter._refine_elements
        # Get elements to refine
        elements = self.mesh.elements[to_refine]
        
        # Create new nodes
        new_nodes = []
        for element in elements:
            # Compute new node positions
            nodes = self.mesh.nodes[element]
            new_node = nodes.mean(axis=0)
            new_nodes.append(new_node)
        
        # Add new nodes to mesh
        n_old_nodes = len(self.mesh.nodes)
        self.mesh.nodes = np.vstack([self.mesh.nodes, new_nodes])
        
        # Create new elements
        new_elements = []
        for i, element in enumerate(elements):
            new_node = n_old_nodes + i
            # Split element into smaller elements
            for j in range(len(element)):
                new_elements.append([
                    element[j],
                    element[(j + 1) % len(element)],
                    new_node
                ])
        
        # Update mesh elements
        self.mesh.elements = np.vstack([
            self.mesh.elements[~to_refine],
            new_elements
        ])
        
        return True


class SolutionBasedAdapter(MeshAdapter):
    """Solution-based mesh adaptation."""
    
    def __init__(self, mesh: 'Mesh', solution: np.ndarray,
                 max_iterations: int = 10, target_error: float = 0.1,
                 solution_threshold: float = 0.5):
        """Initialize solution-based adapter.
        
        Args:
            mesh: Mesh to adapt
            solution: Solution field
            max_iterations: Maximum number of adaptation iterations
            target_error: Target error level
            solution_threshold: Threshold for solution variation
        """
        super().__init__(mesh, max_iterations, target_error)
        self.solution = solution
        self.solution_threshold = solution_threshold
    
    def adapt(self) -> bool:
        """Adapt mesh based on solution characteristics.
        
        Returns:
            True if adaptation converged
        """
        for iteration in range(self.max_iterations):
            # Analyze solution
            variations = self._analyze_solution()
            
            # Check convergence
            if np.max(variations) < self.solution_threshold:
                return True
            
            # Mark elements for refinement
            to_refine = variations > self.solution_threshold
            
            # Refine elements
            if not self._refine_elements(to_refine):
                return False
        
        return False
    
    def _analyze_solution(self) -> np.ndarray:
        """Analyze solution variations.
        
        Returns:
            Array of solution variations
        """
        variations = np.zeros(len(self.mesh.elements))
        
        for i, element in enumerate(self.mesh.elements):
            values = self.solution[element]
            
            # Compute solution variation
            if self.mesh.dim == 2:
                # For 2D, use max-min difference
                variations[i] = np.max(values) - np.min(values)
            else:
                # For 3D, use standard deviation
                variations[i] = np.std(values)
        
        return variations
    
    def _refine_elements(self, to_refine: np.ndarray) -> bool:
        """Refine selected elements.
        
        Args:
            to_refine: Boolean array indicating elements to refine
            
        Returns:
            True if refinement was successful
        """
        # Implementation similar to ErrorBasedAdapter._refine_elements
        # Get elements to refine
        elements = self.mesh.elements[to_refine]
        
        # Create new nodes
        new_nodes = []
        for element in elements:
            # Compute new node positions
            nodes = self.mesh.nodes[element]
            new_node = nodes.mean(axis=0)
            new_nodes.append(new_node)
        
        # Add new nodes to mesh
        n_old_nodes = len(self.mesh.nodes)
        self.mesh.nodes = np.vstack([self.mesh.nodes, new_nodes])
        
        # Create new elements
        new_elements = []
        for i, element in enumerate(elements):
            new_node = n_old_nodes + i
            # Split element into smaller elements
            for j in range(len(element)):
                new_elements.append([
                    element[j],
                    element[(j + 1) % len(element)],
                    new_node
                ])
        
        # Update mesh elements
        self.mesh.elements = np.vstack([
            self.mesh.elements[~to_refine],
            new_elements
        ])
        
        return True 
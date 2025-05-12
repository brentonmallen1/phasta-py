"""IGES file support module.

This module provides tools for reading and processing IGES files, including
geometry extraction and mesh generation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import logging
from pathlib import Path
import platform
import os
import re

if TYPE_CHECKING:
    from phasta.mesh.base import Mesh
    from phasta.mesh.cad import CADMeshGenerator

logger = logging.getLogger(__name__)


class IGESReader:
    """IGES file reader and parser."""
    
    def __init__(self):
        """Initialize IGES reader."""
        self.entities = {}
        self.parameters = {}
        self.directory = {}
        self.start = {}
        self.global_section = {}
        self.terminate = {}
    
    def read_file(self, file_path: Union[str, Path]):
        """Read IGES file.
        
        Args:
            file_path: Path to IGES file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"IGES file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            self._parse_file(f)
    
    def _parse_file(self, file_handle):
        """Parse IGES file.
        
        Args:
            file_handle: File handle
        """
        # Read all lines
        lines = file_handle.readlines()
        
        # Parse each section
        self._parse_start_section(lines)
        self._parse_global_section(lines)
        self._parse_directory_section(lines)
        self._parse_parameter_section(lines)
        self._parse_terminate_section(lines)
    
    def _parse_start_section(self, lines: List[str]):
        """Parse start section of IGES file.
        
        Args:
            lines: List of file lines
        """
        start_lines = [line for line in lines if line[72] == 'S']
        for line in start_lines:
            sequence_number = int(line[73:80])
            content = line[0:72].rstrip()
            self.start[sequence_number] = content
    
    def _parse_global_section(self, lines: List[str]):
        """Parse global section of IGES file.
        
        Args:
            lines: List of file lines
        """
        global_lines = [line for line in lines if line[72] == 'G']
        if not global_lines:
            raise ValueError("No global section found in IGES file")
        
        # Parse global parameters
        global_data = global_lines[0][0:72].split(';')
        for param in global_data:
            if not param.strip():
                continue
            key, value = param.split(':', 1)
            self.global_section[key.strip()] = value.strip()
    
    def _parse_directory_section(self, lines: List[str]):
        """Parse directory section of IGES file.
        
        Args:
            lines: List of file lines
        """
        dir_lines = [line for line in lines if line[72] == 'D']
        for i in range(0, len(dir_lines), 2):
            if i + 1 >= len(dir_lines):
                break
                
            # Parse directory entry
            entry = dir_lines[i][0:72] + dir_lines[i+1][0:72]
            sequence_number = int(entry[0:8])
            
            # Extract entity information
            entity_type = int(entry[8:16])
            parameter_pointer = int(entry[16:24])
            structure = int(entry[24:32])
            line_font = int(entry[32:40])
            level = int(entry[40:48])
            view = int(entry[48:56])
            transform = int(entry[56:64])
            label = entry[64:72].strip()
            
            self.directory[sequence_number] = {
                'type': entity_type,
                'parameter_pointer': parameter_pointer,
                'structure': structure,
                'line_font': line_font,
                'level': level,
                'view': view,
                'transform': transform,
                'label': label
            }
    
    def _parse_parameter_section(self, lines: List[str]):
        """Parse parameter section of IGES file.
        
        Args:
            lines: List of file lines
        """
        param_lines = [line for line in lines if line[72] == 'P']
        current_entity = None
        current_params = []
        
        for line in param_lines:
            sequence_number = int(line[73:80])
            content = line[0:72].rstrip()
            
            # Check if this is a new entity
            if content.endswith(';'):
                if current_entity is not None:
                    self.parameters[current_entity] = current_params
                current_params = []
                current_entity = sequence_number
            
            # Add parameter data
            current_params.append(content)
        
        # Add last entity
        if current_entity is not None:
            self.parameters[current_entity] = current_params
    
    def _parse_terminate_section(self, lines: List[str]):
        """Parse terminate section of IGES file.
        
        Args:
            lines: List of file lines
        """
        term_lines = [line for line in lines if line[72] == 'T']
        if not term_lines:
            raise ValueError("No terminate section found in IGES file")
        
        # Parse terminate section
        term_data = term_lines[0][0:72].split()
        self.terminate = {
            'start_section': int(term_data[0]),
            'global_section': int(term_data[1]),
            'directory_section': int(term_data[2]),
            'parameter_section': int(term_data[3]),
            'terminate_section': int(term_data[4])
        }


class IGESGeometry:
    """IGES geometry representation."""
    
    def __init__(self, reader: IGESReader):
        """Initialize IGES geometry.
        
        Args:
            reader: IGES reader
        """
        self.reader = reader
        self.curves = {}
        self.surfaces = {}
        self.solids = {}
        self._extract_geometry()
    
    def _extract_geometry(self):
        """Extract geometry from IGES file."""
        # Extract curves
        self._extract_curves()
        
        # Extract surfaces
        self._extract_surfaces()
        
        # Extract solids
        self._extract_solids()
    
    def _extract_curves(self):
        """Extract curves from IGES file."""
        for seq_num, entry in self.reader.directory.items():
            if entry['type'] in [100, 102, 110, 112, 114, 116, 126]:
                # Extract curve parameters
                params = self._parse_parameters(seq_num)
                if params:
                    self.curves[seq_num] = {
                        'type': entry['type'],
                        'parameters': params
                    }
    
    def _extract_surfaces(self):
        """Extract surfaces from IGES file."""
        for seq_num, entry in self.reader.directory.items():
            if entry['type'] in [114, 118, 120, 122, 128, 140, 141, 142, 143, 144]:
                # Extract surface parameters
                params = self._parse_parameters(seq_num)
                if params:
                    self.surfaces[seq_num] = {
                        'type': entry['type'],
                        'parameters': params
                    }
    
    def _extract_solids(self):
        """Extract solids from IGES file."""
        for seq_num, entry in self.reader.directory.items():
            if entry['type'] in [150, 152, 154, 156, 158, 160, 162, 164, 168, 180, 182, 184, 186, 190, 192, 194]:
                # Extract solid parameters
                params = self._parse_parameters(seq_num)
                if params:
                    self.solids[seq_num] = {
                        'type': entry['type'],
                        'parameters': params
                    }
    
    def _parse_parameters(self, sequence_number: int) -> Optional[Dict]:
        """Parse parameters for an entity.
        
        Args:
            sequence_number: Entity sequence number
            
        Returns:
            Parsed parameters or None if not found
        """
        if sequence_number not in self.reader.parameters:
            return None
        
        params = {}
        param_data = ''.join(self.reader.parameters[sequence_number])
        
        # Split parameters
        param_list = param_data.split(',')
        param_list = [p.strip() for p in param_list]
        
        # Remove entity type and count
        entity_type = int(param_list[0])
        param_list = param_list[1:]
        
        # Parse based on entity type
        if entity_type in [100, 102]:  # Circular arc
            params['center'] = np.array([float(x) for x in param_list[0:3]])
            params['normal'] = np.array([float(x) for x in param_list[3:6]])
            params['start_point'] = np.array([float(x) for x in param_list[6:9]])
            params['end_point'] = np.array([float(x) for x in param_list[9:12]])
        
        elif entity_type in [110, 112, 114, 116]:  # Line, polyline, etc.
            num_points = int(param_list[0])
            points = []
            for i in range(num_points):
                idx = 1 + i * 3
                points.append(np.array([float(x) for x in param_list[idx:idx+3]]))
            params['points'] = np.array(points)
        
        elif entity_type in [118, 120, 122]:  # Ruled surface, surface of revolution, etc.
            params['type'] = entity_type
            params['data'] = param_list
        
        return params


class IGESMeshGenerator:
    """IGES-based mesh generator."""
    
    def __init__(self, quality_control: Optional[Dict] = None):
        """Initialize IGES mesh generator.
        
        Args:
            quality_control: Mesh quality control parameters
        """
        self.quality_control = quality_control or {
            'min_angle': 30.0,
            'max_angle': 150.0,
            'aspect_ratio': 5.0,
            'skewness': 0.8,
            'orthogonality': 0.7
        }
        self.reader = IGESReader()
        self.geometry = None
    
    def generate_mesh(self, iges_file: Union[str, Path]) -> 'Mesh':
        """Generate mesh from IGES file.
        
        Args:
            iges_file: Path to IGES file
            
        Returns:
            Generated mesh
        """
        # Read IGES file
        self.reader.read_file(iges_file)
        
        # Extract geometry
        self.geometry = IGESGeometry(self.reader)
        
        # Generate mesh
        mesh = self._generate_mesh()
        
        # Apply quality control
        mesh = self._apply_quality_control(mesh)
        
        return mesh
    
    def _generate_mesh(self) -> 'Mesh':
        """Generate mesh from geometry.
        
        Returns:
            Generated mesh
        """
        # Initialize mesh data
        nodes = []
        elements = []
        node_map = {}  # Map from geometry points to node indices
        
        # Process surfaces
        for surface_id, surface in self.geometry.surfaces.items():
            if surface['type'] in [118, 120, 122]:  # Ruled surface, surface of revolution, etc.
                # Generate mesh for surface
                surface_nodes, surface_elements = self._mesh_surface(surface)
                
                # Add nodes
                for node in surface_nodes:
                    node_key = tuple(node)
                    if node_key not in node_map:
                        node_map[node_key] = len(nodes)
                        nodes.append(node)
                
                # Add elements with correct node indices
                for element in surface_elements:
                    element_nodes = [node_map[tuple(nodes[i])] for i in element]
                    elements.append(element_nodes)
        
        # Convert to numpy arrays
        nodes = np.array(nodes)
        elements = np.array(elements)
        
        # Create mesh
        from phasta.mesh.base import Mesh
        return Mesh(nodes, elements)
    
    def _mesh_surface(self, surface: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh for a surface.
        
        Args:
            surface: Surface data
            
        Returns:
            Tuple of (nodes, elements)
        """
        # This is a simplified implementation
        # In practice, you would use a proper surface meshing algorithm
        
        # For now, create a simple grid mesh
        num_points = 10  # Number of points in each direction
        nodes = []
        elements = []
        
        # Generate nodes
        for i in range(num_points):
            for j in range(num_points):
                # Create a simple grid point
                x = i / (num_points - 1)
                y = j / (num_points - 1)
                z = 0.0
                nodes.append([x, y, z])
        
        # Generate elements
        for i in range(num_points - 1):
            for j in range(num_points - 1):
                # Create a quad element
                n1 = i * num_points + j
                n2 = i * num_points + j + 1
                n3 = (i + 1) * num_points + j + 1
                n4 = (i + 1) * num_points + j
                elements.append([n1, n2, n3, n4])
        
        return np.array(nodes), np.array(elements)
    
    def _apply_quality_control(self, mesh: 'Mesh') -> 'Mesh':
        """Apply quality control to mesh.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Quality-controlled mesh
        """
        # Apply quality control parameters
        min_angle = self.quality_control.get('min_angle', 30.0)
        max_angle = self.quality_control.get('max_angle', 150.0)
        aspect_ratio = self.quality_control.get('aspect_ratio', 5.0)
        skewness = self.quality_control.get('skewness', 0.8)
        orthogonality = self.quality_control.get('orthogonality', 0.7)
        
        # Check element quality
        valid_elements = []
        for element in mesh.elements:
            # Calculate element quality metrics
            element_nodes = mesh.nodes[element]
            
            # Check angles
            angles = self._calculate_angles(element_nodes)
            if not (min_angle <= angles.min() and angles.max() <= max_angle):
                continue
            
            # Check aspect ratio
            if self._calculate_aspect_ratio(element_nodes) > aspect_ratio:
                continue
            
            # Check skewness
            if self._calculate_skewness(element_nodes) > skewness:
                continue
            
            # Check orthogonality
            if self._calculate_orthogonality(element_nodes) < orthogonality:
                continue
            
            valid_elements.append(element)
        
        # Create new mesh with valid elements
        return Mesh(mesh.nodes, np.array(valid_elements))
    
    def _calculate_angles(self, element_nodes: np.ndarray) -> np.ndarray:
        """Calculate angles in an element.
        
        Args:
            element_nodes: Element node coordinates
            
        Returns:
            Array of angles
        """
        # Calculate vectors between nodes
        vectors = np.diff(element_nodes, axis=0)
        vectors = np.vstack((vectors, element_nodes[0] - element_nodes[-1]))
        
        # Calculate angles
        angles = []
        for i in range(len(vectors)):
            v1 = vectors[i]
            v2 = vectors[(i + 1) % len(vectors)]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(np.degrees(angle))
        
        return np.array(angles)
    
    def _calculate_aspect_ratio(self, element_nodes: np.ndarray) -> float:
        """Calculate aspect ratio of an element.
        
        Args:
            element_nodes: Element node coordinates
            
        Returns:
            Aspect ratio
        """
        # Calculate edge lengths
        edges = np.diff(element_nodes, axis=0)
        edges = np.vstack((edges, element_nodes[0] - element_nodes[-1]))
        lengths = np.linalg.norm(edges, axis=1)
        
        return lengths.max() / lengths.min()
    
    def _calculate_skewness(self, element_nodes: np.ndarray) -> float:
        """Calculate skewness of an element.
        
        Args:
            element_nodes: Element node coordinates
            
        Returns:
            Skewness
        """
        # Calculate ideal and actual angles
        ideal_angle = 90.0  # For quad elements
        actual_angles = self._calculate_angles(element_nodes)
        
        # Calculate maximum deviation from ideal angle
        max_deviation = np.max(np.abs(actual_angles - ideal_angle))
        return max_deviation / 90.0
    
    def _calculate_orthogonality(self, element_nodes: np.ndarray) -> float:
        """Calculate orthogonality of an element.
        
        Args:
            element_nodes: Element node coordinates
            
        Returns:
            Orthogonality
        """
        # Calculate vectors between nodes
        vectors = np.diff(element_nodes, axis=0)
        vectors = np.vstack((vectors, element_nodes[0] - element_nodes[-1]))
        
        # Calculate dot products between adjacent vectors
        dot_products = []
        for i in range(len(vectors)):
            v1 = vectors[i]
            v2 = vectors[(i + 1) % len(vectors)]
            dot_products.append(np.abs(np.dot(v1, v2)))
        
        # Calculate orthogonality as average of dot products
        return 1.0 - np.mean(dot_products)


def generate_mesh_from_iges(iges_file: Union[str, Path],
                          quality_control: Optional[Dict] = None) -> 'Mesh':
    """Generate mesh from IGES file.
    
    Args:
        iges_file: Path to IGES file
        quality_control: Mesh quality control parameters
        
    Returns:
        Generated mesh
    """
    generator = IGESMeshGenerator(quality_control)
    return generator.generate_mesh(iges_file) 
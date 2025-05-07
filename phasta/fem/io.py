"""I/O module for finite element method."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import meshio
from pathlib import Path


class MeshIO:
    """Class for reading and writing mesh data."""
    
    def __init__(self):
        """Initialize mesh I/O handler."""
        pass
    
    def read_mesh(self, filename: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Read mesh from file.
        
        Args:
            filename: Path to mesh file
            
        Returns:
            Tuple of (nodes, elements, metadata)
        """
        mesh = meshio.read(filename)
        
        # Extract nodes and elements
        nodes = mesh.points
        elements = {}
        for cell_type, cell_data in mesh.cells:
            elements[cell_type] = cell_data
        
        # Extract metadata
        metadata = {
            'cell_data': mesh.cell_data,
            'point_data': mesh.point_data,
            'field_data': mesh.field_data
        }
        
        return nodes, elements, metadata
    
    def write_mesh(self, filename: str, nodes: np.ndarray, 
                  elements: Dict[str, np.ndarray],
                  metadata: Optional[Dict] = None) -> None:
        """Write mesh to file.
        
        Args:
            filename: Path to output file
            nodes: Node coordinates
            elements: Dictionary of element connectivity arrays
            metadata: Optional mesh metadata
        """
        if metadata is None:
            metadata = {}
        
        # Create mesh object
        mesh = meshio.Mesh(
            points=nodes,
            cells=[(cell_type, cell_data) for cell_type, cell_data in elements.items()],
            cell_data=metadata.get('cell_data', {}),
            point_data=metadata.get('point_data', {}),
            field_data=metadata.get('field_data', {})
        )
        
        # Write mesh
        mesh.write(filename)


class SolutionIO:
    """Class for reading and writing solution data."""
    
    def __init__(self):
        """Initialize solution I/O handler."""
        pass
    
    def read_solution(self, filename: str) -> Dict[str, np.ndarray]:
        """Read solution from file.
        
        Args:
            filename: Path to solution file
            
        Returns:
            Dictionary of solution fields
        """
        # Read solution data
        data = np.load(filename)
        
        # Convert to dictionary
        solution = {}
        for key in data.keys():
            solution[key] = data[key]
        
        return solution
    
    def write_solution(self, filename: str, solution: Dict[str, np.ndarray]) -> None:
        """Write solution to file.
        
        Args:
            filename: Path to output file
            solution: Dictionary of solution fields
        """
        np.savez(filename, **solution)


class PHASTAIO:
    """Class for reading and writing PHASTA format files."""
    
    def __init__(self):
        """Initialize PHASTA I/O handler."""
        pass
    
    def read_phasta_mesh(self, directory: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Read PHASTA format mesh.
        
        Args:
            directory: Directory containing PHASTA mesh files
            
        Returns:
            Tuple of (nodes, elements, metadata)
        """
        # Read geombc.dat.* files
        nodes = []
        elements = []
        metadata = {}
        
        # Find all geombc.dat.* files
        directory = Path(directory)
        mesh_files = sorted(directory.glob('geombc.dat.*'))
        
        for file in mesh_files:
            with open(file, 'r') as f:
                # Read header
                n_nodes = int(f.readline().strip())
                n_elements = int(f.readline().strip())
                
                # Read nodes
                for _ in range(n_nodes):
                    coords = list(map(float, f.readline().strip().split()))
                    nodes.append(coords)
                
                # Read elements
                for _ in range(n_elements):
                    conn = list(map(int, f.readline().strip().split()))
                    elements.append(conn)
        
        # Convert to numpy arrays
        nodes = np.array(nodes)
        elements = np.array(elements)
        
        return nodes, elements, metadata
    
    def write_phasta_mesh(self, directory: str, nodes: np.ndarray,
                         elements: np.ndarray, metadata: Optional[Dict] = None,
                         n_procs: int = 1) -> None:
        """Write PHASTA format mesh.
        
        Args:
            directory: Output directory
            nodes: Node coordinates
            elements: Element connectivity
            metadata: Optional mesh metadata
            n_procs: Number of processor files to write
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Partition mesh if needed
        if n_procs > 1:
            # Simple partitioning for now
            n_nodes_per_proc = len(nodes) // n_procs
            n_elements_per_proc = len(elements) // n_procs
            
            for i in range(n_procs):
                # Get node and element ranges for this processor
                node_start = i * n_nodes_per_proc
                node_end = (i + 1) * n_nodes_per_proc if i < n_procs - 1 else len(nodes)
                elem_start = i * n_elements_per_proc
                elem_end = (i + 1) * n_elements_per_proc if i < n_procs - 1 else len(elements)
                
                # Write processor file
                with open(directory / f'geombc.dat.{i}', 'w') as f:
                    # Write header
                    f.write(f'{node_end - node_start}\n')
                    f.write(f'{elem_end - elem_start}\n')
                    
                    # Write nodes
                    for node in nodes[node_start:node_end]:
                        f.write(' '.join(map(str, node)) + '\n')
                    
                    # Write elements
                    for elem in elements[elem_start:elem_end]:
                        f.write(' '.join(map(str, elem)) + '\n')
        else:
            # Write single file
            with open(directory / 'geombc.dat.0', 'w') as f:
                # Write header
                f.write(f'{len(nodes)}\n')
                f.write(f'{len(elements)}\n')
                
                # Write nodes
                for node in nodes:
                    f.write(' '.join(map(str, node)) + '\n')
                
                # Write elements
                for elem in elements:
                    f.write(' '.join(map(str, elem)) + '\n')
    
    def read_phasta_solution(self, directory: str) -> Dict[str, np.ndarray]:
        """Read PHASTA format solution.
        
        Args:
            directory: Directory containing PHASTA solution files
            
        Returns:
            Dictionary of solution fields
        """
        solution = {}
        
        # Find all restart.* files
        directory = Path(directory)
        solution_files = sorted(directory.glob('restart.*'))
        
        for file in solution_files:
            with open(file, 'r') as f:
                # Read header
                n_fields = int(f.readline().strip())
                
                # Read field names
                field_names = []
                for _ in range(n_fields):
                    field_names.append(f.readline().strip())
                
                # Read field data
                for field_name in field_names:
                    n_values = int(f.readline().strip())
                    values = []
                    for _ in range(n_values):
                        values.append(float(f.readline().strip()))
                    solution[field_name] = np.array(values)
        
        return solution
    
    def write_phasta_solution(self, directory: str, solution: Dict[str, np.ndarray],
                            n_procs: int = 1) -> None:
        """Write PHASTA format solution.
        
        Args:
            directory: Output directory
            solution: Dictionary of solution fields
            n_procs: Number of processor files to write
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Write solution files
        for i in range(n_procs):
            with open(directory / f'restart.{i}', 'w') as f:
                # Write header
                f.write(f'{len(solution)}\n')
                
                # Write field names
                for field_name in solution.keys():
                    f.write(f'{field_name}\n')
                
                # Write field data
                for field_name, values in solution.items():
                    f.write(f'{len(values)}\n')
                    for value in values:
                        f.write(f'{value}\n') 
"""Global assembly for finite element method."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import sparse
from .assembly import ElementAssembly


class GlobalAssembly:
    """Class for assembling global system matrices and vectors."""
    
    def __init__(self, element_type: str, order: int = 1):
        """Initialize global assembly.
        
        Args:
            element_type: Type of element ('line', 'tri', 'quad', 'tet', 'hex')
            order: Polynomial order of the element
        """
        self.element_type = element_type
        self.order = order
        self.element_assembly = ElementAssembly(element_type, order)
    
    def assemble_mass_matrix(self, nodes: np.ndarray, elements: np.ndarray) -> sparse.csr_matrix:
        """Assemble global mass matrix.
        
        Args:
            nodes: Physical coordinates of all nodes, shape (n_nodes, dim)
            elements: Element connectivity, shape (n_elements, n_nodes_per_element)
            
        Returns:
            Global mass matrix in CSR format
        """
        n_nodes = nodes.shape[0]
        n_elements = elements.shape[0]
        n_nodes_per_element = elements.shape[1]
        
        # Pre-allocate arrays for COO format
        rows = []
        cols = []
        data = []
        
        # Loop over elements
        for e in range(n_elements):
            # Get element nodes
            element_nodes = nodes[elements[e]]
            
            # Compute element mass matrix
            M_e = self.element_assembly.compute_mass_matrix(element_nodes)
            
            # Add to global matrix
            for i in range(n_nodes_per_element):
                for j in range(n_nodes_per_element):
                    rows.append(elements[e, i])
                    cols.append(elements[e, j])
                    data.append(M_e[i, j])
        
        # Create sparse matrix
        M = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        
        return M
    
    def assemble_stiffness_matrix(self, nodes: np.ndarray, elements: np.ndarray, 
                                k: Union[float, np.ndarray] = 1.0) -> sparse.csr_matrix:
        """Assemble global stiffness matrix.
        
        Args:
            nodes: Physical coordinates of all nodes, shape (n_nodes, dim)
            elements: Element connectivity, shape (n_elements, n_nodes_per_element)
            k: Material property (constant or array for each element)
            
        Returns:
            Global stiffness matrix in CSR format
        """
        n_nodes = nodes.shape[0]
        n_elements = elements.shape[0]
        n_nodes_per_element = elements.shape[1]
        
        # Pre-allocate arrays for COO format
        rows = []
        cols = []
        data = []
        
        # Loop over elements
        for e in range(n_elements):
            # Get element nodes
            element_nodes = nodes[elements[e]]
            
            # Get material property for this element
            k_e = k[e] if isinstance(k, np.ndarray) else k
            
            # Compute element stiffness matrix
            K_e = self.element_assembly.compute_stiffness_matrix(element_nodes, k_e)
            
            # Add to global matrix
            for i in range(n_nodes_per_element):
                for j in range(n_nodes_per_element):
                    rows.append(elements[e, i])
                    cols.append(elements[e, j])
                    data.append(K_e[i, j])
        
        # Create sparse matrix
        K = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        
        return K
    
    def assemble_load_vector(self, nodes: np.ndarray, elements: np.ndarray,
                           f: Union[float, callable]) -> np.ndarray:
        """Assemble global load vector.
        
        Args:
            nodes: Physical coordinates of all nodes, shape (n_nodes, dim)
            elements: Element connectivity, shape (n_elements, n_nodes_per_element)
            f: Source term (constant or function of coordinates)
            
        Returns:
            Global load vector
        """
        n_nodes = nodes.shape[0]
        n_elements = elements.shape[0]
        
        # Initialize global load vector
        F = np.zeros(n_nodes)
        
        # Loop over elements
        for e in range(n_elements):
            # Get element nodes
            element_nodes = nodes[elements[e]]
            
            # Compute element load vector
            F_e = self.element_assembly.compute_load_vector(element_nodes, f)
            
            # Add to global vector
            F[elements[e]] += F_e
        
        return F
    
    def assemble_convection_matrix(self, nodes: np.ndarray, elements: np.ndarray,
                                 v: Union[np.ndarray, callable]) -> sparse.csr_matrix:
        """Assemble global convection matrix.
        
        Args:
            nodes: Physical coordinates of all nodes, shape (n_nodes, dim)
            elements: Element connectivity, shape (n_elements, n_nodes_per_element)
            v: Velocity field (constant vector or function of coordinates)
            
        Returns:
            Global convection matrix in CSR format
        """
        n_nodes = nodes.shape[0]
        n_elements = elements.shape[0]
        n_nodes_per_element = elements.shape[1]
        
        # Pre-allocate arrays for COO format
        rows = []
        cols = []
        data = []
        
        # Loop over elements
        for e in range(n_elements):
            # Get element nodes
            element_nodes = nodes[elements[e]]
            
            # Get velocity for this element
            if callable(v):
                # Evaluate velocity at element centroid
                centroid = np.mean(element_nodes, axis=0)
                v_e = v(centroid)
            else:
                v_e = v
            
            # Compute element convection matrix
            C_e = self.element_assembly.compute_convection_matrix(element_nodes, v_e)
            
            # Add to global matrix
            for i in range(n_nodes_per_element):
                for j in range(n_nodes_per_element):
                    rows.append(elements[e, i])
                    cols.append(elements[e, j])
                    data.append(C_e[i, j])
        
        # Create sparse matrix
        C = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        
        return C
    
    def apply_dirichlet_bc(self, K: sparse.csr_matrix, F: np.ndarray,
                          dirichlet_nodes: np.ndarray, dirichlet_values: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Apply Dirichlet boundary conditions.
        
        Args:
            K: Global stiffness matrix
            F: Global load vector
            dirichlet_nodes: Indices of nodes with Dirichlet BC
            dirichlet_values: Values at Dirichlet nodes
            
        Returns:
            Modified stiffness matrix and load vector
        """
        # Create copies to avoid modifying originals
        K_mod = K.copy()
        F_mod = F.copy()
        
        # Modify load vector
        for i, node in enumerate(dirichlet_nodes):
            # Get column of stiffness matrix
            col = K_mod.getcol(node).toarray().flatten()
            # Subtract contribution from load vector
            F_mod -= col * dirichlet_values[i]
            # Zero out row and column
            K_mod[node, :] = 0
            K_mod[:, node] = 0
            # Set diagonal to 1
            K_mod[node, node] = 1
            # Set load vector value
            F_mod[node] = dirichlet_values[i]
        
        return K_mod, F_mod 
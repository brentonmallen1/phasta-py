"""Element assembly for finite element method."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from .basis import LagrangeShapeFunction
from .metrics import ElementMetrics
from .integration import ElementIntegrator


class ElementAssembly:
    """Class for assembling element matrices and vectors."""
    
    def __init__(self, element_type: str, order: int = 1):
        """Initialize element assembly.
        
        Args:
            element_type: Type of element ('line', 'tri', 'quad', 'tet', 'hex')
            order: Polynomial order of the element
        """
        self.element_type = element_type
        self.order = order
        self.integrator = ElementIntegrator(element_type, order)
        self.shape_function = LagrangeShapeFunction(element_type, order)
        self.metrics = ElementMetrics(element_type, order)
    
    def compute_mass_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """Compute element mass matrix.
        
        Args:
            nodes: Physical coordinates of element nodes, shape (n_nodes, dim)
            
        Returns:
            Mass matrix, shape (n_nodes, n_nodes)
        """
        n_nodes = nodes.shape[0]
        M = np.zeros((n_nodes, n_nodes))
        
        def mass_func(xi, nodes):
            N = self.shape_function.evaluate(xi)
            return np.outer(N, N)
        
        # Integrate mass matrix
        for i in range(n_nodes):
            for j in range(n_nodes):
                def mij_func(xi, nodes):
                    N = self.shape_function.evaluate(xi)
                    return N[:, i] * N[:, j]
                M[i, j] = self.integrator.integrate(nodes, mij_func)
        
        return M
    
    def compute_stiffness_matrix(self, nodes: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Compute element stiffness matrix.
        
        Args:
            nodes: Physical coordinates of element nodes, shape (n_nodes, dim)
            k: Material property (e.g., thermal conductivity)
            
        Returns:
            Stiffness matrix, shape (n_nodes, n_nodes)
        """
        n_nodes = nodes.shape[0]
        K = np.zeros((n_nodes, n_nodes))
        
        def stiffness_func(xi, nodes):
            dN_phys = self.metrics.compute_physical_derivatives(xi, nodes)
            return np.einsum('ijk,ilk->ijl', dN_phys, dN_phys)
        
        # Integrate stiffness matrix
        for i in range(n_nodes):
            for j in range(n_nodes):
                def kij_func(xi, nodes):
                    dN_phys = self.metrics.compute_physical_derivatives(xi, nodes)
                    return np.sum(dN_phys[:, i] * dN_phys[:, j], axis=1)
                K[i, j] = k * self.integrator.integrate(nodes, kij_func)
        
        return K
    
    def compute_load_vector(self, nodes: np.ndarray, f: Union[float, callable]) -> np.ndarray:
        """Compute element load vector.
        
        Args:
            nodes: Physical coordinates of element nodes, shape (n_nodes, dim)
            f: Source term (constant or function of coordinates)
            
        Returns:
            Load vector, shape (n_nodes,)
        """
        n_nodes = nodes.shape[0]
        F = np.zeros(n_nodes)
        
        if isinstance(f, (int, float)):
            # Constant source term
            def load_func(xi, nodes):
                N = self.shape_function.evaluate(xi)
                return f * N
        else:
            # Function source term
            def load_func(xi, nodes):
                N = self.shape_function.evaluate(xi)
                # Convert natural coordinates to physical coordinates
                x_phys = np.zeros((xi.shape[0], nodes.shape[1]))
                for i in range(xi.shape[0]):
                    x_phys[i] = np.sum(N[i, :, np.newaxis] * nodes, axis=0)
                f_vals = f(x_phys)
                return f_vals[:, np.newaxis] * N
        
        # Integrate load vector
        for i in range(n_nodes):
            def fi_func(xi, nodes):
                N = self.shape_function.evaluate(xi)
                return N[:, i] * load_func(xi, nodes)[:, 0]
            F[i] = self.integrator.integrate(nodes, fi_func)
        
        return F
    
    def compute_convection_matrix(self, nodes: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute element convection matrix.
        
        Args:
            nodes: Physical coordinates of element nodes, shape (n_nodes, dim)
            v: Velocity vector, shape (dim,)
            
        Returns:
            Convection matrix, shape (n_nodes, n_nodes)
        """
        n_nodes = nodes.shape[0]
        C = np.zeros((n_nodes, n_nodes))
        
        def convection_func(xi, nodes):
            N = self.shape_function.evaluate(xi)
            dN_phys = self.metrics.compute_physical_derivatives(xi, nodes)
            return np.einsum('i,ijk->jk', v, dN_phys) * N
        
        # Integrate convection matrix
        for i in range(n_nodes):
            for j in range(n_nodes):
                def cij_func(xi, nodes):
                    N = self.shape_function.evaluate(xi)
                    dN_phys = self.metrics.compute_physical_derivatives(xi, nodes)
                    return N[:, i] * np.sum(v * dN_phys[:, j], axis=1)
                C[i, j] = self.integrator.integrate(nodes, cij_func)
        
        return C
    
    def compute_boundary_vector(self, nodes: np.ndarray, g: Union[float, callable]) -> np.ndarray:
        """Compute element boundary vector.
        
        Args:
            nodes: Physical coordinates of element nodes, shape (n_nodes, dim)
            g: Boundary condition (constant or function of coordinates)
            
        Returns:
            Boundary vector, shape (n_nodes,)
        """
        n_nodes = nodes.shape[0]
        G = np.zeros(n_nodes)
        
        if isinstance(g, (int, float)):
            # Constant boundary condition
            def boundary_func(xi, nodes):
                N = self.shape_function.evaluate(xi)
                return g * N
        else:
            # Function boundary condition
            def boundary_func(xi, nodes):
                N = self.shape_function.evaluate(xi)
                # Convert natural coordinates to physical coordinates
                x_phys = np.zeros((xi.shape[0], nodes.shape[1]))
                for i in range(xi.shape[0]):
                    x_phys[i] = np.sum(N[i, :, np.newaxis] * nodes, axis=0)
                g_vals = g(x_phys)
                return g_vals[:, np.newaxis] * N
        
        # Integrate boundary vector
        for i in range(n_nodes):
            def gi_func(xi, nodes):
                N = self.shape_function.evaluate(xi)
                return N[:, i] * boundary_func(xi, nodes)[:, 0]
            G[i] = self.integrator.integrate(nodes, gi_func)
        
        return G

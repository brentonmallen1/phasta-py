"""
Test suite for the compressible flow solver.

This module contains tests for:
1. Basic solver functionality
2. Boundary conditions
3. Turbulence models
4. Flux computation
5. Time integration
6. Validation cases
"""

import unittest
import numpy as np
from phasta.solver.compressible.solver import CompressibleSolver
from phasta.solver.compressible.boundary_conditions import (
    WallBoundary, InletBoundary, OutletBoundary, 
    PeriodicBoundary, SymmetryBoundary
)
from phasta.solver.compressible.turbulence_models import (
    KEpsilonModel, KOmegaModel, SmagorinskyModel,
    TurbulenceModelConfig
)

class TestCompressibleSolver(unittest.TestCase):
    """Test cases for the compressible flow solver."""
    
    def setUp(self):
        """Set up test cases."""
        # Create a simple 2D mesh
        self.mesh = self._create_test_mesh()
        
        # Create solver instance
        self.solver = CompressibleSolver(
            mesh=self.mesh,
            config={
                "gamma": 1.4,
                "prandtl": 0.72,
                "cfl": 0.5,
                "time_integration": "rk4"
            }
        )
        
    def _create_test_mesh(self):
        """Create a simple 2D test mesh."""
        # Create a 10x10 grid
        x = np.linspace(0, 1, 11)
        y = np.linspace(0, 1, 11)
        X, Y = np.meshgrid(x, y)
        
        # Create nodes
        nodes = np.column_stack((X.flatten(), Y.flatten()))
        
        # Create elements (triangles)
        elements = []
        for i in range(10):
            for j in range(10):
                # Create two triangles for each cell
                n1 = i * 11 + j
                n2 = n1 + 1
                n3 = n1 + 11
                n4 = n3 + 1
                
                elements.append([n1, n2, n3])
                elements.append([n2, n4, n3])
                
        return {
            "nodes": nodes,
            "elements": np.array(elements),
            "boundaries": {
                "wall": np.arange(0, 11),  # Bottom wall
                "inlet": np.arange(0, 121, 11),  # Left inlet
                "outlet": np.arange(10, 121, 11),  # Right outlet
                "symmetry": np.arange(110, 121)  # Top symmetry
            }
        }
        
    def test_initialization(self):
        """Test solver initialization."""
        self.assertIsNotNone(self.solver)
        self.assertEqual(self.solver.mesh["nodes"].shape[0], 121)
        self.assertEqual(self.solver.mesh["elements"].shape[0], 200)
        
    def test_boundary_conditions(self):
        """Test boundary condition application."""
        # Create boundary conditions
        wall = WallBoundary(temperature=300.0)
        inlet = InletBoundary(mach=0.5, pressure=101325.0, temperature=300.0)
        outlet = OutletBoundary(pressure=101325.0)
        symmetry = SymmetryBoundary()
        
        # Apply boundary conditions
        solution = np.ones((121, 5))  # [rho, rho*u, rho*v, rho*w, rho*E]
        
        # Test wall boundary
        solution_wall = wall.apply(solution.copy(), self.mesh, 
                                 self.mesh["boundaries"]["wall"])
        self.assertTrue(np.allclose(solution_wall[self.mesh["boundaries"]["wall"], 1:4], 0.0))
        
        # Test inlet boundary
        solution_inlet = inlet.apply(solution.copy(), self.mesh,
                                   self.mesh["boundaries"]["inlet"])
        self.assertFalse(np.allclose(solution_inlet[self.mesh["boundaries"]["inlet"], 1:4], 0.0))
        
        # Test outlet boundary
        solution_outlet = outlet.apply(solution.copy(), self.mesh,
                                     self.mesh["boundaries"]["outlet"])
        self.assertFalse(np.allclose(solution_outlet[self.mesh["boundaries"]["outlet"], 1:4], 0.0))
        
        # Test symmetry boundary
        solution_symmetry = symmetry.apply(solution.copy(), self.mesh,
                                         self.mesh["boundaries"]["symmetry"])
        self.assertTrue(np.allclose(solution_symmetry[self.mesh["boundaries"]["symmetry"], 2], 0.0))
        
    def test_turbulence_models(self):
        """Test turbulence model implementation."""
        # Create turbulence model configurations
        k_eps_config = TurbulenceModelConfig(
            model_type="rans",
            model_name="k-epsilon",
            wall_function=True
        )
        
        k_omega_config = TurbulenceModelConfig(
            model_type="rans",
            model_name="k-omega",
            wall_function=True
        )
        
        les_config = TurbulenceModelConfig(
            model_type="les",
            model_name="smagorinsky",
            model_params={"C_s": 0.17}
        )
        
        # Create models
        k_eps = KEpsilonModel(k_eps_config)
        k_omega = KOmegaModel(k_omega_config)
        les = SmagorinskyModel(les_config)
        
        # Create test data
        solution = np.ones((121, 7))  # [rho, rho*u, rho*v, rho*w, rho*E, k, eps/omega]
        solution[:, 5] = 0.1  # k
        solution[:, 6] = 0.01  # eps/omega
        
        grad_u = np.zeros((121, 3, 3))
        grad_u[:, 0, 1] = 1.0  # du/dy
        
        # Test eddy viscosity computation
        mu_t_keps = k_eps.compute_eddy_viscosity(solution, self.mesh, grad_u)
        mu_t_komega = k_omega.compute_eddy_viscosity(solution, self.mesh, grad_u)
        mu_t_les = les.compute_eddy_viscosity(solution, self.mesh, grad_u)
        
        self.assertTrue(np.all(mu_t_keps > 0))
        self.assertTrue(np.all(mu_t_komega > 0))
        self.assertTrue(np.all(mu_t_les > 0))
        
        # Test source term computation
        source_keps = k_eps.compute_source_terms(solution, self.mesh, grad_u)
        source_komega = k_omega.compute_source_terms(solution, self.mesh, grad_u)
        source_les = les.compute_source_terms(solution, self.mesh, grad_u)
        
        self.assertEqual(source_keps.shape, (121, 2))
        self.assertEqual(source_komega.shape, (121, 2))
        self.assertEqual(source_les.shape, (121, 2))
        
    def test_flux_computation(self):
        """Test flux computation."""
        # Create test data
        solution = np.ones((121, 5))
        solution[:, 1:4] = 0.5  # Velocity
        solution[:, 4] = 2.0  # Energy
        
        # Compute fluxes
        fluxes = self.solver.compute_fluxes(solution)
        
        self.assertEqual(fluxes.shape, (200, 5))  # 200 elements, 5 variables
        self.assertTrue(np.all(np.isfinite(fluxes)))
        
    def test_time_integration(self):
        """Test time integration."""
        # Create initial solution
        solution = np.ones((121, 5))
        solution[:, 1:4] = 0.5
        solution[:, 4] = 2.0
        
        # Integrate one step
        dt = 0.001
        solution_new = self.solver.integrate_time(solution, dt)
        
        self.assertEqual(solution_new.shape, solution.shape)
        self.assertTrue(np.all(np.isfinite(solution_new)))
        
    def test_sod_shock_tube(self):
        """Test Sod shock tube problem."""
        # Create 1D mesh
        x = np.linspace(0, 1, 101)
        mesh_1d = {
            "nodes": x.reshape(-1, 1),
            "elements": np.array([[i, i+1] for i in range(100)]),
            "boundaries": {
                "inlet": np.array([0]),
                "outlet": np.array([100])
            }
        }
        
        # Create solver
        solver_1d = CompressibleSolver(
            mesh=mesh_1d,
            config={
                "gamma": 1.4,
                "prandtl": 0.72,
                "cfl": 0.5,
                "time_integration": "rk4"
            }
        )
        
        # Set initial conditions
        solution = np.ones((101, 5))
        # Left state
        solution[:50, 0] = 1.0  # rho
        solution[:50, 1] = 0.0  # rho*u
        solution[:50, 4] = 2.5  # rho*E
        # Right state
        solution[50:, 0] = 0.125  # rho
        solution[50:, 1] = 0.0  # rho*u
        solution[50:, 4] = 0.25  # rho*E
        
        # Run simulation
        t_end = 0.2
        dt = 0.001
        t = 0.0
        
        while t < t_end:
            solution = solver_1d.integrate_time(solution, dt)
            t += dt
            
        # Check results
        # Density should be higher on the left
        self.assertTrue(np.mean(solution[:50, 0]) > np.mean(solution[50:, 0]))
        # Pressure should be higher on the left
        p_left = (1.4 - 1.0) * (solution[:50, 4] - 
                               0.5 * solution[:50, 1]**2 / solution[:50, 0])
        p_right = (1.4 - 1.0) * (solution[50:, 4] - 
                                0.5 * solution[50:, 1]**2 / solution[50:, 0])
        self.assertTrue(np.mean(p_left) > np.mean(p_right))
        
if __name__ == "__main__":
    unittest.main() 
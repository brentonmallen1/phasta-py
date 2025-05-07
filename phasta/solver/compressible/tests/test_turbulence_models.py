"""
Unit tests for turbulence models.
"""

import numpy as np
import pytest
from phasta.solver.compressible.turbulence_models import (
    TurbulenceModelConfig,
    KEpsilonModel,
    KOmegaModel,
    SSTModel,
    SmagorinskyModel
)

@pytest.fixture
def config():
    """Create default turbulence model configuration."""
    return TurbulenceModelConfig(
        model_type="rans",
        model_name="sst",
        wall_function=True,
        wall_function_type="automatic"
    )

@pytest.fixture
def solution():
    """Create test solution array."""
    n_nodes = 100
    solution = np.zeros((n_nodes, 7))  # [rho, u, v, w, E, k, omega]
    
    # Set some initial values
    solution[:, 0] = 1.0  # density
    solution[:, 1:4] = 1.0  # velocity
    solution[:, 4] = 2.0  # energy
    solution[:, 5] = 0.1  # k
    solution[:, 6] = 1.0  # omega
    
    return solution

@pytest.fixture
def mesh():
    """Create test mesh."""
    return {
        "nodes": np.random.rand(100, 3),
        "elements": np.random.randint(0, 100, (50, 8)),
        "boundary_faces": {
            "wall": np.array([0, 1, 2]),
            "inlet": np.array([3, 4]),
            "outlet": np.array([5, 6])
        }
    }

@pytest.fixture
def grad_u():
    """Create test velocity gradient."""
    n_nodes = 100
    grad_u = np.zeros((n_nodes, 3, 3))
    
    # Set some initial values
    grad_u[:, 0, 0] = 1.0  # du/dx
    grad_u[:, 1, 1] = 1.0  # dv/dy
    grad_u[:, 2, 2] = 1.0  # dw/dz
    
    return grad_u

def test_sst_eddy_viscosity(config, solution, mesh, grad_u):
    """Test SST model eddy viscosity computation."""
    model = SSTModel(config)
    mu_t = model.compute_eddy_viscosity(solution, mesh, grad_u)
    
    # Check dimensions
    assert mu_t.shape == (len(solution),)
    
    # Check physical constraints
    assert np.all(mu_t >= 0)
    
    # Check order of magnitude
    rho = solution[:, 0]
    k = solution[:, 5]
    omega = solution[:, 6]
    mu_t_expected = rho * k / omega
    assert np.all(np.abs(mu_t - mu_t_expected) < 1e-10)

def test_sst_source_terms(config, solution, mesh, grad_u):
    """Test SST model source terms computation."""
    model = SSTModel(config)
    source_terms = model.compute_source_terms(solution, mesh, grad_u)
    
    # Check dimensions
    assert source_terms.shape == (len(solution), 2)
    
    # Check physical constraints
    assert np.all(source_terms[:, 0] >= -1e-10)  # Production should be positive
    assert np.all(source_terms[:, 1] >= -1e-10)  # Production should be positive

def test_sst_blending_functions(config, solution, mesh, grad_u):
    """Test SST model blending functions."""
    model = SSTModel(config)
    
    # Test F1
    F1 = model._compute_F1(solution, mesh, grad_u)
    assert np.all((F1 >= 0) & (F1 <= 1))
    
    # Test F2
    F2 = model._compute_F2(solution, mesh, grad_u)
    assert np.all((F2 >= 0) & (F2 <= 1))

def test_sst_wall_functions(config, solution, mesh, grad_u):
    """Test SST model with wall functions."""
    model = SSTModel(config)
    mu_t = model.compute_eddy_viscosity(solution, mesh, grad_u)
    
    # Check wall function application
    assert np.all(mu_t[mesh["boundary_faces"]["wall"]] == 0)

def test_sst_model_constants(config):
    """Test SST model constants."""
    model = SSTModel(config)
    
    # Check model constants
    assert model.alpha_1 == 0.31
    assert model.alpha_2 == 0.44
    assert model.beta_1 == 0.075
    assert model.beta_2 == 0.0828
    assert model.beta_star == 0.09
    assert model.sigma_k1 == 0.85
    assert model.sigma_k2 == 1.0
    assert model.sigma_omega1 == 0.5
    assert model.sigma_omega2 == 0.856
    assert model.a1 == 0.31

def test_sst_custom_parameters(config):
    """Test SST model with custom parameters."""
    custom_params = {
        "alpha_1": 0.35,
        "beta_1": 0.085
    }
    config.model_params = custom_params
    
    model = SSTModel(config)
    
    # Check custom parameters
    assert model.alpha_1 == 0.35
    assert model.beta_1 == 0.085
    
    # Check other parameters remain unchanged
    assert model.alpha_2 == 0.44
    assert model.beta_2 == 0.0828 
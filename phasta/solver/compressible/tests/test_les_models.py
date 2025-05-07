"""
Unit tests for LES models.
"""

import numpy as np
import pytest
from ..les_models import WALEModel, VremanModel, HybridRANSLES
from ..turbulence_models import TurbulenceModelConfig

@pytest.fixture
def config():
    """Create default LES model configuration."""
    return TurbulenceModelConfig(
        model_type="les",
        model_name="wale",
        wall_function=False
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

def test_wale_eddy_viscosity(config, solution, mesh, grad_u):
    """Test WALE model eddy viscosity computation."""
    model = WALEModel(config)
    mu_t = model.compute_eddy_viscosity(solution, mesh, grad_u)
    
    # Check dimensions
    assert mu_t.shape == (len(solution),)
    
    # Check physical constraints
    assert np.all(mu_t >= 0)
    assert np.all(np.isfinite(mu_t))

def test_vreman_eddy_viscosity(config, solution, mesh, grad_u):
    """Test Vreman model eddy viscosity computation."""
    config.model_name = "vreman"
    model = VremanModel(config)
    mu_t = model.compute_eddy_viscosity(solution, mesh, grad_u)
    
    # Check dimensions
    assert mu_t.shape == (len(solution),)
    
    # Check physical constraints
    assert np.all(mu_t >= 0)
    assert np.all(np.isfinite(mu_t))

def test_hybrid_rans_les_eddy_viscosity(config, solution, mesh, grad_u):
    """Test Hybrid RANS/LES model eddy viscosity computation."""
    config.model_name = "hybrid_rans_les"
    model = HybridRANSLES(config)
    mu_t = model.compute_eddy_viscosity(solution, mesh, grad_u)
    
    # Check dimensions
    assert mu_t.shape == (len(solution),)
    
    # Check physical constraints
    assert np.all(mu_t >= 0)
    assert np.all(np.isfinite(mu_t))

def test_wale_source_terms(config, solution, mesh, grad_u):
    """Test WALE model source terms computation."""
    model = WALEModel(config)
    source_terms = model.compute_source_terms(solution, mesh, grad_u)
    
    # Check dimensions
    assert source_terms.shape == (len(solution), 2)
    
    # Check that source terms are zero for LES models
    assert np.all(source_terms == 0)

def test_vreman_source_terms(config, solution, mesh, grad_u):
    """Test Vreman model source terms computation."""
    config.model_name = "vreman"
    model = VremanModel(config)
    source_terms = model.compute_source_terms(solution, mesh, grad_u)
    
    # Check dimensions
    assert source_terms.shape == (len(solution), 2)
    
    # Check that source terms are zero for LES models
    assert np.all(source_terms == 0)

def test_hybrid_rans_les_source_terms(config, solution, mesh, grad_u):
    """Test Hybrid RANS/LES model source terms computation."""
    config.model_name = "hybrid_rans_les"
    model = HybridRANSLES(config)
    source_terms = model.compute_source_terms(solution, mesh, grad_u)
    
    # Check dimensions
    assert source_terms.shape == (len(solution), 2)
    
    # Check physical constraints
    assert np.all(np.isfinite(source_terms))

def test_wale_model_constants(config):
    """Test WALE model constants."""
    model = WALEModel(config)
    assert model.C_w == 0.325

def test_vreman_model_constants(config):
    """Test Vreman model constants."""
    config.model_name = "vreman"
    model = VremanModel(config)
    assert model.C_v == 0.07

def test_hybrid_rans_les_model_constants(config):
    """Test Hybrid RANS/LES model constants."""
    config.model_name = "hybrid_rans_les"
    model = HybridRANSLES(config)
    assert model.C_des == 0.65
    assert model.C_des_ks == 0.61

def test_custom_parameters():
    """Test custom parameter initialization."""
    custom_params = {
        "C_w": 0.35,
        "C_v": 0.08,
        "C_des": 0.70,
        "C_des_ks": 0.65
    }
    
    # Test WALE model
    config = TurbulenceModelConfig(
        model_type="les",
        model_name="wale",
        model_params={"C_w": custom_params["C_w"]}
    )
    model = WALEModel(config)
    assert model.C_w == custom_params["C_w"]
    
    # Test Vreman model
    config = TurbulenceModelConfig(
        model_type="les",
        model_name="vreman",
        model_params={"C_v": custom_params["C_v"]}
    )
    model = VremanModel(config)
    assert model.C_v == custom_params["C_v"]
    
    # Test Hybrid RANS/LES model
    config = TurbulenceModelConfig(
        model_type="les",
        model_name="hybrid_rans_les",
        model_params={
            "C_des": custom_params["C_des"],
            "C_des_ks": custom_params["C_des_ks"]
        }
    )
    model = HybridRANSLES(config)
    assert model.C_des == custom_params["C_des"]
    assert model.C_des_ks == custom_params["C_des_ks"] 
"""
Unit tests for transition models.
"""

import numpy as np
import pytest
from ..transition_models import TransitionModelConfig, GammaReThetaModel, KKLOmegaModel
from ..turbulence_models import TurbulenceModelConfig, SSTModel

@pytest.fixture
def config_gamma_retheta():
    """Create default γ-Reθ model configuration."""
    return TransitionModelConfig(
        model_type="gamma-retheta",
        turbulence_model="sst",
        wall_function=False,
        model_params={
            "c_a1": 2.0,
            "c_a2": 0.06,
            "c_e1": 1.0,
            "c_e2": 50.0,
            "c_theta_t": 0.03,
            "sigma_gamma": 1.0,
            "sigma_theta": 2.0
        }
    )

@pytest.fixture
def config_k_kl_omega():
    """Create default k-kl-ω model configuration."""
    return TransitionModelConfig(
        model_type="k-kl-omega",
        turbulence_model="sst",
        wall_function=False,
        model_params={
            "sigma_k": 1.0,
            "sigma_kl": 1.0,
            "sigma_omega": 2.0,
            "alpha": 0.52,
            "beta": 0.072,
            "beta_star": 0.09
        }
    )

@pytest.fixture
def solution():
    """Create test solution array."""
    n_points = 10
    return np.array([
        [1.0, 1.0, 0.0, 0.0, 2.5, 0.1, 1.0, 0.5, 200.0]  # [rho, u, v, w, E, k, omega, gamma/kl, re_theta]
        for _ in range(n_points)
    ])

@pytest.fixture
def mesh():
    """Create test mesh."""
    return {
        "nodes": np.random.rand(10, 3),
        "elements": np.array([[0, 1, 2, 3] for _ in range(5)]),
        "boundary_faces": np.array([[0, 1, 2] for _ in range(3)])
    }

@pytest.fixture
def grad_u():
    """Create test velocity gradient."""
    n_points = 10
    return np.array([
        [[0.1, 0.0, 0.0],
         [0.0, 0.1, 0.0],
         [0.0, 0.0, 0.1]]
        for _ in range(n_points)
    ])

def test_gamma_retheta_source_terms(config_gamma_retheta, solution, mesh, grad_u):
    """Test γ-Reθ model source terms computation."""
    model = GammaReThetaModel(config_gamma_retheta)
    turbulence_model = SSTModel(TurbulenceModelConfig())
    
    source_terms = model.compute_source_terms(solution, mesh, grad_u, turbulence_model)
    
    assert source_terms.shape == (len(solution), 2)
    assert np.all(np.isfinite(source_terms))
    
    # Check production terms
    assert np.all(source_terms[:, 0] >= 0)  # S_gamma production
    assert np.all(source_terms[:, 1] >= 0)  # S_theta production

def test_gamma_retheta_eddy_viscosity_modification(config_gamma_retheta, solution):
    """Test γ-Reθ model eddy viscosity modification."""
    model = GammaReThetaModel(config_gamma_retheta)
    mu_t = np.ones(len(solution))
    
    modified_mu_t = model.compute_eddy_viscosity_modification(mu_t, solution)
    
    assert modified_mu_t.shape == mu_t.shape
    assert np.all(np.isfinite(modified_mu_t))
    assert np.all(modified_mu_t <= mu_t)  # Eddy viscosity should be reduced by intermittency

def test_gamma_retheta_critical_re_theta(config_gamma_retheta, solution, mesh):
    """Test γ-Reθ model critical Reθ computation."""
    model = GammaReThetaModel(config_gamma_retheta)
    
    re_theta_crit = model._compute_re_theta_crit(solution, mesh)
    
    assert re_theta_crit.shape == (len(solution),)
    assert np.all(np.isfinite(re_theta_crit))
    assert np.all(re_theta_crit > 0)

def test_k_kl_omega_source_terms(config_k_kl_omega, solution, mesh, grad_u):
    """Test k-kl-ω model source terms computation."""
    model = KKLOmegaModel(config_k_kl_omega)
    
    source_terms = model.compute_source_terms(solution, mesh, grad_u)
    
    assert source_terms.shape == (len(solution), 3)
    assert np.all(np.isfinite(source_terms))
    
    # Check production terms
    assert np.all(source_terms[:, 0] >= 0)  # S_k production
    assert np.all(source_terms[:, 1] >= 0)  # S_kl production
    assert np.all(source_terms[:, 2] >= 0)  # S_omega production

def test_k_kl_omega_production_terms(config_k_kl_omega, solution, grad_u):
    """Test k-kl-ω model production terms computation."""
    model = KKLOmegaModel(config_k_kl_omega)
    
    P_k = model._compute_production_k(solution, grad_u)
    P_kl = model._compute_production_kl(solution, grad_u)
    
    assert P_k.shape == (len(solution),)
    assert P_kl.shape == (len(solution),)
    assert np.all(np.isfinite(P_k))
    assert np.all(np.isfinite(P_kl))
    assert np.all(P_k >= 0)
    assert np.all(P_kl >= 0)

def test_model_constants(config_gamma_retheta, config_k_kl_omega):
    """Test model constants initialization."""
    gamma_retheta = GammaReThetaModel(config_gamma_retheta)
    k_kl_omega = KKLOmegaModel(config_k_kl_omega)
    
    # Check γ-Reθ model constants
    assert gamma_retheta.c_a1 == 2.0
    assert gamma_retheta.c_a2 == 0.06
    assert gamma_retheta.c_e1 == 1.0
    assert gamma_retheta.c_e2 == 50.0
    assert gamma_retheta.c_theta_t == 0.03
    assert gamma_retheta.sigma_gamma == 1.0
    assert gamma_retheta.sigma_theta == 2.0
    
    # Check k-kl-ω model constants
    assert k_kl_omega.sigma_k == 1.0
    assert k_kl_omega.sigma_kl == 1.0
    assert k_kl_omega.sigma_omega == 2.0
    assert k_kl_omega.alpha == 0.52
    assert k_kl_omega.beta == 0.072
    assert k_kl_omega.beta_star == 0.09

def test_custom_parameters():
    """Test custom parameter initialization."""
    custom_params = {
        "c_a1": 2.5,
        "c_a2": 0.08,
        "c_e1": 1.2,
        "c_e2": 60.0,
        "c_theta_t": 0.04,
        "sigma_gamma": 1.2,
        "sigma_theta": 2.2
    }
    
    config = TransitionModelConfig(
        model_type="gamma-retheta",
        turbulence_model="sst",
        wall_function=False,
        model_params=custom_params
    )
    
    model = GammaReThetaModel(config)
    
    assert model.c_a1 == custom_params["c_a1"]
    assert model.c_a2 == custom_params["c_a2"]
    assert model.c_e1 == custom_params["c_e1"]
    assert model.c_e2 == custom_params["c_e2"]
    assert model.c_theta_t == custom_params["c_theta_t"]
    assert model.sigma_gamma == custom_params["sigma_gamma"]
    assert model.sigma_theta == custom_params["sigma_theta"] 
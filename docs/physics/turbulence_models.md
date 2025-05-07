# Turbulence Models Documentation

## Overview
This document describes the implementation of various turbulence models in PHASTA-Py, including RANS models (k-ε, k-ω, SST), transition models (γ-Reθ, k-kl-ω), and LES models.

## Available Models

### RANS Models

#### k-ε Model
The standard k-ε model is implemented with the following features:
- Eddy viscosity computation: μ_t = ρ C_μ k²/ε
- Source terms for k and ε equations
- Production and destruction terms
- Wall function integration

#### k-ω Model
The k-ω model includes:
- Eddy viscosity computation: μ_t = ρ k/ω
- Source terms for k and ω equations
- Production and destruction terms
- Wall function integration

#### SST Model
Menter's SST model features:
- Eddy viscosity computation with blending functions
- Source terms for k and ω equations
- Cross-diffusion term
- Wall function integration
- Blending functions F1 and F2

### Transition Models

#### γ-Reθ Model
The γ-Reθ transition model includes:
- Intermittency equation
- Momentum thickness Reynolds number equation
- Critical Reθ computation
- Eddy viscosity modification
- Integration with turbulence models

#### k-kl-ω Model
The k-kl-ω model features:
- Turbulent kinetic energy equation
- Laminar kinetic energy equation
- Specific dissipation rate equation
- Production and destruction terms
- Wall function integration

### LES Models

#### Smagorinsky Model
The Smagorinsky LES model includes:
- Eddy viscosity computation: μ_t = ρ (C_s Δ)² |S|
- Filter width computation
- Strain rate magnitude calculation

## Configuration

Turbulence models are configured using the `TurbulenceModelConfig` class:

```python
@dataclass
class TurbulenceModelConfig:
    model_type: str  # "rans" or "les"
    model_name: str  # e.g., "k-epsilon", "k-omega", "sst", "smagorinsky"
    wall_function: bool = False
    wall_function_type: str = "automatic"
    wall_function_params: Dict[str, float] = None
    transition_model: Optional[str] = None
    model_params: Dict[str, float] = None
```

## Usage

### Creating Turbulence Models
```python
from phasta.solver.compressible.turbulence_models import (
    TurbulenceModelConfig,
    KEpsilonModel,
    KOmegaModel,
    SSTModel,
    SmagorinskyModel
)

# Create configuration
config = TurbulenceModelConfig(
    model_type="rans",
    model_name="sst",
    wall_function=True,
    wall_function_type="automatic"
)

# Create model
model = SSTModel(config)
```

### Computing Eddy Viscosity
```python
# Compute eddy viscosity
mu_t = model.compute_eddy_viscosity(solution, mesh, grad_u)

# Compute source terms
source_terms = model.compute_source_terms(solution, mesh, grad_u)
```

## Integration with Transition Models

Transition models can be integrated with turbulence models:

```python
from phasta.solver.compressible.transition_models import (
    TransitionModelConfig,
    GammaReThetaModel,
    KKLOmegaModel
)

# Create transition model configuration
transition_config = TransitionModelConfig(
    model_type="gamma-retheta",
    turbulence_model="sst",
    wall_function=True
)

# Create transition model
transition_model = GammaReThetaModel(transition_config)

# Compute source terms with transition
source_terms = transition_model.compute_source_terms(
    solution, mesh, grad_u, turbulence_model
)
```

## Best Practices

1. **Model Selection**
   - Use k-ε for simple flows and wall-bounded flows
   - Use k-ω for boundary layers and adverse pressure gradients
   - Use SST for complex flows with separation
   - Use γ-Reθ for transition prediction
   - Use k-kl-ω for natural transition
   - Use Smagorinsky for LES simulations

2. **Wall Functions**
   - Use wall functions for high Reynolds number flows
   - Choose appropriate wall function type based on y+
   - Consider enhanced wall treatment for complex flows

3. **Transition Models**
   - Use γ-Reθ for bypass transition
   - Use k-kl-ω for natural transition
   - Ensure proper integration with turbulence models

4. **LES Models**
   - Use appropriate filter width
   - Consider wall damping for near-wall regions
   - Use appropriate C_s value for different flows

## Validation

The turbulence models have been validated against:
1. Channel flow at Reτ = 395
2. Flat plate boundary layer
3. Backward-facing step flow
4. Transitional flows
5. LES benchmark cases

## References

1. Launder, B. E., & Spalding, D. B. (1974). The numerical computation of turbulent flows.
2. Menter, F. R. (1994). Two-equation eddy-viscosity turbulence models for engineering applications.
3. Langtry, R. B., & Menter, F. R. (2009). Correlation-based transition modeling for unstructured parallelized computational fluid dynamics codes.
4. Smagorinsky, J. (1963). General circulation experiments with the primitive equations. 
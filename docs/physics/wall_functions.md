# Wall Functions Documentation

## Overview
Wall functions are used to model the near-wall region in turbulent flows, providing a bridge between the wall and the fully turbulent region. This document describes the implementation of various wall function approaches in PHASTA-Py.

## Available Wall Functions

### Standard Wall Functions
The standard wall functions implement the traditional approach with a clear separation between the viscous sublayer and the log layer:

- Viscous sublayer (y+ < 11.0): u+ = y+
- Log layer (y+ ≥ 11.0): u+ = (1/κ) * ln(E * y+)

where:
- u+ is the dimensionless velocity
- y+ is the dimensionless wall distance
- κ is the von Karman constant (default: 0.41)
- E is the wall roughness parameter (default: 9.0)

### Enhanced Wall Treatment
The enhanced wall treatment provides a smooth transition between the viscous sublayer and log layer using blending functions:

1. Viscous sublayer solution: u+_vis = y+
2. Log layer solution: u+_log = (1/κ) * ln(E * y+)
3. Blending function: φ = tanh((y+/y+_switch)⁴)
4. Final solution: u+ = u+_vis * (1-φ) + u+_blend * φ

where u+_blend is computed using an exponential blending function.

### Automatic Wall Treatment
The automatic wall treatment provides a continuous solution across all y+ values:

1. Viscous sublayer solution: u+_vis = y+
2. Log layer solution: u+_log = (1/κ) * ln(E * y+)
3. Smooth transition: u+ = u+_vis + (u+_log - u+_vis) * (1 - exp(-(y+* - y+_switch)/b))

where:
- y+* = max(y+, y+_switch)
- b is the blending factor (default: 0.1)

## Configuration

Wall functions are configured using the `WallFunctionConfig` class:

```python
@dataclass
class WallFunctionConfig:
    kappa: float = 0.41  # von Karman constant
    E: float = 9.0      # Wall function constant
    y_plus_switch: float = 11.0  # Switch point between viscous and log layers
    y_plus_cutoff: float = 300.0  # Cutoff for wall functions
    blending_factor: float = 0.1  # Blending factor for enhanced wall treatment
```

## Usage

### Creating Wall Functions
```python
from phasta.solver.compressible.wall_functions import (
    WallFunctionConfig,
    create_wall_functions
)

# Create configuration
config = WallFunctionConfig()

# Create wall functions
wall_functions = create_wall_functions("standard", config)  # or "enhanced" or "automatic"
```

### Computing Wall Quantities
```python
# Compute y+
y_plus = wall_functions.compute_y_plus(y, u_tau, nu)

# Compute u+
u_plus = wall_functions.compute_u_plus(y_plus)

# Compute wall shear stress
tau_wall, u_tau = wall_functions.compute_tau_wall(y, u, rho, mu)
```

## Integration with Turbulence Models

Wall functions are integrated with turbulence models through the `TurbulenceModel` class:

```python
class TurbulenceModel:
    def __init__(self, config: TurbulenceModelConfig):
        self.config = config
        if config.wall_function:
            self.wall_functions = create_wall_functions(
                config.wall_function_type,
                WallFunctionConfig(**config.wall_function_params)
            )
```

## Best Practices

1. **Mesh Requirements**
   - For standard wall functions: y+ > 30
   - For enhanced wall treatment: y+ < 1
   - For automatic wall treatment: any y+

2. **Model Selection**
   - Use standard wall functions for simple flows and coarse meshes
   - Use enhanced wall treatment for complex flows and fine meshes
   - Use automatic wall treatment for general-purpose applications

3. **Parameter Tuning**
   - Adjust κ and E for different wall roughness
   - Modify y+_switch for different flow regimes
   - Tune blending_factor for enhanced wall treatment

## Validation

The wall functions implementation has been validated against:
1. Channel flow at Reτ = 395
2. Flat plate boundary layer
3. Backward-facing step flow

## References

1. Launder, B. E., & Spalding, D. B. (1974). The numerical computation of turbulent flows.
2. Menter, F. R. (1994). Two-equation eddy-viscosity turbulence models for engineering applications.
3. Knopp, T., et al. (2006). A new wall function strategy for complex turbulent flows. 
# Turbulence Models

This document describes the turbulence modeling capabilities in PHASTA.

## Overview

PHASTA supports a wide range of turbulence models, from simple algebraic models to advanced LES and hybrid approaches. The implementation is modular and extensible, allowing for easy addition of new models.

## Available Models

### RANS Models

#### k-ε Model
- Standard k-ε implementation
- Realizable k-ε variant
- RNG k-ε variant
- Wall functions and low-Re modifications

#### k-ω Model
- Standard k-ω implementation
- SST variant
- Wall functions and low-Re modifications

#### Spalart-Allmaras
- One-equation model
- Original and modified variants
- Wall functions

### LES Models

#### Smagorinsky
- Standard Smagorinsky model
- Wall damping functions
- Filter width calculation

#### Dynamic Smagorinsky
- Dynamic coefficient calculation
- Test filtering
- Coefficient averaging
- Numerical stability measures

#### Wall-Adapted LES
- Wall damping
- Near-wall treatment
- Filter width adaptation

### Hybrid RANS/LES

#### Detached Eddy Simulation (DES)
- DES97 formulation
- DDES variant
- IDDES variant
- RANS-LES interface treatment

#### Wall-Modeled LES
- Wall stress modeling
- Interface treatment
- Blending functions

### Dynamic Subgrid Models

#### Dynamic Coefficient Calculation
- Germano identity
- Lilly's method
- Coefficient averaging
- Numerical stability

#### Mixed Dynamic Models
- Dynamic Smagorinsky
- Dynamic mixed scale
- Dynamic one-equation

## Implementation Details

### Model Selection
```python
config = TurbulenceModelConfig(
    model_type="les",  # or "rans"
    model_name="dynamic_smagorinsky",  # model specific name
    wall_function=True,
    wall_function_type="automatic",
    model_params={
        "cs": 0.17,  # Smagorinsky constant
        "filter_width_ratio": 2.0
    }
)
```

### Wall Treatment
- Automatic wall treatment
- Enhanced wall treatment
- Wall roughness modeling
- Wall functions

### Numerical Methods
- Filter width calculation
- Test filtering
- Coefficient averaging
- Interface treatment
- Blending functions

## Best Practices

### Model Selection
1. For wall-bounded flows:
   - Use wall-modeled LES or hybrid RANS/LES
   - Consider wall functions for high-Re flows
2. For free shear flows:
   - Use standard LES or dynamic models
3. For complex flows:
   - Use hybrid RANS/LES with appropriate interface treatment

### Grid Requirements
1. LES:
   - First cell y+ < 1 for wall-resolved LES
   - Adequate resolution in shear layers
2. Hybrid RANS/LES:
   - RANS region: y+ < 1
   - LES region: Δ+ < 100
3. Wall-modeled LES:
   - First cell y+ < 100
   - Adequate resolution in outer layer

### Numerical Settings
1. Time step:
   - CFL < 1 for LES
   - Smaller time steps for dynamic models
2. Filter width:
   - Use appropriate filter width ratio
   - Consider grid anisotropy
3. Averaging:
   - Use sufficient averaging steps
   - Consider local averaging

## Validation

### Benchmark Cases
1. Channel flow
2. Backward-facing step
3. Periodic hill
4. NACA 0012 airfoil
5. Cylinder flow

### Performance Metrics
1. Mean velocity profiles
2. Reynolds stresses
3. Wall shear stress
4. Separation/reattachment
5. Pressure distribution

## Future Work

### Planned Improvements
1. Advanced wall modeling
2. Improved interface treatment
3. Better numerical stability
4. Performance optimization
5. Additional models

### Research Directions
1. Machine learning for model coefficients
2. Adaptive model selection
3. Uncertainty quantification
4. Multi-scale modeling
5. GPU acceleration 
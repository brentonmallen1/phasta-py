# Wall Functions

This document describes the wall function implementations in PHASTA.

## Overview

Wall functions are used to model the near-wall region in turbulent flows, allowing for coarser grids while maintaining accuracy. PHASTA implements various wall function approaches suitable for different flow regimes and turbulence models.

## Available Wall Functions

### Automatic Wall Treatment

#### Features
- Automatic switching between wall functions and low-Re approach
- Smooth blending between regions
- Works with all turbulence models
- Handles transition regions

#### Implementation
```python
config = WallFunctionConfig(
    type="automatic",
    blending_function="exponential",
    y_plus_transition=11.0,
    blending_factor=0.1
)
```

### Enhanced Wall Treatment

#### Features
- Two-layer approach
- Viscous sublayer resolution
- Buffer layer treatment
- Log-law region modeling

#### Implementation
```python
config = WallFunctionConfig(
    type="enhanced",
    y_plus_cutoff=5.0,
    buffer_layer_thickness=30.0,
    blending_function="tanh"
)
```

### Wall Roughness

#### Features
- Equivalent sand grain roughness
- Roughness function
- Roughness parameterization
- Transitional roughness

#### Implementation
```python
config = WallFunctionConfig(
    type="roughness",
    roughness_height=1e-5,
    roughness_constant=0.5,
    roughness_type="uniform"
)
```

## Implementation Details

### Wall Function Selection
1. Automatic:
   - General purpose
   - Works with all models
   - Handles transition
2. Enhanced:
   - Better near-wall resolution
   - More accurate for complex flows
   - Higher computational cost
3. Roughness:
   - For rough walls
   - Industrial applications
   - Environmental flows

### Numerical Methods

#### Wall Distance Calculation
- Exact distance calculation
- Approximate methods
- Parallel implementation
- Efficient algorithms

#### Wall Shear Stress
- Local wall shear stress
- Wall-normal gradient
- Friction velocity
- Wall units

#### Blending Functions
- Exponential blending
- Hyperbolic tangent
- Polynomial blending
- Smooth transitions

## Best Practices

### Grid Requirements
1. Automatic:
   - y+ < 300
   - At least 10 cells in boundary layer
2. Enhanced:
   - y+ < 5 in viscous sublayer
   - 20-30 cells in boundary layer
3. Roughness:
   - y+ > 30
   - Adequate resolution of roughness

### Model Compatibility
1. RANS:
   - All wall functions
   - Model-specific modifications
2. LES:
   - Wall-modeled LES
   - Hybrid approaches
3. Hybrid RANS/LES:
   - Interface treatment
   - Blending functions

### Numerical Settings
1. Time step:
   - CFL < 1
   - Smaller near walls
2. Convergence:
   - Stricter near walls
   - Monitor wall quantities
3. Initialization:
   - Proper wall distance
   - Initial y+ distribution

## Validation

### Benchmark Cases
1. Flat plate
2. Channel flow
3. Backward-facing step
4. Periodic hill
5. Rough wall channel

### Performance Metrics
1. Wall shear stress
2. Velocity profiles
3. Friction coefficient
4. Heat transfer
5. Separation/reattachment

## Future Work

### Planned Improvements
1. Advanced blending functions
2. Better transition handling
3. Improved roughness modeling
4. Performance optimization
5. Additional wall functions

### Research Directions
1. Machine learning for wall functions
2. Adaptive wall treatment
3. Uncertainty quantification
4. Multi-scale wall modeling
5. GPU acceleration 
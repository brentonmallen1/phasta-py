# Numerical Methods

This document describes the numerical methods implemented in PHASTA-Py for solving the compressible Navier-Stokes equations.

## Time Integration

### Explicit Methods
- **Runge-Kutta Methods (RK2, RK3, RK4)**
  - Classical explicit time integration schemes
  - Order of accuracy matches the scheme number
  - CFL-based timestep selection
  - Suitable for smooth flows

- **Strong Stability Preserving (SSP-RK3)**
  - Third-order accurate
  - Preserves strong stability properties
  - Better suited for problems with discontinuities
  - Recommended for shock-capturing

- **Total Variation Diminishing (TVD-RK3)**
  - Third-order accurate
  - Prevents spurious oscillations
  - Ideal for problems with shocks
  - Combines well with TVD limiters

### Implicit Methods
- **Backward Euler**
  - First-order accurate
  - Unconditionally stable
  - Suitable for stiff problems
  - Requires linear/nonlinear solver

- **Crank-Nicolson**
  - Second-order accurate
  - Unconditionally stable
  - Better accuracy than Backward Euler
  - Requires linear/nonlinear solver

### Advanced Time Integration
- **Dual Time-Stepping**
  - Combines physical and pseudo-time
  - Accelerates convergence to steady state
  - Suitable for both steady and unsteady problems
  - Flexible choice of pseudo-time integrator

- **Local Time-Stepping**
  - Element-wise timestep selection
  - Improves efficiency for multi-scale problems
  - Based on local CFL condition
  - Maintains stability across elements

## Shock Capturing

### TVD Schemes
- **Superbee Limiter**
  - Second-order accurate
  - Sharp shock resolution
  - Prevents oscillations
  - Good for strong shocks

### WENO Reconstruction
- **Fifth-Order WENO**
  - High-order accuracy in smooth regions
  - Non-oscillatory near discontinuities
  - Smoothness-based stencil selection
  - Robust for complex flows

### Artificial Viscosity
- **Second-Derivative Based**
  - Adaptive coefficient
  - Shock detection
  - Smooth transition regions
  - Minimal dissipation in smooth regions

## Limiting Strategies

### Slope Limiters
- **Minmod Limiter**
  - First-order accurate at extrema
  - Second-order in smooth regions
  - Conservative
  - Good for general-purpose use

### Flux Limiters
- **Superbee Limiter**
  - Sharp shock resolution
  - Second-order accurate
  - TVD property
  - Good for strong shocks

- **Van Leer Limiter**
  - Smooth limiter
  - Second-order accurate
  - Less aggressive than Superbee
  - Good for moderate shocks

### Pressure Limiters
- **Van Leer Based**
  - Ensures positive pressure
  - Prevents pressure oscillations
  - Conservative
  - Good for compressible flows

## Usage Examples

### Basic Time Integration
```python
from phasta.solver.compressible import TimeIntegrationConfig, create_time_integrator

# Create configuration
config = TimeIntegrationConfig(
    scheme="explicit_rk",
    order=4,
    cfl=0.5
)

# Create integrator
integrator = create_time_integrator(config)

# Use in solver
dt = integrator.compute_timestep(solution, mesh)
solution = integrator.integrate(solution, mesh, residual)
```

### Shock Capturing
```python
from phasta.solver.compressible import ShockCapturingConfig, create_shock_capturing

# Create configuration
config = ShockCapturingConfig(
    scheme="weno",
    kappa=1.0,
    epsilon=1e-6
)

# Create shock capturing scheme
scheme = create_shock_capturing(config)

# Use in solver
u_left, u_right = scheme.compute_reconstruction(solution, dx)
```

### Limiting
```python
from phasta.solver.compressible import LimiterConfig, create_limiter

# Create configuration
config = LimiterConfig(
    scheme="slope",
    beta=1.0,
    epsilon=1e-6
)

# Create limiter
limiter = create_limiter(config)

# Use in solver
du_limited = limiter.compute_slope_limiter(solution, dx)
```

## Best Practices

1. **Time Integration Selection**
   - Use explicit methods for smooth flows
   - Use implicit methods for stiff problems
   - Consider dual time-stepping for steady problems
   - Use local time-stepping for multi-scale problems

2. **Shock Capturing**
   - Use TVD schemes for strong shocks
   - Use WENO for high-order accuracy
   - Use artificial viscosity as a backup
   - Combine with appropriate limiters

3. **Limiting**
   - Use slope limiters for general-purpose
   - Use flux limiters for strong shocks
   - Use pressure limiters for compressible flows
   - Adjust parameters based on problem

## References

1. Shu, C.-W. (1998). Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws. In Advanced numerical approximation of nonlinear hyperbolic equations (pp. 325-432).

2. Sweby, P. K. (1984). High resolution schemes using flux limiters for hyperbolic conservation laws. SIAM Journal on Numerical Analysis, 21(5), 995-1011.

3. Van Leer, B. (1979). Towards the ultimate conservative difference scheme. V. A second-order sequel to Godunov's method. Journal of Computational Physics, 32(1), 101-136. 
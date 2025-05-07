# Integration Tests

This document describes the integration tests implemented in PHASTA-Py to verify the correctness of the compressible flow solver implementation.

## Overview

Integration tests verify that different components of the solver work together correctly by comparing numerical solutions with analytical solutions or experimental data. Each test case is designed to validate specific aspects of the solver:

1. Flow over flat plate (Blasius solution)
2. Flow over cylinder (drag and wake)
3. Shock tube (Sod problem)
4. Supersonic flow over wedge (oblique shock)

## Test Cases

### 1. Flow over Flat Plate

**Purpose**: Validates the solver's ability to capture boundary layer development and viscous effects.

**Test Details**:
- Compares numerical solution with Blasius solution
- Validates boundary layer thickness
- Checks velocity profile accuracy
- Tests convergence with mesh refinement

**Key Parameters**:
- Reynolds number: Re_x = 1000
- Mach number: M = 0.2
- Wall temperature: T_w = 300K
- Inlet pressure: p = 1 atm

**Validation Metrics**:
- Maximum velocity profile error < 10%
- Boundary layer thickness error < 10%
- Second-order convergence rate

### 2. Flow over Cylinder

**Purpose**: Validates the solver's ability to handle separated flows and compute aerodynamic forces.

**Test Details**:
- Compares drag coefficient with experimental data
- Validates wake characteristics
- Checks recirculation zone
- Tests convergence with mesh refinement

**Key Parameters**:
- Reynolds number: Re_D = 100
- Mach number: M = 0.2
- Cylinder diameter: D = 1.0
- Wall temperature: T_w = 300K

**Validation Metrics**:
- Drag coefficient error < 20%
- Wake width error < 20%
- Presence of recirculation zone
- Convergence of drag coefficient

### 3. Shock Tube (Sod Problem)

**Purpose**: Validates the solver's ability to capture shocks, contact discontinuities, and expansion fans.

**Test Details**:
- Compares with exact solution
- Validates shock position and strength
- Checks contact discontinuity
- Tests convergence with mesh refinement

**Key Parameters**:
- Left state: ρ = 1.0, u = 0.0, p = 1.0
- Right state: ρ = 0.125, u = 0.0, p = 0.1
- Specific heat ratio: γ = 1.4

**Validation Metrics**:
- Maximum density error < 10%
- Maximum velocity error < 10%
- Maximum pressure error < 10%
- Shock position error < 5%
- Contact discontinuity position error < 5%

### 4. Supersonic Flow over Wedge

**Purpose**: Validates the solver's ability to capture oblique shocks and compute post-shock conditions.

**Test Details**:
- Compares with oblique shock theory
- Validates shock angle
- Checks post-shock conditions
- Tests convergence with mesh refinement

**Key Parameters**:
- Inlet Mach number: M = 2.0
- Wedge angle: θ = 15°
- Inlet pressure: p = 1 atm
- Inlet temperature: T = 300K

**Validation Metrics**:
- Shock angle error < 2°
- Post-shock Mach number error < 10%
- Pressure ratio error < 10%
- Density ratio error < 10%

## Running Tests

To run all integration tests:
```bash
pytest phasta-py/phasta/solver/compressible/tests/test_*.py -v
```

To run a specific test:
```bash
pytest phasta-py/phasta/solver/compressible/tests/test_flat_plate.py -v
```

## Test Output

Each test generates:
1. Numerical solution data
2. Comparison with analytical/exact solution
3. Error metrics
4. Convergence study results

## Best Practices

1. **Mesh Resolution**:
   - Start with coarse mesh for quick testing
   - Use fine mesh for final validation
   - Perform convergence study with multiple mesh sizes

2. **Convergence Criteria**:
   - Use appropriate convergence tolerance (typically 1e-6)
   - Check both residual and solution convergence
   - Monitor conservation properties

3. **Error Analysis**:
   - Compute both pointwise and integral errors
   - Check convergence rates
   - Compare with theoretical expectations

4. **Visualization**:
   - Plot solution profiles
   - Compare with analytical solutions
   - Visualize error distributions

## References

1. Blasius, H. (1908). Grenzschichten in Flüssigkeiten mit kleiner Reibung. Zeitschrift für Mathematik und Physik, 56(1), 1-37.

2. Sod, G. A. (1978). A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws. Journal of Computational Physics, 27(1), 1-31.

3. Anderson, J. D. (2019). Fundamentals of Aerodynamics. McGraw-Hill Education.

4. Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics. Springer. 
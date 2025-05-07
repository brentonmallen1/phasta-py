# Validation Cases

This document describes the validation cases used to verify the accuracy and reliability of the PHASTA-Py compressible flow solver.

## Transonic Flow over NACA 0012 Airfoil

### Overview
This case validates the solver's ability to handle mixed subsonic/supersonic flows with shock waves. The NACA 0012 airfoil is a standard test case for transonic flow validation.

### Flow Conditions
- Mach number: 0.8
- Angle of attack: 1.25°
- Reynolds number: 9.0 × 10^6
- Temperature: 300 K
- Pressure: 1 atm

### Validation Metrics
1. Pressure coefficient distribution
2. Shock location
3. Lift coefficient
4. Drag coefficient

### Expected Results
- Maximum pressure coefficient error: < 0.1
- RMS pressure coefficient error: < 0.05
- Shock location error: < 0.05 chord
- Lift coefficient error: < 0.05
- Drag coefficient error: < 0.005

### Mesh Requirements
- Minimum resolution: 200 × 100 points
- First cell height: 1e-6 chord
- Domain size: 20 chord lengths upstream, 10 chord lengths above/below

## Supersonic Flow over Cone

### Overview
This case validates the solver's ability to handle supersonic flows with conical shock waves. The cone is a fundamental geometry for supersonic flow validation.

### Flow Conditions
- Mach number: 2.0
- Cone half-angle: 10°
- Reynolds number: 1.0 × 10^6
- Temperature: 300 K
- Pressure: 1 atm

### Validation Metrics
1. Shock angle
2. Surface pressure distribution
3. Surface heat transfer
4. Boundary layer thickness

### Expected Results
- Shock angle error: < 1°
- Surface pressure error: < 5%
- Heat transfer error: < 10%
- Boundary layer thickness error: < 5%

### Mesh Requirements
- Minimum resolution: 300 × 150 points
- First cell height: 1e-6 cone length
- Domain size: 5 cone lengths upstream, 3 cone lengths above

## Hypersonic Flow over Blunt Body

### Overview
This case validates the solver's ability to handle hypersonic flows with strong shock waves and high-temperature effects. The blunt body is a standard test case for hypersonic flow validation.

### Flow Conditions
- Mach number: 8.0
- Body radius: 1.0 m
- Reynolds number: 1.0 × 10^6
- Temperature: 300 K
- Pressure: 1 atm

### Validation Metrics
1. Stand-off distance
2. Surface pressure distribution
3. Surface heat transfer
4. Shock layer thickness

### Expected Results
- Stand-off distance error: < 5%
- Surface pressure error: < 10%
- Heat transfer error: < 15%
- Shock layer thickness error: < 10%

### Mesh Requirements
- Minimum resolution: 400 × 200 points
- First cell height: 1e-6 body radius
- Domain size: 10 body radii upstream, 5 body radii above

## Turbulent Boundary Layer

### Overview
This case validates the solver's ability to handle turbulent boundary layers. The flat plate is a standard test case for turbulent boundary layer validation.

### Flow Conditions
- Mach number: 0.2
- Reynolds number: 1.0 × 10^7
- Temperature: 300 K
- Pressure: 1 atm

### Validation Metrics
1. Velocity profile
2. Skin friction coefficient
3. Boundary layer thickness
4. Displacement thickness

### Expected Results
- Velocity profile error: < 5%
- Skin friction error: < 10%
- Boundary layer thickness error: < 5%
- Displacement thickness error: < 5%

### Mesh Requirements
- Minimum resolution: 500 × 200 points
- First cell height: y+ < 1
- Domain size: 10 plate lengths, 5 plate lengths above

## Shock-Boundary Layer Interaction

### Overview
This case validates the solver's ability to handle shock-boundary layer interactions. The compression ramp is a standard test case for shock-boundary layer interaction validation.

### Flow Conditions
- Mach number: 2.25
- Ramp angle: 24°
- Reynolds number: 5.0 × 10^6
- Temperature: 300 K
- Pressure: 1 atm

### Validation Metrics
1. Separation length
2. Surface pressure distribution
3. Surface heat transfer
4. Boundary layer recovery

### Expected Results
- Separation length error: < 10%
- Surface pressure error: < 5%
- Heat transfer error: < 15%
- Boundary layer recovery error: < 10%

### Mesh Requirements
- Minimum resolution: 600 × 300 points
- First cell height: y+ < 1
- Domain size: 10 ramp lengths upstream, 5 ramp lengths above

## Running Validation Cases

To run a validation case:

```python
from phasta.solver.compressible.tests import (
    test_airfoil,
    test_cone,
    test_blunt_body,
    test_boundary_layer,
    test_shock_interaction
)

# Run airfoil test
test_airfoil.test_airfoil_flow()

# Run cone test
test_cone.test_cone_flow()

# Run blunt body test
test_blunt_body.test_blunt_body_flow()

# Run boundary layer test
test_boundary_layer.test_boundary_layer_flow()

# Run shock interaction test
test_shock_interaction.test_shock_interaction_flow()
```

## Best Practices

1. **Mesh Resolution**
   - Use appropriate mesh resolution for each case
   - Ensure first cell height is small enough for boundary layer resolution
   - Use mesh refinement in regions of high gradients

2. **Convergence Criteria**
   - Use appropriate convergence criteria for each case
   - Monitor residuals and solution changes
   - Check conservation of mass, momentum, and energy

3. **Error Analysis**
   - Compare with experimental data or analytical solutions
   - Compute error metrics (maximum error, RMS error)
   - Check convergence rates

4. **Visualization**
   - Plot pressure contours
   - Plot velocity vectors
   - Plot temperature contours
   - Plot boundary layer profiles

## References

1. AGARD-AR-138, "A Selection of Experimental Test Cases for the Validation of CFD Codes"
2. AIAA Paper 79-1478, "Experimental Investigation of Transonic Flow over a NACA 0012 Airfoil"
3. AIAA Paper 85-0163, "Experimental Investigation of Supersonic Flow over a Cone"
4. AIAA Paper 89-1739, "Experimental Investigation of Hypersonic Flow over a Blunt Body"
5. AIAA Paper 93-0287, "Experimental Investigation of Turbulent Boundary Layer Flow"
6. AIAA Paper 95-0587, "Experimental Investigation of Shock-Boundary Layer Interaction" 
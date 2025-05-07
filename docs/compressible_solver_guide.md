# PHASTA-Py Compressible Flow Solver User Guide

This guide provides detailed information for users of the compressible flow solver in PHASTA-Py.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Input/Output](#inputoutput)
5. [Post-processing](#post-processing)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## Introduction

The compressible flow solver in PHASTA-Py is a high-performance computational fluid dynamics (CFD) solver for compressible flows. It supports:

- 2D and 3D simulations
- Structured and unstructured meshes
- Various boundary conditions
- Multiple turbulence models
- Parallel computing

### Key Features

- High-order spatial discretization
- Multiple time integration schemes
- TVD schemes for shock capturing
- RANS and LES turbulence modeling
- VTK output for visualization

## Installation

### Requirements

- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib
- H5Py
- MPI4Py (for parallel computing)

### Installation Steps

1. Install Python dependencies:

```bash
pip install numpy scipy matplotlib h5py mpi4py
```

2. Install PHASTA-Py:

```bash
pip install phasta-py
```

3. Verify installation:

```python
import phasta
print(phasta.__version__)
```

## Configuration

### Solver Configuration

The solver is configured using a dictionary:

```python
config = {
    # Physical parameters
    "gamma": 1.4,  # Ratio of specific heats
    "prandtl": 0.72,  # Prandtl number
    
    # Numerical parameters
    "cfl": 0.5,  # CFL number
    "time_integration": "rk4",  # Time integration scheme
    
    # Turbulence modeling
    "turbulence_model": {
        "type": "rans",  # RANS or LES
        "model": "k-epsilon",  # Turbulence model
        "wall_function": True  # Use wall functions
    }
}
```

### Available Options

#### Time Integration Schemes

- `"euler"`: First-order explicit Euler
- `"rk2"`: Second-order Runge-Kutta
- `"rk4"`: Fourth-order Runge-Kutta
- `"bdf2"`: Second-order backward difference

#### Turbulence Models

RANS Models:
- `"k-epsilon"`: Standard k-epsilon model
- `"k-omega"`: Standard k-omega model
- `"sst"`: Menter's SST model

LES Models:
- `"smagorinsky"`: Smagorinsky model
- `"wale"`: WALE model
- `"dynamic"`: Dynamic Smagorinsky model

### Boundary Conditions

#### Inlet Boundary

```python
from phasta.solver.compressible.boundary_conditions import InletBoundary

# Subsonic inlet
inlet = InletBoundary(
    mach=0.5,  # Mach number
    pressure=101325.0,  # Pressure (Pa)
    temperature=300.0  # Temperature (K)
)
```

#### Outlet Boundary

```python
from phasta.solver.compressible.boundary_conditions import OutletBoundary

# Subsonic outlet with back pressure
outlet = OutletBoundary(
    pressure=101325.0  # Pressure (Pa), None for supersonic
)
```

#### Wall Boundary

```python
from phasta.solver.compressible.boundary_conditions import WallBoundary

# Isothermal wall
wall = WallBoundary(
    temperature=300.0,  # Temperature (K)
    isothermal=True  # Isothermal or adiabatic
)
```

#### Symmetry Boundary

```python
from phasta.solver.compressible.boundary_conditions import SymmetryBoundary

# Symmetry plane
symmetry = SymmetryBoundary()
```

## Input/Output

### Mesh Input

The solver accepts meshes in various formats:

1. Dictionary format:

```python
mesh = {
    "nodes": nodes,  # Nx3 array of node coordinates
    "elements": elements,  # Mx3 array of element connectivity
    "boundaries": {
        "inlet": inlet_nodes,  # Array of inlet node indices
        "outlet": outlet_nodes,  # Array of outlet node indices
        "wall": wall_nodes,  # Array of wall node indices
        "symmetry": symmetry_nodes  # Array of symmetry node indices
    }
}
```

2. VTK format:

```python
mesh = phasta.mesh.read_vtk("mesh.vtk")
```

3. GMSH format:

```python
mesh = phasta.mesh.read_gmsh("mesh.msh")
```

### Solution Output

The solver can output solutions in various formats:

1. VTK format:

```python
solver.save_vtk("solution.vtk", solution)
```

2. HDF5 format:

```python
solver.save_hdf5("solution.h5", solution)
```

3. CSV format:

```python
solver.save_csv("solution.csv", solution)
```

### Output Variables

The solution array contains the following variables:

- `solution[:, 0]`: Density (kg/m³)
- `solution[:, 1]`: x-momentum (kg/m²/s)
- `solution[:, 2]`: y-momentum (kg/m²/s)
- `solution[:, 3]`: z-momentum (kg/m²/s)
- `solution[:, 4]`: Total energy (J/m³)
- `solution[:, 5]`: Turbulent kinetic energy (m²/s²) (if using RANS)
- `solution[:, 6]`: Turbulent dissipation rate (m²/s³) (if using k-epsilon)

## Post-processing

### Visualization

The solver provides various visualization tools:

1. Plot solution variables:

```python
solver.plot_solution(solution, variable="pressure")
```

2. Plot residuals:

```python
solver.plot_residuals()
```

3. Plot forces:

```python
solver.plot_forces()
```

### Data Analysis

The solver provides various analysis tools:

1. Compute forces:

```python
forces = solver.compute_forces(solution)
```

2. Compute heat transfer:

```python
heat_transfer = solver.compute_heat_transfer(solution)
```

3. Compute flow statistics:

```python
stats = solver.compute_statistics(solution)
```

## Troubleshooting

### Common Issues

1. **Negative Pressure**

Error message:
```
Negative pressure detected at node 123
```

Solution:
- Increase CFL number
- Use TVD scheme
- Check boundary conditions
- Refine mesh

2. **CFL Violation**

Error message:
```
CFL condition violated: CFL = 1.2 > 0.5
```

Solution:
- Decrease time step
- Increase CFL number
- Refine mesh

3. **Turbulence Model Issues**

Error message:
```
Turbulent kinetic energy negative at node 456
```

Solution:
- Use wall functions
- Check inlet conditions
- Refine mesh near walls

### Debugging

1. Enable debug output:

```python
solver = CompressibleSolver(
    mesh=mesh,
    config={"debug": True}
)
```

2. Check residuals:

```python
residuals = solver.compute_residuals(solution)
print(residuals)
```

3. Check boundary conditions:

```python
solver.check_boundary_conditions()
```

## FAQ

### General Questions

1. **What is the maximum Mach number supported?**

The solver supports Mach numbers up to 10.0. For higher Mach numbers, use the hypersonic solver.

2. **What is the minimum Reynolds number for turbulence modeling?**

The minimum Reynolds number for turbulence modeling is 1000. For lower Reynolds numbers, use the laminar solver.

3. **What is the maximum number of processors supported?**

The solver supports up to 1024 processors for parallel computing.

### Performance Questions

1. **How can I improve performance?**

- Use appropriate mesh size
- Use parallel computing
- Use wall functions
- Use TVD schemes

2. **How can I reduce memory usage?**

- Use appropriate mesh size
- Use appropriate time step
- Use appropriate turbulence model
- Use appropriate boundary conditions

3. **How can I improve accuracy?**

- Use higher-order schemes
- Use finer mesh
- Use appropriate turbulence model
- Use appropriate boundary conditions

### Technical Questions

1. **What is the order of accuracy?**

- Spatial: 2nd order
- Temporal: 1st to 4th order
- Turbulence: 1st order

2. **What is the stability limit?**

- CFL < 1.0 for explicit schemes
- CFL < 10.0 for implicit schemes

3. **What is the convergence criterion?**

- Residual < 1e-6
- Maximum iterations = 1000

## Conclusion

This guide has covered the basics of using the compressible flow solver in PHASTA-Py. For more information, see the [Tutorial](compressible_solver_tutorial.md) and [Developer Guide](compressible_solver_developer_guide.md). 
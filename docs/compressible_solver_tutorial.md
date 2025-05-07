# PHASTA-Py Compressible Flow Solver Tutorial

This tutorial provides step-by-step instructions for using the compressible flow solver in PHASTA-Py.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Examples](#basic-examples)
3. [Advanced Examples](#advanced-examples)
4. [Best Practices](#best-practices)

## Getting Started

### Installation

First, install PHASTA-Py and its dependencies:

```bash
pip install phasta-py numpy scipy matplotlib h5py
```

### Basic Setup

Here's a minimal example to get started:

```python
import numpy as np
from phasta.solver.compressible import CompressibleSolver

# Create a simple 2D mesh
x = np.linspace(0, 1, 11)
y = np.linspace(0, 1, 11)
X, Y = np.meshgrid(x, y)
nodes = np.column_stack((X.flatten(), Y.flatten()))

# Create elements (triangles)
elements = []
for i in range(10):
    for j in range(10):
        n1 = i * 11 + j
        n2 = n1 + 1
        n3 = n1 + 11
        n4 = n3 + 1
        elements.append([n1, n2, n3])
        elements.append([n2, n4, n3])

# Create mesh dictionary
mesh = {
    "nodes": nodes,
    "elements": np.array(elements),
    "boundaries": {
        "inlet": np.arange(0, 121, 11),  # Left boundary
        "outlet": np.arange(10, 121, 11),  # Right boundary
        "wall": np.arange(0, 11),  # Bottom boundary
        "symmetry": np.arange(110, 121)  # Top boundary
    }
}

# Create solver
solver = CompressibleSolver(
    mesh=mesh,
    config={
        "gamma": 1.4,
        "prandtl": 0.72,
        "cfl": 0.5,
        "time_integration": "rk4"
    }
)

# Set initial conditions
solution = np.ones((121, 5))  # [rho, rho*u, rho*v, rho*w, rho*E]
solution[:, 1:4] = 0.0  # Zero velocity
solution[:, 4] = 2.5e5  # Energy

# Run simulation
t_end = 1.0
dt = 0.001
t = 0.0

while t < t_end:
    solution = solver.integrate_time(solution, dt)
    t += dt
```

## Basic Examples

### 1. Subsonic Flow Over a Flat Plate

```python
import numpy as np
from phasta.solver.compressible import CompressibleSolver
from phasta.solver.compressible.boundary_conditions import (
    WallBoundary, InletBoundary, OutletBoundary
)

# Create mesh (similar to previous example)
mesh = create_flat_plate_mesh()

# Create boundary conditions
inlet = InletBoundary(
    mach=0.5,
    pressure=101325.0,
    temperature=300.0
)
outlet = OutletBoundary(pressure=101325.0)
wall = WallBoundary(temperature=300.0)

# Create solver
solver = CompressibleSolver(
    mesh=mesh,
    config={
        "gamma": 1.4,
        "prandtl": 0.72,
        "cfl": 0.5,
        "time_integration": "rk4"
    }
)

# Set initial conditions
solution = np.ones((n_nodes, 5))
solution[:, 1:4] = 0.0
solution[:, 4] = 2.5e5

# Run simulation
t_end = 1.0
dt = 0.001
t = 0.0

while t < t_end:
    # Apply boundary conditions
    solution = inlet.apply(solution, mesh, mesh["boundaries"]["inlet"])
    solution = outlet.apply(solution, mesh, mesh["boundaries"]["outlet"])
    solution = wall.apply(solution, mesh, mesh["boundaries"]["wall"])
    
    # Integrate time
    solution = solver.integrate_time(solution, dt)
    t += dt
    
    # Save results periodically
    if t % 0.1 < dt:
        solver.save_vtk(f"solution_{t:.1f}.vtk", solution)
```

### 2. Supersonic Flow in a Nozzle

```python
import numpy as np
from phasta.solver.compressible import CompressibleSolver
from phasta.solver.compressible.boundary_conditions import (
    InletBoundary, OutletBoundary, WallBoundary
)

# Create mesh
mesh = create_nozzle_mesh()

# Create boundary conditions
inlet = InletBoundary(
    mach=2.0,
    pressure=101325.0,
    temperature=300.0
)
outlet = OutletBoundary(pressure=None)  # Supersonic outlet
wall = WallBoundary(temperature=300.0)

# Create solver
solver = CompressibleSolver(
    mesh=mesh,
    config={
        "gamma": 1.4,
        "prandtl": 0.72,
        "cfl": 0.5,
        "time_integration": "rk4"
    }
)

# Set initial conditions
solution = np.ones((n_nodes, 5))
solution[:, 1:4] = 0.0
solution[:, 4] = 2.5e5

# Run simulation
t_end = 1.0
dt = 0.001
t = 0.0

while t < t_end:
    # Apply boundary conditions
    solution = inlet.apply(solution, mesh, mesh["boundaries"]["inlet"])
    solution = outlet.apply(solution, mesh, mesh["boundaries"]["outlet"])
    solution = wall.apply(solution, mesh, mesh["boundaries"]["wall"])
    
    # Integrate time
    solution = solver.integrate_time(solution, dt)
    t += dt
    
    # Save results periodically
    if t % 0.1 < dt:
        solver.save_vtk(f"solution_{t:.1f}.vtk", solution)
```

## Advanced Examples

### 1. Turbulent Flow Over a NACA Airfoil

```python
import numpy as np
from phasta.solver.compressible import CompressibleSolver
from phasta.solver.compressible.boundary_conditions import (
    InletBoundary, OutletBoundary, WallBoundary
)
from phasta.solver.compressible.turbulence_models import (
    KEpsilonModel, TurbulenceModelConfig
)

# Create mesh
mesh = create_naca_mesh()

# Create boundary conditions
inlet = InletBoundary(
    mach=0.2,
    pressure=101325.0,
    temperature=300.0
)
outlet = OutletBoundary(pressure=101325.0)
wall = WallBoundary(temperature=300.0)

# Create turbulence model
turb_config = TurbulenceModelConfig(
    model_type="rans",
    model_name="k-epsilon",
    wall_function=True
)
turb_model = KEpsilonModel(turb_config)

# Create solver
solver = CompressibleSolver(
    mesh=mesh,
    config={
        "gamma": 1.4,
        "prandtl": 0.72,
        "cfl": 0.5,
        "time_integration": "rk4",
        "turbulence_model": turb_model
    }
)

# Set initial conditions
solution = np.ones((n_nodes, 7))  # [rho, rho*u, rho*v, rho*w, rho*E, k, eps]
solution[:, 1:4] = 0.0
solution[:, 4] = 2.5e5
solution[:, 5] = 1e-6  # k
solution[:, 6] = 1e-7  # eps

# Run simulation
t_end = 1.0
dt = 0.001
t = 0.0

while t < t_end:
    # Apply boundary conditions
    solution = inlet.apply(solution, mesh, mesh["boundaries"]["inlet"])
    solution = outlet.apply(solution, mesh, mesh["boundaries"]["outlet"])
    solution = wall.apply(solution, mesh, mesh["boundaries"]["wall"])
    
    # Integrate time
    solution = solver.integrate_time(solution, dt)
    t += dt
    
    # Save results periodically
    if t % 0.1 < dt:
        solver.save_vtk(f"solution_{t:.1f}.vtk", solution)
```

### 2. LES of Flow Around a Cylinder

```python
import numpy as np
from phasta.solver.compressible import CompressibleSolver
from phasta.solver.compressible.boundary_conditions import (
    InletBoundary, OutletBoundary, WallBoundary
)
from phasta.solver.compressible.turbulence_models import (
    SmagorinskyModel, TurbulenceModelConfig
)

# Create mesh
mesh = create_cylinder_mesh()

# Create boundary conditions
inlet = InletBoundary(
    mach=0.1,
    pressure=101325.0,
    temperature=300.0
)
outlet = OutletBoundary(pressure=101325.0)
wall = WallBoundary(temperature=300.0)

# Create LES model
les_config = TurbulenceModelConfig(
    model_type="les",
    model_name="smagorinsky",
    model_params={"C_s": 0.17}
)
les_model = SmagorinskyModel(les_config)

# Create solver
solver = CompressibleSolver(
    mesh=mesh,
    config={
        "gamma": 1.4,
        "prandtl": 0.72,
        "cfl": 0.5,
        "time_integration": "rk4",
        "turbulence_model": les_model
    }
)

# Set initial conditions
solution = np.ones((n_nodes, 5))
solution[:, 1:4] = 0.0
solution[:, 4] = 2.5e5

# Run simulation
t_end = 1.0
dt = 0.001
t = 0.0

while t < t_end:
    # Apply boundary conditions
    solution = inlet.apply(solution, mesh, mesh["boundaries"]["inlet"])
    solution = outlet.apply(solution, mesh, mesh["boundaries"]["outlet"])
    solution = wall.apply(solution, mesh, mesh["boundaries"]["wall"])
    
    # Integrate time
    solution = solver.integrate_time(solution, dt)
    t += dt
    
    # Save results periodically
    if t % 0.1 < dt:
        solver.save_vtk(f"solution_{t:.1f}.vtk", solution)
```

## Best Practices

### 1. Mesh Generation

- Use appropriate mesh resolution for your problem
- Refine mesh in regions of high gradients
- Use structured meshes for simple geometries
- Use unstructured meshes for complex geometries

### 2. Boundary Conditions

- Use appropriate boundary conditions for your problem
- Set inlet conditions based on flow regime (subsonic/supersonic)
- Use wall functions for turbulent flows
- Use symmetry conditions when possible

### 3. Time Integration

- Use appropriate time step based on CFL condition
- Use higher-order time integration for accuracy
- Use TVD schemes for shock capturing
- Monitor residuals for convergence

### 4. Turbulence Modeling

- Use RANS models for steady flows
- Use LES models for unsteady flows
- Use wall functions for high Reynolds numbers
- Validate results against experimental data

### 5. Post-processing

- Save results regularly
- Use appropriate visualization tools
- Compute relevant flow quantities
- Compare with analytical solutions

### 6. Performance

- Use appropriate mesh size
- Use parallel computing for large problems
- Profile code for bottlenecks
- Optimize critical sections

### 7. Debugging

- Enable debug output
- Check boundary conditions
- Monitor solution variables
- Use validation cases

### 8. Documentation

- Document your setup
- Document your results
- Document your conclusions
- Share your code

## Conclusion

This tutorial has covered the basics of using the compressible flow solver in PHASTA-Py. For more information, see the [User Guide](compressible_solver_guide.md) and [Developer Guide](compressible_solver_developer_guide.md). 
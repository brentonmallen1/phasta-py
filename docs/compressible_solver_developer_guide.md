# PHASTA-Py Compressible Flow Solver Developer Guide

This guide provides detailed information for developers working on the compressible flow solver in PHASTA-Py.

## Table of Contents
1. [Architecture](#architecture)
2. [Code Organization](#code-organization)
3. [Core Components](#core-components)
4. [Adding New Features](#adding-new-features)
5. [Testing](#testing)
6. [Performance Optimization](#performance-optimization)
7. [Debugging](#debugging)
8. [Contributing](#contributing)

## Architecture

The compressible flow solver follows a modular architecture with the following key components:

```
phasta-py/
├── solver/
│   └── compressible/
│       ├── solver.py           # Main solver class
│       ├── boundary_conditions.py  # Boundary condition implementations
│       ├── turbulence_models.py    # Turbulence model implementations
│       ├── flux.py             # Flux computation
│       ├── time_integration.py # Time integration schemes
│       └── validation/         # Validation cases
├── mesh/                      # Mesh handling
├── utils/                     # Utility functions
└── tests/                     # Test suite
```

### Key Design Principles

1. **Modularity**: Each component is self-contained and has a well-defined interface
2. **Extensibility**: New features can be added by implementing new classes
3. **Performance**: Critical sections are optimized for speed
4. **Maintainability**: Code is well-documented and follows consistent style

## Code Organization

### Main Solver Class

The `CompressibleSolver` class in `solver.py` is the main entry point:

```python
class CompressibleSolver:
    def __init__(self, mesh, config):
        """Initialize solver.
        
        Args:
            mesh: Mesh data
            config: Solver configuration
        """
        self.mesh = mesh
        self.config = config
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize solver components."""
        self.boundary_conditions = self._create_boundary_conditions()
        self.turbulence_model = self._create_turbulence_model()
        self.flux_scheme = self._create_flux_scheme()
        self.time_integrator = self._create_time_integrator()
        
    def integrate_time(self, solution, dt):
        """Integrate solution in time.
        
        Args:
            solution: Current solution
            dt: Time step
            
        Returns:
            Updated solution
        """
        return self.time_integrator.integrate(solution, dt)
```

### Boundary Conditions

Boundary conditions are implemented in `boundary_conditions.py`:

```python
class BoundaryCondition:
    def apply(self, solution, mesh, bc_nodes):
        """Apply boundary condition.
        
        Args:
            solution: Current solution
            mesh: Mesh data
            bc_nodes: Boundary node indices
            
        Returns:
            Modified solution
        """
        raise NotImplementedError
```

### Turbulence Models

Turbulence models are implemented in `turbulence_models.py`:

```python
class TurbulenceModel:
    def compute_eddy_viscosity(self, solution, mesh, grad_u):
        """Compute eddy viscosity.
        
        Args:
            solution: Current solution
            mesh: Mesh data
            grad_u: Velocity gradient
            
        Returns:
            Eddy viscosity
        """
        raise NotImplementedError
```

## Core Components

### Flux Computation

The flux computation is handled by the `FluxScheme` class:

```python
class FluxScheme:
    def compute_flux(self, left_state, right_state, normal):
        """Compute flux between two states.
        
        Args:
            left_state: Left state
            right_state: Right state
            normal: Face normal
            
        Returns:
            Flux vector
        """
        raise NotImplementedError
```

### Time Integration

Time integration schemes are implemented in `time_integration.py`:

```python
class TimeIntegrator:
    def integrate(self, solution, dt):
        """Integrate solution in time.
        
        Args:
            solution: Current solution
            dt: Time step
            
        Returns:
            Updated solution
        """
        raise NotImplementedError
```

## Adding New Features

### Adding a New Boundary Condition

1. Create a new class in `boundary_conditions.py`:

```python
class NewBoundary(BoundaryCondition):
    def __init__(self, param1, param2):
        super().__init__("new_boundary")
        self.param1 = param1
        self.param2 = param2
        
    def apply(self, solution, mesh, bc_nodes):
        # Implement boundary condition
        return modified_solution
```

2. Add tests in `tests/test_boundary_conditions.py`:

```python
def test_new_boundary():
    boundary = NewBoundary(param1=1.0, param2=2.0)
    solution = np.ones((100, 5))
    modified = boundary.apply(solution, mesh, bc_nodes)
    # Add assertions
```

### Adding a New Turbulence Model

1. Create a new class in `turbulence_models.py`:

```python
class NewTurbulenceModel(TurbulenceModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize model parameters
        
    def compute_eddy_viscosity(self, solution, mesh, grad_u):
        # Implement eddy viscosity computation
        return mu_t
        
    def compute_source_terms(self, solution, mesh, grad_u):
        # Implement source terms
        return source_terms
```

2. Add tests in `tests/test_turbulence_models.py`:

```python
def test_new_turbulence_model():
    config = TurbulenceModelConfig(
        model_type="rans",
        model_name="new_model"
    )
    model = NewTurbulenceModel(config)
    # Add tests
```

## Testing

### Unit Tests

Write unit tests for each component:

```python
class TestFluxScheme(unittest.TestCase):
    def setUp(self):
        self.flux_scheme = FluxScheme()
        
    def test_flux_computation(self):
        left_state = np.array([1.0, 0.0, 0.0, 0.0, 2.5e5])
        right_state = np.array([0.125, 0.0, 0.0, 0.0, 0.25e5])
        normal = np.array([1.0, 0.0, 0.0])
        
        flux = self.flux_scheme.compute_flux(left_state, right_state, normal)
        self.assertEqual(flux.shape, (5,))
        self.assertTrue(np.all(np.isfinite(flux)))
```

### Integration Tests

Write integration tests for the full solver:

```python
class TestCompressibleSolver(unittest.TestCase):
    def setUp(self):
        self.mesh = create_test_mesh()
        self.solver = CompressibleSolver(
            mesh=self.mesh,
            config={"cfl": 0.5}
        )
        
    def test_sod_shock_tube(self):
        solution = create_sod_initial_conditions()
        t_end = 0.2
        dt = 0.001
        
        while t < t_end:
            solution = self.solver.integrate_time(solution, dt)
            t += dt
            
        # Check results
        self.assertTrue(check_shock_position(solution))
        self.assertTrue(check_pressure_jump(solution))
```

### Validation Cases

Add validation cases in `validation/`:

```python
class ValidationCase:
    def __init__(self):
        self.mesh = self._create_mesh()
        self.solver = self._create_solver()
        
    def run(self):
        solution = self._set_initial_conditions()
        return self._run_simulation(solution)
        
    def plot_results(self):
        # Plot results
        pass
```

## Performance Optimization

### Profiling

Use Python's profiling tools to identify bottlenecks:

```python
import cProfile
import pstats

def profile_solver():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run solver
    solver = CompressibleSolver(mesh, config)
    solution = solver.run()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()
```

### Optimization Techniques

1. **Vectorization**
   - Use NumPy operations instead of loops
   - Avoid Python-level loops in critical sections

2. **Memory Management**
   - Pre-allocate arrays
   - Use views instead of copies
   - Minimize temporary allocations

3. **Parallel Computing**
   - Use MPI for distributed memory
   - Use OpenMP for shared memory
   - Use GPU acceleration (coming soon)

## Debugging

### Debug Output

Enable debug output in the solver:

```python
class CompressibleSolver:
    def __init__(self, mesh, config):
        self.debug = config.get("debug", False)
        
    def _debug_print(self, message):
        if self.debug:
            print(f"[DEBUG] {message}")
```

### Error Handling

Implement proper error handling:

```python
class SolverError(Exception):
    """Base class for solver errors."""
    pass

class CFLViolationError(SolverError):
    """Raised when CFL condition is violated."""
    pass

class NegativePressureError(SolverError):
    """Raised when negative pressure is detected."""
    pass
```

### Logging

Use Python's logging module:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class CompressibleSolver:
    def integrate_time(self, solution, dt):
        logger.debug(f"Integrating time step: dt={dt}")
        # ...
        logger.info(f"Time step completed: t={self.t}")
```

## Contributing

### Code Style

Follow PEP 8 style guide:

```python
# Good
def compute_flux(left_state, right_state, normal):
    """Compute flux between states."""
    return flux

# Bad
def computeFlux(leftState,rightState,normal):
    return flux
```

### Documentation

Document all public interfaces:

```python
def compute_flux(left_state, right_state, normal):
    """Compute flux between two states.
    
    Args:
        left_state: Left state vector [rho, rho*u, rho*v, rho*w, rho*E]
        right_state: Right state vector
        normal: Face normal vector
        
    Returns:
        Flux vector
        
    Raises:
        ValueError: If states are invalid
    """
    pass
```

### Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Update documentation
6. Submit pull request

### Code Review

Review checklist:
- [ ] Code follows style guide
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No performance regression
- [ ] No memory leaks
- [ ] Error handling implemented
- [ ] Logging added 
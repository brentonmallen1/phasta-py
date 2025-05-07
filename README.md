# PHASTA Python Implementation

This is a Python implementation of PHASTA (Parallel Hierarchic Adaptive Stabilized Transient Analysis), a Computational Fluid Dynamics (CFD) solver that supports modeling compressible or incompressible, laminar or turbulent, steady or unsteady flows in 3D using unstructured grids.

## Features

- Incompressible and compressible flow solvers
- Support for both CPU and GPU acceleration (NVIDIA CUDA and Apple Metal)
- Parallel computing capabilities via MPI
- Modern Python interface with NumPy/SciPy integration
- Comprehensive test suite and validation cases

## Installation

### Prerequisites

- Python 3.9 or higher
- MPI implementation (e.g., OpenMPI, MPICH)
- For GPU support:
  - NVIDIA GPU with CUDA support, or
  - Apple Silicon with Metal support

### Basic Installation

```bash
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Usage

Basic example of running a simulation:

```python
from phasta import Solver, Mesh, Config

# Create configuration
config = Config(
    solver_type="incompressible",
    time_step=0.001,
    num_steps=1000
)

# Load mesh
mesh = Mesh.from_file("path/to/mesh.vtk")

# Create and run solver
solver = Solver(config, mesh)
solver.run()
```

## Development

### Running Tests

```bash
pytest
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- pylint for linting

To run all style checks:

```bash
black .
isort .
mypy .
pylint phasta
```

## License

See the LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

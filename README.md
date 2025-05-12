# PHASTA Python Implementation

This is a Python implementation of PHASTA (Parallel Hierarchic Adaptive Stabilized Transient Analysis), a Computational Fluid Dynamics (CFD) solver that supports modeling compressible or incompressible, laminar or turbulent, steady or unsteady flows in 3D using unstructured grids.

## Features

- Incompressible and compressible flow solvers
- Support for both CPU and GPU acceleration (NVIDIA CUDA and Apple Metal)
- Parallel computing capabilities via MPI
- Modern Python interface with NumPy/SciPy integration
- Comprehensive test suite and validation cases

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/phasta-py.git
cd phasta-py
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Install core dependencies:
```bash
pip install numpy scipy pandas matplotlib pyvista vtk
```

### Running Your First Example

The lid-driven cavity case is a good starting point. This example demonstrates:
- Basic mesh generation
- Incompressible flow solver
- Basic visualization

1. Navigate to the examples directory:
```bash
cd examples/notebooks
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Open `lid_driven_cavity.ipynb` and run the cells in sequence.

### Optional Dependencies

For advanced features, you may want to install additional dependencies:

```bash
# GPU acceleration
pip install pycuda pyopencl metal-python

# Parallel computing
pip install mpi4py h5py netCDF4

# Advanced visualization
pip install pyvista vtk
```

### Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Check your Python version:
```bash
python --version  # Should be 3.8 or higher
```

3. Verify the installation:
```bash
python -c "import phasta; print(phasta.__version__)"
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

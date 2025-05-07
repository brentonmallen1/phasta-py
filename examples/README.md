# PHASTA-Py Examples

This directory contains example scripts and notebooks demonstrating various features and use cases of PHASTA-Py.

## Directory Structure

- `mesh_generation/`: Examples of generating different types of meshes
- `simulations/`: Complete simulation examples
- `acceleration/`: Examples of using different acceleration backends
- `visualization/`: Examples of visualizing results
- `parallel/`: Examples of parallel computing features
- `notebooks/`: Jupyter notebooks with interactive examples

## Getting Started

1. Start with `mesh_generation/basic_meshes.py` to learn about mesh generation
2. Try `simulations/lid_driven_cavity.py` for a simple simulation
3. Explore `visualization/plot_results.py` to learn about visualization
4. Check `acceleration/gpu_example.py` for GPU acceleration
5. Look at `parallel/parallel_mesh.py` for parallel computing examples

## Requirements

All examples require the following packages:
- numpy
- matplotlib
- meshio
- pyvista (for 3D visualization)
- mpi4py (for parallel examples)
- torch (for GPU acceleration)

Install them using:
```bash
pip install numpy matplotlib meshio pyvista mpi4py torch
```

## Running Examples

Most examples can be run directly with Python:
```bash
python examples/mesh_generation/basic_meshes.py
```

For parallel examples, use MPI:
```bash
mpirun -n 4 python examples/parallel/parallel_mesh.py
```

For GPU examples, ensure you have CUDA installed:
```bash
python examples/acceleration/gpu_example.py
```

## Contributing

Feel free to contribute new examples or improve existing ones. Please follow these guidelines:
1. Include clear documentation
2. Add comments explaining key concepts
3. Use consistent formatting
4. Include expected output or results 
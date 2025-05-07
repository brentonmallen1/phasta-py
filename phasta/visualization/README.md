# PHASTA-Py Visualization Module

This module provides visualization utilities for PHASTA-Py, allowing users to visualize meshes and simulation results in both 2D and 3D.

## Features

- 2D and 3D mesh visualization
- Solution field visualization (scalar fields)
- Velocity field visualization (vector fields)
- Support for saving visualizations to files
- Interactive 3D visualization using PyVista
- High-quality 2D plots using Matplotlib

## Dependencies

- NumPy
- Matplotlib
- PyVista
- meshio

## Usage

### Mesh Visualization

```python
from phasta.visualization.plotter import plot_mesh_2d, plot_mesh_3d

# Plot a 2D mesh
plot_mesh_2d(
    mesh,
    show_cells=True,
    show_points=True,
    title='2D Mesh',
    output_file='mesh_2d.png'
)

# Plot a 3D mesh
plot_mesh_3d(
    mesh,
    show_cells=True,
    show_points=True,
    title='3D Mesh',
    output_file='mesh_3d.png'
)
```

### Solution Field Visualization

```python
from phasta.visualization.plotter import plot_solution_2d, plot_solution_3d

# Plot a 2D solution field
plot_solution_2d(
    mesh,
    solution,
    field_name='Temperature',
    title='Temperature Distribution',
    output_file='temperature_2d.png'
)

# Plot a 3D solution field
plot_solution_3d(
    mesh,
    solution,
    field_name='Temperature',
    title='Temperature Distribution',
    output_file='temperature_3d.png'
)
```

### Velocity Field Visualization

```python
from phasta.visualization.plotter import plot_velocity_field_2d, plot_velocity_field_3d

# Plot a 2D velocity field
plot_velocity_field_2d(
    mesh,
    velocity,  # shape: (n_points, 2)
    title='Velocity Field',
    output_file='velocity_2d.png'
)

# Plot a 3D velocity field
plot_velocity_field_3d(
    mesh,
    velocity,  # shape: (n_points, 3)
    title='Velocity Field',
    output_file='velocity_3d.png'
)
```

## Example

See `examples/mesh_visualization.py` for a complete example of generating and visualizing various test meshes.

## Notes

- For 3D visualization, PyVista provides an interactive interface where you can rotate, zoom, and pan the visualization.
- All visualization functions support saving the output to files in various formats (PNG, PDF, etc.).
- The visualization functions are designed to work with meshio mesh objects, which are the standard mesh format used in PHASTA-Py.
- For large meshes, consider using the `show_points=False` option to improve performance.
- The 3D visualization uses PyVista's GPU-accelerated rendering when available. 
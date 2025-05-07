"""Example script demonstrating mesh generation and visualization capabilities.

This script generates various test meshes and visualizes them using the
mesh generation and visualization utilities.
"""

import numpy as np
from phasta.mesh.generator import (
    generate_pipe_mesh,
    generate_channel_mesh,
    generate_cavity_mesh,
    generate_cylinder_mesh
)
from phasta.visualization.plotter import (
    plot_mesh_2d,
    plot_mesh_3d,
    plot_velocity_field_2d,
    plot_velocity_field_3d,
    plot_streamlines_2d,
    plot_streamlines_3d,
    plot_isosurface_3d,
    plot_slice_3d,
    plot_animation_3d
)


def main():
    """Generate and visualize example meshes."""
    # Generate a pipe mesh
    pipe_mesh = generate_pipe_mesh(
        length=10.0,
        radius=1.0,
        n_axial=50,
        n_radial=20,
        n_circumferential=40,
        output_file='pipe_mesh.vtk'
    )
    
    # Visualize pipe mesh
    plot_mesh_3d(
        pipe_mesh,
        show_cells=True,
        show_points=True,
        title='Pipe Mesh',
        output_file='pipe_mesh.png'
    )
    
    # Create a sample velocity field for the pipe
    n_points = len(pipe_mesh.points)
    velocity = np.zeros((n_points, 3))
    
    # Parabolic velocity profile in the pipe
    for i in range(n_points):
        x, y, z = pipe_mesh.points[i]
        r = np.sqrt(y**2 + z**2)
        if r < 1.0:  # Inside pipe
            velocity[i, 0] = 1.0 - (r/1.0)**2  # Parabolic profile
            velocity[i, 1] = 0.0
            velocity[i, 2] = 0.0
    
    # Visualize velocity field
    plot_velocity_field_3d(
        pipe_mesh,
        velocity,
        title='Pipe Flow Velocity Field',
        output_file='pipe_velocity.png'
    )
    
    # Visualize streamlines
    plot_streamlines_3d(
        pipe_mesh,
        velocity,
        n_points=50,
        title='Pipe Flow Streamlines',
        output_file='pipe_streamlines.png'
    )
    
    # Generate a channel mesh
    channel_mesh = generate_channel_mesh(
        length=10.0,
        height=2.0,
        width=2.0,
        nx=50,
        ny=20,
        nz=20,
        output_file='channel_mesh.vtk'
    )
    
    # Create a sample temperature field for the channel
    temperature = np.zeros(n_points)
    for i in range(n_points):
        x, y, z = channel_mesh.points[i]
        temperature[i] = np.exp(-(x-5.0)**2/10.0) * np.sin(np.pi*y/2.0)
    
    # Visualize temperature field with isosurfaces
    plot_isosurface_3d(
        channel_mesh,
        temperature,
        field_name='Temperature',
        levels=[-0.5, 0.0, 0.5],
        title='Channel Temperature Isosurfaces',
        output_file='channel_temperature_isosurfaces.png'
    )
    
    # Visualize temperature field with a slice
    plot_slice_3d(
        channel_mesh,
        temperature,
        field_name='Temperature',
        normal=(0, 1, 0),
        origin=(0, 1, 0),
        title='Channel Temperature Slice',
        output_file='channel_temperature_slice.png'
    )
    
    # Generate a cavity mesh
    cavity_mesh = generate_cavity_mesh(
        size=1.0,
        n=20,
        output_file='cavity_mesh.vtk'
    )
    
    # Create a time-dependent velocity field for the cavity
    n_steps = 20
    times = np.linspace(0, 2*np.pi, n_steps)
    velocities = []
    
    for t in times:
        velocity = np.zeros((len(cavity_mesh.points), 3))
        for i in range(len(cavity_mesh.points)):
            x, y, z = cavity_mesh.points[i]
            velocity[i, 0] = np.sin(t) * np.sin(np.pi*y) * np.sin(np.pi*z)
            velocity[i, 1] = np.cos(t) * np.sin(np.pi*x) * np.sin(np.pi*z)
            velocity[i, 2] = np.sin(t) * np.sin(np.pi*x) * np.sin(np.pi*y)
        velocities.append(velocity)
    
    # Create animation of velocity field
    plot_animation_3d(
        cavity_mesh,
        fields={'velocity': velocities},
        times=times,
        title='Cavity Flow Animation',
        output_file='cavity_flow.mp4',
        fps=10
    )
    
    # Generate a cylinder mesh
    cylinder_mesh = generate_cylinder_mesh(
        radius=0.5,
        length=10.0,
        n_radial=20,
        n_circumferential=40,
        n_axial=50,
        output_file='cylinder_mesh.vtk'
    )
    
    # Create a sample pressure field for the cylinder
    pressure = np.zeros(len(cylinder_mesh.points))
    for i in range(len(cylinder_mesh.points)):
        x, y, z = cylinder_mesh.points[i]
        r = np.sqrt(y**2 + z**2)
        theta = np.arctan2(z, y)
        pressure[i] = np.exp(-(x-5.0)**2/10.0) * np.cos(2*theta) / (r + 0.1)
    
    # Visualize pressure field with isosurfaces
    plot_isosurface_3d(
        cylinder_mesh,
        pressure,
        field_name='Pressure',
        levels=[-0.5, 0.0, 0.5],
        title='Cylinder Pressure Isosurfaces',
        output_file='cylinder_pressure_isosurfaces.png'
    )


if __name__ == '__main__':
    main() 
"""
Advanced visualization tools for PHASTA results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PHASTAPlotter:
    """Advanced plotting tools for PHASTA results."""
    
    def __init__(self, results):
        """Initialize with simulation results."""
        self.results = results
        self.setup_style()
    
    def setup_style(self):
        """Set up plotting style."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_field(self, field, title=None, cmap='jet', ax=None, **kwargs):
        """Plot a 2D field with enhanced visualization."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(field, cmap=cmap, **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        if title:
            ax.set_title(title)
        return ax
    
    def plot_velocity_field(self, title="Velocity Field", **kwargs):
        """Plot velocity field with streamlines."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Magnitude
        self.plot_field(np.sqrt(self.results.u**2 + self.results.v**2),
                       title="Velocity Magnitude", ax=ax1, **kwargs)
        
        # Streamlines
        ax2.streamplot(self.results.x, self.results.y,
                      self.results.u, self.results.v,
                      density=1.5, color='k', linewidth=0.5)
        ax2.set_title("Streamlines")
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_temperature_field(self, title="Temperature Field", **kwargs):
        """Plot temperature field with heat flux vectors."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Temperature
        self.plot_field(self.results.temperature,
                       title="Temperature", ax=ax1, **kwargs)
        
        # Heat flux
        qx, qy = self.results.heat_flux
        ax2.quiver(self.results.x, self.results.y, qx, qy,
                  scale=50, color='k')
        ax2.set_title("Heat Flux Vectors")
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_turbulence_field(self, title="Turbulence Field", **kwargs):
        """Plot turbulence quantities."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Turbulent kinetic energy
        self.plot_field(self.results.k,
                       title="Turbulent Kinetic Energy", ax=ax1, **kwargs)
        
        # Dissipation rate
        self.plot_field(self.results.epsilon,
                       title="Dissipation Rate", ax=ax2, **kwargs)
        
        # Eddy viscosity
        self.plot_field(self.results.nu_t,
                       title="Eddy Viscosity", ax=ax3, **kwargs)
        
        # Wall distance
        self.plot_field(self.results.y_plus,
                       title="Wall Distance (y+)", ax=ax4, **kwargs)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_phase_field(self, title="Phase Field", **kwargs):
        """Plot phase field with interface."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Phase fraction
        self.plot_field(self.results.phase_fraction,
                       title="Phase Fraction", ax=ax1, **kwargs)
        
        # Interface
        ax2.contour(self.results.x, self.results.y,
                   self.results.phase_fraction, [0.5],
                   colors='k', linewidths=2)
        ax2.set_title("Phase Interface")
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_radiation_field(self, title="Radiation Field", **kwargs):
        """Plot radiation quantities."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Intensity
        self.plot_field(self.results.radiation_intensity,
                       title="Radiation Intensity", ax=ax1, **kwargs)
        
        # Heat flux
        self.plot_field(self.results.radiation_heat_flux,
                       title="Radiation Heat Flux", ax=ax2, **kwargs)
        
        # Absorption
        self.plot_field(self.results.absorption,
                       title="Absorption", ax=ax3, **kwargs)
        
        # Scattering
        self.plot_field(self.results.scattering,
                       title="Scattering", ax=ax4, **kwargs)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_profiles(self, x_pos, title="Profiles", **kwargs):
        """Plot various profiles at a given x position."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Velocity profile
        y, u = self.results.get_velocity_profile(x_pos)
        ax1.plot(u, y)
        ax1.set_title("Velocity Profile")
        ax1.set_xlabel("Velocity")
        ax1.set_ylabel("Height")
        ax1.grid(True)
        
        # Temperature profile
        y, T = self.results.get_temperature_profile(x_pos)
        ax2.plot(T, y)
        ax2.set_title("Temperature Profile")
        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Height")
        ax2.grid(True)
        
        # Turbulence profiles
        y, k = self.results.get_turbulence_profile(x_pos)
        ax3.plot(k, y)
        ax3.set_title("Turbulent Kinetic Energy Profile")
        ax3.set_xlabel("k")
        ax3.set_ylabel("Height")
        ax3.grid(True)
        
        # Phase fraction profile
        y, phi = self.results.get_phase_profile(x_pos)
        ax4.plot(phi, y)
        ax4.set_title("Phase Fraction Profile")
        ax4.set_xlabel("Phase Fraction")
        ax4.set_ylabel("Height")
        ax4.grid(True)
        
        plt.suptitle(f"{title} at x = {x_pos}")
        plt.tight_layout()
        return fig
    
    def plot_3d(self, field, title=None, **kwargs):
        """Create 3D visualization using Plotly."""
        fig = go.Figure(data=[go.Surface(z=field, **kwargs)])
        if title:
            fig.update_layout(title=title)
        return fig
    
    def plot_animation(self, field_sequence, title=None, **kwargs):
        """Create animation of field evolution."""
        fig = go.Figure()
        
        for i, field in enumerate(field_sequence):
            fig.add_trace(go.Frame(
                data=[go.Heatmap(z=field)],
                name=f"frame{i}"
            ))
        
        if title:
            fig.update_layout(title=title)
        return fig
    
    def plot_comparison(self, fields, labels, title=None, **kwargs):
        """Compare multiple fields."""
        fig, axes = plt.subplots(1, len(fields), figsize=(5*len(fields), 5))
        
        for ax, field, label in zip(axes, fields, labels):
            self.plot_field(field, title=label, ax=ax, **kwargs)
        
        if title:
            plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_statistics(self, title="Statistics", **kwargs):
        """Plot statistical analysis of results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram of velocity magnitude
        ax1.hist(np.sqrt(self.results.u**2 + self.results.v**2).flatten(),
                bins=50, density=True)
        ax1.set_title("Velocity Magnitude Distribution")
        ax1.set_xlabel("Velocity")
        ax1.set_ylabel("Frequency")
        
        # Temperature distribution
        ax2.hist(self.results.temperature.flatten(),
                bins=50, density=True)
        ax2.set_title("Temperature Distribution")
        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Frequency")
        
        # Turbulence intensity
        ax3.hist(self.results.k.flatten(),
                bins=50, density=True)
        ax3.set_title("Turbulent Kinetic Energy Distribution")
        ax3.set_xlabel("k")
        ax3.set_ylabel("Frequency")
        
        # Phase fraction distribution
        ax4.hist(self.results.phase_fraction.flatten(),
                bins=50, density=True)
        ax4.set_title("Phase Fraction Distribution")
        ax4.set_xlabel("Phase Fraction")
        ax4.set_ylabel("Frequency")
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig 
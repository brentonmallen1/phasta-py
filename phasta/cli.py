"""Command-line interface for PHASTA-Py.

This module provides a command-line interface for running PHASTA-Py simulations
and performing common operations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from phasta.core.mesh import Mesh
from phasta.core.field import Field
from phasta.acceleration import get_best_available_backend


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PHASTA-Py: Parallel Hierarchic Adaptive Stabilized Transient Analysis'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Run simulation command
    run_parser = subparsers.add_parser('run', help='Run a simulation')
    run_parser.add_argument('config_file', type=str, help='Path to configuration file')
    run_parser.add_argument('--output-dir', type=str, help='Output directory')
    run_parser.add_argument('--backend', type=str, choices=['cpu', 'cuda', 'metal'],
                          help='Acceleration backend to use')
    
    # Convert mesh command
    convert_parser = subparsers.add_parser('convert-mesh', help='Convert mesh between formats')
    convert_parser.add_argument('input_file', type=str, help='Input mesh file')
    convert_parser.add_argument('output_file', type=str, help='Output mesh file')
    convert_parser.add_argument('--format', type=str, help='Output format')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize simulation results')
    viz_parser.add_argument('input_file', type=str, help='Input file to visualize')
    viz_parser.add_argument('--field', type=str, help='Field to visualize')
    viz_parser.add_argument('--output', type=str, help='Output image file')
    
    return parser.parse_args()


def run_simulation(config_file: str, output_dir: Optional[str] = None,
                  backend: Optional[str] = None) -> None:
    """Run a simulation with the given configuration."""
    # TODO: Implement simulation runner
    print(f"Running simulation with config: {config_file}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    if backend:
        print(f"Using backend: {backend}")


def convert_mesh(input_file: str, output_file: str, format: Optional[str] = None) -> None:
    """Convert a mesh between different formats."""
    # TODO: Implement mesh conversion
    print(f"Converting mesh from {input_file} to {output_file}")
    if format:
        print(f"Output format: {format}")


def visualize_results(input_file: str, field: Optional[str] = None,
                     output: Optional[str] = None) -> None:
    """Visualize simulation results."""
    # TODO: Implement visualization
    print(f"Visualizing results from {input_file}")
    if field:
        print(f"Field to visualize: {field}")
    if output:
        print(f"Output file: {output}")


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == 'run':
        run_simulation(args.config_file, args.output_dir, args.backend)
    elif args.command == 'convert-mesh':
        convert_mesh(args.input_file, args.output_file, args.format)
    elif args.command == 'visualize':
        visualize_results(args.input_file, args.field, args.output)
    else:
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)


if __name__ == '__main__':
    main()

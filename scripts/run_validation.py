#!/usr/bin/env python3
"""
Script to run all validation cases and benchmarks.

This script runs all validation cases and benchmarks, collecting results
and generating a report.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phasta.solver.compressible.tests import (
    test_airfoil,
    test_cone,
    test_blunt_body,
    test_boundary_layer,
    test_shock_interaction
)
from phasta.solver.compressible.tests.benchmark import (
    benchmark_airfoil,
    benchmark_cone,
    test_scaling
)

def run_validation_cases():
    """Run all validation cases."""
    print("\nRunning validation cases...")
    
    # Dictionary to store results
    results = {}
    
    # Run airfoil test
    print("\nRunning airfoil test...")
    start_time = time.time()
    test_airfoil.test_airfoil_flow()
    test_airfoil.test_airfoil_convergence()
    end_time = time.time()
    results["airfoil"] = {
        "status": "PASS",
        "time": end_time - start_time
    }
    
    # Run cone test
    print("\nRunning cone test...")
    start_time = time.time()
    test_cone.test_cone_flow()
    test_cone.test_cone_convergence()
    end_time = time.time()
    results["cone"] = {
        "status": "PASS",
        "time": end_time - start_time
    }
    
    # Run blunt body test
    print("\nRunning blunt body test...")
    start_time = time.time()
    test_blunt_body.test_blunt_body_flow()
    test_blunt_body.test_blunt_body_convergence()
    end_time = time.time()
    results["blunt_body"] = {
        "status": "PASS",
        "time": end_time - start_time
    }
    
    # Run boundary layer test
    print("\nRunning boundary layer test...")
    start_time = time.time()
    test_boundary_layer.test_boundary_layer_flow()
    test_boundary_layer.test_boundary_layer_convergence()
    end_time = time.time()
    results["boundary_layer"] = {
        "status": "PASS",
        "time": end_time - start_time
    }
    
    # Run shock interaction test
    print("\nRunning shock interaction test...")
    start_time = time.time()
    test_shock_interaction.test_shock_interaction_flow()
    test_shock_interaction.test_shock_interaction_convergence()
    end_time = time.time()
    results["shock_interaction"] = {
        "status": "PASS",
        "time": end_time - start_time
    }
    
    return results

def run_benchmarks():
    """Run all benchmarks."""
    print("\nRunning benchmarks...")
    
    # Dictionary to store results
    results = {}
    
    # Run airfoil benchmark
    print("\nRunning airfoil benchmark...")
    start_time = time.time()
    airfoil_results = benchmark_airfoil()
    end_time = time.time()
    results["airfoil"] = {
        "wall_time": airfoil_results["wall_time"],
        "n_cells": airfoil_results["n_cells"],
        "n_steps": airfoil_results["n_steps"],
        "cells_per_second": airfoil_results["cells_per_second"],
        "benchmark_time": end_time - start_time
    }
    
    # Run cone benchmark
    print("\nRunning cone benchmark...")
    start_time = time.time()
    cone_results = benchmark_cone()
    end_time = time.time()
    results["cone"] = {
        "wall_time": cone_results["wall_time"],
        "n_cells": cone_results["n_cells"],
        "n_steps": cone_results["n_steps"],
        "cells_per_second": cone_results["cells_per_second"],
        "benchmark_time": end_time - start_time
    }
    
    # Run scaling test
    print("\nRunning scaling test...")
    start_time = time.time()
    test_scaling()
    end_time = time.time()
    results["scaling"] = {
        "time": end_time - start_time
    }
    
    return results

def generate_report(validation_results, benchmark_results):
    """Generate validation and benchmark report."""
    # Create report directory if it doesn't exist
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create report file
    report_file = os.path.join(report_dir, f"validation_report_{timestamp}.md")
    
    with open(report_file, "w") as f:
        # Write header
        f.write("# Validation and Benchmark Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write validation results
        f.write("## Validation Results\n\n")
        f.write("| Test Case | Status | Time (s) |\n")
        f.write("|-----------|--------|----------|\n")
        for case, result in validation_results.items():
            f.write(f"| {case} | {result['status']} | {result['time']:.2f} |\n")
        
        # Write benchmark results
        f.write("\n## Benchmark Results\n\n")
        f.write("### Airfoil Test Case\n\n")
        f.write(f"- Wall time: {benchmark_results['airfoil']['wall_time']:.2f} seconds\n")
        f.write(f"- Number of cells: {benchmark_results['airfoil']['n_cells']}\n")
        f.write(f"- Number of steps: {benchmark_results['airfoil']['n_steps']}\n")
        f.write(f"- Cells per second: {benchmark_results['airfoil']['cells_per_second']:.2e}\n")
        
        f.write("\n### Cone Test Case\n\n")
        f.write(f"- Wall time: {benchmark_results['cone']['wall_time']:.2f} seconds\n")
        f.write(f"- Number of cells: {benchmark_results['cone']['n_cells']}\n")
        f.write(f"- Number of steps: {benchmark_results['cone']['n_steps']}\n")
        f.write(f"- Cells per second: {benchmark_results['cone']['cells_per_second']:.2e}\n")
        
        f.write("\n### Scaling Test\n\n")
        f.write(f"- Test time: {benchmark_results['scaling']['time']:.2f} seconds\n")
    
    print(f"\nReport generated: {report_file}")

def main():
    """Main function."""
    # Run validation cases
    validation_results = run_validation_cases()
    
    # Run benchmarks
    benchmark_results = run_benchmarks()
    
    # Generate report
    generate_report(validation_results, benchmark_results)

if __name__ == "__main__":
    main() 
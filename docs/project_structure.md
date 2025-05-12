# PHASTA Project Structure

## Overview
This document describes the organization of the PHASTA codebase. The project follows a standard Python package structure with clear separation of concerns.

## Directory Structure

```
phasta/
├── docs/                      # Documentation
│   ├── api/                   # API documentation
│   ├── tutorials/             # Tutorial documentation
│   ├── feature_tracking.md    # Feature implementation status
│   └── enhancement_plan.md    # Future enhancements
│
├── examples/                  # Example code and tutorials
│   ├── tutorials/            # Jupyter notebook tutorials
│   │   ├── 01_basic_flow_solver.ipynb
│   │   ├── 02_turbulence_modeling.ipynb
│   │   └── 03_heat_transfer.ipynb
│   ├── cases/               # Example simulation cases
│   └── visualization/       # Visualization tools and examples
│
├── phasta/                   # Core package code
│   ├── core/                # Core functionality
│   ├── models/              # Physics models
│   ├── solvers/             # Numerical solvers
│   ├── utils/               # Utility functions
│   └── visualization/       # Visualization tools
│
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── validation/         # Validation tests
│
├── validation_data/        # Validation data and benchmarks
│
├── scripts/                # Utility scripts
│
├── pyproject.toml         # Project configuration
├── setup.py              # Package setup
├── requirements.txt      # Python dependencies
└── README.md            # Project overview
```

## Key Components

### Documentation (`docs/`)
- API documentation
- Tutorials
- Feature tracking
- Enhancement plans
- Best practices

### Examples (`examples/`)
- Interactive tutorials
- Example cases
- Visualization examples
- Best practice demonstrations

### Core Package (`phasta/`)
- Core functionality
- Physics models
- Numerical solvers
- Utility functions
- Visualization tools

### Tests (`tests/`)
- Unit tests
- Integration tests
- Validation tests
- Performance benchmarks

### Validation Data (`validation_data/`)
- Benchmark cases
- Experimental data
- Reference solutions

### Scripts (`scripts/`)
- Build scripts
- Development tools
- Utility scripts

## File Organization Guidelines

1. **Code Organization**
   - Keep related functionality together
   - Use clear, descriptive names
   - Follow Python package conventions

2. **Documentation**
   - Keep documentation close to code
   - Use consistent formatting
   - Include examples

3. **Testing**
   - Organize tests by type
   - Include validation cases
   - Maintain test data

4. **Examples**
   - Provide clear, working examples
   - Include comments and explanations
   - Use consistent style

## Migration Plan

1. **Phase 1: Consolidation**
   - Move all code to `phasta-py/`
   - Consolidate duplicate directories
   - Remove unnecessary files

2. **Phase 2: Reorganization**
   - Implement new directory structure
   - Update import paths
   - Update documentation

3. **Phase 3: Cleanup**
   - Remove old directories
   - Update build scripts
   - Verify functionality

## Notes
- Keep the structure clean and organized
- Document any changes
- Maintain backward compatibility
- Follow Python best practices 
# Development Guide

This guide provides instructions for setting up the development environment and contributing to the PHASTA Python implementation.

## Prerequisites

- Python 3.11 or higher
- Git
- Make (optional, but recommended)
- CUDA toolkit (for GPU acceleration)
- MPI (for parallel processing)

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/phasta-py.git
cd phasta-py
```

2. Create and activate a virtual environment:
```bash
# Using Make
make install

# Or manually
python3.11 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev,docs]"
pre-commit install
```

## Development Workflow

### Code Style

The project uses several tools to maintain code quality:

- Black for code formatting
- isort for import sorting
- Ruff for linting
- MyPy for type checking
- pre-commit hooks for automated checks

To run all checks:
```bash
make lint
```

To format code:
```bash
make format
```

### Testing

Run tests with:
```bash
make test
```

Test categories:
- Unit tests: `pytest tests/unit`
- Integration tests: `pytest tests/integration`
- GPU tests: `pytest tests/gpu`
- MPI tests: `pytest tests/mpi`

### Documentation

Build documentation:
```bash
make docs
```

Documentation is built using Sphinx and includes:
- API reference
- Tutorials
- Examples
- Development guides

## Project Structure

```
phasta-py/
├── phasta/              # Main package
│   ├── core/           # Core functionality
│   ├── solvers/        # Flow solvers
│   ├── models/         # Physics models
│   └── utils/          # Utility functions
├── tests/              # Test suite
├── docs/               # Documentation
├── examples/           # Example notebooks
└── scripts/            # Utility scripts
```

## Best Practices

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints
   - Write docstrings for all public functions
   - Keep functions small and focused

2. **Testing**
   - Write tests for new features
   - Maintain test coverage
   - Use appropriate test markers
   - Test edge cases

3. **Documentation**
   - Update docstrings when changing code
   - Add examples for new features
   - Keep README and guides up to date

4. **Version Control**
   - Use meaningful commit messages
   - Create feature branches
   - Keep commits focused and atomic
   - Rebase before merging

5. **Performance**
   - Profile code for bottlenecks
   - Use appropriate data structures
   - Optimize critical paths
   - Consider GPU acceleration

## Common Tasks

### Adding a New Feature

1. Create a feature branch
2. Implement the feature
3. Add tests
4. Update documentation
5. Run all checks
6. Submit a pull request

### Debugging

1. Use logging for debugging
2. Enable debug mode in configuration
3. Use appropriate debuggers
4. Check GPU memory usage
5. Monitor MPI communication

### Performance Optimization

1. Profile code with cProfile
2. Use line_profiler for detailed analysis
3. Monitor memory usage
4. Optimize critical sections
5. Consider parallelization

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Python version
   - Verify dependencies
   - Check compiler settings

2. **Test Failures**
   - Check test environment
   - Verify test data
   - Check GPU availability

3. **Performance Issues**
   - Profile code
   - Check memory usage
   - Verify parallelization

### Getting Help

- Check documentation
- Search issues
- Ask on mailing list
- Contact maintainers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run all checks
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
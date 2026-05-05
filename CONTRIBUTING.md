# Contributing to GPUStack

Thank you for your interest in contributing to GPUStack! This document provides guidelines and information about contributing to this project.

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- NVIDIA GPU with compatible drivers (for GPU worker nodes)

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gpustack.git
   cd gpustack
   ```

3. Set up the development environment:
   ```bash
   pip install -e .
   ```

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Include steps to reproduce the problem
- Describe the expected vs actual behavior
- Include your environment details (OS, Python version, GPU model)

### Pull Requests

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure tests pass:
   ```bash
   uv run pytest
   ```

3. Follow the existing code style and conventions
4. Write clear, descriptive commit messages
5. Push to your fork and submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for new code
- Add docstrings for public functions and classes
- Run hack/lint.sh to check code style and formatting

### Testing

- Write unit tests for new functionality
- Ensure all existing tests pass
- Test with both CPU and GPU configurations when applicable

## Development Guidelines

### Project Structure

- `gpustack/` - Main package source code
- `tests/` - Test files
- `docs/` - Documentation
- `docker/` - Docker-related files

### Commit Messages

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Questions?

If you have questions about contributing, feel free to open an issue or reach out to the maintainers.

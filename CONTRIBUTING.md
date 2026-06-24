# Contributing to MINet

Thank you for your interest in contributing! We welcome contributions of all kinds — bug reports, feature requests, code improvements, and documentation.

## Getting Started

1. Fork the repository and clone it locally.
2. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Install the development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Code Style

- Follow PEP 8 conventions.
- Keep lines under 120 characters where practical.
- Use meaningful variable names.
- Add docstrings for new classes and functions.

## Pull Request Process

1. Ensure your code passes the CI checks (we run flake8 linting and an import test automatically).
2. Update the README if your change affects usage or installation.
3. Describe what your change does and why in the PR description.
4. Link any related issues.

## Reporting Bugs

When reporting a bug, please include:

- Your Python and PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- The full error traceback
- Steps to reproduce the issue
- Operating system and CUDA version (if applicable)

## Feature Requests

Open an issue with the "enhancement" label. Describe the feature, why it would be useful, and (optionally) how you imagine it could be implemented.

## Questions

For general questions, feel free to open a discussion or contact the authors directly.

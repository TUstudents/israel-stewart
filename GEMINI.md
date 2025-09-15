# Project Overview

This project is a Python implementation of relativistic hydrodynamics using the Israel-Stewart formalism. It provides a framework for simulating relativistic fluid dynamics in curved spacetime with second-order viscous corrections and full tensor algebra support.

**Key Technologies:**

*   **Programming Language:** Python 3.12+
*   **Package Manager:** uv
*   **Core Libraries:** numpy, scipy, sympy, matplotlib, numba, h5py
*   **Code Quality:**
    *   **Linting & Formatting:** ruff
    *   **Type Checking:** mypy
*   **Testing:** pytest

# Building and Running

The following commands are used for building, running, and testing the project. They are executed using the `uv` package manager.

*   **Installation:**
    ```bash
    # Install with uv
    uv sync
    ```

*   **Development Installation:**
    ```bash
    # Install with development tools
    uv sync --extra dev
    ```

*   **Running Tests:**
    ```bash
    # Run all tests
    uv run pytest

    # Run tests without the "slow" marker
    uv run pytest -m "not slow"

    # Run tests with coverage report
    uv run pytest --cov
    ```

*   **Code Quality Checks:**
    ```bash
    # Check code style
    uv run ruff check

    # Format code
    uv run ruff format

    # Type checking
    uv run mypy israel_stewart
    ```

*   **Development Environment:**
    ```bash
    # Start Jupyter Lab for interactive development
    uv run jupyter lab

    # Run package in development mode
    uv run python -m israel_stewart
    ```

# Development Conventions

*   **Code Style:** The project uses `ruff` for code formatting and linting. The configuration can be found in the `pyproject.toml` file.
*   **Type Checking:** `mypy` is used for static type checking. The configuration is in `pyproject.toml`.
*   **Testing:** The project uses `pytest` for testing. Test files are located in the `israel_stewart/tests` directory. The `pyproject.toml` file contains the pytest configuration.
*   **Pre-commit Hooks:** The project uses pre-commit hooks to enforce code quality standards. The configuration is in `.pre-commit-config.yaml`.
*   **Documentation:** The documentation is built using Sphinx and is located in the `docs` directory.
*   **Modular Architecture:** The project is organized into the following modules:
    *   `core`: Tensor algebra and spacetime geometry.
    *   `equations`: Physics equations (Israel-Stewart).
    *   `solvers`: Numerical time evolution methods.
    *   `benchmarks`: Benchmark tests against known solutions.
    *   `stochastic`, `rg_analysis`, `linearization`: Advanced analysis tools.

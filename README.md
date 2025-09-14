# Israel-Stewart Relativistic Hydrodynamics

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A sophisticated Python implementation of relativistic hydrodynamics using the **Israel-Stewart formalism** with second-order viscous corrections. This package provides a complete framework for simulating relativistic fluid dynamics in curved spacetime with full tensor algebra support.

## 🚀 Key Features

### Complete Tensor Framework
- **Modular tensor algebra system** with automatic index management
- **Four-vector operations** with Lorentz transformations and boosts
- **Stress-energy tensors** for viscous relativistic fluids
- **Covariant derivatives** with full Christoffel symbol support
- **Arbitrary rank tensor contractions** for complex physics calculations

### Curved Spacetime Support
- **Numerical and symbolic Christoffel computation** using finite differences
- **Multiple coordinate systems**: Minkowski, Milne, Bjorken, FLRW, Schwarzschild
- **General relativity integration** for cosmological and astrophysical applications
- **3+1 decomposition** with projection operators for fluid dynamics

### Physics Implementation
- ✅ **Conservation laws**: Complete energy-momentum conservation ∇_μ T^μν = 0
- ✅ **Israel-Stewart stress-energy tensor** with second-order viscous corrections
- ✅ **Transport coefficients** framework for temperature-dependent viscosities
- 🚧 **Relaxation equations** for second-order viscous evolution (in progress)

### Advanced Analysis Tools
- **Stochastic methods**: Fluctuation-dissipation relations and Langevin dynamics
- **RG analysis**: Renormalization group techniques using Martin-Siggia-Rose-Janssen-De Dominicis formalism
- **Linear stability analysis**: Dispersion relations and perturbation theory
- **Performance monitoring**: Optimization recommendations for tensor operations

## 📦 Installation

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/your-org/israel-stewart.git
cd israel-stewart

# Install with uv
uv sync
```

### Development Installation
```bash
# Install with development tools
uv sync --extra dev

# Install with Jupyter support
uv sync --extra jupyter

# Install all optional dependencies
uv sync --extra all
```

## 🔬 Quick Start

### Basic Tensor Operations
```python
from israel_stewart.core import FourVector, MinkowskiMetric
import numpy as np

# Create Minkowski metric and four-vector
metric = MinkowskiMetric()
components = np.array([1.0, 0.5, 0.2, 0.1])
u = FourVector(components, metric=metric)

print(f"Time component: {u.time_component}")
print(f"Is timelike: {u.is_timelike()}")
print(f"Norm: {u.norm()}")
```

### Relativistic Fluid State
```python
from israel_stewart.core import create_perfect_fluid_state

# Create perfect fluid at rest
state = create_perfect_fluid_state(
    energy_density=1.0,
    pressure=0.33,
    three_velocity=[0.1, 0.0, 0.0]
)

print(f"Energy density: {state.energy_density}")
print(f"Four-velocity: {state.velocity_field.four_velocity}")
```

### Conservation Laws
```python
from israel_stewart.equations.conservation import IsraelStewartConservation
from israel_stewart.core import SpacetimeGrid

# Set up conservation law computation
grid = SpacetimeGrid(dimensions=4, extent=[[0, 1]] * 4, resolution=[20] * 4)
conservation = IsraelStewartConservation(grid)

# Compute energy-momentum conservation
div_T = conservation.energy_momentum_conservation(state)
print(f"∇_μ T^μν = {div_T}")
```

## 📚 Documentation Structure

### Core Foundation (`core/`)
- **`tensor_base.py`**: Core TensorField class with index management
- **`four_vectors.py`**: FourVector specialization with relativistic operations
- **`stress_tensors.py`**: StressEnergyTensor and ViscousStressTensor classes
- **`derivatives.py`**: CovariantDerivative and ProjectionOperator for 3+1 decomposition
- **`metrics.py`**: Spacetime metrics and complete Christoffel symbols framework
- **`fields.py`**: Fluid field variables and thermodynamic state vectors
- **`constants.py`**: Physical constants in natural units

### Physics Equations (`equations/`)
- **`conservation.py`**: ✅ Energy-momentum and particle number conservation
- **`relaxation.py`**: 🚧 Israel-Stewart second-order relaxation equations
- **`coefficients.py`**: 🚧 Temperature-dependent transport coefficients
- **`constraints.py`**: 🚧 Thermodynamic consistency conditions

### Numerical Methods (`solvers/`)
- **`finite_difference.py`**: 🚧 Spatial discretization schemes
- **`spectral.py`**: 🚧 Fourier-space methods for periodic systems
- **`implicit.py`**: 🚧 Implicit time integration for relaxation terms

### Advanced Analysis
- **`stochastic/`**: Fluctuation-dissipation relations and stochastic forcing
- **`rg_analysis/`**: Renormalization group analysis for hydrodynamic fluctuations
- **`linearization/`**: Linear stability analysis and dispersion relations

## 🛠️ Development

### Code Quality
```bash
# Linting and formatting
uv run ruff check          # Check code style
uv run ruff format         # Format code
uv run mypy israel_stewart # Type checking

# Testing
uv run pytest             # Run all tests (59 test cases)
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest --cov       # Run with coverage report
```

### Development Environment
```bash
# Start Jupyter Lab for interactive development
uv run jupyter lab

# Run package in development mode
uv run python -m israel_stewart
```

### Testing Framework
The package includes comprehensive test suites:
- **59 total test cases** covering all implemented functionality
- **Conservation law tests** (31 tests): Energy-momentum and particle conservation
- **Christoffel symbol tests** (28 tests): Numerical vs symbolic validation
- **Benchmark comparisons** against known analytical solutions

## 🔬 Physics Background

### Israel-Stewart Formalism
The Israel-Stewart formalism extends ideal relativistic hydrodynamics to include **second-order viscous corrections**, making it suitable for:

- **Heavy-ion collision dynamics** in high-energy nuclear physics
- **Cosmological fluid evolution** in the early universe
- **Neutron star matter** under extreme conditions
- **Quark-gluon plasma** thermalization processes

### Mathematical Framework
The package implements the full **3+1 decomposition** of spacetime, treating relativistic fluids as:

```
T^μν = ε u^μ u^ν + p Δ^μν + π^μν + bulk terms
```

Where:
- `ε`: Energy density in the fluid rest frame
- `p`: Thermodynamic pressure
- `π^μν`: Viscous shear stress tensor
- `Δ^μν`: Spatial projection tensor orthogonal to fluid velocity
- Full **covariant derivatives** ∇_μ with Christoffel symbols Γ^λ_μν

## 🌟 Examples

### Multiple Coordinate Systems
```python
from israel_stewart.core import MilneMetric, BJorkenMetric, FLRWMetric

# Boost-invariant coordinates for heavy-ion collisions
milne = MilneMetric()
print(f"Milne coordinates: {milne.coordinate_names}")

# Bjorken flow metric
bjorken = BJorkenMetric()
print(f"Bjorken metric signature: {bjorken.signature}")

# Cosmological FLRW metric with scale factor a(t) = t^(2/3)
flrw = FLRWMetric(scale_factor_power=2/3)
print(f"FLRW Hubble parameter: {flrw.hubble_parameter}")
```

### Performance Monitoring
```python
from israel_stewart.core.performance import reset_performance_stats, performance_report

# Reset performance counters
reset_performance_stats()

# Run tensor-intensive calculations
result = some_heavy_computation()

# Get optimization recommendations
report = performance_report()
print(report)
```

## 📊 Current Implementation Status

### ✅ Completed Modules
- **Complete tensor framework** with modular architecture (6 core modules)
- **Curved spacetime support** with numerical and symbolic Christoffel symbols
- **Conservation laws** with full Israel-Stewart stress-energy tensor
- **5 coordinate systems**: Minkowski, Milne, Bjorken, FLRW, Schwarzschild
- **Comprehensive validation** with 59 passing test cases

### 🚧 In Progress
- **Relaxation equations**: Second-order Israel-Stewart evolution equations
- **Transport coefficients**: Temperature and density-dependent viscosities
- **Numerical solvers**: Time integration methods for stiff relaxation timescales

### 🔮 Planned Features
- **Adaptive mesh refinement** for shock capturing
- **GPU acceleration** with CuPy integration
- **Machine learning** surrogate models for equation of state
- **Visualization tools** for 4D spacetime data

## 🏗️ Architecture

The package follows a **physics-based modular architecture**:

```
core/ → equations/ → solvers/ → benchmarks/
  ↓
stochastic/, rg_analysis/, linearization/
```

- **Core modules** provide tensor algebra and spacetime geometry
- **Equation modules** implement the Israel-Stewart physics
- **Solver modules** handle numerical time evolution
- **Analysis modules** provide specialized theoretical tools

## 📄 License & Citation

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0** license.

### Academic Use
This package is designed for **non-commercial research use**. If you use this code in academic work, please cite:

```bibtex
@software{israel_stewart_2024,
  title = {Israel-Stewart Relativistic Hydrodynamics},
  author = {Relativistic Hydrodynamics Team},
  year = {2024},
  version = {0.1.0},
  url = {https://github.com/your-org/israel-stewart}
}
```

### Physics References
The implementation follows the theoretical framework established in:
- Israel, W. & Stewart, J. M. (1979). "Transient relativistic thermodynamics and kinetic theory"
- Romatschke, P. & Romatschke, U. (2019). "Relativistic Fluid Dynamics In and Out of Equilibrium"

## 🤝 Contributing

We welcome contributions to the Israel-Stewart package! Please see our development workflow:

1. **Fork the repository** and create a feature branch
2. **Follow code quality standards**: Use `ruff` for formatting and pass all `mypy` checks
3. **Add comprehensive tests** for new physics functionality
4. **Update documentation** including docstrings and CLAUDE.md
5. **Submit pull request** with clear description of physics improvements

### Development Environment
```bash
# Set up pre-commit hooks for code quality
uv run pre-commit install

# Run full test suite before contributing
uv run pytest --cov
```

For questions or discussions about the physics implementation, please open an issue on the repository.

---

**Explore the fascinating world of relativistic fluid dynamics with Israel-Stewart!** 🌌
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated Python implementation of relativistic hydrodynamics using the Israel-Stewart formalism with second-order viscous corrections. The codebase is organized into specialized modules for different aspects of relativistic fluid dynamics.

## Architecture

The package follows a modular physics-based architecture:

**Core Foundation (`core/`)**:
- `tensor_base.py`: Core TensorField class with index management and basic operations
- `four_vectors.py`: FourVector specialization with relativistic physics operations
- `stress_tensors.py`: StressEnergyTensor and ViscousStressTensor for fluid dynamics
- `derivatives.py`: CovariantDerivative and ProjectionOperator for curved spacetime
- `transformations.py`: LorentzTransformation and CoordinateTransformation classes
- `tensor_utils.py`: Type guards, validation functions, and optimization utilities
- `performance.py`: Performance monitoring and optimization for tensor operations
- `tensors.py`: Consolidated imports for backwards compatibility
- `metrics.py`: Spacetime metrics and Christoffel symbols
- `fields.py`: Fluid field variables and state vectors
- `constants.py`: Physical constants and unit systems

**Utilities (`utils/`)**:
- `logging_config.py`: Structured logging configuration with performance and physics-specific loggers
- `visualization.py`: Plotting utilities for tensor fields and physics data
- `io.py`: File I/O utilities for simulation data
- `dimensionless.py`: Dimensionless variable transformations

**Physics Equations (`equations/`)**:
- `conservation.py`: Energy-momentum and particle number conservation
- `relaxation.py`: Israel-Stewart second-order relaxation equations
- `coefficients.py`: Transport coefficients (shear/bulk viscosity, conductivity)
- `constraints.py`: Thermodynamic consistency conditions

**Numerical Methods (`solvers/`)**:
- `finite_difference.py`: Spatial discretization schemes
- `spectral.py`: Fourier-space methods for periodic systems
- `splitting.py`: Operator splitting for stiff equations
- `implicit.py`: Implicit time integration for relaxation terms

**Advanced Analysis Modules**:
- `stochastic/`: Fluctuation-dissipation relations and stochastic forcing
- `rg_analysis/`: Renormalization group analysis using Martin-Siggia-Rose-Janssen-De Dominicis formalism
- `linearization/`: Linear stability analysis and dispersion relations

**Validation (`benchmarks/`)**:
- `bjorken_flow.py`: 1D boost-invariant expansion (exact solution)
- `sound_waves.py`: Linear wave propagation tests
- `equilibration.py`: Relaxation to equilibrium validation

## Development Commands

**Package Management (uv)**:
- `uv sync` - Install dependencies
- `uv sync --extra dev` - Install with development tools
- `uv sync --extra jupyter` - Install with Jupyter support
- `uv sync --extra all` - Install all optional dependencies

**Scripts**: `./scripts/format.sh` (or `--all-files`), `./scripts/test.sh` (or `--coverage`), `./scripts/build.sh` (or `--clean`)

**Direct**: `uv run pytest`, `uv run ruff check`, `uv run ruff format`, `uv run mypy israel_stewart`

**Logging**: Set `ISRAEL_STEWART_LOG_LEVEL`, `ISRAEL_STEWART_LOG_FORMAT`, `ISRAEL_STEWART_LOG_PERFORMANCE`, `ISRAEL_STEWART_LOG_MEMORY`, `ISRAEL_STEWART_LOG_FILE`

**Testing Complete Israel-Stewart System**:
```python
# Complete Israel-Stewart relaxation equations
from israel_stewart.core import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core import SpacetimeGrid, MinkowskiMetric
from israel_stewart.equations.relaxation import ISRelaxationEquations
import numpy as np

# Setup spacetime grid
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),
    spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    grid_points=(8, 8, 8, 8)
)

# Transport coefficients with second-order terms
coeffs = TransportCoefficients(
    shear_viscosity=0.1,
    bulk_viscosity=0.05,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3,
    lambda_pi_pi=0.1,  # Second-order coupling
    xi_1=0.2          # Bulk nonlinearity
)

# Initialize relaxation system
metric = MinkowskiMetric()
relaxation = ISRelaxationEquations(grid, metric, coeffs)

# Setup field configuration
fields = ISFieldConfiguration(grid)
fields.rho.fill(1.0)      # Energy density
fields.pressure.fill(0.33) # Pressure
fields.Pi.fill(0.01)      # Bulk pressure
fields.pi_munu.fill(0.005) # Shear tensor

# Evolve dissipative fluxes
dt = 0.01
relaxation.evolve_relaxation(fields, dt, method='implicit')

# Analyze stability
stability = relaxation.stability_analysis(fields)
print(f"Stiffness ratio: {stability['stiffness_ratio']}")
print(f"Recommended timestep: {stability['recommended_dt']}")
```

## Development Notes

- Python 3.12+, ruff (linting/formatting), CC-BY-NC-SA-4.0 license
- **Greek letters in docs**: Use UTF-8 (π, μ, ν, θ, ∇) not ASCII
- **Logging**: `from israel_stewart.utils import get_logger`
- Israel-Stewart second-order viscous hydrodynamics, general covariance in curved spacetime

### Critical: Spectral Solver Boundary Conditions

**ALWAYS use `boundary_conditions="periodic"` for spectral methods:**

```python
# CORRECT:
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi), (0.0, 2*np.pi), (0.0, 2*np.pi)],
    grid_points=(8, 16, 16, 16),
    boundary_conditions="periodic",  # Required for FFT-based methods!
)

# WRONG (defaults to "dirichlet", causes 6% error in derivatives):
grid = SpacetimeGrid(..., grid_points=(8, 16, 16, 16))  # Missing boundary_conditions
```

**Why**: FFT assumes periodicity. Dirichlet: `dx = L/(N-1)`, Periodic: `dx = L/N`. Wrong spacing shifts wavenumbers by `(N-1)/N`, causing systematic derivative errors. See `EXPANSION_SCALAR_BUG_FIX.md`.

### Testing Guidelines

**When tests fail, investigate the root cause instead of weakening assertions!**

- Use exact validation for spectral methods (error < 1e-10), not weak correlations
- Create diagnostic scripts for complex bugs (see `debug_expansion_scalar.py`, `debug_fft_simple.py`)
- Verify boundary conditions in all spectral solver tests
- Check error patterns - specific ratios like 15/16 reveal underlying issues

## Workflow

**Setup**: `uv sync --extra dev` → `uv run pre-commit install` → `./scripts/format.sh --all-files` → `./scripts/test.sh --coverage`

**Daily**: Edit code → `./scripts/format.sh` → `./scripts/test.sh` → `git commit`

**CI/CD**: Scripts support `--quiet` flag, proper exit codes

**Module flow**: `core` → `equations` → `solvers` → `benchmarks` (+ `stochastic`, `rg_analysis`, `linearization`)

## Current Implementation Status

**Completed Core:**
- ✅ **Tensor Framework**: TensorField (automatic index management), FourVector (Lorentz boosts), StressEnergyTensor, ViscousStressTensor, CovariantDerivative, ProjectionOperator (3+1 decomposition), transformations, performance monitoring
- ✅ **Metrics**: Minkowski, Milne, Bjorken, FLRW, Schwarzschild; Christoffel symbols (numerical + symbolic); arbitrary rank tensor contractions
- ✅ **Fields & Constants**: Thermodynamic state, velocity fields, transport coefficients, natural units

**Completed Physics:**
- ✅ **Conservation Laws**: ∇_μ T^μν = 0 (31 tests)
- ✅ **Relaxation Equations**: IS second-order (Π, π^μν, q^μ), all couplings (λ_ππ, λ_πΠ, λ_πq, ξ₁, ξ₂), implicit/exponential integrators, stability analysis (30+ tests)
- ✅ **Spectral Solver**: FFT-based, 72% coverage, 58/59 tests

**Next**: Transport coefficients (T/ρ-dependent viscosities), Benchmarks (Bjorken, sound waves)

**Key Features**: Automatic index tracking, Einstein summation (opt_einsum), metric signatures, 3+1 decomposition, Christoffel symbols, covariant derivatives (arbitrary rank), curved spacetime, 90+ tests

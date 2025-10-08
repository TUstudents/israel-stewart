# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated Python implementation of relativistic hydrodynamics using the Israel-Stewart formalism with second-order viscous corrections. The codebase uses a **pure 3D spatial architecture** with time as an evolution parameter, achieving 95% memory reduction compared to 4D spacetime storage.

## Architecture: 3+1D Pure Spatial Formulation

**Key Concept**: Fields are stored as pure 3D spatial arrays `(nx, ny, nz)`. Time evolution is handled by numerical integrators (RK4, IMEX), not by storing 4D spacetime arrays.

**Memory Efficiency**: For 64³ grid with 12 fields:
- Old 4D architecture (nt=20): ~1500 MB
- New 3D architecture: ~75 MB
- **Reduction: 95%**

## Core Modules

**Spatial Grids (`core/`)**:
- `spacegrid.py`: Pure 3D spatial grid (recommended for all simulations)
- `spacetime_grid.py`: Legacy 4D grid (kept for backward compatibility, metadata only)
- `fields.py`: ISFieldConfiguration with pure 3D field storage
- `metrics.py`: Spacetime metrics and Christoffel symbols
- `constants.py`: Physical constants and unit systems

**Tensor Framework (`core/`)**:
- `tensor_base.py`: Core TensorField class with index management
- `four_vectors.py`: FourVector specialization with relativistic operations
- `stress_tensors.py`: StressEnergyTensor and ViscousStressTensor
- `derivatives.py`: CovariantDerivative and ProjectionOperator
- `transformations.py`: Lorentz and coordinate transformations
- `tensor_utils.py`: Validation and optimization utilities
- `performance.py`: Performance monitoring
- `tensors.py`: Consolidated imports for backward compatibility

**Utilities (`utils/`)**:
- `streaming.py`: Buffered snapshot writing with constant memory
- `io.py`: HDF5 trajectory I/O utilities
- `logging_config.py`: Structured logging with performance tracking
- `visualization.py`: Plotting utilities
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

## Quick Start: Pure 3D Architecture

**Recommended: SpaceGrid with Streaming Snapshots**

```python
from israel_stewart.core import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics
import numpy as np

# Create pure 3D spatial grid (no time dimension)
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(64, 64, 64),  # Just spatial: (nx, ny, nz)
    boundary_conditions="periodic"
)

# Initialize 3D fields (direct indexing, no time slice)
fields = ISFieldConfiguration(grid)
X, Y, Z = grid.meshgrid()

# Direct 3D field initialization
fields.rho[:] = 1.0 + 0.1 * np.sin(X)  # Pure 3D array (64, 64, 64)
fields.pressure[:] = fields.rho / 3.0
fields.u_mu[..., 0] = 1.0  # Rest frame

# Transport coefficients
coeffs = TransportCoefficients(
    shear_viscosity=0.1,
    bulk_viscosity=0.05,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3
)

# Create spectral solver (pure 3D operations)
hydro = SpectralISHydrodynamics(grid, fields, coeffs)

# Evolve with streaming snapshots (constant memory)
hydro.evolve(
    t_final=10.0,
    snapshot_config={
        "filename": "output.h5",
        "interval": 0.1,         # Save every 0.1 time units
        "buffer_size": 20,       # Buffer 20 snapshots before flushing
        "save_initial": True
    }
)

# Memory usage: ~75 MB (constant)
# vs old 4D approach: ~1500 MB (growing with nt)
```

## Development Notes

- Python 3.12+, ruff (linting/formatting), CC-BY-NC-SA-4.0 license
- **Greek letters in docs**: Use UTF-8 (π, μ, ν, θ, ∇) not ASCII
- **Logging**: `from israel_stewart.utils import get_logger`
- **Shebang convention**: Use `#!/usr/bin/env -S uv run python` for executable run scripts (e.g., `run_*.py`). Do NOT use shebangs in verification/diagnostic scripts - run them explicitly with `uv run python script.py`
- **Sign convention**: Convention B (Landau-Lifshitz) for stress tensor: `T^μν = (ε+p)u^μu^ν + p·g^μν + Π·Δ^μν - π^μν + q^μu^ν + q^νu^μ`. The MINUS sign treats π^μν as dissipative correction opposing flow, matching dispersion matrix convention.
- Israel-Stewart second-order viscous hydrodynamics, general covariance in curved spacetime

### SpaceGrid vs SpacetimeGrid

**Use SpaceGrid (Pure 3D) for all new code:**

```python
# ✅ RECOMMENDED: Pure 3D SpaceGrid
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(64, 64, 64),  # (nx, ny, nz)
    boundary_conditions="periodic"
)
fields = ISFieldConfiguration(grid)
fields.rho[:] = 1.0  # Direct 3D indexing
```

**SpacetimeGrid (Legacy 4D) - Deprecated:**

```python
# ❌ OLD: 4D SpacetimeGrid (deprecated, use only for backward compatibility)
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),  # Confusing: not used for evolution
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(1, 64, 64, 64),  # (nt, nx, ny, nz) - wasteful
    boundary_conditions="periodic"
)
fields = ISFieldConfiguration(grid)
# ISFieldConfiguration now always uses pure 3D storage internally
fields.rho[:] = 1.0  # Works, but SpaceGrid is clearer
```

**Key Differences:**
- **SpaceGrid**: 3D spatial grid, time is evolution parameter (modern)
- **SpacetimeGrid**: 4D spacetime grid, kept for metadata/backward compatibility (legacy)
- **Memory**: SpaceGrid uses 95% less memory (no time dimension)
- **Clarity**: SpaceGrid makes the 3+1D architecture explicit

### Critical: Spectral Solver Boundary Conditions

**ALWAYS use `boundary_conditions="periodic"` for spectral methods:**

```python
# CORRECT:
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(64, 64, 64),
    boundary_conditions="periodic",  # Required for FFT-based methods!
)

# WRONG (defaults to "dirichlet", causes 6% error in derivatives):
grid = SpaceGrid(..., grid_points=(64, 64, 64))  # Missing boundary_conditions
```

**Why**: FFT assumes periodicity. Dirichlet: `dx = L/(N-1)`, Periodic: `dx = L/N`. Wrong spacing shifts wavenumbers by `(N-1)/N`, causing systematic derivative errors. See `EXPANSION_SCALAR_BUG_FIX.md`.

### Linear Regime Detection

**For small perturbations, use linearized momentum conversion to avoid spurious harmonics:**

The spectral solver automatically detects linear regime when:
- `|δρ| < 0.1` (density perturbation < 10% of background)
- `|v| < 0.1` (velocity perturbation small)

**Why this matters**: The nonlinear momentum-to-velocity conversion `du/dt = [d(h·u)/dt - u·dh/dt]/h` creates spurious 2nd harmonics in the linear regime. When conditions are met, the solver uses linearized form `du/dt = d(h·u)/dt / h₀` where h₀ = 4/3 is the background enthalpy for radiation fluid.

**Implementation**: `israel_stewart/solvers/spectral.py:_convert_momentum_to_velocity_derivative_with_fields()`

### Testing Guidelines

**When tests fail, investigate the root cause instead of weakening assertions!**

- Use exact validation for spectral methods (error < 1e-10), not weak correlations
- Create diagnostic scripts for complex bugs (see `verify_spectral_solver_physics/` for examples)
- Verify boundary conditions in all spectral solver tests
- Check error patterns - specific ratios like 15/16 reveal underlying issues

#### Numerical Truncation vs Bugs

**Know when to accept small drift vs investigate:**

- **RHS at t=0**: Demand exact accuracy (< 1e-10 error). If RHS doesn't match analytical prediction initially, investigate.
- **Long-time eigenmode evolution**: Accept small drift (~6-8% at t=0.1 on 32³ grid). This is expected numerical truncation from discretization, not a bug.
- **Convergence test**: If error scales as O(dt^4) for RK4, it's truncation. If error increases with smaller timesteps, investigate.
- **Error patterns**: Specific ratios (e.g., 15/16, 63/64) or systematic biases indicate bugs in spacing/indexing, not truncation.

**Diagnostic tools**: Use scripts in `verify_spectral_solver_physics/` to compare numerical vs analytical RHS, track eigenmode structure, check for spurious harmonics. Example:
```bash
uv run python verify_spectral_solver_physics/compare_analytical_vs_numerical_rhs.py
```

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
- ✅ **Spectral Solver**: FFT-based with linear regime detection, periodic boundary conditions
- ✅ **Benchmarks**: Bjorken flow, sound wave propagation, equilibration dynamics (executable via `run_*.py` scripts)

**Key Features**: Automatic index tracking, Einstein summation (opt_einsum), metric signatures, 3+1 decomposition, Christoffel symbols, covariant derivatives (arbitrary rank), curved spacetime, comprehensive validation suite

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
- **Sign convention** (CRITICAL): For (-,+,+,+) signature: `T^μν = (ε+p)u^μu^ν + p·g^μν + Π·Δ^μν + π^μν + q^μu^ν + q^νu^μ`. ALL dissipative terms have PLUS signs. This follows from IReD paper eq. (5) after metric conversion (see `docs/IRED_THEORY.md` Section 1.3).
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

### Israel-Stewart Regime of Applicability

**Israel-Stewart hydrodynamics has a fundamental physical regime where it is valid.** Operating outside this regime leads to unphysical results and numerical instabilities.

#### Regime Condition (Wagner & Gavassino 2024)

For plane wave modes with wavenumber k and frequency ω:
```
|τω| ≲ 1
```

where τ is the relaxation time (max of τ_π, τ_Π).

**Physical interpretation**: The relaxation time must be smaller than or comparable to the oscillation period. If τω >> 1, dissipative fluxes cannot relax fast enough to track hydrodynamic variables.

**For sound waves**: ω ≈ k·c_s, so the condition becomes:
```
k ≲ 1/(τ·c_s)
```

For radiation fluid (c_s = 1/√3 ≈ 0.577) with typical τ ~ 0.5:
```
k_max ≈ 1/(0.5 × 0.577) ≈ 3.5

Recommended: k_max ≤ 4 (with safety margin)
```

#### Practical Guidelines

**✓ Valid regimes:**
- Low-moderate wavenumbers: k ≲ 4 for τ ~ 0.5
- Smooth flows (spatial variations over scales >> mean free path)
- Temporal variations on timescales ≳ τ

**✗ Invalid regimes:**
- High wavenumbers: k > 1/(τ·c_s) → instability expected
- Sharp shocks or discontinuities
- Far-from-equilibrium dynamics

**Example regime check:**
```python
# Estimate maximum frequency
k_max = np.max(np.abs(grid.wave_numbers))
c_s = 1.0 / np.sqrt(3.0)  # Radiation fluid
omega_max = k_max * c_s

# Check regime
tau_max = max(coeffs.shear_relaxation_time, coeffs.bulk_relaxation_time)
regime_param = abs(tau_max * omega_max)

if regime_param > 1.0:
    logger.warning(
        f"|τω| = {regime_param:.2f} > 1. Outside Israel-Stewart regime. "
        "Expect unphysical results. Reduce k_max or relaxation times."
    )
```

#### Reference

Wagner & Gavassino, "The regime of applicability of Israel-Stewart hydrodynamics" (2024), arXiv:2309.14828v2. See `HIGH_K_INSTABILITY_RESOLUTION.md` for detailed analysis.

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

### IReD Formulation and Transport Coefficients

**This codebase implements the IReD (Inverse-Reynolds-Dominance) formulation of Israel-Stewart hydrodynamics** as described in Wagner, Palermo, Ambrus (2022). IReD is formally equivalent to the DNMR (Denicol-Niemi-Molnár-Rischke) approach up to second order in gradients, but eliminates O(Kn²) parabolic terms by construction.

#### Form B Relaxation Equations (Standard IReD)

**CRITICAL**: The relaxation equations use Form B structure (WITHOUT `/τ` in source terms):

```python
# ✅ CORRECT (Form B - IReD formulation):
dΠ/dt = -Π/τ_Π - ζθ + J_terms
dπ^μν/dt = -π^μν/τ_π - 2ησ^μν + J_terms
dq^μ/dt = -q^μ/τ_q - κΔ^μν∂_νT + J_terms

# ❌ WRONG (Form A - causes numerical instability):
dΠ/dt = -Π/τ_Π - ζθ/τ_Π + J_terms
```

**Why Form B?** This implements operator splitting (separate relaxation and forcing), not algebraic solution. Form A causes severe numerical instabilities despite appearing in some dispersion relation derivations.

**Implementation**: See `israel_stewart/equations/relaxation.py:200-348`
- Line 227: `first_order = -self.coeffs.bulk_viscosity * theta` (Form B for bulk)
- Line 289: `first_order = 2.0 * self.coeffs.shear_viscosity * sigma_munu` (Form B for shear)

#### IReD vs DNMR: Key Differences

| Aspect | DNMR | IReD |
|--------|------|------|
| **Approach** | Eigenmode decomposition | Direct asymptotic matching |
| **K terms** | K^{μ₁...μℓ} ≠ 0 (parabolic) | K^{μ₁...μℓ} = 0 (by construction) |
| **Relaxation times** | Inverse eigenvalues: τ^(ℓ) = 1/ω_ℓ | Weighted averages: τ_Π = Σ_r τ^(0)_{0r} C^(0)_r |
| **Convergence** | τ decreases with order | τ increases with order |
| **Accuracy** | IReD ≻ DNMR ≻ tDNMR ≻ NS ≻ 2ndOH | (Wagner & Gavassino 2024) |

**Formal equivalence**: IReD ≡ DNMR at second order. Transport coefficients related by Table II in IReD.pdf (Wagner et al. 2022).

#### Current Transport Coefficients

**Status**: ✅ **IReD hard sphere benchmark implementation available** (Phase 14A complete)

**Option 1: IReD Hard Sphere Gas (Quantitatively Accurate)**

```python
from israel_stewart.equations.ired_simple import HardSphereIReD

# Use IReD hard sphere benchmark (Wagner et al. 2022, Tables III-IV)
model = HardSphereIReD(
    temperature=0.4,  # 400 MeV
    cross_section=1.0,  # 1 fm²
    truncation="41"  # 41-moment accuracy
)

# All transport coefficients computed from kinetic theory
coeffs = TransportCoefficients(
    shear_viscosity=model.shear_viscosity(),  # η = 1.2678/(σβ)
    shear_relaxation_time=model.shear_relaxation_time(),  # τ_π = 1.6552 λ_mfp
    lambda_pi_pi=model.tau_pi_pi(),  # τ_ππ = 1.6944 τ_π
    delta_pi_pi=model.delta_pi_pi(),  # δ_ππ = 4/3
    lambda_pi_V=model.lambda_pi_V(),  # λ_πV from IReD Table III
    # ... all 10+ second-order coefficients
)

# Validation: η/s, regime parameters, etc.
print(f"η/s = {model.eta_over_s():.4f}")
print(f"|τω| at k=2 fm⁻¹: {model.regime_parameter(2.0):.2f}")
```

**Option 2: Phenomenological (Legacy, Exploratory Work)**

```python
coeffs = TransportCoefficients(
    # First-order (required)
    shear_viscosity=0.1,      # η (phenomenological)
    bulk_viscosity=0.05,      # ζ (phenomenological)

    # Relaxation times (required)
    shear_relaxation_time=0.5,  # τ_π (phenomenological)
    bulk_relaxation_time=0.3,   # τ_Π (phenomenological)

    # IReD second-order coefficients (optional, phenomenological)
    delta_Pi_Pi=0.0,   # δ_ΠΠ (bulk self-coupling to expansion)
    lambda_Pi_pi=0.0,  # λ_Ππ (bulk-shear coupling)
    lambda_pi_pi=0.0,  # λ_ππ (shear self-coupling)
    # Note: For rigorous values, use HardSphereIReD from ired_simple.py
)
```

**IReD Implementation Details**:
- **Module**: `israel_stewart/equations/ired_simple.py`
- **Tests**: 29/29 passing in `test_ired_coefficients.py`
- **Validates against**: IReD paper Tables III-IV (< 0.01% error)
- **Coefficients**: First-order (η, ζ=0, D) + relaxation times (τ_π, τ_V) + 10 second-order couplings
- **Accuracy**: 41-moment truncation converges to 0.03% of exact kinetic theory
- **Limitations**: Hard sphere gas only (constant cross-section, massless, conformal)

**For other systems**: Full collision matrix solver (Phase 14B, future work) or use phenomenological Option 2

#### Verification

**Run the IReD verification script** to check implementation correctness:

```bash
uv run python verify_ired_implementation.py
```

**This checks**:
1. ✅ Form B structure in relaxation equations (no `/τ` in source terms)
2. ✅ Regime applicability warning triggers appropriately (|τω| > 1)
3. ✅ Required transport coefficients present
4. ✅ Numerical RHS matches analytical predictions

**Expected output**: 12/16 checks pass (4 failures are acceptable: optional coefficients and method lookup).

#### Documentation

**Comprehensive theory**: `docs/IRED_THEORY.md` (~12,000 words)
- Part I: Theoretical foundation (Boltzmann, moments, Landau frame)
- Part II: IReD vs DNMR approaches
- Part III: Relaxation equations with explicit formulas
- Part IV: Regime of applicability (|τω| ≲ 1)
- Part V: Formal equivalence proof
- Part VI: Implementation guide and code analysis
- Part VII: Numerical benchmarks

**Quick reference**: `docs/IRED_QUICK_REFERENCE.md`
- One-page lookup for key equations
- Transport coefficient formulas
- IReD ↔ DNMR conversion table
- Code variable cross-reference

**Historical context**: See `DISPERSION_MATRIX_PARADOX.md` and `HIGH_K_INSTABILITY_RESOLUTION.md` for resolution of Form A vs Form B confusion.

#### Key References

- **IReD formulation**: Wagner, Palermo, Ambrus (2022), arXiv:2208.02506 - `docs/IReD.pdf`
- **Regime applicability**: Wagner & Gavassino (2024), arXiv:2309.14828v2 - `docs/regime of applicability.pdf`
- **DNMR approach**: Denicol et al. (2012), arXiv:1202.4551
- **Conformal case**: Baier et al. (2008), arXiv:0712.2451 - `docs/JHEP042008100.pdf`

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
- ✅ **Conservation Laws**: ∇_μ T^μν = 0 (31 static tests), ✅ **Dynamic Conservation** (12 tests in `test_dynamic_conservation.py`):
  - Global conservation: ∫ρ d³x, ∫(ρu^i) d³x, ∫n d³x conserved during evolution (3 tests passing ✓)
  - Local balance: ∂_t ρ + ∇·T^{i0} = 0, ∂_t(ρu^j) + ∇·T^{ij} = 0 pointwise (3 tests)
  - Constraint maintenance: V^μ u_μ = 0, π^μν u_μ = 0, u·u = -1 throughout evolution (3 tests)
  - Physical scenarios: Sound waves, diffusion, Bjorken expansion (3 tests)
- ✅ **Relaxation Equations**: IS second-order (Π, π^μν, V^μ Landau frame), all couplings (λ_ππ, λ_πΠ, λ_πV, ξ₁, ξ₂), implicit/exponential integrators, stability analysis (21/21 tests passing)
- ✅ **IReD Transport Coefficients** (Phase 14A): Hard sphere gas benchmark from kinetic theory (Wagner et al. 2022), validates against IReD Tables III-IV (29/29 tests passing)
- ✅ **Spectral Solver**: FFT-based with linear regime detection, periodic boundary conditions
- ✅ **Benchmarks**: Bjorken flow, sound wave propagation, equilibration dynamics (executable via `run_*.py` scripts)

**Key Features**: Automatic index tracking, Einstein summation (opt_einsum), metric signatures, 3+1 decomposition, Christoffel symbols, covariant derivatives (arbitrary rank), curved spacetime, comprehensive validation suite

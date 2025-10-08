# Israel-Stewart Relativistic Hydrodynamics

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://pypi.org/classifiers/)

A comprehensive Python framework for relativistic hydrodynamics using the **Israel-Stewart formalism** with second-order viscous corrections. This package provides production-ready numerical tools for simulating relativistic fluid dynamics in curved spacetime with complete tensor algebra support and advanced numerical methods.

## 🔬 Scientific Overview

The Israel-Stewart formalism extends ideal relativistic hydrodynamics beyond the first-order Navier-Stokes approximation by incorporating **second-order viscous corrections** and **finite relaxation times**. This framework is essential for accurate modeling of:

- **Heavy-ion collision dynamics** in relativistic nuclear physics
- **Quark-gluon plasma** evolution and thermalization
- **Cosmological fluid evolution** in the early universe
- **Neutron star matter** under extreme conditions
- **Relativistic turbulence** and instability analysis

### Mathematical Foundation

The package implements the complete **3+1 decomposition** of spacetime with the Israel-Stewart stress-energy tensor:

```
T^μν = (ε+p) u^μ u^ν + p Δ^μν + Π Δ^μν - π^μν + q^μ u^ν + q^ν u^μ
```

Where:
- `ε`: Energy density in the fluid rest frame
- `p`: Thermodynamic pressure
- `π^μν`: Traceless viscous shear stress tensor (MINUS sign = Convention B)
- `Π`: Bulk viscous pressure
- `q^μ`: Heat flux four-vector
- `Δ^μν = g^μν + u^μ u^ν`: Spatial projection tensor

**Note**: The MINUS sign for π^μν follows Convention B (Landau-Lifshitz), treating shear stress as a dissipative correction opposing flow. This ensures consistency with dispersion relations in linear stability analysis.

The evolution equations include second-order relaxation dynamics:
- **Energy-momentum conservation**: `∇_μ T^μν = 0`
- **Shear relaxation**: `τ_π ∂_t π^μν + π^μν = -2η σ^μν + ...`
- **Bulk relaxation**: `τ_Π ∂_t Π + Π = -ξ ∇_μ u^μ + ...`

### Pure 3D Spatial Architecture

The package uses a **pure 3+1 formulation** where fields are stored as 3D spatial arrays `(nx, ny, nz)` with time as an evolution parameter, **not** as 4D spacetime arrays `(nt, nx, ny, nz)`.

**Memory Efficiency**: For a 64³ spatial grid with 12 fields:
- **Pure 3D architecture**: ~75 MB (constant during evolution)
- **Legacy 4D storage (nt=20)**: ~1500 MB (grows with time steps)
- **Reduction**: **95% less memory** ✨

Time evolution is handled by numerical integrators (RK4, IMEX, split-step) that advance the state forward, not by storing full 4D spacetime history. This enables:
- **Larger spatial grids** for higher resolution
- **Longer simulations** without memory constraints
- **Streaming output** with constant memory footprint

**Implementation**: Use `SpaceGrid` for pure 3D spatial grids (recommended) instead of legacy `SpacetimeGrid`.

## 🚀 Key Features

### Memory-Efficient Architecture
- **Pure 3D spatial storage** with 95% memory reduction vs 4D spacetime arrays
- **Streaming output** with constant memory footprint for long simulations
- **Time evolution** via numerical integrators (RK4, IMEX, split-step), not 4D storage

### Complete Tensor Framework
- **Modular tensor algebra** with automatic covariant/contravariant index tracking
- **Arbitrary rank tensor operations** with optimized Einstein summation
- **Four-vector operations** including Lorentz boosts and proper time evolution
- **Stress-energy tensors** using Convention B (Landau-Lifshitz) for consistent dispersion relations
- **Covariant derivatives** with complete Christoffel symbol computation

### Curved Spacetime Support
- **Multiple coordinate systems**: Minkowski, Milne, Bjorken, FLRW, Schwarzschild metrics
- **Numerical Christoffel symbols** via finite difference derivatives on arbitrary grids
- **Symbolic Christoffel symbols** through automatic differentiation with SymPy
- **General relativity integration** for cosmological and astrophysical applications
- **Performance optimization** with cached metric computations

### Physics Implementation Status
- ✅ **Conservation laws** (∇_μ T^μν = 0): Complete implementation with 31 test cases
- ✅ **Israel-Stewart relaxation equations**: Full second-order viscous evolution with 30+ tests
- ✅ **Transport coefficients**: Enhanced framework with second-order coupling terms
- ✅ **Thermodynamic fields**: Complete state vector with validation constraints
- 🚧 **Extended transport models**: Temperature and density-dependent viscosities

### Advanced Numerical Methods
- **Finite difference schemes**: Conservative, upwind, and WENO methods for spatial discretization
- **Implicit time integration**: Backward Euler, IMEX Runge-Kutta, and exponential integrators
- **Operator splitting**: Strang, Lie-Trotter, adaptive, and physics-based splitting
- **Spectral methods**: FFT-based high-accuracy solvers for periodic problems
- **Stability analysis**: Automatic stiffness detection and timestep recommendations

## 📦 Installation

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Development Installation
```bash
# Clone the repository
git clone https://github.com/your-org/israel-stewart.git
cd israel-stewart

# Install with all development dependencies
uv sync --extra dev

# Set up pre-commit hooks for code quality
uv run pre-commit install
```

### Optional Dependencies
```bash
# Install with Jupyter notebook support
uv sync --extra jupyter

# Install all optional dependencies
uv sync --extra all
```

### Core Dependencies
- **numpy** ≥1.24.0: High-performance numerical arrays
- **scipy** ≥1.10.0: Scientific computing and optimization
- **sympy** ≥1.12: Symbolic mathematics for analytical computations
- **matplotlib** ≥3.6.0: Scientific plotting and visualization
- **numba** ≥0.57.0: JIT compilation for performance-critical code
- **h5py** ≥3.8.0: HDF5 data storage for large simulations

## 🔬 Quick Start Examples

### Basic Israel-Stewart System
```python
from israel_stewart.core import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.solvers.spectral import SpectralISHydrodynamics
import numpy as np

# Setup pure 3D spatial grid (recommended)
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(64, 64, 64),  # Pure 3D: (nx, ny, nz)
    boundary_conditions="periodic"
)

# Transport coefficients with second-order terms
coeffs = TransportCoefficients(
    shear_viscosity=0.1,
    bulk_viscosity=0.05,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3,
    lambda_pi_pi=0.1,  # Second-order coupling
    xi_1=0.2           # Bulk nonlinearity
)

# Initialize field configuration (pure 3D arrays)
fields = ISFieldConfiguration(grid)
X, Y, Z = grid.meshgrid()

# Direct 3D field initialization
fields.rho[:] = 1.0 + 0.1 * np.sin(X)  # Energy density
fields.pressure[:] = fields.rho / 3.0   # Pressure (radiation fluid)
fields.u_mu[..., 0] = 1.0               # Rest frame (u^t = 1)
fields.Pi[:] = 0.01                      # Bulk pressure
fields.pi_munu[..., 1, 1] = 0.005       # Shear tensor component

# Create spectral solver with pure 3D operations
solver = SpectralISHydrodynamics(grid, fields, coeffs)

# Evolve with streaming snapshots (constant memory)
solver.evolve(
    t_final=10.0,
    dt=0.01,
    snapshot_config={
        "filename": "output.h5",
        "interval": 0.1,      # Save every 0.1 time units
        "buffer_size": 20     # Buffer before flushing to disk
    }
)
```

### Running Complete Benchmarks
```python
# Execute validated physics benchmarks with analytical solutions

# 1D Bjorken flow (boost-invariant expansion)
from israel_stewart.benchmarks import BjorkenFlowBenchmark
bjorken = BjorkenFlowBenchmark(
    tau_range=(0.5, 5.0),
    transport_coeffs=coeffs
)
bjorken.run_validation()
bjorken.plot_comparison()  # Compare with exact solution

# Sound wave propagation in viscous fluids
from israel_stewart.benchmarks import SoundWaveBenchmark
sound = SoundWaveBenchmark(
    wave_number=8.0,
    grid_points=(64, 64, 64),
    transport_coeffs=coeffs
)
sound.run_validation()
sound.plot_dispersion_relation()

# Relaxation to thermal equilibrium
from israel_stewart.benchmarks import EquilibrationBenchmark
equil = EquilibrationBenchmark(
    initial_perturbation=0.1,
    transport_coeffs=coeffs
)
equil.run_validation()
equil.plot_relaxation_curves()
```

Or run complete validation directly:
```bash
./run_bjorken_benchmark.py      # 1D boost-invariant expansion
./run_sound_wave_benchmark.py    # Linear wave propagation
./run_equilibration_benchmark.py # Relaxation to equilibrium
```

### Conservation Law Validation
```python
from israel_stewart.equations.conservation import ConservationLaws
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.metrics import MinkowskiMetric

# Setup pure 3D grid
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(-1.0, 1.0)] * 3,
    grid_points=(64, 64, 64),
    boundary_conditions="periodic"
)

# Initialize conservation law system
metric = MinkowskiMetric()
conservation = ConservationLaws(grid, metric)

# Compute energy-momentum conservation
div_T = conservation.energy_momentum_conservation(fields, coeffs)
print(f"Conservation violation: {np.max(np.abs(div_T)):.2e}")

# Particle number conservation
div_N = conservation.particle_number_conservation(fields)
print(f"Particle conservation: {np.max(np.abs(div_N)):.2e}")
```

## 🏗️ Architecture and Implementation

### Modular Physics-Based Design
```
israel_stewart/
├── core/           # Foundation: tensors, metrics, fields, spatial grids
├── equations/      # Physics: conservation laws, IS relaxation equations
├── solvers/        # Numerical methods: spectral, finite difference, implicit
├── benchmarks/     # Validation: Bjorken flow, sound waves, equilibration
├── stochastic/     # Advanced: fluctuation-dissipation relations
├── rg_analysis/    # Theory: renormalization group techniques
└── linearization/  # Analysis: stability and dispersion relations
```

### Core Implementation (✅ Production Ready)
- **`tensor_base.py`**: Complete TensorField class with automatic index management
- **`spacegrid.py`**: Pure 3D spatial grids with 95% memory reduction
- **`metrics.py`**: Multiple spacetime metrics (Minkowski, Milne, Bjorken, FLRW, Schwarzschild)
- **`fields.py`**: ISFieldConfiguration with pure 3D storage and Convention B stress tensor
- **`derivatives.py`**: Covariant derivatives and projection operators for curved spacetime

### Solver Implementation (✅ Production Ready)
- **Spectral solver**: FFT-based with linear regime detection and periodic boundary conditions
- **Finite difference**: Conservative schemes with adaptive timestep control
- **Implicit methods**: IMEX and exponential integrators for stiff relaxation equations
- **Benchmarks**: Complete validation suite with analytical solutions (run_*.py scripts)

## 🧪 Testing and Validation

### Comprehensive Test Suite
- **Extensive test coverage** across all physics modules with continuous integration
- **Physics validation**: Benchmark comparisons with exact analytical solutions
- **Numerical verification**: Conservation law accuracy and convergence analysis
- **Performance benchmarks**: Tensor operation optimization and scaling tests

### Test Categories
- **Core tensor framework**: Tensor operations, index management, Einstein summation
- **Conservation laws**: Energy-momentum and particle conservation validation
- **Relaxation equations**: Second-order Israel-Stewart evolution with coupling terms
- **Numerical methods**: Spectral solver convergence, stability, and accuracy validation
- **Curved spacetime**: Christoffel symbol computation and metric validation

### Running Tests
```bash
# Full test suite with coverage
./scripts/test.sh --coverage

# Fast tests only (excludes benchmarks)
uv run pytest -m "not slow"

# Specific physics module
uv run pytest israel_stewart/tests/test_conservation.py -v

# Performance benchmarks
uv run pytest -m benchmark --benchmark-only
```

### Verification and Diagnostic Tools

The `verify_spectral_solver_physics/` directory contains specialized diagnostic scripts for detailed validation:

```bash
# Compare numerical RHS with analytical predictions
uv run python verify_spectral_solver_physics/compare_analytical_vs_numerical_rhs.py

# Track eigenmode structure preservation during evolution
uv run python verify_spectral_solver_physics/track_eigenmode_structure.py

# Verify SVD nullspace accuracy for dispersion matrix
uv run python verify_spectral_solver_physics/check_svd_nullspace.py

# Check for spurious harmonic generation
uv run python verify_spectral_solver_physics/check_spurious_harmonics.py
```

These tools provide:
- **RHS validation**: Verify that numerical right-hand-side matches analytical predictions
- **Eigenmode tracking**: Monitor how eigenmodes drift during time evolution
- **Dispersion analysis**: Compare numerical dispersion relations with theory
- **Convergence testing**: Verify proper scaling with timestep and grid resolution

See `verify_spectral_solver_physics/summary_findings.md` for detailed investigation results and validation methodology.

## 📚 Documentation and Examples

### Development Workflow
```bash
# Code formatting and quality
./scripts/format.sh              # Multi-pass ruff formatting
uv run mypy israel_stewart       # Type checking
./scripts/test.sh               # Comprehensive testing

# Build and validation
./scripts/build.sh --clean      # Clean package build
```

### Physics Examples
The `benchmarks/` directory contains validated physics examples with executable runners:
- **`bjorken_flow.py`**: 1D boost-invariant expansion with exact solutions → Run with `./run_bjorken_benchmark.py`
- **`sound_waves.py`**: Linear wave propagation in relativistic media → Run with `./run_sound_wave_benchmark.py`
- **`equilibration.py`**: Relaxation to thermal equilibrium → Run with `./run_equilibration_benchmark.py`

Each benchmark provides complete validation against analytical solutions with visualization and error analysis.

### Advanced Features
```python
# Performance monitoring with automatic optimization
from israel_stewart.core.performance import monitor_performance, performance_report

@monitor_performance
def compute_stress_tensor(fields, metric):
    # Automatically tracked for bottlenecks
    return fields.compute_israel_stewart_tensor(metric)

# Get optimization recommendations
report = performance_report()
print(report.optimization_suggestions)
```

## 🔬 Current Implementation Status

### ✅ Production-Ready Components
- **Pure 3D spatial architecture** with 95% memory reduction vs 4D storage
- **Complete tensor framework** with automatic index management and Einstein summation
- **Full Israel-Stewart physics** including all second-order coupling terms and Convention B stress tensor
- **Spectral and finite difference solvers** with linear regime detection and adaptive timestep control
- **Multiple spacetime metrics** (Minkowski, Milne, Bjorken, FLRW, Schwarzschild) with Christoffel symbols
- **Complete benchmark suite**: Bjorken flow, sound wave propagation, equilibration dynamics
- **Comprehensive validation**: Physics tests, convergence analysis, and diagnostic tools

### 📈 Performance Characteristics
- **Memory efficiency**: Pure 3D storage with streaming output for constant memory footprint
- **Tensor operations**: Optimized with `opt_einsum` and performance monitoring
- **Scalability**: Tested on grids up to 128³ spatial points with linear regime detection
- **Numerical stability**: Automatic stiffness detection, timestep adaptation, and convergence testing

## 🌟 Physics Applications

### Heavy-Ion Collision Dynamics
```python
from israel_stewart.core import MilneMetric, BJorkenMetric

# Boost-invariant coordinates for RHIC/LHC collisions
milne_metric = MilneMetric()
bjorken_flow = ISFieldConfiguration(milne_grid)

# Initialize quark-gluon plasma state
bjorken_flow.initialize_bjorken_profile(
    initial_energy_density=30.0,  # GeV/fm³
    initial_temperature=0.3,      # GeV
    longitudinal_expansion=True
)
```

### Cosmological Applications
```python
# Friedmann-Lemaître-Robertson-Walker metric
flrw_metric = FLRWMetric(scale_factor_power=2/3)  # Matter-dominated universe

# Dark matter + radiation fluid
cosmic_fields = ISFieldConfiguration(cosmic_grid)
cosmic_fields.setup_two_component_fluid(
    matter_density=0.26,
    radiation_density=0.74
)
```

### Neutron Star Applications
```python
# Schwarzschild metric for strong gravitational fields
schwarzschild = SchwarzschildMetric(mass=1.4)  # Solar masses

# High-density nuclear matter
nuclear_matter = ISFieldConfiguration(stellar_grid)
nuclear_matter.setup_nuclear_eos(
    baryon_density=0.5,  # fm⁻³
    temperature=10.0     # MeV
)
```

## 📄 License and Attribution

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**. This package is designed for **academic and research use only**.

### Citation
If you use this code in published research, please cite:

```bibtex
@software{israel_stewart_2024,
  title = {Israel-Stewart Relativistic Hydrodynamics: A Python Framework},
  author = {Relativistic Hydrodynamics Team},
  year = {2024},
  version = {0.1.0},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/your-org/israel-stewart},
  note = {Python package for second-order viscous relativistic hydrodynamics}
}
```

### Physics References
The theoretical framework implemented in this package is based on:

1. **W. Israel and J.M. Stewart** (1979). "Transient relativistic thermodynamics and kinetic theory." *Ann. Phys.* **118**, 341-372.
2. **P. Romatschke and U. Romatschke** (2019). "Relativistic Fluid Dynamics In and Out of Equilibrium." *Cambridge University Press*.
3. **G.S. Denicol and J. Noronha** (2016). "Analytical attractor and the divergence of the slow-roll expansion in relativistic hydrodynamics." *Phys. Rev. D* **94**, 054040.

## 🤝 Contributing

We welcome contributions from the relativistic hydrodynamics community!

### Development Guidelines
1. **Follow the physics**: Ensure theoretical accuracy and proper covariant formulation
2. **Maintain code quality**: All contributions must pass `ruff`, `mypy`, and test suite
3. **Add comprehensive tests**: Physics changes require validation against analytical solutions
4. **Document thoroughly**: Include docstrings with proper Greek letter notation (π, μ, ν, λ, ξ, etc.)
5. **Performance awareness**: Use the built-in performance monitoring for optimization

### Development Environment Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/israel-stewart.git
cd israel-stewart

# Development installation with all tools
uv sync --extra dev --extra jupyter

# Set up development hooks
uv run pre-commit install

# Run full validation before contributing
./scripts/format.sh --all-files
./scripts/test.sh --coverage
./scripts/build.sh --clean
```

For questions about the physics implementation or to discuss new features, please open an issue on the repository.

---

**Explore the fundamental physics of relativistic matter with Israel-Stewart!** 🌌⚛️

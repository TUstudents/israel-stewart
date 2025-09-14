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

**Code Quality**:
- `uv run ruff check` - Run linting
- `uv run ruff format` - Format code
- `uv run mypy israel_stewart` - Type checking
- `uv run pytest` - Run tests
- `uv run pytest -m "not slow"` - Skip slow tests
- `uv run pytest --cov` - Run tests with coverage

**Development Environment**:
- `uv run jupyter lab` - Start JupyterLab
- `uv run python -m israel_stewart` - Run package

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

- Requires Python 3.12+
- Uses ruff for linting and formatting (replaces black/flake8/isort)
- Licensed under CC-BY-NC-SA-4.0 (non-commercial research use)
- **Greek Letters in Documentation**: When writing physics equations in docstrings, always use proper UTF-8 Greek letters (π, μ, ν, λ, ξ, ρ, σ, τ, θ, ω, Π, Λ, Σ, Ω, ∇) instead of ASCII approximations (pi, mu, nu, lambda, xi, rho, sigma, tau, theta, omega, Pi, Lambda, Sigma, Omega, nabla). This ensures mathematical clarity and professional appearance in documentation.
- Core tensor framework is fully implemented with modular architecture
- Physics equations modules (`equations/`) are placeholder files ready for implementation
- Solver modules (`solvers/`) are placeholder files ready for implementation
- The physics follows the Israel-Stewart second-order viscous hydrodynamics formalism
- Tensor operations should respect general covariance and work in curved spacetime
- Numerical methods should handle the stiff relaxation timescales in the IS equations
- The RG analysis module implements field-theoretic renormalization group techniques for hydrodynamic fluctuations

## Module Dependencies

The typical flow is: `core` → `equations` → `solvers` → `benchmarks`, with `stochastic`, `rg_analysis`, and `linearization` as specialized analysis tools that depend on the base physics implementation.

## Current Implementation Status

**Completed Core Modules:**
- ✅ **Tensor Framework**: Fully modularized tensor algebra system
  - `tensor_base.py`: TensorField class with automatic index management
  - `four_vectors.py`: FourVector with Lorentz boosts and relativistic operations
  - `stress_tensors.py`: StressEnergyTensor and ViscousStressTensor classes
  - `derivatives.py`: CovariantDerivative and ProjectionOperator for 3+1 decomposition
  - `transformations.py`: Lorentz and coordinate transformations
  - `tensor_utils.py`: Validation, type guards, and optimization utilities
  - `performance.py`: Performance monitoring with operation timing and recommendations
- ✅ **Metrics & Christoffel Symbols**: Complete curved spacetime support
  - `MinkowskiMetric`: Flat spacetime with both signature conventions
  - `GeneralMetric`: Arbitrary curved spacetimes with symbolic/numerical support
  - `MilneMetric`: Boost-invariant coordinates for relativistic heavy-ion collisions
  - `BJorkenMetric`: Specialized Bjorken flow implementation
  - `FLRWMetric`: Cosmological Friedmann-Lemaître-Robertson-Walker metric
  - `SchwarzschildMetric`: Black hole spacetime metric
  - **Numerical Christoffel computation**: Finite difference derivatives on arbitrary grids
  - **Symbolic Christoffel computation**: Automatic symbolic differentiation
  - **Arbitrary rank tensor contractions**: Support for tensors of any rank
  - **Grid integration**: Direct integration with SpacetimeGrid coordinate systems
- ✅ **Fields**: Thermodynamic state, velocity fields, and transport coefficients
- ✅ **Constants**: Physical constants in natural units with validation functions

**Completed Physics Modules:**
- ✅ **Conservation Laws** (`equations/conservation.py`): Complete energy-momentum conservation ∇_μ T^μν = 0
  - Full Israel-Stewart stress-energy tensor construction
  - Covariant divergence computation with Christoffel symbols
  - Particle number conservation ∇_μ N^μ = 0
  - Evolution equation extraction from conservation laws
  - Comprehensive validation with 31 test cases
- ✅ **Relaxation Equations** (`equations/relaxation.py`): Complete Israel-Stewart second-order viscous evolution
  - **Full relaxation dynamics**: Bulk pressure Π, shear tensor π^μν, heat flux q^μ evolution
  - **All second-order coupling terms**: λ_ππ, λ_πΠ, λ_πq, ξ₁, ξ₂ nonlinear coefficients
  - **Multiple numerical methods**: Explicit, implicit, and exponential integrators
  - **Stability analysis**: Automatic stiffness detection and timestep recommendations
  - **Symbolic equation framework**: SymPy-based exact derivative computation
  - **Performance optimization**: Efficient tensor contractions with monitoring
  - **Enhanced transport coefficients**: Complete second-order coefficient framework
  - **Comprehensive validation**: 30+ test cases covering all physics and numerical methods

**Next Implementation Priority:**
1. **Transport Coefficients** (`equations/coefficients.py`): Extended temperature and density-dependent viscosities
2. **Numerical Solvers** (`solvers/`): Specialized time integration schemes for stiff IS equations
3. **Advanced Field Dynamics** (`equations/constraints.py`): Thermodynamic consistency enforcement

**Key Features Implemented:**
- Automatic covariant/contravariant index tracking
- Metric signature handling (mostly-plus vs mostly-minus conventions)
- Einstein summation with opt_einsum optimization
- 3+1 fluid decomposition with projection operators
- **Complete Christoffel symbol framework**: Numerical and symbolic computation
- **Curved spacetime support**: Full general relativistic Israel-Stewart hydrodynamics
- **Arbitrary coordinate systems**: Minkowski, Milne, FLRW, Schwarzschild metrics
- **Robust covariant derivatives**: Support for tensors of arbitrary rank
- Performance monitoring for computational bottlenecks
- Comprehensive validation with physics error checking (90+ total test cases)
- **Complete Israel-Stewart implementation**: Production-ready second-order viscous hydrodynamics
- **Advanced numerical methods**: Implicit solvers, exponential integrators, stability analysis
- **Full second-order physics**: All coupling terms, nonlinear relaxation, thermodynamic constraints
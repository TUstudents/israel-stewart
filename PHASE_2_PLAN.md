# Phase 2: Extended Transport Coefficients Implementation

## Current Status Analysis

**✅ Phase 1 Complete:** Vectorized kinematic quantities implementation
- All kinematic methods (expansion, shear, vorticity, temperature gradient) working with proper covariant derivatives
- Physics validation passing (tracelessness, orthogonality, antisymmetry)
- Vectorized implementation with significant performance improvements

**✅ Core Infrastructure Complete:**
- Tensor framework, metrics, Christoffel symbols
- Conservation laws with full validation (31 test cases)
- Relaxation equations with all coupling terms (30+ test cases)
- Spectral solver (602 lines, production-ready)

**❌ Critical Missing Modules (Empty Files):**
- `equations/coefficients.py` (0 lines) - Transport coefficient calculation
- `equations/constraints.py` (0 lines) - Thermodynamic consistency
- `solvers/finite_difference.py` (0 lines) - Spatial discretization
- `solvers/implicit.py` (0 lines) - Implicit time integration
- `solvers/splitting.py` (0 lines) - Operator splitting
- All `benchmarks/*.py` (0 lines) - Physics validation benchmarks

## Phase 2 Implementation Plan

### Priority 1: Extended Transport Coefficients (`equations/coefficients.py`)

**Objective:** Implement physically realistic, temperature and density-dependent transport coefficients for Israel-Stewart hydrodynamics.

**Key Features:**
1. **Basic Viscosities:**
   - Shear viscosity η(T, ρ) with kinetic theory scaling
   - Bulk viscosity ζ(T, ρ) with QCD-inspired models
   - Thermal conductivity κ(T, ρ)

2. **Second-Order Coefficients:**
   - All λ coefficients (λ_ππ, λ_πΠ, λ_πq, etc.)
   - Relaxation times τ_π, τ_Π, τ_q with proper temperature scaling
   - Nonlinear coefficients ξ₁, ξ₂ for bulk pressure

3. **Physical Models:**
   - Kinetic theory expressions for dilute gas
   - QCD-inspired models for dense matter
   - Interpolation schemes for crossover regions
   - Speed of sound constraints and thermodynamic consistency

4. **Temperature/Density Dependencies:**
   - Arrhenius-type temperature scaling
   - Critical point behavior near phase transitions
   - Proper dimensional analysis and natural units

### Priority 2: Implicit Time Integration (`solvers/implicit.py`)

**Objective:** Implement robust implicit solvers specifically designed for stiff Israel-Stewart relaxation equations.

**Key Features:**
1. **Backward Euler Methods:**
   - Newton-Krylov solvers for nonlinear systems
   - Adaptive timestep control based on stiffness analysis
   - Preconditioning for Israel-Stewart operator structure

2. **IMEX (Implicit-Explicit) Schemes:**
   - Implicit treatment of stiff relaxation terms
   - Explicit treatment of hyperbolic transport
   - High-order IMEX-RK methods (2nd, 3rd, 4th order)

3. **Specialized IS Solvers:**
   - Exponential time differencing for relaxation terms
   - Block-structured preconditioning for coupled fluxes
   - Stability analysis and automatic timestep selection

### Priority 3: Bjorken Flow Benchmark (`benchmarks/bjorken_flow.py`)

**Objective:** Implement the canonical 1D boost-invariant expansion benchmark with exact solutions.

**Key Features:**
1. **Exact Solutions:**
   - Analytical Bjorken flow for ideal hydrodynamics
   - Israel-Stewart corrections for viscous case
   - Temperature evolution with proper thermodynamics

2. **Numerical Validation:**
   - Convergence testing for temporal discretization
   - Accuracy assessment vs analytical solutions
   - Performance benchmarking for different solvers

3. **Physics Testing:**
   - Second-order coefficient validation
   - Causality constraint verification
   - Relaxation time scale physics

## Implementation Strategy

1. **Start with `coefficients.py`** - This enables realistic physics in all subsequent tests
2. **Build `implicit.py`** - Essential for stable integration of stiff equations
3. **Validate with `bjorken_flow.py`** - Provides exact solutions for rigorous testing
4. **Extended testing** - Use benchmark to validate coefficient models and solver accuracy

## Expected Outcomes

- **Physics realism:** Temperature/density-dependent coefficients matching QCD/kinetic theory
- **Numerical stability:** Robust implicit solvers handling relaxation time stiffness
- **Rigorous validation:** Exact benchmark solutions ensuring implementation correctness
- **Performance optimization:** Efficient solvers enabling large-scale simulations

This phase will transform the framework from a proof-of-concept with constant coefficients to a production-ready simulation tool with physically realistic transport properties and robust numerical methods.

## Detailed Implementation Tasks

### Task 1: Transport Coefficients Module

**File:** `israel_stewart/equations/coefficients.py`

**Classes to implement:**
1. `TransportCoefficientCalculator` - Main coefficient computation engine
2. `KineticTheoryModel` - Dilute gas kinetic theory expressions
3. `QCDInspiredModel` - Dense matter QCD-based coefficients
4. `InterpolationModel` - Smooth interpolation between regimes

**Key methods:**
- `compute_shear_viscosity(T, rho, **kwargs)`
- `compute_bulk_viscosity(T, rho, **kwargs)`
- `compute_thermal_conductivity(T, rho, **kwargs)`
- `compute_relaxation_times(T, rho, **kwargs)`
- `compute_second_order_coefficients(T, rho, **kwargs)`

**Physics models:**
- Chapman-Enskog kinetic theory for η, ζ, κ
- QCD trace anomaly for bulk viscosity
- Critical point scaling near phase transitions
- Thermodynamic consistency constraints

### Task 2: Implicit Solver Module

**File:** `israel_stewart/solvers/implicit.py`

**Classes to implement:**
1. `ImplicitISolver` - Main implicit solver interface
2. `BackwardEulerSolver` - 1st order backward Euler
3. `IMEXRungeKuttaSolver` - High-order IMEX schemes
4. `ExponentialIntegrator` - ETD methods for relaxation

**Key methods:**
- `solve_implicit_step(fields, dt, method='newton_krylov')`
- `compute_jacobian(fields)` - Analytical/numerical Jacobian
- `precondition_system(matrix)` - IS-specific preconditioning
- `adaptive_timestep(fields, error_tol)` - Automatic timestep control

**Numerical methods:**
- Newton-Krylov for nonlinear systems
- GMRES/BiCGStab for linear solves
- Stiffness detection and adaptation
- Error estimation and timestep control

### Task 3: Bjorken Flow Benchmark

**File:** `israel_stewart/benchmarks/bjorken_flow.py`

**Classes to implement:**
1. `BjorkenFlowSolution` - Analytical solution container
2. `BjorkenBenchmark` - Numerical vs analytical comparison
3. `ConvergenceTest` - Temporal/spatial convergence analysis

**Key methods:**
- `analytical_solution(tau, transport_coeffs)` - Exact solutions
- `run_numerical_simulation(solver, grid, coeffs)` - Numerical evolution
- `compare_solutions(numerical, analytical)` - Error analysis
- `convergence_study(solvers, grid_refinements)` - Convergence rates

**Physics validation:**
- Energy-momentum conservation verification
- Causality constraint checking
- Relaxation time scale physics
- Second-order coefficient sensitivity analysis

## Success Criteria

✅ **Coefficients Module:**
- Temperature/density-dependent viscosities matching literature values
- All second-order coefficients computed consistently
- Thermodynamic constraints satisfied
- Comprehensive unit tests with physical limits

✅ **Implicit Solver:**
- Stable evolution for stiff relaxation equations
- Adaptive timestep achieving target accuracy
- Performance competitive with explicit methods
- Robust convergence for large CFL numbers

✅ **Bjorken Benchmark:**
- Numerical solutions matching analytical within 1% error
- Demonstrated convergence for all implemented solvers
- Physics validation for coefficient models
- Performance benchmarking and optimization

This comprehensive implementation will establish the Israel-Stewart framework as a robust, physically accurate, and computationally efficient tool for relativistic hydrodynamics simulations.
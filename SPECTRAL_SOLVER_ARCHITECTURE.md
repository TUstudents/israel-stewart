# SpectralISHydrodynamics Solver Architecture

## Overview: Dual-Mode Solver

The `SpectralISHydrodynamics` solver operates in **two distinct modes**:

1. **Mode A: 4D Spacetime Constraint Solver** - Refines analytical solutions across entire spacetime
2. **Mode B: Traditional Time Evolution** - Forward time-stepping from initial conditions

## Architecture Flux Diagram

```
╔══════════════════════════════════════════════════════════════════╗
║              SpectralISHydrodynamics Solver Architecture          ║
╚══════════════════════════════════════════════════════════════════╝

INPUT: Initial Conditions
├─ MODE A: 4D Spacetime (nt, nx, ny, nz) - all time slices initialized
└─ MODE B: 3D Snapshot (nx, ny, nz) at t=0 only

                           ↓

        ┌─────────────────────────────────────┐
        │   SpectralISHydrodynamics Class     │
        │   (spectral.py lines 1023-2061)     │
        └─────────────────────────────────────┘
                           ↓
        ┌─────────────────┴─────────────────┐
        │                                   │
    MODE A (Single Shot)            MODE B (Time Loop)
    ─────────────────              ──────────────────
    time_step(dt)                  evolve(t_final)
    lines 1135-1148                lines 2019-2057
        │                                   │
        ↓                                   ↓
    Refine 4D solution              while t < t_final:
    across spacetime                    dt = adaptive_time_step()
                                        time_step(dt)
                                        t += dt
                                                        │
    ┌───────────────────────────────────────────────────┘
    │
    ↓
╔═══════════════════════════════════════════════════════╗
║              time_step(dt, method)                     ║
║              Core Integration Engine                   ║
║              lines 1135-1148                           ║
╚═══════════════════════════════════════════════════════╝
    │
    ├─ method='split_step' (default) ───→ _split_step_advance(dt)
    │                                      lines 1150-1166
    │
    └─ method='spectral_imex' ───────────→ _spectral_imex_advance(dt)
                                           lines 1168-1240 (IMEX RK2)

╔═══════════════════════════════════════════════════════╗
║         _split_step_advance(dt)                        ║
║         Operator Splitting Method                      ║
║         (Strang Splitting for 2nd-order accuracy)      ║
╚═══════════════════════════════════════════════════════╝

    Step 1: Linear Terms (Spectral, dt/2)
    ══════════════════════════════════════
    spectral.advance_linear_terms(fields, dt/2)
    lines 763-795
        ↓
    ┌─────────────────────────────────────┐
    │  Viscous Diffusion (Fourier Space)  │
    │  • Bulk: Π → exp(-ζ∇² dt) Π       │
    │  • Shear: π^μν → exp(-η∇² dt) π   │
    │  • Heat flux damping                │
    │  [Exponential integrator for stiff  │
    │   diffusion terms]                  │
    └─────────────────────────────────────┘
        ↓

    Step 2: Conservation Laws (Real Space, dt)
    ═══════════════════════════════════════════
    _advance_conservation_laws(dt)
    lines 1412-1442
        ↓
    ┌─────────────────────────────────────┐
    │ ConservationLaws.evolution_equations│
    │  ∂_μ T^μν = 0                      │
    │  Returns: drho_dt, dmom_dt         │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ _rk2_conservation_step (lines 1480) │
    │  RK2 time integration:             │
    │  • Stage 1: k1 = f(t, y)           │
    │  • Stage 2: k2 = f(t+dt/2, y+k1/2) │
    │  • Update: y += dt * k2            │
    │                                     │
    │  CRITICAL CONVERSION:              │
    │  du^i/dt = (1/ρ)[d(ρu^i)/dt       │
    │              - u^i·dρ/dt]          │
    │  [Momentum → Velocity derivative]  │
    └─────────────────────────────────────┘
        ↓

    Step 3: Relaxation Terms (Real Space, dt)
    ══════════════════════════════════════════
    _advance_relaxation_terms(dt)
    lines 1588-1597
        ↓
    ┌─────────────────────────────────────┐
    │ ISRelaxationEquations.evolve_rel    │
    │  Israel-Stewart evolution:         │
    │  • ∂Π/∂t = -Π/τ_Π + sources        │
    │  • ∂π^μν/∂t = -π/τ_π + sources     │
    │  • ∂q^μ/∂t = -q/τ_q + sources      │
    │  [Full nonlinear IS equations]     │
    └─────────────────────────────────────┘
        ↓

    Step 4: Linear Terms (Spectral, dt/2)
    ══════════════════════════════════════
    spectral.advance_linear_terms(fields, dt/2)
    (Same as Step 1 - completes Strang splitting)

╔═══════════════════════════════════════════════════════╗
║      Alternative: _spectral_imex_advance(dt)          ║
║      IMEX Runge-Kutta ARS(2,2,2) Scheme               ║
║      lines 1168-1240                                   ║
║      [2nd-order, L-stable for stiff problems]         ║
╚═══════════════════════════════════════════════════════╝

    γ = 1 - 1/√2 ≈ 0.292893218

    Implicit Stage 1: Y₁ = y^n + h·γ·G(Y₁)
    ═══════════════════════════════════════════
    _solve_implicit_stage() via Newton-Krylov
    lines 1714-1790
        ↓
    G(Y) = Stiff terms:
        • Viscous diffusion: ζ∇²Π, η∇²π^μν
        • Linear relaxation: -Π/τ_Π, -π/τ_π
        ↓
    Solve: (I - γh·∂G/∂y)·Y₁ = y^n
    Using: newton_krylov with LGMRES
        ↓

    Explicit/Implicit Stage 2: Y₂ = y^n + h·F(Y₁) + h·(1-γ)·G(Y₁) + h·γ·G(Y₂)
    ═════════════════════════════════════════════════════════════════════════════
    F(Y) = Explicit nonlinear terms:
        • Conservation: ∂_μ T^μν = 0
        • Nonlinear IS sources
        ↓
    G(Y) = Stiff linear terms (as above)
        ↓

    Final Update: y^{n+1} = y^n + h/2·[F(Y₁)+F(Y₂)] + h·[(1-γ)·G(Y₁)+γ·G(Y₂)]

╔═══════════════════════════════════════════════════════╗
║           SpectralISolver Class                        ║
║           Low-Level Spectral Operations                ║
║           lines 32-1020                                ║
╚═══════════════════════════════════════════════════════╝

    Fourier Transform Operations
    ═══════════════════════════════
    • adaptive_fft/ifft - Automatic real/complex FFT selection
    • spatial_derivative(field, dir) - ∂_i f = IFFT(ik_i·FFT(f))
    • spatial_gradient(field) - ∇f = (∂_x, ∂_y, ∂_z) [optimized]
    • spatial_divergence(vector) - ∇·v computed in k-space
    • laplacian(field) - ∇²f = IFFT(-k²·FFT(f))

    Performance Features:
    ─────────────────────
    • Memory pooling for FFT workspaces
    • Cached k-vectors for different FFT shapes
    • Real FFT optimization (50% memory, 30% speedup)
    • Vectorized multi-direction derivatives

    Physics Operators
    ══════════════════
    • apply_viscous_operator - exp(-νk²dt) damping
    • apply_bulk_viscous_operator - Israel-Stewart Π evolution
    • spectral_convolution - Nonlinear terms via FFT with dealiasing

╔═══════════════════════════════════════════════════════╗
║              Physics Modules (External)                ║
╚═══════════════════════════════════════════════════════╝

    ConservationLaws                    ISRelaxationEquations
    (equations/conservation.py)         (equations/relaxation.py)
    ─────────────────────              ─────────────────────────
    • stress_energy_tensor()           • evolve_relaxation()
      T^μν = (ρ+p)u^μu^ν + pg^μν       • _bulk_rhs() - Π sources
             + π^μν + q^μu^ν + ...     • _shear_rhs() - π^μν sources
                                        • _heat_flux_rhs() - q^μ sources
    • divergence_T()
      ∂_μ T^μν using covariant          With full nonlinear couplings:
      derivatives + Christoffel         λ_ππ, λ_πΠ, λ_πq, ξ₁, ξ₂

    • evolution_equations()
      Returns: {drho_dt, dmom_dt}      Returns: {dPi_dt, dpi_dt, dq_dt}

╔═══════════════════════════════════════════════════════╗
║              Adaptive Time Stepping                    ║
║              lines 1984-2017                           ║
╚═══════════════════════════════════════════════════════╝

    CFL Condition (Advection):
    ──────────────────────────
    dt_CFL = c · min(dx, dy, dz) / max(|u|)
    where c = 0.5 (CFL factor)

    Viscous Constraint (Diffusion):
    ────────────────────────────────
    dt_visc = 0.5 · min(dx², dy², dz²) / η
    [Parabolic CFL for diffusion]

    Relaxation Constraint (Israel-Stewart):
    ────────────────────────────────────────
    dt_relax = 0.1 · min(τ_π, τ_Π, τ_q)
    [Resolve relaxation timescales]

    Final: dt = min(dt_CFL, dt_visc, dt_relax, max_dt)

╔═══════════════════════════════════════════════════════╗
║                  OUTPUT                                ║
╚═══════════════════════════════════════════════════════╝

    MODE A (4D Spacetime Constraint Solver):
    ────────────────────────────────────────
    • Input: Entire 4D spacetime initialized with analytical solution
    • Operation: One call to time_step(dt) refines via spectral projection
    • Output: Refined 4D solution with ∂_μ T^μν ≈ 0 enforced
    • Use Case: Verification against known solutions, spectral accuracy tests

    MODE B (Time Evolution Solver):
    ────────────────────────────────
    • Input: 3D initial conditions at t=0 only
    • Operation: evolve(t_final) loops time_step(dt) with adaptive dt
    • Output: Time series fields(t=0, dt, 2dt, ..., t_final)
    • Use Case: Physics simulations, dynamics studies, initial value problems
    • Features: Adaptive timestep, optional output_callback at each step
```

## Key Implementation Details

### Conservation Law Integration (Critical)

The momentum-velocity conversion (lines 1444-1478) is essential for correct physics:

```python
# Conservation laws give: d(ρu^i)/dt (momentum density)
# But we evolve: u^i (velocity)
# Product rule: d(ρu^i)/dt = ρ·du^i/dt + u^i·dρ/dt
# Solve for: du^i/dt = (1/ρ)[d(ρu^i)/dt - u^i·dρ/dt]
```

Without this conversion, sound waves propagate at incorrect speeds.

### Operator Splitting Accuracy

Strang splitting provides 2nd-order accuracy:
```
Step 1: L(dt/2)     [Linear operators]
Step 2: N(dt)       [Nonlinear operators]
Step 3: L(dt/2)     [Linear operators]
```
This is more accurate than Lie splitting: L(dt) → N(dt)

### IMEX Method Advantages

The ARS(2,2,2) IMEX-RK scheme:
- L-stable (good for stiff relaxation)
- 2nd-order accurate
- Implicit treatment of stiff diffusion/relaxation
- Explicit treatment of nonlinear advection
- Better than split-step for very stiff problems (small τ_π, τ_Π)

## Performance Characteristics

### Spectral Method Advantages
- Spectral accuracy: exp(-cN) error convergence for smooth solutions
- Efficient FFT: O(N log N) for derivatives vs O(N²) finite difference
- Exact differentiation in Fourier space: no numerical dispersion
- Natural periodic boundaries

### Computational Costs
- FFT: ~3 transforms per spatial derivative
- Split-step: ~10-15 FFTs per timestep
- IMEX: ~20-30 FFTs per timestep (implicit solve overhead)
- Memory: O(N) with workspace reuse, O(N log N) without

### Scalability
- Grid size: Currently tested up to (100, 32, 32, 32) = 3.2M points
- Memory optimization: Array pooling, FFT workspace caching
- Real FFT: 50% memory reduction for real fields

## Physics Validation Status

### Currently Tested (Mode A Benchmarks)
✅ Spectral derivative accuracy
✅ Conservation law enforcement (∂_μ T^μν ≈ 0)
✅ Multi-mode superposition (linearity)
✅ Viscous term inclusion
✅ Spatial convergence (spectral accuracy)

### Not Yet Tested (Mode B Required)
❌ Actual wave propagation dynamics
❌ Viscous damping rates over time
❌ Long-time energy conservation
❌ Nonlinear steepening and shock formation
❌ Israel-Stewart relaxation to Navier-Stokes limit
❌ Initial value problem evolution accuracy

## Recommendations

### For Verification Work (Current Benchmarks)
Use **Mode A** with analytical solutions to test:
- Spatial discretization accuracy
- Conservation law satisfaction
- Spectral convergence rates

### For Physics Simulations
Use **Mode B** with `evolve(t_final)` to simulate:
- Bjorken flow expansion
- Gubser flow dynamics
- Sound wave propagation
- Shock tube problems
- Heavy-ion collision evolution

### Next Steps
Implement **Mode B benchmarks** using `evolve()`:
1. Sound wave propagation with actual time evolution
2. Viscous damping rate measurements
3. Long-time stability tests (100+ periods)
4. Initial value problem with known analytic solution (e.g., Gubser flow)
5. Shock tube test (Riemann problem)

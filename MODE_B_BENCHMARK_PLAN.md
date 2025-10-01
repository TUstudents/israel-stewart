# Mode B Benchmark Plan: Time Evolution Validation

## Overview

Mode B benchmarks test the **time evolution capabilities** of `SpectralISHydrodynamics` using the `evolve(t_final)` method. These validate actual hydrodynamic dynamics, not just spectral operator correctness.

## Benchmark Suite Design

### Test 1: Sound Wave Propagation (Time Evolution)

**Physics**: Track wave packet moving through domain

**Setup**:
```python
# Initial condition at t=0 only
grid = SpacetimeGrid(
    time_range=(0.0, 10.0),  # Long enough for multiple periods
    spatial_ranges=[(0.0, 2*π)]*3,
    grid_points=(1, 64, 64, 64),  # Single time slice initially
    boundary_conditions='periodic'
)

# Gaussian wave packet
rho_0 = 1.0
amplitude = 0.01
x0, y0, z0 = π, π, π  # Center
sigma = 0.5

X, Y, Z = grid.spatial_meshgrid()
r_squared = (X-x0)**2 + (Y-y0)**2 + (Z-z0)**2
rho_initial = rho_0 + amplitude * np.exp(-r_squared/(2*sigma**2))

# Add initial velocity (x-direction propagation)
c_s = np.sqrt(1/3)
u_x_initial = (c_s/rho_0) * amplitude * np.exp(-r_squared/(2*sigma**2))

# Evolve forward in time
hydro.evolve(t_final=10.0, output_callback=save_snapshots)
```

**Validation**:
1. **Wave Speed**: Measure distance traveled / time → compare to c_s = 1/√3
   - Expected: v_measured ≈ 0.577 ± 1%
2. **Amplitude Preservation**: Track peak height over time
   - Ideal fluid: amplitude constant within 2%
3. **Dispersion**: Measure wave packet width σ(t)
   - Good numerics: σ(t) ≈ σ(0) (no artificial broadening)
4. **Phase Coherence**: FFT analysis should show single k-mode

**Success Criteria**:
- Wave speed error < 1%
- Amplitude decay < 2% (ideal fluid)
- Dispersion < 5% after 10 crossing times

---

### Test 2: Viscous Damping Dynamics

**Physics**: Exponential amplitude decay with viscosity

**Analytical Solution**:
```
ρ(t,x) = ρ₀ + A(t)·sin(kx)
A(t) = A₀ · exp(-γt)
γ = (η + ζ)k²/(2ρ₀)  [Navier-Stokes damping rate]
```

**Setup**:
```python
# Single-mode standing wave
k = 1.0
amplitude = 0.05  # Larger for better SNR
eta = 0.1  # Shear viscosity
zeta = 0.05  # Bulk viscosity

coeffs = TransportCoefficients(
    shear_viscosity=eta,
    bulk_viscosity=zeta,
    shear_relaxation_time=0.5,  # Test Israel-Stewart
    bulk_relaxation_time=0.3
)

# Initial: standing wave
rho_initial = rho_0 + amplitude * np.sin(k*X)

# Evolve for ~5 damping times
t_damp = 1.0 / gamma
hydro.evolve(t_final=5*t_damp, output_callback=measure_amplitude)
```

**Validation**:
1. **Damping Rate**: Fit A(t) = A₀ exp(-γt) to time series
   - Extract γ_numerical from exponential fit
   - Compare to γ_analytical = (η+ζ)k²/(2ρ₀)
   - Relative error < 10%

2. **Israel-Stewart vs Navier-Stokes**:
   - Run with τ_π = 0 (Navier-Stokes limit)
   - Run with τ_π > 0 (Israel-Stewart)
   - Verify IS has slower initial decay (relaxation delay)

3. **Wavenumber Dependence**:
   - Test k = [0.5, 1.0, 2.0]
   - Verify γ ∝ k² scaling

**Success Criteria**:
- Damping rate error < 10% for each k
- k² scaling verified: R² > 0.99
- Israel-Stewart delay observed (τ_π signature)

---

### Test 3: Long-Time Energy Conservation

**Physics**: Monitor total energy over extended simulation

**Setup**:
```python
# Complex multi-mode initial condition
rho_initial = rho_0
for k, A in zip([0.5, 1.0, 1.5], [0.01, 0.008, 0.005]):
    rho_initial += A * np.sin(k*X) * np.cos(k*Y)

# Ideal fluid (no dissipation)
coeffs = TransportCoefficients(
    shear_viscosity=0.0,
    bulk_viscosity=0.0
)

# Long evolution: 100 wave periods
T_period = 2*π / (c_s * k_min)
t_final = 100 * T_period

# Monitor energy at each step
energies = []
def track_energy(t, step, fields):
    E = np.sum(fields.rho) * volume_element
    energies.append((t, E))

hydro.evolve(t_final=t_final, output_callback=track_energy)
```

**Validation**:
1. **Energy Drift**: ΔE/E₀ over entire run
   - Expected: < 1% for 100 periods
   - Plot E(t) to check for systematic drift

2. **Poincaré Recurrence**: Should see near-periodic return
   - For integrable system with few modes
   - Check ||ρ(t=T) - ρ(t=0)|| at recurrence time T

3. **Spectral Mode Preservation**:
   - FFT at t=0 and t=t_final
   - Verify same modes present (no spurious coupling)

**Success Criteria**:
- Energy drift < 1% over 100 periods
- No exponential instabilities (max|ρ| bounded)
- Mode structure preserved in FFT

---

### Test 4: Bjorken Flow Evolution

**Physics**: 1D boost-invariant expansion (exact solution available)

**Analytical Solution**:
```
τ = √(t² - z²)  [Proper time]
T(τ) ∝ τ^(-1/3)  [Temperature scaling]
ρ(τ) ∝ τ^(-4/3)  [Energy density]
```

**Setup**:
```python
# Use Milne coordinates for Bjorken flow
from israel_stewart.core.metrics import MilneMetric

grid = SpacetimeGrid(
    coordinate_system='milne',
    time_range=(0.1, 10.0),  # τ_0 to τ_max
    spatial_ranges=[(-5, 5), (0, 2*π), (0, 2*π)],  # η_s, x, y
    grid_points=(100, 1, 16, 16),
    boundary_conditions='mixed'  # Open in η_s, periodic in x,y
)

# Initial condition at τ=0.1 fm/c
T_0 = 0.3  # GeV (QGP temperature)
rho_initial = temperature_to_energy_density(T_0)

# Evolve with viscosity
eta_over_s = 0.2  # η/s ratio
coeffs = TransportCoefficients.from_eta_over_s(
    eta_over_s=eta_over_s,
    temperature=T_0
)

hydro.evolve(t_final=10.0)
```

**Validation**:
1. **Temperature Scaling**: T(τ) vs analytical
   - Ideal: T ∝ τ^(-1/3)
   - Viscous: Slower cooling (viscous heating)
   - Measure exponent α from T ∝ τ^(-α)

2. **Viscous Corrections**:
   - Measure π^ηη (longitudinal pressure)
   - Compare to analytical π^ηη ≈ 4η/(3τ)

3. **Equation of State**:
   - Verify P = ρ/3 maintained (conformal)
   - Check thermodynamic consistency

**Success Criteria**:
- Ideal: α = 1/3 ± 2%
- Viscous: α < 1/3 (viscous heating signature)
- π^ηη matches analytical within 15%

---

### Test 5: Shock Tube (Riemann Problem)

**Physics**: Nonlinear wave steepening and shock formation

**Setup**:
```python
# Sod shock tube initial conditions
# Left state:  ρ_L = 1.0, P_L = 1.0, u_L = 0
# Right state: ρ_R = 0.125, P_R = 0.1, u_R = 0
# Interface at x = 0.5

grid = SpacetimeGrid(
    time_range=(0.0, 0.2),
    spatial_ranges=[(0.0, 1.0), (0.0, 0.1), (0.0, 0.1)],
    grid_points=(1, 512, 4, 4),  # High resolution in x
    boundary_conditions='outflow'  # Need outflow boundaries
)

# Discontinuous initial condition
x = grid.coordinates['x']
rho_initial = np.where(x < 0.5, 1.0, 0.125)
P_initial = np.where(x < 0.5, 1.0, 0.1)

# Small viscosity for shock capturing
coeffs = TransportCoefficients(
    shear_viscosity=0.01,  # Artificial viscosity
    bulk_viscosity=0.01
)

hydro.evolve(t_final=0.2)
```

**Validation**:
1. **Wave Structure**: Identify 3 waves
   - Rarefaction fan (left)
   - Contact discontinuity (middle)
   - Shock (right)

2. **Wave Speeds**: Measure from x-t diagram
   - Compare to exact Riemann solver solution
   - Shock speed vs Rankine-Hugoniot conditions

3. **Entropy Production**: Monitor at shock
   - Physical: entropy increase across shock
   - Check conservation of conserved quantities

**Success Criteria**:
- Wave structure qualitatively correct
- Shock speed within 5% of exact
- No spurious oscillations (spectral ringing controlled)

---

## Implementation Plan

### File: `benchmarks/run_spectral_mode_b_benchmark.py`

```python
class SpectralModeBBenchmark:
    """Time evolution benchmarks using evolve() method."""

    def test_wave_propagation(self) -> BenchmarkResult:
        """Test 1: Sound wave with actual time evolution"""

    def test_viscous_damping_dynamics(self) -> BenchmarkResult:
        """Test 2: Measure damping rates over time"""

    def test_long_time_conservation(self) -> BenchmarkResult:
        """Test 3: Energy drift over 100 periods"""

    def test_bjorken_flow(self) -> BenchmarkResult:
        """Test 4: 1D boost-invariant expansion"""

    def test_shock_tube(self) -> BenchmarkResult:
        """Test 5: Riemann problem with shock formation"""
```

### Execution Strategy

1. **Start with Test 1-3** (simpler, well-defined analytics)
   - Validate basic time evolution
   - Check energy conservation
   - Measure viscous damping

2. **Add Test 4** (requires Milne coordinates)
   - Tests realistic heavy-ion physics
   - Validates metric integration

3. **Add Test 5** (most challenging)
   - Requires outflow boundaries (not fully implemented)
   - Tests shock capturing ability
   - May expose spectral ringing issues

### Success Criteria

**Minimum Viable**:
- Tests 1-3 passing with < 10% error
- Documents actual time evolution capability

**Full Validation**:
- All 5 tests passing
- Demonstrates solver ready for production physics

### Estimated Time

- Test 1 (Wave propagation): 2 hours
- Test 2 (Viscous damping): 2 hours
- Test 3 (Energy conservation): 1 hour
- Test 4 (Bjorken flow): 3 hours (metric complexity)
- Test 5 (Shock tube): 4 hours (boundary conditions)

**Total**: ~12 hours for complete Mode B benchmark suite

## Differences from Mode A Benchmarks

| Aspect | Mode A (Current) | Mode B (Planned) |
|--------|------------------|------------------|
| **Initialization** | Entire 4D spacetime | Only t=0 |
| **Evolution** | One `time_step()` call | Loop via `evolve()` |
| **Time** | All time slices present | Forward stepping |
| **Tests** | Spectral operators | Physics dynamics |
| **Validation** | Conservation at t=0 | Evolution over time |
| **Metrics** | Spatial accuracy | Temporal accuracy |

## Open Questions

1. **Grid Initialization**: Does `SpacetimeGrid` with (nt, nx, ny, nz) auto-grow the time dimension during `evolve()`?
   - Need to check if grid is fixed or dynamic
   - May need different initialization for Mode B

2. **Boundary Conditions**: Current tests use periodic
   - Shock tube needs outflow boundaries
   - May require new boundary condition implementation

3. **Output Storage**: `evolve()` has `output_callback`
   - How to efficiently store time series?
   - Need IO utilities for trajectory data

4. **Metric Integration**: Bjorken test needs `MilneMetric`
   - Is metric switching fully working?
   - Need to validate coordinate transformations

## Next Steps

1. **Investigate grid behavior** in Mode B
2. **Implement Test 1** (wave propagation) as prototype
3. **Run and validate** against analytical solution
4. **Iterate** on remaining tests based on Test 1 learnings

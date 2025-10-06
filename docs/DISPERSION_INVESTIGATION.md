# Israel-Stewart Dispersion Relation Damping Investigation

## Problem Statement

**Observed Discrepancy**:
- Analytical dispersion relation predicts: **γ_analytical = 0.509634** /time
- Numerical simulation measures: **γ_measured ≈ -0.025** /time (negative = mode growth!)

**Test Configuration**:
- Wave number: k = 8.0
- Transport coefficients: η=0.08, ζ=0.04, τ_π=0.5, τ_Π=0.3
- Grid: (32, 32, 16), domain size: 2π
- Method: split_step
- Simulation time: 3 wave periods

**Key Finding**: The Fourier mode amplitude |ρ_k(t)| is **growing** instead of decaying, indicating either:
1. Error in analytical dispersion relation formula
2. Error in numerical implementation of IS equations
3. Incorrect transport coefficient values causing instability

---

## Investigation Phases

### Phase 1: Documentation ✅ (COMPLETED)
Create comprehensive reference materials for debugging:
- [x] `IS_DISPERSION_FLOWCHART.md` - Complete code flow diagram
- [x] `IS_EQUATIONS_REFERENCE.md` - All equations with implementation details
- [x] `DAMPING_DEBUG_GUIDE.md` - Quick lookup guide

### Phase 2: Analytical Dispersion Validation

**Goal**: Verify the analytical dispersion relation is correct

#### 2.1 Dispersion Matrix Construction
File: `israel_stewart/benchmarks/sound_waves.py:374-433`

Verify the 4×4 linearized IS matrix for variables `[δε, δv_x, δΠ, δπ_xx]`:

**Row 0: Energy Conservation**
```python
matrix[0, 0] = -1j * omega          # ∂_t ε term
matrix[0, 1] = 1j * k * enthalpy    # ∂_x v_x term
```
**Check**: Does this match ∂_t ε + ∂_x[(ε+p)v_x] = 0? ✓

**Row 1: Momentum Conservation**
```python
matrix[1, 0] = 1j * k * cs_squared  # ∂_x ε term (pressure force)
matrix[1, 1] = -1j * omega * enthalpy  # ∂_t v_x term
matrix[1, 2] = 1j * k               # ∂_x Π term (bulk viscous force)
matrix[1, 3] = 1j * k               # ∂_x π_xx term (shear viscous force)
```
**Check**: Does this match (ε+p)∂_t v_x + c_s²∂_x ε + ∂_x Π + ∂_x π_xx = 0? ✓

**Row 2: Bulk Relaxation**
```python
matrix[2, 1] = 1j * zeta * k        # Source: ζ·k·v_x
matrix[2, 2] = 1.0 - 1j * omega * tau_Pi  # Linear: (1 - iωτ_Π)
```
**CRITICAL CHECK**: Israel-Stewart bulk equation is:
```
τ_Π DΠ/Dτ + Π = -ζθ
```
For plane wave: `DΠ/Dτ = -iω Π`, `θ = ∇·u = ikv_x`
```
-iωτ_Π Π + Π = -ζ(ikv_x)
Π(1 - iωτ_Π) = -ζ(ikv_x)
```
**Question**: Should source term be `+iζkv_x` or `-iζkv_x`?

**Row 3: Shear Relaxation**
```python
matrix[3, 1] = 1j * (4.0 / 3.0) * eta * k  # Source: (4/3)η·k·v_x
matrix[3, 3] = 1.0 - 1j * omega * tau_pi  # Linear: (1 - iωτ_π)
```
**CRITICAL CHECK**: Israel-Stewart shear equation is:
```
τ_π Dπ^μν/Dτ + π^μν = 2ησ^μν
```
For longitudinal wave: `σ_xx = (2/3)∂_x v_x = (2/3)ikv_x`
```
-iωτ_π π_xx + π_xx = 2η(2/3)(ikv_x)
π_xx(1 - iωτ_π) = (4/3)η(ikv_x)
```
**Question**: Factor of (4/3) is correct ✓

#### 2.2 Root Finding
File: `israel_stewart/benchmarks/sound_waves.py:293-372`

**Initial Guesses** (line 333-338):
```python
cs_estimate * k                        # Sound mode frequency
-gamma_viscous                         # Damping (capped at 30% of ω)
```

**Issue**: Damping is capped to avoid numerical issues, but this might bias root finding.

**Determinant Check** (line 362):
```python
det_check = abs(self._determinant_function(omega, k))
if det_check < 1e-8:  # Accept as solution
```

**Action Items**:
1. [ ] Print determinant matrix for k=8.0
2. [ ] Verify det(M) for known roots
3. [ ] Check if capping damping excludes correct solutions
4. [ ] Compare with analytical formula from literature

#### 2.3 Literature Comparison

Israel-Stewart dispersion relation (Kovtun 2019, Grozdanov 2019):
```
ω = c_s k - i(ζ + 4η/3)k²/(ε+p) × [1/(1 + (ωτ)²)] + O(k³)
```

For our parameters (ε=1, p=1/3, η=0.08, ζ=0.04):
```
enthalpy = 4/3
γ_NS = (ζ + 4η/3)k²/(ε+p) = (0.04 + 0.107)×64/1.33 = 7.05
```

But IS has relaxation suppression factor!

**Action Items**:
1. [ ] Derive exact IS dispersion relation analytically
2. [ ] Compare with numerical root finding
3. [ ] Check if relaxation times cause instability for these parameters

---

### Phase 3: Numerical Implementation Validation

#### 3.1 Split-Step Method
File: `israel_stewart/solvers/spectral.py:1136-1152`

```python
def _split_step_advance(self, dt):
    # Step 1: Linear diffusion (spectral)
    self.spectral.advance_linear_terms(self.fields, dt/2)

    # Step 2: Conservation laws
    self._advance_conservation_laws(dt)

    # Step 3: Relaxation terms
    self._advance_relaxation_terms(dt)

    # Step 4: Linear diffusion (spectral)
    self.spectral.advance_linear_terms(self.fields, dt/2)
```

**Question**: What is in `advance_linear_terms()`? Should it include `-Π/τ` or not?

**Relaxation Evolution** (line 1861-1870):
```python
def _advance_relaxation_terms(self, dt):
    self.relaxation.evolve_relaxation(self.fields, dt)
```

**Critical Check** - `relaxation.evolve_relaxation()` (relaxation.py:724-759):
Default method is `"explicit"`:
```python
def _explicit_evolution(self, fields, dt):
    rhs = self.compute_relaxation_rhs(fields)
    fields.from_dissipative_vector(
        fields.to_dissipative_vector() + dt * rhs
    )
```

**Check `_bulk_rhs()`** (relaxation.py:200-234):
```python
linear = -Pi / self.coeffs.bulk_relaxation_time
first_order = -self.coeffs.bulk_viscosity * theta
```

**Sign Check**:
- Linear term: `-Π/τ_Π` ✓ (correct damping)
- Source term: `-ζθ` where `θ = ∇·u`

**For sound wave**: `θ = ik·δv_x` (positive for compression)
- If `δv_x > 0` (rightward velocity), then `θ > 0` (expansion)
- Bulk pressure source: `-ζθ < 0` (reduces Π for expansion)

**Question**: Is sign of θ correct in spectral derivatives?

#### 3.2 Expansion Scalar Computation
File: `israel_stewart/solvers/spectral.py:1872-1895`

```python
def _compute_expansion_scalar(self):
    velocity_spatial = u_mu[..., 1:4]  # u^1, u^2, u^3
    theta = self.spectral.spatial_divergence(velocity_spatial)
    return theta
```

**Critical**: What does `spatial_divergence()` return for spectral method?

**Action Items**:
1. [ ] Verify `spatial_divergence()` in spectral space
2. [ ] Check if FFT derivatives have correct sign
3. [ ] Test with simple divergence-free field (should give θ=0)

#### 3.3 IMEX Method
File: `israel_stewart/solvers/spectral.py:1154-1269`

**Implicit term G(y)** (line 1658-1663):
```python
stiff_terms["Pi"] = -fields.Pi / self.coeffs.bulk_relaxation_time
```
This is **correct**: G(Π) = -Π/τ_Π

**Explicit term F(y)** should contain: `-ζθ + nonlinear`

**Check** `_compute_relaxation_sources()` (line 1310-1315):
```python
if self._integration_mode == "spectral_imex" and self.coeffs is not None:
    if getattr(self.coeffs, "bulk_relaxation_time", None):
        dPi_dt += self.fields.Pi / self.coeffs.bulk_relaxation_time  # ADD back!
```

**Logic**:
1. `compute_relaxation_rhs()` returns: `-Π/τ - ζθ + nonlinear`
2. In IMEX mode, we **add back** `+Π/τ` to get F(y) = `-ζθ + nonlinear`
3. Implicit solve handles G(y) = `-Π/τ` separately

**Verification**:
- Full RHS = F(y) + G(y) = `(-ζθ + nonlinear) + (-Π/τ)` ✓

---

### Phase 4: Transport Coefficient Analysis

#### 4.1 Stability Analysis

For k=8.0, current coefficients:
- η = 0.08, ζ = 0.04
- τ_π = 0.5, τ_Π = 0.3
- ω_sound ≈ 6.0

**Dimensionless parameters**:
- ωτ_Π = 6.0 × 0.3 = 1.8
- ωτ_π = 6.0 × 0.5 = 3.0

**Relaxation suppression factor**: `1/(1 + (ωτ)²)`
- Bulk: 1/(1 + 1.8²) = 1/4.24 = 0.236
- Shear: 1/(1 + 3.0²) = 1/10 = 0.1

**Effective damping**:
```
γ_eff ≈ (ζ + 4η/3) × k² × suppression / enthalpy
γ_bulk = 0.04 × 64 × 0.236 / 1.33 = 0.45
γ_shear = 0.107 × 64 × 0.1 / 1.33 = 0.51
γ_total ≈ 0.51  (matches analytical!)
```

**But simulation shows growth!** This suggests numerical error, not analytical error.

#### 4.2 Test Cases

**Test 1: Navier-Stokes Limit** (τ → 0)
- Set τ_Π = 0.01, τ_π = 0.01
- Expected: Strong damping, no relaxation effects

**Test 2: Inviscid Limit** (η, ζ → 0)
- Set η = 0, ζ = 0
- Expected: No damping, pure oscillation

**Test 3: Large Viscosity**
- Set η = 0.5, ζ = 0.3
- Expected: Strong damping γ > 1.0

---

### Phase 5: Sign Convention Audit

#### Critical Sign Checks

| Location | Equation | Implementation | Status |
|----------|----------|----------------|--------|
| Dispersion matrix row 2 | `Π(1-iωτ_Π) = -ζ(ikv_x)` | `matrix[2,2] = 1-iωτ_Π` | ✓ |
| Dispersion matrix row 3 | `π(1-iωτ_π) = (4/3)η(ikv_x)` | `matrix[3,3] = 1-iωτ_π` | ✓ |
| Bulk RHS linear | `dΠ/dt = -Π/τ_Π + ...` | `linear = -Pi/tau_Pi` | ✓ |
| Bulk RHS source | `... -ζθ` | `first_order = -zeta*theta` | ? |
| IMEX implicit G(y) | `G_Π = -Π/τ_Π` | `stiff = -Pi/tau_Pi` | ✓ |
| IMEX explicit F(y) | Should NOT have `-Π/τ` | Adds back `+Π/τ` | ✓ |
| Expansion scalar | `θ = ∇·u` | `div(velocity_spatial)` | ? |

**Most Suspicious**: Expansion scalar sign in FFT derivatives

---

## Debugging Checklist

### Quick Diagnostics

**1. Print dispersion matrix** (k=8.0):
```python
benchmark = NumericalSoundWaveBenchmark(...)
analytical = benchmark.analytical
matrix = analytical._build_dispersion_matrix(
    complex(6.0, -0.5),  # Trial omega
    np.array([8.0, 0.0, 0.0])
)
print("Dispersion matrix:\n", matrix)
print("Determinant:", np.linalg.det(matrix))
```

**2. Verify expansion scalar**:
```python
# Initialize with v_x = sin(kx)
fields.u_mu[..., 1] = 0.01 * np.sin(8.0 * X)
theta = solver._compute_expansion_scalar()
# Expected: theta = 0.01 * 8.0 * cos(8.0*X) at x=0: theta(0) = 0.08
```

**3. Test relaxation RHS**:
```python
fields.Pi[:] = 0.1
fields.u_mu[..., 1] = 0.01 * np.sin(8.0 * X)
theta = solver._compute_expansion_scalar()
dPi_dt = relaxation._bulk_rhs(fields.Pi, fields.pi_munu, theta)
# Expected: dPi_dt = -0.1/0.3 + (-0.04)*theta = -0.333 + ...
```

**4. Monitor Fourier mode**:
```python
# Track |ρ_k| directly in simulation
k_index = 8
for t in time_points:
    rho_k = np.fft.fftn(fields.rho - 1.0)[k_index, 0, 0]
    print(f"t={t:.3f}, |ρ_k|={abs(rho_k):.6f}")
# Should decay exponentially: |ρ_k| = A₀ exp(-γt)
```

---

## Expected Findings

### Hypothesis 1: FFT Sign Error
**Symptom**: Expansion scalar has wrong sign
**Test**: θ should be positive for v_x = sin(kx) with k>0
**Fix**: Negate spectral derivative or change source term sign

### Hypothesis 2: Relaxation Time Error
**Symptom**: Instability at ωτ > 1
**Test**: Reduce τ_Π, τ_π by 10×
**Fix**: Use smaller relaxation times or different initial conditions

### Hypothesis 3: Initial Condition Mismatch
**Symptom**: Transients dominate short simulations
**Test**: Run for 20 wave periods instead of 3
**Fix**: Use eigenmode initialization (already done)

### Hypothesis 4: IMEX Source Term Bug
**Symptom**: Double-counting or missing terms in F(y)
**Test**: Compare split_step vs spectral_imex methods
**Fix**: Verify `_integration_mode` logic

---

## Investigation Results (2025-10-06)

### Phase 1: Timestep Fix ✅ COMPLETED

**Problem Found**: CFL timestep calculation only considered wave propagation, not relaxation stability.

**Fix Applied** (`sound_waves.py:1370-1381`):
```python
dt_cfl_wave = dt_factor * dx / max(sound_speed, 0.1)
dt_cfl_relax = 0.01 * min(tau_Pi, tau_pi)
dt_cfl = min(dt_cfl_wave, dt_cfl_relax)
```

**Result**: Simulation is now **stable** (γ > 0 instead of negative growth).

### Phase 2: Damping Measurement Issue ⚠️ IN PROGRESS

**New Problem**: Even with stable timestep (dt=0.003), measured damping is **5× too small**:
- Analytical: γ = 0.510 /time
- Measured: γ = 0.104 /time (from mode energy)
- Error: ~80%

**Key Findings from Debug Analysis**:

1. **Fourier mode amplitudes oscillate instead of decaying monotonically**
   - |ρ_k(t)| oscillates with period ~1.0 (sound wave period)
   - |v_k(t)| oscillates out of phase with |ρ_k(t)|
   - Energy exchange between kinetic and potential forms

2. **Total mode energy also oscillates**
   - Even |ρ_k|² + (4/3)|v_k|² shows ~20% oscillations
   - Energy initially **increases** from 10000 to 13000 (t=0 to t=0.5)
   - Then decays slowly with oscillations

3. **Energy growth indicates initialization problem**
   - If initial conditions matched true eigenmode, energy would decay monotonically
   - Growth suggests mismatch between initialized state and true IS eigenmode

**Hypothesis**: The eigenmode initialization in `setup_initial_conditions()` doesn't correctly account for the Israel-Stewart dissipative flux ratios, causing transient energy injection and slow equilibration to the true eigenmode.

**Next Steps**:
1. ✅ Verified eigenmode ratios are correctly extracted (v_x=0.564, Π=0.079, π_xx=0.148)
2. ✅ Found SVD fails for ill-conditioned matrices - switched to eigenvalue decomposition
3. ✅ Tested longer simulations - damping does NOT converge (jumps between ±0.5)

### Phase 3: Root Cause Analysis ✅ COMPLETED

**Findings**:

1. **SVD numerical instability** (`sound_waves.py:1184`):
   - Condition number: 2.4×10^17 (at limit of double precision)
   - SVD residual: ||M·v|| = 1.12 (100% error!)
   - Eigenvalue decomposition residual: ||M·v|| = 8×10^-15 (excellent)
   - **Fix applied**: Replaced SVD with `np.linalg.eig()`

2. **Eigenmode ratios verified correct**:
   - Eigenvector satisfies M·v ≈ 0 within machine precision
   - Ratios: v_x/ρ = 0.564, Π/ρ = 0.079, π_xx/ρ = 0.148
   - Match theoretical IS dispersion relation

3. **Persistent initialization problem**:
   - Energy initially increases (10000 → 13000 at t=0→0.5)
   - Mode energy oscillates ~20% instead of smooth decay
   - Damping measurements wildly inconsistent: t=1→γ=+0.64, t=2→γ=-0.66, t=4→γ=+0.16
   - Does NOT converge with longer simulation time

**Hypothesis**: The issue is NOT in eigenmode extraction but in how the eigenvector components are interpreted and applied to initialize the fields. The eigenvector gives δε, δv_x, δΠ, δπ_xx in a specific basis that may not match how fields are defined in the code.

## Next Steps

1. **Execute Phase 2**: Print matrices, verify determinants
2. **Execute Phase 3**: Test expansion scalar, relaxation RHS
3. **Execute Phase 4**: Try different transport coefficients
4. **Execute Phase 5**: Complete sign audit

**Priority**: Focus on expansion scalar and relaxation source terms first.

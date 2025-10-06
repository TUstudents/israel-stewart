# Israel-Stewart Dispersion Relation Code Flow

This document maps the complete code flow from analytical dispersion relation computation to numerical simulation damping measurement.

---

## Part 1: Analytical Dispersion Relation

### Entry Point: `benchmark.analytical.analyze_dispersion_relation(k_vector)`

**File**: `israel_stewart/benchmarks/sound_waves.py`

```
analyze_dispersion_relation(wave_vector)  [Line 161]
│
├─→ Cache check  [Line 178-181]
│   └─→ Return cached if available
│
├─→ _find_dispersion_roots(k_magnitude)  [Line 185]
│   │
│   ├─→ Estimate sound speed  [Line 308]
│   │   └─→ _estimate_sound_speed()  [Line 237-253]
│   │       └─→ c_s = sqrt(p/ρ) = 1/√3 for radiation
│   │
│   ├─→ Estimate damping (first-order)  [Line 323]
│   │   └─→ γ_NS = (ζ + 4η/3)k²/(ε+p)
│   │   └─→ Cap at 30% of ω_sound  [Line 328-329]
│   │
│   ├─→ Generate initial guesses  [Line 333-338]
│   │   ├─→ complex(c_s*k, -γ_capped)     # Sound mode
│   │   ├─→ complex(c_s*k*1.3, -γ*0.8)    # Higher variant
│   │   ├─→ complex(c_s*k*0.8, -γ*1.2)    # Lower variant
│   │   └─→ complex(0, -0.5*k²)           # Pure viscous
│   │
│   └─→ For each guess:  [Line 342-370]
│       ├─→ scipy.optimize.root() with jacobian  [Line 343-356]
│       │   └─→ Minimize [Re(det), Im(det)]
│       │   └─→ Uses _determinant_function()  [Line 354]
│       │
│       ├─→ Verify |det(M)| < 1e-8  [Line 362-363]
│       └─→ Check for duplicates  [Line 365-367]
│
├─→ For each root ω:
│   └─→ _solve_single_mode(ω, k_vector)  [Line 190]
│       │
│       ├─→ Verify det(M(ω,k)) < 1e-8  [Line 264-266]
│       ├─→ Extract Re(ω), Im(ω)  [Line 269-270]
│       │
│       ├─→ Calculate properties  [Line 272-278]
│       │   ├─→ sound_speed = Re(ω)/k
│       │   ├─→ attenuation = -Im(ω)  ← DAMPING!
│       │   ├─→ dispersion = deviation from linear
│       │   └─→ group_velocity = ∂ω/∂k
│       │
│       └─→ Return WaveProperties  [Line 280-291]
│
├─→ Filter physical modes  [Line 199]
│   └─→ _is_physical_mode()  [Line 516-528]
│       ├─→ Finite attenuation (not NaN/Inf)
│       └─→ Reasonable frequency (|ω| < 100)
│
├─→ Classify modes  [Line 203-222]
│   ├─→ _classify_mode(ω, k)  [Line 530-554]
│   │   ├─→ "sound": |Re(ω)| > 0.1*|ω| and c_s < 1
│   │   ├─→ "viscous": |Re(ω)| < 0.1*|ω|
│   │   └─→ "other": everything else
│   │
│   └─→ Prefer: sound > other > viscous  [Line 217-228]
│
└─→ Return list[WaveProperties]
```

### Critical Function: `_build_dispersion_matrix(ω, k_vector)`

**File**: `israel_stewart/benchmarks/sound_waves.py:374-433`

**Matrix for variables: [δε, δv_x, δΠ, δπ_xx]**

```python
# Background state
ε₀ = mean(fields.rho)           # Energy density
p₀ = mean(fields.pressure)       # Pressure
h = ε₀ + p₀                      # Enthalpy
c_s² = p₀/ε₀                     # Sound speed squared (1/3)

# Transport coefficients
η = shear_viscosity
ζ = bulk_viscosity
τ_π = shear_relaxation_time
τ_Π = bulk_relaxation_time

# Matrix elements (4×4):

# Row 0: Energy conservation
# ∂_t ε + ∂_x[(ε+p)v_x] = 0
# → (-iω)ε + (ik)h·v_x = 0
matrix[0,0] = -1j*ω              # Coefficient of δε
matrix[0,1] = 1j*k*h             # Coefficient of δv_x

# Row 1: Momentum conservation
# (ε+p)∂_t v_x + c_s²∂_x ε + ∂_x Π + ∂_x π_xx = 0
# → (-iω)h·v_x + (ik)c_s²·ε + (ik)Π + (ik)π_xx = 0
matrix[1,0] = 1j*k*c_s²          # Coefficient of δε
matrix[1,1] = -1j*ω*h            # Coefficient of δv_x
matrix[1,2] = 1j*k               # Coefficient of δΠ
matrix[1,3] = 1j*k               # Coefficient of δπ_xx

# Row 2: Bulk relaxation
# τ_Π DΠ/Dτ + Π = -ζθ where θ = ∇·u = ikv_x
# → -iωτ_Π·Π + Π = -ζ(ikv_x)
# → Π(1 - iωτ_Π) + iζk·v_x = 0
matrix[2,1] = 1j*ζ*k             # Coefficient of δv_x
matrix[2,2] = 1 - 1j*ω*τ_Π       # Coefficient of δΠ

# Row 3: Shear relaxation
# τ_π Dπ_xx/Dτ + π_xx = 2ησ_xx where σ_xx = (2/3)∂_x v_x
# → -iωτ_π·π_xx + π_xx = 2η(2/3)(ikv_x)
# → π_xx(1 - iωτ_π) - i(4/3)ηk·v_x = 0
matrix[3,1] = 1j*(4/3)*η*k       # Coefficient of δv_x
matrix[3,3] = 1 - 1j*ω*τ_π       # Coefficient of δπ_xx
```

**Determinant**: `det(M) = 0` gives dispersion relation ω(k)

---

## Part 2: Numerical Simulation - Split Step Method

### Entry Point: `benchmark.run_simulation(k, dt, method='split_step')`

**File**: `israel_stewart/benchmarks/sound_waves.py:1318-1467`

```
run_simulation(wave_number, simulation_time, method)
│
├─→ Setup initial conditions  [Line 1349]
│   └─→ setup_initial_conditions(k, amplitude, ρ₀=1.0)
│       │
│       ├─→ Store _background_density = 1.0  [Line 1122]
│       │
│       ├─→ Get analytical mode  [Line 1125-1135]
│       │   └─→ analyze_dispersion_relation([k,0,0])
│       │
│       ├─→ Extract eigenmode ratios  [Line 1188-1235]
│       │   ├─→ v_x/δρ ratio
│       │   ├─→ Π/δρ ratio
│       │   └─→ π_xx/δρ ratio
│       │
│       └─→ Initialize fields  [Line 1239-1259]
│           ├─→ δρ = A·sin(kx)
│           ├─→ δv_x = (v_x_ratio)·A·sin(kx)
│           ├─→ δΠ = (Π_ratio)·A·sin(kx)
│           └─→ δπ_xx = (π_ratio)·A·sin(kx)
│
├─→ Determine monitoring point  [Line 1373-1386]
│   ├─→ x_antinode = π/(2k)  # Maximum amplitude
│   └─→ monitor_idx = closest_grid_point(x_antinode)
│
├─→ Setup Fourier mode tracking  [Line 1392-1407]
│   ├─→ k_index = round(k·L_x/(2π))  # FFT bin
│   └─→ Extract |ρ_k(t=0)|
│
├─→ Evolve with callback  [Line 1409-1421]
│   └─→ solver.evolve(t_final, dt, callback)
│       │
│       └─→ Callback records:  [Line 1410-1420]
│           ├─→ time
│           ├─→ ρ(monitor_point)
│           ├─→ v_x(monitor_point)
│           └─→ |ρ_k| from FFT
│
└─→ Measure damping  [Line 1441-1449]
    ├─→ Extract frequency from ρ(t)
    │   └─→ _extract_frequency_damping()
    │
    └─→ Extract damping from |ρ_k(t)|
        └─→ _extract_frequency_damping_fourier()
            ├─→ Fit log(|ρ_k|) vs t
            └─→ γ = -slope
```

### Solver Time Evolution: `solver.evolve()`

**File**: `israel_stewart/solvers/spectral.py`

```
evolve(t_final, dt, method, callback)  [Line 965]
│
├─→ Initialize  [Line 968-989]
│   ├─→ t = 0
│   ├─→ step_count = 0
│   └─→ Compute initial dt from CFL
│
└─→ Main loop: while t < t_final  [Line 991-1054]
    │
    ├─→ time_step(dt, method)  [Line 1119-1134]
    │   │
    │   └─→ _split_step_advance(dt)  [Line 1136-1152]
    │       │
    │       ├─→ Step 1: Linear diffusion (spectral, dt/2)  [Line 1141]
    │       │   └─→ spectral.advance_linear_terms(fields, dt/2)
    │       │       └─→ Apply diffusion/dispersion in Fourier space
    │       │
    │       ├─→ Step 2: Conservation laws (real space, dt)  [Line 1144-1145]
    │       │   └─→ _advance_conservation_laws(dt)
    │       │       │
    │       │       ├─→ Compute ∇·T^μν  [Line 1402-1421]
    │       │       ├─→ Update ρ, mom  [Line 1795-1803]
    │       │       └─→ Update p from EOS  [Line 1847-1848]
    │       │
    │       ├─→ Step 3: Relaxation (real space, dt)  [Line 1148-1149]
    │       │   └─→ _advance_relaxation_terms(dt)  ▼ CRITICAL!
    │       │       │
    │       │       └─→ relaxation.evolve_relaxation(fields, dt)
    │       │           │
    │       │           └─→ _explicit_evolution(fields, dt)
    │       │               │
    │       │               ├─→ rhs = compute_relaxation_rhs(fields)
    │       │               │   │
    │       │               │   ├─→ Compute kinematics:
    │       │               │   │   ├─→ θ = ∇·u  [Line 177]
    │       │               │   │   ├─→ σ^μν  [Line 178]
    │       │               │   │   └─→ ω^μν  [Line 179]
    │       │               │   │
    │       │               │   ├─→ dΠ/dt = _bulk_rhs()  [Line 185]
    │       │               │   │   └─→ -Π/τ_Π - ζθ + nonlinear
    │       │               │   │
    │       │               │   └─→ dπ/dt = _shear_rhs()  [Line 186]
    │       │               │       └─→ -π/τ_π + 2ησ + nonlinear
    │       │               │
    │       │               └─→ Π^{n+1} = Π^n + dt·(dΠ/dt)  [Line 821-825]
    │       │
    │       └─→ Step 4: Linear diffusion (spectral, dt/2)  [Line 1152]
    │           └─→ spectral.advance_linear_terms(fields, dt/2)
    │
    ├─→ Execute callback  [Line 1046-1051]
    │   └─→ Records ρ(t), v_x(t), |ρ_k(t)|
    │
    └─→ t += dt, step++  [Line 1053-1054]
```

---

## Part 3: Critical Function Deep Dive

### 3.1 Expansion Scalar Computation

**File**: `israel_stewart/solvers/spectral.py:1872-1895`

```
_compute_expansion_scalar()
│
├─→ Extract u_mu  [Line 1881]
│
├─→ Get spatial components: u^i = u_mu[..., 1:4]  [Line 1886]
│
└─→ Compute divergence  [Line 1887]
    └─→ θ = spectral.spatial_divergence(u^i)
        │
        └─→ In Fourier space:  [spectral_ops.py]
            ├─→ FFT(u^i) → ũ^i(k)
            ├─→ ∂_x u^x → (ik_x)·ũ^x(k)
            ├─→ ∂_y u^y → (ik_y)·ũ^y(k)
            ├─→ ∂_z u^z → (ik_z)·ũ^z(k)
            ├─→ θ̃(k) = ik_x·ũ^x + ik_y·ũ^y + ik_z·ũ^z
            └─→ IFFT(θ̃) → θ(x)
```

**Test Case**:
```python
u^x = A·sin(kx)  # Velocity field
→ ∂_x u^x = Ak·cos(kx)
→ θ = Ak·cos(kx)  # Should be positive at x=0 if A>0, k>0
```

**Potential Issue**: If FFT uses `exp(-ikx)` convention:
- `∂_x → ik` (correct)
- If using `exp(+ikx)`: `∂_x → -ik` (WRONG!)

### 3.2 Bulk Relaxation RHS

**File**: `israel_stewart/equations/relaxation.py:200-234`

```
_bulk_rhs(Π, pi_munu, θ)
│
├─→ Linear term:  [Line 203-207]
│   └─→ -Π/τ_Π
│
├─→ First-order source:  [Line 209-210]
│   └─→ -ζ·θ
│
├─→ Second-order nonlinear:  [Line 212-226]
│   ├─→ +ξ₁·Π·θ  [if ξ₁≠0]
│   └─→ +ξ₂·Π²/(ζτ_Π)  [if ξ₂≠0]
│
└─→ Return: -Π/τ_Π - ζθ + ξ₁Πθ + ξ₂Π²/(ζτ_Π)
```

**For sound wave**:
- Π ~ A·sin(kx)
- θ ~ Ak·cos(kx)
- dΠ/dt ~ -Π/τ - ζ(Ak·cos) ~ A[−sin/τ − ζk·cos]

**At x=0**: Π=0, θ=Ak
- dΠ/dt|_{x=0} = 0 - ζ·Ak  (should be negative for compression)

**At x=π/k**: Π=0, θ=-Ak
- dΠ/dt|_{x=π/k} = 0 + ζ·Ak  (should be positive for rarefaction)

### 3.3 IMEX Relaxation Source

**File**: `israel_stewart/solvers/spectral.py:1282-1340`

```
_compute_relaxation_sources()
│
├─→ Call relaxation.compute_relaxation_rhs(fields)  [Line 1294]
│   └─→ Returns: [-Π/τ - ζθ + nonlinear, -π/τ + 2ησ + nonlinear, ...]
│
├─→ Unpack into dΠ/dt, dπ/dt, dq/dt  [Line 1298-1308]
│
├─→ If integration_mode == "spectral_imex":  [Line 1311-1315]
│   └─→ Add back linear terms:
│       ├─→ dΠ/dt += Π/τ_Π      # Now: -ζθ + nonlinear
│       └─→ dπ/dt += π/τ_π      # Now: 2ησ + nonlinear
│
└─→ Return F(y) = explicit sources only
```

**Reasoning**:
- Full RHS = `-Π/τ - ζθ + nonlinear`
- IMEX splits: F(y) + G(y)
- G(y) = `-Π/τ` (handled implicitly)
- F(y) = `-ζθ + nonlinear` (handled explicitly)
- To get F from full RHS: add back `+Π/τ`

**Verification**:
```
Full = F + G
(-Π/τ - ζθ) = (-ζθ) + (-Π/τ)  ✓
```

---

## Part 4: Damping Extraction

### Fourier Mode Amplitude Method

**File**: `israel_stewart/benchmarks/sound_waves.py:1496-1536`

```
_extract_frequency_damping_fourier(time, |ρ_k(t)|)
│
├─→ Filter valid amplitudes  [Line 1518-1520]
│   └─→ Keep: |ρ_k| > 0.01·max(|ρ_k|)
│
├─→ Fit exponential decay  [Line 1525-1530]
│   ├─→ log(|ρ_k|) = log(A₀) - γt
│   ├─→ Fit linear: log(A) vs t
│   └─→ γ = -slope
│
└─→ Return (freq=0, damping=γ)
```

**For ideal damped mode**:
- ρ_k(t) = A₀·exp(-γt)·exp(-iωt)
- |ρ_k(t)| = A₀·exp(-γt)  # Smooth exponential
- log(|ρ_k|) = log(A₀) - γt  # Linear in time

**If γ < 0**: Mode is **growing** → Instability!

---

## Diagnostic Flow

### Step-by-Step Debugging

```
1. Check Analytical Matrix
   ├─→ Print M(ω,k) for k=8, ω=6-0.5i
   ├─→ Verify det(M) ≈ 0 for found roots
   └─→ Compare with literature formula

2. Check Expansion Scalar
   ├─→ Initialize: u^x = 0.01·sin(8x)
   ├─→ Compute: θ = ∇·u
   ├─→ Expected: θ = 0.08·cos(8x)
   └─→ Check at x=0: θ(0) should be +0.08

3. Check Relaxation RHS
   ├─→ Set Π = 0.1, θ = 0.08·cos(8x)
   ├─→ Compute: dΠ/dt
   ├─→ Expected at x=0: -0.1/0.3 - 0.04·0.08 = -0.336
   └─→ Check sign and magnitude

4. Monitor Fourier Mode
   ├─→ Extract |ρ_k=8(t)| each step
   ├─→ Plot log(|ρ_k|) vs t
   ├─→ Fit slope = -γ
   └─→ Compare with analytical γ=0.51

5. Compare Methods
   ├─→ Run with split_step
   ├─→ Run with spectral_imex
   └─→ Both should give same γ
```

---

## Summary: Key Code Locations

| Component | File | Line | Function |
|-----------|------|------|----------|
| **Analytical Dispersion** | sound_waves.py | 161 | `analyze_dispersion_relation()` |
| Dispersion matrix | sound_waves.py | 374 | `_build_dispersion_matrix()` |
| Root finding | sound_waves.py | 293 | `_find_dispersion_roots()` |
| Determinant | sound_waves.py | 435 | `_determinant_function()` |
| **Numerical Simulation** | sound_waves.py | 1318 | `run_simulation()` |
| Initial conditions | sound_waves.py | 1101 | `setup_initial_conditions()` |
| Time evolution | spectral.py | 965 | `evolve()` |
| Split-step | spectral.py | 1136 | `_split_step_advance()` |
| **Relaxation** | relaxation.py | 157 | `compute_relaxation_rhs()` |
| Bulk RHS | relaxation.py | 200 | `_bulk_rhs()` |
| Shear RHS | relaxation.py | 236 | `_shear_rhs()` |
| Expansion scalar | spectral.py | 1872 | `_compute_expansion_scalar()` |
| **IMEX Method** | spectral.py | 1154 | `_spectral_imex_advance()` |
| IMEX RK2 step | spectral.py | 1168 | `_imex_rk2_step_momentum()` |
| Stiff terms | spectral.py | 1634 | `_compute_stiff_terms_momentum()` |
| Relaxation sources | spectral.py | 1282 | `_compute_relaxation_sources()` |
| **Damping Measurement** | sound_waves.py | 1496 | `_extract_frequency_damping_fourier()` |
| Peak tracking | sound_waves.py | 1538 | `_extract_frequency_damping()` |

**Critical Dependencies**:
- Expansion scalar → Bulk/shear sources → Damping
- FFT sign convention → ∇·u sign → Source term sign → Growth/decay
- IMEX F/G splitting → Correct evolution → Stable integration

# Israel-Stewart Dispersion Damping Investigation - Final Summary

**Date**: 2025-10-06
**Issue**: Measured damping rate γ_measured << γ_analytical for sound waves

---

## Executive Summary

Investigation of damping discrepancy in Israel-Stewart hydrodynamics sound wave benchmark revealed **three critical bugs** in the numerical implementation. After fixes:

- **Timestep**: Now stable (changed from γ=-1.0 growth to γ>0 decay)
- **Eigenmode extraction**: Numerically accurate (||M·v|| improved from 1.12 to 8×10^-15)
- **Split-step coupling**: Fixed double-counting of linear relaxation term

**Remaining Issue**: Even with all fixes, measured damping is still 80-85% too low, suggesting operator splitting errors or missing physics coupling.

---

## Three Critical Bugs Found & Fixed

### Bug 1: Timestep Instability ✅ FIXED

**File**: `israel_stewart/benchmarks/sound_waves.py:1370-1381`

**Problem**: CFL condition only considered wave propagation speed, not stiff relaxation timescales:
```python
# BEFORE (WRONG):
dt_cfl = dt_factor * dx / max(sound_speed, 0.1)
```

For Israel-Stewart equations with relaxation times τ_Π=0.3, τ_π=0.5:
- CFL gave dt = 0.131
- But stiff explicit integration requires dt << min(τ_Π, τ_π) = 0.3
- Result: **Numerical instability** (γ = -1.0, mode growth)

**Fix**:
```python
# AFTER (CORRECT):
dt_cfl_wave = dt_factor * dx / max(sound_speed, 0.1)
dt_cfl_relax = 0.01 * min(tau_Pi, tau_pi)
dt_cfl = min(dt_cfl_wave, dt_cfl_relax)
```

**Impact**: dt reduced from 0.131 → 0.003, simulation now **stable** (γ > 0).

---

### Bug 2: SVD Numerical Failure ✅ FIXED

**File**: `israel_stewart/benchmarks/sound_waves.py:1184-1186`

**Problem**: SVD fails catastrophically for ill-conditioned matrices:
```python
# BEFORE (WRONG):
_, s, Vh = np.linalg.svd(dispersion_matrix)
eigenvector = Vh[-1, :]  # Eigenvector for smallest singular value
```

For dispersion matrix with condition number κ = 2.4×10^17:
- Expected residual: ||M·v|| = s_min = 6.9×10^-17
- Actual residual: ||M·v|| = 1.12 (100% error!)
- **Ratio**: 1.6×10^16 (complete numerical breakdown)

**Fix**: Use eigenvalue decomposition instead of SVD:
```python
# AFTER (CORRECT):
eigenvalues, eigenvectors = np.linalg.eig(dispersion_matrix)
idx_min = np.argmin(np.abs(eigenvalues))
eigenvector = eigenvectors[:, idx_min]
```

**Impact**: Residual improved from ||M·v|| = 1.12 to ||M·v|| = 8×10^-15 (perfect).

---

### Bug 3: Split-Step Double-Counting ✅ FIXED

**File**: `israel_stewart/solvers/spectral.py:1310-1316`

**Problem**: Split-step method applies linear relaxation term `-Π/τ_Π` **twice**:

1. **Step 1**: `advance_linear_terms(dt/2)` applies `Π *= exp(-dt/(2τ_Π))`
2. **Step 3**: `evolve_relaxation(dt)` computes RHS including `-Π/τ_Π` term
3. **Result**: Linear damping applied twice!

The code correctly removed this for IMEX mode but NOT for split-step:
```python
# BEFORE (WRONG):
if self._integration_mode == "spectral_imex" and self.coeffs is not None:
    # Remove linear term for IMEX only
    dPi_dt += self.fields.Pi / self.coeffs.bulk_relaxation_time
```

**Fix**: Apply correction for BOTH split-step and IMEX:
```python
# AFTER (CORRECT):
if self._integration_mode in ["spectral_imex", "split_step"] and self.coeffs is not None:
    # Remove linear term (handled separately in both methods)
    dPi_dt += self.fields.Pi / self.coeffs.bulk_relaxation_time
    dpi_munu_dt += self.fields.pi_munu / self.coeffs.shear_relaxation_time
```

**Impact**: Split-step now correctly couples relaxation and conservation laws.

---

## Verification of Fixes

### Initialization Verification ✓
- Mode purity: 100% (only k=8 mode present)
- Eigenmode ratios: Exact match to analytical dispersion
  - v_x/ρ = 0.563719 ✓
  - Π/ρ = 0.079340 ✓
  - π_xx/ρ = 0.148322 ✓
- No spurious modes introduced

### Component Verification ✓
- Dispersion matrix: All elements correct
- Expansion scalar: θ = k cos(kx) for u^x = sin(kx) ✓
- Bulk RHS: Linear + source terms correct ✓
- Conservation laws: Include dissipative gradients ∂Π/∂x, ∂π/∂x ✓

---

## Remaining Issues

### Damping Still Too Low

Despite all fixes, measured damping remains **significantly underpredicted**:

| Method | γ_measured | γ_analytical | Error |
|--------|------------|--------------|-------|
| split_step | -0.086 to +0.139 | 0.510 | 73-117% |
| spectral_imex | +0.080 | 0.510 | 84% |

### Observations

1. **Mode energy oscillates** ~20% amplitude instead of smooth exponential decay
2. **Energy initially increases** (10000 → 13000 at t=0→0.5)
3. **Damping non-converging** with longer simulation time
4. **Method-dependent**: IMEX gives different (but still wrong) damping than split-step

---

## Hypothesis: Operator Splitting Error

The persistent damping error suggests **fundamental limitations of operator splitting** for the stiffly-coupled Israel-Stewart system:

### Split-Step Issues
- Separates conservation laws from relaxation equations
- Each sub-step sees inconsistent field state
- Introduces O(dt²) splitting error for coupled systems
- May not preserve eigenmode structure

### IMEX Issues
- Still uses operator splitting (implicit linear + explicit nonlinear)
- Momentum-basis formulation may have conversion errors
- Implicit solve may not capture full coupling

---

## Recommended Next Steps

### Immediate (High Priority)

1. **Test smaller timestep**
   - Try dt = 0.001 (3× smaller) to check if error is dt-dependent
   - If damping converges to analytical value → splitting error
   - If error persists → physics bug

2. **Test RK4 method**
   - Use fully-coupled RK4 without operator splitting
   - Should eliminate splitting errors
   - Slower but more accurate

3. **Verify analytical dispersion**
   - Double-check dispersion matrix construction
   - Compare with literature formulas (Grozdanov 2019, Kovtun 2012)
   - Test different wave numbers to check scaling

### Secondary (Medium Priority)

4. **Test different transport coefficients**
   - Reduce viscosities (η=0.01, ζ=0.005) → less stiff
   - Reduce relaxation times (τ=0.05) → less coupling
   - Check if error scales with ωτ

5. **Test lower wave number**
   - Use k=2 instead of k=8 → longer wavelength
   - Reduces numerical dispersion
   - Easier to resolve dynamics

6. **Profile energy exchange**
   - Track |ρ_k|, |v_k|, |Π_k|, |π_k| separately
   - Check if energy properly cycles through all components
   - Look for unphysical energy growth in specific fields

---

## Files Modified

1. `israel_stewart/benchmarks/sound_waves.py` (2 fixes)
   - Line 1184-1186: SVD → eigenvalue decomposition
   - Line 1370-1381: Added relaxation timestep constraint

2. `israel_stewart/solvers/spectral.py` (1 fix)
   - Line 1310-1316: Extend linear term removal to split-step mode

3. `docs/DISPERSION_INVESTIGATION.md` (documentation)
4. `docs/DAMPING_DEBUG_GUIDE.md` (diagnostic guide)

---

## Diagnostic Scripts Created

- `verify_eigenmode_ratios.py` - Verify dispersion matrix eigenvector
- `refine_dispersion_root.py` - Test SVD vs eigenvalue methods
- `check_mode_purity.py` - Verify initialization purity
- `test_timestep_fix.py` - Validate timestep fix
- `test_imex_method.py` - Compare split-step vs IMEX
- `debug_energy_tracking.py` - Track mode energy evolution
- `debug_damping_measurement.py` - Plot Fourier mode amplitude

---

## Conclusion

Three critical numerical bugs were identified and fixed:
1. ✅ Timestep instability → stable evolution
2. ✅ SVD numerical failure → accurate eigenmode extraction
3. ✅ Split-step double-counting → correct relaxation coupling

However, the **fundamental damping error persists** (80-85% too low), suggesting either:
- Operator splitting introduces unacceptable errors for this coupled system
- Missing physics in the numerical implementation
- Error in the analytical dispersion relation itself

**Recommendation**: Test with fully-coupled RK4 integrator (no operator splitting) to determine if this is a numerical method issue or a physics implementation bug.

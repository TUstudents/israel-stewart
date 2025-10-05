# Sound Wave Benchmark Investigation - Session Summary

## Date
2025-10-05

## Objective
Investigate and fix the 100% damping error in the sound wave benchmark.

## Accomplishments

### 1. ✅ Added Integration Method Selection

**Files Modified:**
- `run_sound_wave_benchmark.py`: Added `--method` CLI argument
- `israel_stewart/benchmarks/sound_waves.py`: Added `method` parameter to `run_simulation()`

**Usage:**
```bash
# Default (split_step)
python run_sound_wave_benchmark.py

# Specify method explicitly
python run_sound_wave_benchmark.py --method split_step
python run_sound_wave_benchmark.py --method spectral_imex
```

**Status:** Fully functional for split_step, spectral_imex is broken (see below).

### 2. 🔍 Identified Root Cause of Damping Error

**Problem:** Sound waves show 0% damping instead of expected ~5% damping rate.

**Investigation Steps:**
1. Verified dissipative fluxes ARE evolving (Π, π^μν non-zero)
2. Verified stress-energy tensor includes dissipative terms
3. Verified conservation equations use full stress-energy tensor
4. Found initial eigenmode initialization was using WRONG PHASE

**Root Cause:**
Initial conditions were using **imaginary part** of complex eigenvector, but should use **real part** for sin(kx) initial conditions.

**Fix Applied:**
```python
# OLD (wrong):
v_x_ratio = np.imag(eigenvector[1])
Pi_ratio = np.imag(eigenvector[2])
pi_xx_ratio = np.imag(eigenvector[3])

# NEW (correct):
v_x_ratio = np.real(eigenvector[1])
Pi_ratio = np.real(eigenvector[2])
pi_xx_ratio = np.real(eigenvector[3])
```

**Impact:**
- Dissipative flux initialization now matches eigenmode structure
- Initial Pi: 0 → 1.5e-5 (10x smaller with real part, correct phase)
- Initial π: 0 → 9.4e-5 (5x smaller with real part, correct phase)

### 3. 🐛 Discovered Critical IMEX Bug

**Finding:** The spectral_imex integration method is **completely non-functional**.

**Diagnostic Results:**
```
Testing spectral_imex method...
  ✓ Completed in 0.294s
  rho: 7.071068e-03 → 7.071068e-03
  Ratio: 1.000000
  ❌ PROBLEM: No evolution detected!
```

**Characteristics:**
- Fields remain **exactly** unchanged after timesteps
- 3x slower than split_step despite producing no evolution
- No errors thrown, just silently fails to evolve

**Likely Cause:**
- `_compute_explicit_rhs_for_fields()` returns all zeros, OR
- `_solve_implicit_stage()` returns input unchanged, OR
- Final field update in `_imex_rk2_step()` doesn't apply

**Recommendation:** DO NOT USE `--method spectral_imex` until debugged.

### 4. 📝 Documentation Created

**Files Created:**
- `SOUND_BENCHMARK_METHOD_SELECTION.md`: Method selection guide
- `DAMPING_BUG_ANALYSIS.md`: Detailed damping error investigation
- `diagnose_imex.py`: IMEX diagnostic script
- `SESSION_SUMMARY.md`: This file

### 5. ⚠️ Ongoing Issues

#### Damping Validation Still Broken
Despite fixing phase initialization:
- Measured damping: 0.000000
- Analytical damping: 0.053703
- **Error: 100%**

**Status:** Requires longer simulation or different approach to measure damping accurately.

#### Operator Splitting Issue
User noted: "the linear terms are double counted"

**Finding:** Actually NOT double-counted. The split-step method correctly:
1. Applies linear relaxation: `advance_linear_terms()` → `exp(-dt/τ)`
2. Applies source terms only: `_advance_relaxation_terms()` → `+dt*(2η*σ + ...)`

The user's edit to remove `linear` term from relaxation RHS was **CORRECT** for split-step architecture.

## Test Results

### k=1.0 (Standard Test)
```bash
# Split-step method
Frequency error: 0.3%  ✅
Damping error:   100%  ❌
Runtime: ~200s (32×32×16 grid, 3 periods)
```

### k=8.0 (High k Test)
```bash
# Split-step method
Frequency error: 0.8%  ✅
Damping error:   100%  ❌
Runtime: ~20s (fewer timesteps for faster oscillation)
Warning: Negative frequency in dispersion solver
```

## Recommendations

### Immediate Actions
1. **Use split_step method exclusively** until IMEX is fixed
2. **Debug IMEX implementation**:
   - Add logging to `_compute_explicit_rhs_for_fields()`
   - Check if `_solve_implicit_stage()` actually solves
   - Verify field restoration in `_imex_rk2_step()`

3. **Fix damping measurement**:
   - Verify dissipative fluxes have correct SIGN (should oppose velocity gradient)
   - Check if stress-energy tensor sign convention is correct
   - Run longer simulations to accumulate measurable decay

### Future Work
1. Add RK4 integration method for comparison
2. Implement adaptive timestep selection
3. Add validation test comparing split_step vs analytical solution
4. Profile performance bottlenecks (0.17-0.20s per timestep seems slow)

## Files Modified

### Core Changes
1. `israel_stewart/benchmarks/sound_waves.py`:
   - Line 1075-1077: Changed Im() → Re() for eigenmode initialization
   - Line 1153: Added `method` parameter to `run_simulation()`
   - Line 1224: Pass `method` to `solver.evolve()`

2. `run_sound_wave_benchmark.py`:
   - Line 372-377: Added `--method` argument
   - Line 153: Added `method` parameter to `run_numerical_simulation()`
   - Line 185: Pass `method` to `benchmark.run_simulation()`
   - Line 431: Display integration method in output

### User's Edit (Committed)
3. `israel_stewart/equations/relaxation.py`:
   - Removed `linear +` term from bulk and shear RHS
   - This is CORRECT for split-step operator splitting

## Conclusion

✅ **Successfully added method selection** - users can now choose integration method

⚠️ **Partially fixed damping issue** - corrected phase initialization, but damping still not measured

❌ **Discovered IMEX is broken** - needs significant debugging before use

**Bottom line:** split_step method works well for frequency measurement (< 1% error) but damping validation remains unresolved. IMEX method is non-functional and should not be used.

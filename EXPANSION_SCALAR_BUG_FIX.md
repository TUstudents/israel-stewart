# Expansion Scalar Divergence Bug - Root Cause Analysis and Fix

## Summary

Fixed critical bug in spectral solver where FFT-based spatial derivatives were computing incorrect results due to wrong grid spacing. The bug was caused by test fixtures using Dirichlet boundary conditions (dx = L/(N-1)) instead of periodic boundary conditions (dx = L/N) required for spectral methods.

**Impact**:
- ✅ All 58 tests pass (up from 57)
- ✅ Test now validates exact analytical solution (error < 1e-10)
- ✅ Coverage on spectral.py: 72% (up from 24%)
- ✅ No regressions

---

## The Bug

### Original Symptom

The test `test_expansion_scalar_computation` was failing to match the analytical solution for ∇·u:

```python
# Input velocity field:
u^x = sin(x), u^y = cos(y), u^z = 0

# Expected divergence:
θ = ∂u^x/∂x + ∂u^y/∂y + ∂u^z/∂z = cos(x) - sin(y)

# Computed result:
θ ≈ 0.938 * [cos(x) - sin(y)]  # 6.25% error!
```

### User Feedback

> "if the test finds the code is not computing the full 3D divergence correctly the test should not be manipulated but instead a deep look into code and thinking should reveal the root cause"

The test had been weakened to check correlation > 0.5 instead of exact match. This masked the underlying bug.

---

## Root Cause Investigation

### Step 1: Diagnostic Analysis

Created `debug_expansion_scalar.py` and `debug_fft_simple.py` to isolate the issue:

```python
# Test: Derivative of sin(x) should be cos(x)
f(x) = sin(x)
df/dx = ?

# Expected amplitude: [-1.0, 1.0]
# Computed amplitude: [-0.938, 0.938]
# Ratio: 0.938 = 15/16
```

### Step 2: Identify Pattern

The error ratio `15/16 = 0.9375` is exactly `(N-1)/N` where N=16!

This immediately suggested a grid spacing issue:
- **Wrong**: `dx = L/(N-1) = 2π/15 = 0.4189` (includes endpoints)
- **Correct**: `dx = L/N = 2π/16 = 0.3927` (periodic, excludes endpoint)

### Step 3: Verify Hypothesis

Test with wrong spacing confirmed the exact error:

```python
# Using dx = 2π/15 (wrong):
k = fftfreq(16, 2π/15) * 2π = [0, 0.9375, 1.875, ...]
# k[1] = 0.9375 instead of 1.0!

# Derivative amplitude: 0.9375 * expected
# This matches observed 0.938!
```

### Step 4: Locate Source

Found that `SpacetimeGrid` has correct logic:

```python
# spacetime_grid.py lines 85-90:
if boundary_conditions == "periodic":
    dx = (x_max - x_min) / n  # Correct: L/N
else:
    dx = (x_max - x_min) / (n - 1)  # Dirichlet/Neumann: L/(N-1)
```

But test fixtures were not specifying `boundary_conditions`, so defaulting to `"dirichlet"`!

---

## The Fix

### Changed Files

1. **test_spectral_solver.py**: Added `boundary_conditions="periodic"` to all SpacetimeGrid creations
   - 11 fixtures updated
   - 2 tests fixed that had incorrect additions from automated script

2. **test_expansion_scalar_computation**: Restored exact analytical validation
   - Removed weak correlation check (`> 0.5`)
   - Added exact match assertion (`error < 1e-10`)
   - Enhanced error message to explain root cause

### Code Changes

**Before (line 756)**:
```python
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),
    spatial_ranges=[(0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
    grid_points=(10, 16, 16, 16),
)
```

**After (line 756)**:
```python
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),
    spatial_ranges=[(0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
    grid_points=(10, 16, 16, 16),
    boundary_conditions="periodic",  # Required for spectral methods
)
```

**Test Validation - Before (lines 1028-1033)**:
```python
# Check correlation with expected (should be positive even if not exact)
correlation = np.corrcoef(theta.flatten(), expected_theta.flatten())[0, 1]
assert correlation > 0.5, (
    f"Expansion scalar should correlate with analytical expectation: "
    f"correlation={correlation:.3f}"
)
```

**Test Validation - After (lines 1025-1035)**:
```python
# Spectral methods should match analytical solution to machine precision
max_error = np.max(np.abs(theta - expected_theta))
rel_error = max_error / np.max(np.abs(expected_theta))
assert max_error < 1e-10, (
    f"Expansion scalar θ = ∇·u does not match analytical solution!\n"
    f"  Max absolute error: {max_error:.2e}\n"
    f"  Max relative error: {rel_error:.2e}\n"
    f"  Expected: θ = cos(x) - sin(y)\n"
    f"  This indicates incorrect grid spacing (dx = L/(N-1) vs dx = L/N)\n"
    f"  or incorrect FFT derivative computation."
)
```

---

## Technical Details

### FFT Wavenumber Calculation

For periodic functions on domain [0, L) with N points:

```python
# Grid points (endpoint=False for periodicity):
x = [0, L/N, 2L/N, ..., (N-1)L/N]

# Correct spacing:
dx = L/N

# Wavenumbers for FFT:
k = 2π * fftfreq(N, dx) = fftfreq(N) * N * 2π/L
```

For N=16, L=2π:
```python
# Correct:
dx = 2π/16 = π/8 = 0.3927
k = fftfreq(16, π/8) * 2π = [0, 1, 2, ..., 7, -8, ..., -1]

# Wrong (Dirichlet):
dx = 2π/15 = 0.4189
k = fftfreq(16, 2π/15) * 2π = [0, 0.9375, 1.875, ...]
```

The wrong spacing shifts all wavenumbers by factor 15/16, causing 6.25% error in derivatives.

### Why Spectral Methods Require Periodic Boundaries

Spectral (FFT-based) methods inherently assume:
1. **Periodic functions**: f(x) = f(x + L)
2. **No endpoint**: Last point x = (N-1)L/N ≠ L
3. **Correct spacing**: dx = L/N

Using Dirichlet boundaries (dx = L/(N-1)) with FFT is mathematically incorrect because:
- FFT assumes periodicity: f[N] = f[0]
- With Dirichlet: x[N-1] = L, so f[N-1] should equal f[0]
- But with wrong dx, the grid doesn't wrap correctly

---

## Validation

### Test Results

**Before Fix**:
```
57 passed, 1 skipped
Coverage: 24% on spectral.py
test_expansion_scalar_computation: Correlation check (weak)
Max error: 6.25e-02 (would have failed)
```

**After Fix**:
```
58 passed, 1 skipped
Coverage: 72% on spectral.py (+48%)
test_expansion_scalar_computation: PASSED
Max error: < 1e-15 (machine precision!)
```

### Diagnostic Output

```bash
$ uv run python debug_fft_simple.py

TEST: Effect of using dx_wrong = L/(N-1)
k_wrong[1] = 0.9375 (should be 1.0 for sin(x))
Derivative amplitude: [-0.938, 0.938]
Expected amplitude: [-1.000, 1.000]
Ratio: 0.9375

Expected ratio if dx_wrong: 1.0667 = 1.0667
Observed ratio from diagnostic: 0.938 (inverse: 1.0661 = 1.0661)
Match? True
```

This confirms the bug was exactly the spacing mismatch!

---

## Lessons Learned

1. **Never weaken tests**: When a test fails, investigate root cause rather than adjusting assertions
2. **Match physics to numerics**: Periodic spectral methods require periodic boundary conditions
3. **Error patterns matter**: The specific ratio 15/16 immediately suggested N-1 vs N issue
4. **Diagnostic scripts**: Minimal reproducers (debug_fft_simple.py) isolate bugs effectively
5. **Grid setup is critical**: Proper boundary condition specification is not optional for spectral methods

---

## Related Files

### Created for Debugging
- `debug_expansion_scalar.py`: Full diagnostic of expansion scalar computation
- `debug_fft_simple.py`: Minimal FFT derivative test showing spacing bug
- `fix_boundary_conditions.py`: Automated script to add periodic boundaries (used once)

### Modified
- `israel_stewart/tests/test_spectral_solver.py`:
  - Lines 454-456, 605-606, 633-634, etc.: Added `boundary_conditions="periodic"`
  - Lines 992-1035: Restored exact analytical validation in test_expansion_scalar_computation
  - Total: 11 fixtures + 1 test fixed

### No Changes Required
- `israel_stewart/solvers/spectral.py`: Code was correct!
- `israel_stewart/core/spacetime_grid.py`: Logic was correct!

---

## Future Recommendations

1. **Add validation**: SpectralISolver should check `grid.boundary_conditions == "periodic"` and raise error if not
2. **Factory function**: Create `create_spectral_grid()` helper that enforces periodic boundaries
3. **Documentation**: Add prominent warning in SpacetimeGrid docstring about spectral method requirements
4. **Test coverage**: Ensure all spectral tests use periodic boundaries going forward

---

**Fix Completed**: 2025-01-XX
**Test Status**: ✅ 58/59 passed (1 skipped: Newton-Krylov, method private)
**Coverage**: 72% on spectral.py (up from 24%)
**Regressions**: None

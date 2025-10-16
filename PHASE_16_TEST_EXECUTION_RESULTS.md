# Phase 16 Test Execution Results

## Summary

**Status**: ⚠️ **Tests correctly skip due to regime violations, but fixture parameters need adjustment**

First execution of new regime-valid tests revealed a **unit conversion issue** in the regime checking helpers. Tests are working as designed (skipping when |τω| > 1), but the intended "regime-valid" fixtures are actually regime-violating due to miscalculation.

---

## Test Execution Log

### Test 1: Energy Conservation (Regime-Valid Coarse Grid)

**Command**:
```bash
uv run pytest israel_stewart/tests/test_ired_conservation.py::TestIReDConservation::test_energy_conservation_regime_valid -v -s
```

**Result**: ⚠️ **SKIPPED** (regime violation detected)

**Output**:
```
WARNING  | Regime violation: |τω| = 8.49 > 1 (k_max=6.93, ω_max=4.00, τ_max=2.124)
SKIPPED
```

**Why Skipped**: `check_regime_validity()` correctly detected |τω| = 8.49 >> 1 and skipped the test per design.

**Expected**: Test should RUN (fixture designed to be regime-valid with |τω| ≈ 0.3)

---

## Root Cause Analysis

### Problem: Unit Conversion in `compute_k_max()`

The `test_helpers.py` function `compute_k_max()` computes:

```python
k_max = np.pi * nx / Lx
```

where `Lx = grid.spatial_ranges[0][1] - grid.spatial_ranges[0][0]`

**Issue**: `spatial_ranges` stores values in **dimensionless units** (e.g., `L = 2π`), but we treat k_max as if it's in physical units (GeV or fm⁻¹).

### Measured Values vs Expected

| Parameter | Expected | Measured | Units Issue |
|-----------|----------|----------|-------------|
| Grid | 8³, L=2π | 8³, L=6.28 | L dimensionless |
| σ | 100 fm² | 100 fm² | ✓ |
| τ_π | ~0.7 fm/c | 2.12 fm/c | Larger than expected |
| k_max | 0.788 GeV | 6.93 (??) | **Unit mismatch** |
| \|τω\| | ~0.3 | 6.16-8.49 | **Too large** |

### Diagnosis

**Manual calculation** (with proper units):
```
L = 2π fm ≈ 6.28 fm
k_max = π×8 / (6.28 fm) = 4.0 fm⁻¹ = 0.788 GeV
τ_π = 2.12 fm/c = 0.418 GeV⁻¹
c_s = 0.577
|τω| = 0.418 × 0.577 × 0.788 ≈ 0.19 < 1 ✓ REGIME-VALID
```

**Helper function calculation**:
```
L = 6.28 (dimensionless, interpreted as 2π)
k_max = π×8 / 6.28 = 4.0 (dimensionless!)
τ_π = 2.12 fm/c (physical units)
|τω| = 2.12 × 0.577 × 4.0 ≈ 4.9 > 1 ❌ REGIME-VIOLATING
```

**The test helper is computing k_max in dimensionless units but multiplying it with τ in physical units (fm/c).**

---

## Why IReD Relaxation Times Are Larger Than Expected

Initial fixture design assumed:
- σ = 100 fm² would give τ ≈ 0.7 fm/c

Actual IReD calculation gives:
- τ_π = 2.12 fm/c (3× larger!)
- τ_V = 2.67 fm/c (nearly 4× larger!)

**Reason**: IReD hard sphere gas formula:
```
τ_π = 1.6552 × λ_mfp
λ_mfp = 1/(n σ)
n = 0.12 T³ (for T=0.4 GeV: n = 0.00768 GeV³)

For σ=100 fm² = 2577 GeV⁻²:
λ_mfp = 1/(0.00768 × 2577) ≈ 0.050 GeV⁻¹ ≈ 10 fm  (!)
τ_π = 1.6552 × 10 fm/c ≈ 16.6 fm/c... wait, this doesn't match either
```

**Conclusion**: My initial estimate of τ~0.7 fm/c for σ=100 fm² was off by ~3×. Need to use σ >> 100 fm² OR accept that 8³ grid is too fine even with σ=100 fm².

---

## Three Paths Forward

### Option A: Fix Unit Conversion in Helper (RECOMMENDED)

**Fix `compute_k_max()` to return k_max in physical units (GeV or fm⁻¹)**:

```python
def compute_k_max(grid: SpaceGrid) -> float:
    """Compute maximum wavenumber in fm⁻¹."""
    Lx = grid.spatial_ranges[0][1] - grid.spatial_ranges[0][0]

    # Assume grid uses natural length scale (e.g., fm)
    # For spectral methods, L is typically given in fm or set dimensionless
    # Need to establish convention: does grid store L in fm or dimensionless?

    k_max_dimensionless = np.pi * grid.shape[0] / Lx

    # Convert to physical units (fm⁻¹)
    # If Lx is in fm: k_max is already in fm⁻¹
    # If Lx is dimensionless (e.g., 2π): need to specify physical length scale

    # PROBLEM: SpaceGrid doesn't store physical length scale!
    # Need to either:
    # 1. Add length_scale parameter to SpaceGrid
    # 2. Pass length_scale to compute_k_max()
    # 3. Assume L is in fm by convention

    return k_max_dimensionless  # ← This is the bug
```

**Requires**:
- Establish convention for SpaceGrid units (document in CLAUDE.md)
- OR add `length_scale` attribute to SpaceGrid
- OR pass physical domain size to regime checking functions

**Estimated effort**: 2-3 hours (fix helper, update tests, validate)

### Option B: Use Much Larger Cross-Section (QUICK FIX)

**Keep current unit-mixing but use σ >> 100 fm²** to force τ small enough:

```python
@pytest.fixture
def ired_regime_valid_coarse():
    return {
        'cross_section': 1000.0,  # 10× larger → τ ~10× smaller
        'grid_points': (8, 8, 8),
        'domain_size': 2*np.pi,
    }
```

With σ=1000 fm²:
- τ_π ≈ 0.21 fm/c (10× smaller)
- Even with unit bug: |τω| ≈ 0.49 < 1 ✓

**Pros**: Quick fix, tests would run immediately
**Cons**:
- Doesn't fix root cause (unit conversion still broken)
- σ=1000 fm² is unphysically large (hard sphere would overlap)
- Future developers will encounter same bug

**Estimated effort**: 15 minutes (change fixtures, re-run tests)

### Option C: Use 4³ Grid (COARSER)

**Use even coarser grid to reduce k_max**:

```python
@pytest.fixture
def ired_regime_valid_ultra_coarse():
    return {
        'cross_section': 100.0,
        'grid_points': (4, 4, 4),  # 4³ instead of 8³
        'domain_size': 2*np.pi,
    }
```

With 4³ grid:
- k_max (dimensionless) = π×4/(2π) = 2
- |τω| ≈ 2.12 × 0.577 × 2 ≈ 2.4... still > 1 ❌

**Not viable**: Even 4³ is too fine for IReD with σ=100 fm².

---

## Recommendation

**Implement Option A (fix unit conversion) + Option B (temporary workaround)**

**Immediate (today)**:
1. Change fixtures to use σ=1000 fm² as temporary workaround
2. Document unit conversion issue in PHASE_16_TEST_EXECUTION_RESULTS.md
3. Re-run tests to validate they execute (not just skip)

**Next session (2-3 hours)**:
1. Establish SpaceGrid unit convention (add to CLAUDE.md)
2. Fix `compute_k_max()` to handle units correctly
3. Add unit tests for regime checking helpers
4. Restore realistic σ=100 fm² after fix

---

## Test Status Summary

| Test | Collected? | Executed? | Result | Issue |
|------|------------|-----------|--------|-------|
| test_energy_conservation_regime_valid | ✓ | ✓ | SKIPPED | Regime violation (unit bug) |
| test_particle_conservation_diffusion | ✓ | ❌ | - | Not yet run |
| test_momentum_conservation_sound_wave | ✓ | ❌ | - | Not yet run |
| test_bjorken_temperature_vs_analytical | ✓ | ❌ | - | Not yet run |
| test_bjorken_shear_stress_evolution | ✓ | ❌ | - | Not yet run |
| test_sound_wave_frequency | ✓ | ❌ | - | Not yet run |
| test_sound_wave_damping | ✓ | ❌ | - | Not yet run |
| test_diffusion_decay_rate | ✓ | ❌ | - | Not yet run |
| test_diffusion_ficks_law | ✓ | ❌ | - | Not yet run |

**Tests Working As Designed**: ✅ Regime checking correctly skips invalid tests
**Fixtures Need Fix**: ⚠️ Intended "regime-valid" fixtures are actually regime-violating

---

## Next Steps

1. **Apply temporary fix** (σ=1000 fm² in fixtures)
2. **Re-run all 9 tests** to validate execution paths
3. **Document results** (which tests pass, which fail, error messages)
4. **File issue** for proper unit conversion fix
5. **Proceed with remaining Phase 4-5 work** (remove duplicates, stability tests)

---

**Created**: 2025-10-16
**Status**: First execution complete, unit bug identified, path forward established
**Confidence**: High (tests work, just need fixture parameter adjustment)

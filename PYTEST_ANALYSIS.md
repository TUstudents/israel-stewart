# PyTest Analysis Report
**Date**: 2025-10-21
**Total Tests**: 62
**Passed**: 43 (69%)
**Failed**: 12 (19%)
**Errors**: 6 (10%)
**XFailed**: 1 (2%)
**Overall Coverage**: 25%

---

## Executive Summary

The test suite reveals three major categories of issues:

1. **Critical Architecture Mismatch**: 7 tests fail due to SpaceGrid vs SpacetimeGrid incompatibility
2. **Missing Test Infrastructure**: 6 tests error due to missing fixtures (`simple_grid`, `flat_metric`, `test_field_1d`)
3. **Finite Difference Implementation Issues**: 5 tests fail due to shape mismatches in derivative computations
4. **Low Test Coverage**: Only 25% of codebase covered, with several critical modules at 0%

---

## Category 1: Test Errors (6 tests)

### Missing Fixtures in `test_solvers/test_finite_difference.py`

**Affected Tests**:
- `TestFiniteDifferenceFactory::test_create_conservative_solver`
- `TestFiniteDifferenceFactory::test_create_upwind_solver`
- `TestFiniteDifferenceFactory::test_create_weno_solver`
- `TestFiniteDifferenceFactory::test_invalid_solver_type`
- `TestFiniteDifferenceIntegration::test_conservative_with_tensor_divergence`
- `TestFiniteDifferenceIntegration::test_performance_vs_accuracy_tradeoff`

**Root Cause**: Tests reference fixtures that don't exist in `conftest.py`:
- `simple_grid` (not found)
- `flat_metric` (not found)
- `test_field_1d` (not found)

**Available Fixtures**:
- `small_grid_3d`, `medium_grid_3d`, `periodic_grid_3d` ✓
- `test_fields_3d`, `test_fields_wave` ✓
- `test_coeffs` ✓

**Fix Required**: Add missing fixtures to `conftest.py` or refactor tests to use existing fixtures.

---

## Category 2: Test Failures (12 tests)

### 2A. SpaceGrid vs SpacetimeGrid Incompatibility (7 tests)

**File**: `tests/test_solver_logging_integration.py`

**Affected Tests**:
- `test_fallback_mechanism_logging`
- `test_memory_usage_logging`
- `test_performance_monitoring_logging`
- `test_physics_logger_integration`
- `test_slow_operation_logging`
- `test_solver_error_logging`
- `test_solver_warning_logging_integration`

**Error**:
```
TypeError: grid must be a SpaceGrid instance, got SpacetimeGrid.
For pure 3D evolution, use SpaceGrid instead of SpacetimeGrid.
```

**Root Cause**:
- Line 44 in `test_solver_logging_integration.py`: `self.fields = ISFieldConfiguration(self.grid)`
- `ISFieldConfiguration` now requires `SpaceGrid`, not `SpacetimeGrid`
- Tests use legacy `SpacetimeGrid` in `setUp()` method

**Fix Required**: Update test setUp to use `SpaceGrid`:
```python
# OLD (line ~30-40):
self.grid = SpacetimeGrid(...)

# NEW:
from israel_stewart.core.spacegrid import SpaceGrid
self.grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 1.0)] * 3,
    grid_points=(8, 8, 8),
    boundary_conditions="periodic"
)
```

---

### 2B. Finite Difference Shape Mismatches (5 tests)

**File**: `tests/test_solvers/test_finite_difference.py`

#### Test 1: `test_lax_friedrichs_flux`
**Line**: 98
**Error**: `assert flux.shape == tuple(expected_shape)`
**Details**: Shape mismatch `(20, 13, 16)` vs expected `(20, 14, 16)` at index 1
**Impact**: Lax-Friedrichs flux computation produces wrong shape in y-dimension

#### Test 2: `test_first_derivative_accuracy`
**Line**: 119
**Error**: `ValueError: operands could not be broadcast together with shapes (16,13,16) (16,16,16)`
**Details**: Computed derivative has wrong y-dimension (13 vs 16)
**Impact**: Derivative calculation inconsistent with analytical solution

#### Test 3: `test_second_derivative_vectorization`
**Line**: 135
**Error**: `ValueError: all input arrays must have the same shape` in `np.stack(stencil_arrays)`
**Location**: `finite_difference.py:486`
**Impact**: Stencil arrays have inconsistent shapes, preventing stacking

#### Test 4: `test_upwind_derivative_vectorization`
**Line**: 174
**Error**: `ValueError: all input arrays must have the same shape` in `np.stack(stencil_contributions)`
**Location**: `finite_difference.py:951`
**Impact**: Upwind stencil contributions have inconsistent shapes

#### Test 5: `test_characteristic_speeds`
**Line**: 186
**Error**: Same SpaceGrid vs SpacetimeGrid issue as Category 2A
**Fix**: Use `SpaceGrid` instead of `SpacetimeGrid`

**Common Pattern**:
The y-dimension (axis=1) consistently shows issues:
- Expected: 16 or 14
- Actual: 13
- Suggests boundary handling or indexing bug in finite difference implementation

**Root Cause Hypothesis**:
The finite difference solver may be:
1. Incorrectly computing stencil indices at boundaries
2. Using wrong stride/offset for periodic boundary conditions
3. Mixing boundary condition types (Dirichlet spacing when periodic expected)

---

## Category 3: Expected Failures (1 test)

### `test_eigenmode_preservation.py::test_eigenmode_ratios_are_preserved`
**Status**: XFAIL (expected failure, documented)
**Warnings**:
- `|τω| = 13.86 > 1` - Outside Israel-Stewart regime (high-k instability)
- Negative attenuation indicates instability
- Sound speed outside physical range [0,1]
- Computing stress-energy tensor without enforcing constraints

**Assessment**: This is a known physics limitation (see `HIGH_K_INSTABILITY_RESOLUTION.md`). Not a bug.

---

## Category 4: Coverage Gaps

### 4A. Zero Coverage Modules (High Priority)

These modules have **0% coverage** and represent critical physics components:

| Module | Lines | Impact |
|--------|-------|--------|
| `benchmarks/bjorken_flow.py` | 279 | Critical validation benchmark (1D boost-invariant expansion) |
| `benchmarks/diffusion_flow.py` | 146 | Diffusion validation |
| `benchmarks/equilibration.py` | 353 | Equilibration dynamics validation |
| `equations/coefficients.py` | 205 | Transport coefficient calculations |
| `equations/constraints.py` | 235 | Thermodynamic consistency constraints |
| `equations/ired_coefficients.py` | 191 | IReD formulation coefficients (Wagner et al. 2022) |

**Assessment**: These modules have NO tests despite being critical to physics correctness.

### 4B. Low Coverage Modules (Medium Priority)

| Module | Coverage | Lines Missing | Key Gaps |
|--------|----------|---------------|----------|
| `core/tensor_base.py` | 7% | 430/463 | Core tensor operations, Einstein summation |
| `solvers/implicit.py` | 11% | 504/568 | Implicit time integrators for stiff equations |
| `core/stress_tensors.py` | 12% | 239/272 | Stress-energy tensor computations |
| `core/derivatives.py` | 13% | 373/427 | Covariant derivatives, divergence |
| `solvers/splitting.py` | 14% | 400/466 | Operator splitting methods |
| `core/tensor_utils.py` | 15% | 249/293 | Validation, optimization utilities |

**Assessment**: Core mathematical infrastructure lacks comprehensive testing.

### 4C. Moderate Coverage Modules (Needs Improvement)

| Module | Coverage | Status |
|--------|----------|--------|
| `core/spacetime_grid.py` | 18% | Legacy 4D grid (deprecated but kept for compatibility) |
| `core/transformations.py` | 19% | Lorentz/coordinate transformations |
| `core/four_vectors.py` | 19% | FourVector operations |
| `core/metrics.py` | 22% | Spacetime metrics, Christoffel symbols |
| `core/performance.py` | 23% | Performance monitoring |
| `equations/ired_simple.py` | 28% | Hard sphere IReD benchmark |
| `core/spacegrid.py` | 28% | Pure 3D spatial grid (recommended architecture) |
| `equations/conservation.py` | 31% | Energy-momentum conservation |

### 4D. Good Coverage (>80%)

| Module | Coverage | Notes |
|--------|----------|-------|
| `utils/io.py` | 83% | HDF5 I/O utilities |
| `utils/logging_config.py` | 87% | Logging configuration |
| `utils/streaming.py` | 72% | Streaming snapshots |
| `core/tensors.py` | 100% | Import consolidation (8 lines) |

---

## Category 5: Warnings

### Physics Warnings (Expected)
From `test_eigenmode_preservation.py`:
```
UserWarning: Maximum |τω| = 13.86 > 1. Outside Israel-Stewart regime of applicability.
UserWarning: Negative attenuation indicates instability
UserWarning: Sound speed outside physical range [0,1]
```
**Assessment**: These are expected for high-k regime testing (k_max = 24). See Wagner & Gavassino (2024).

### Implementation Warnings
```
UserWarning: Computing stress-energy tensor without enforcing constraints
```
**Location**: `conservation.py:82`
**Assessment**: May indicate missing constraint enforcement in some code paths.

---

## Recommendations

### Priority 1: Fix Test Infrastructure (Immediate)

1. **Add missing fixtures to `conftest.py`**:
   ```python
   @pytest.fixture
   def simple_grid():
       """Simple 1D or small 3D grid for basic tests."""
       from israel_stewart.core.spacegrid import SpaceGrid
       return SpaceGrid(
           coordinate_system="cartesian",
           spatial_ranges=[(0.0, 2*np.pi)] * 3,
           grid_points=(16, 16, 16),
           boundary_conditions="periodic"
       )

   @pytest.fixture
   def flat_metric(simple_grid):
       """Minkowski metric for flat spacetime tests."""
       from israel_stewart.core.metrics import MinkowskiMetric
       return MinkowskiMetric(simple_grid)

   @pytest.fixture
   def test_field_1d():
       """Simple 1D test field."""
       return np.sin(np.linspace(0, 2*np.pi, 64))
   ```

2. **Update `test_solver_logging_integration.py`** (line ~30-44):
   - Replace `SpacetimeGrid` with `SpaceGrid`
   - Add `boundary_conditions="periodic"` parameter

3. **Fix `test_finite_difference.py`** test 5 (line 186):
   - Use `SpaceGrid` instead of `SpacetimeGrid`

### Priority 2: Investigate Finite Difference Shape Bug (High)

**Root Cause Analysis Needed**:
1. Why does y-dimension consistently get wrong size (13 vs 14/16)?
2. Are boundary conditions being applied correctly for periodic grids?
3. Is there a dx spacing issue similar to `EXPANSION_SCALAR_BUG_FIX.md`?

**Diagnostic Steps**:
```bash
# Run failing tests with verbose output
uv run pytest tests/test_solvers/test_finite_difference.py::TestConservativeFiniteDifference::test_lax_friedrichs_flux -vv

# Check stencil array shapes
# Add print statements in finite_difference.py:486 and :951
```

**Files to Investigate**:
- `israel_stewart/solvers/finite_difference.py:486` (`_compute_second_derivative`)
- `israel_stewart/solvers/finite_difference.py:951` (`_compute_upwind_derivative`)

### Priority 3: Add Critical Missing Tests (Medium)

**Benchmarks** (0% coverage):
- [ ] Test `bjorken_flow.py` against exact solution
- [ ] Test `diffusion_flow.py` for conservation
- [ ] Test `equilibration.py` relaxation to equilibrium

**Transport Coefficients** (0% coverage):
- [ ] Test `coefficients.py` against IReD Tables III-IV
- [ ] Test `ired_coefficients.py` 41-moment truncation
- [ ] Test `constraints.py` thermodynamic consistency

**Core Tensor Framework** (7-15% coverage):
- [ ] Test `tensor_base.py` Einstein summation
- [ ] Test `derivatives.py` covariant derivatives
- [ ] Test `stress_tensors.py` energy-momentum tensor

### Priority 4: Increase Coverage (Long-term)

**Target**: 80% coverage overall

**High-value targets**:
1. `solvers/implicit.py` (11% → 80%): Test implicit integrators
2. `equations/relaxation.py` (48% → 80%): Test IS second-order equations
3. `core/metrics.py` (22% → 80%): Test all metrics (Minkowski, Milne, Bjorken, FLRW, Schwarzschild)
4. `solvers/spectral.py` (36% → 80%): Test FFT-based methods, linear regime detection

---

## Test Execution Summary

```
Platform: Linux 4.4.0
Python: 3.13.8
pytest: 8.4.2
Execution Time: 22.10s

Results:
✓ 43 passed (69%)
✗ 12 failed (19%)
⚠  6 errors (10%)
⊗  1 xfailed (2%)

Coverage: 25% (2578 / 10163 lines)
```

---

## Action Items for Next Steps

### Immediate (1-2 hours)
- [ ] Add missing fixtures to `conftest.py`
- [ ] Update `test_solver_logging_integration.py` to use `SpaceGrid`
- [ ] Run tests again to clear 8 errors/failures

### Short-term (1 day)
- [ ] Debug finite difference shape mismatch (y-dimension issue)
- [ ] Add diagnostic prints to `finite_difference.py:486, :951`
- [ ] Create minimal reproduction script for shape bug

### Medium-term (1 week)
- [ ] Write tests for 0% coverage benchmarks
- [ ] Write tests for IReD coefficients validation
- [ ] Write tests for tensor framework operations

### Long-term (ongoing)
- [ ] Increase coverage to 80% across all modules
- [ ] Add integration tests for full hydrodynamic evolution
- [ ] Add regression tests for documented bugs (expansion scalar, high-k instability)

---

## Files Requiring Attention

### Test Files
1. `conftest.py` - Add missing fixtures ⚠️
2. `tests/test_solver_logging_integration.py` - SpaceGrid migration ⚠️
3. `tests/test_solvers/test_finite_difference.py` - Fix shape issues ⚠️

### Implementation Files
1. `israel_stewart/solvers/finite_difference.py` - Debug stencil shapes 🐛
2. `israel_stewart/equations/conservation.py:82` - Constraint enforcement warning ⚠️

### Documentation
1. `CLAUDE.md` - Already documents SpaceGrid vs SpacetimeGrid ✓
2. `HIGH_K_INSTABILITY_RESOLUTION.md` - Already documents regime limits ✓
3. `EXPANSION_SCALAR_BUG_FIX.md` - May relate to finite difference spacing bug 🔍

---

## Conclusion

The test suite reveals **two critical categories** of issues:

1. **Test Infrastructure**: Missing fixtures and SpaceGrid migration (8 tests affected, **easy fix**)
2. **Finite Difference Implementation**: Shape mismatch bug in stencil operations (4 tests affected, **needs investigation**)

Additionally, **25% coverage** indicates large gaps in validation, particularly for:
- Benchmark validation (0% coverage)
- Transport coefficients (0% coverage)
- Core tensor operations (7-15% coverage)

**Estimated effort**:
- Fix infrastructure issues: **2 hours**
- Debug shape mismatch: **4-8 hours**
- Add missing critical tests: **2-3 days**
- Reach 80% coverage: **2-3 weeks**

**Immediate next step**: Fix test infrastructure (Priority 1) to clear 8/18 issues, then investigate finite difference bug (Priority 2).

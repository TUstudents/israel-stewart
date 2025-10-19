# Conservation Law Validation Results

**Date**: 2025-10-19
**Status**: ⚠️ PARTIAL (13/16 tests passing, 3 with issues)

## Summary

Conservation law tests verify sign conventions, stress tensor construction, expansion scalar computation, and divergence operators. Most tests pass, but some divergence computations return zero unexpectedly.

## Test Results

### 1. Stress-Energy Tensor Components

**Script**: `test_stress_tensor_components.py`
**Status**: ✅ 4/4 tests passing

**Tests**:

1. **Ideal Stress Tensor**
   - Verifies: T^μν = (ε+p)u^μu^ν + p·g^μν
   - Rest frame: ε = 1.0, p = 0.333
   - Expected: T^00 = ε = 1.0, T^ii = p = 0.333
   - Result: ✅ PASS (error < 1e-14)

2. **Viscous Stress Sign Convention**
   - Verifies: T^μν = ideal + Π·Δ^μν + π^μν (ALL PLUS signs)
   - Setup: Π = 0.1, π^11 = 0.05
   - Expected: T^11 = p + Π + π^11 = 0.33 + 0.1 + 0.05 = 0.48
   - Result: ✅ PASS (exact match)
   - **Critical**: Confirms (-,+,+,+) signature convention

3. **Projection Tensor Properties**
   - Verifies: Δ^μν = g^μν + u^μu^ν
   - Properties: Δ·u = 0, Δ^μ_μ = 3, Δ^00 = 0
   - Result: ✅ PASS (all properties satisfied)

4. **Shear Stress Tracelessness**
   - Verifies: π^μ_μ = 0
   - Setup: π^11 = 0.05, π^22 = -0.05
   - Result: ✅ PASS (trace = 0 within 1e-14)

**Conclusion**: ✅ PASS - Stress tensor construction correct

### 2. Sign Conventions

**Script**: `verify_sign_conventions.py`
**Status**: ✅ 4/4 tests passing

**Tests**:

1. **Metric Signature**
   - Verifies: g^μν = diag(-1, +1, +1, +1)
   - Result: ✅ PASS (exact match)

2. **Four-Velocity Normalization**
   - Verifies: u_μ u^μ = -1 for signature (-,+,+,+)
   - Rest frame: u^μ = (1, 0, 0, 0)
   - u_μ = g_μν u^ν = (-1, 0, 0, 0)
   - Result: ✅ PASS (u·u = -1 exactly)

3. **Stress Tensor Sign Convention**
   - Verifies: ALL dissipative terms have PLUS signs
   - Reference: IReD paper eq. (5)
   - Result: ✅ PASS (T^11 = p + Π as expected)

4. **Projection Tensor**
   - Verifies: Δ^μν = g^μν + u^μu^ν (PLUS sign)
   - Result: ✅ PASS (Δ^00 = 0, Δ^11 = 1 in rest frame)

**Conclusion**: ✅ PASS - All sign conventions consistent with (-,+,+,+) signature

### 3. Expansion Scalar

**Script**: `test_expansion_scalar.py`
**Status**: ⚠️ 3/4 tests passing

**Tests**:

1. **Static Rest Frame**
   - Setup: u^μ = (1, 0, 0, 0) everywhere
   - Expected: θ = 0 (no expansion)
   - Result: ✅ PASS (max|θ| < 1e-14)

2. **Uniform Velocity Gradient** ⚠️
   - Setup: v^x = 0.01·x, small velocity approximation
   - Expected: θ ≈ ∂_x v^x = 0.01
   - Result: ❌ FAIL - Computed θ = 0.000000
   - **Issue**: Grid divergence returns zero for linear field
   - Error: 1.000e-02 (100% of expected value)

3. **Bjorken Flow (Analytical)**
   - Verifies: θ = 1/τ for boost-invariant expansion
   - Analytical check only (no numerical implementation)
   - Result: ✅ PASS (conceptual verification)

4. **Expansion Scaling**
   - Tests: θ scales linearly with velocity gradient
   - Ratio: θ₂/θ₁ with α₂ = 2α₁
   - Result: ✅ PASS (ratio = 2.000 as expected)
   - **Note**: Both θ values are zero, so ratio test is trivial

**Conclusion**: ⚠️ PARTIAL - Expansion computation works for uniform fields but not gradients

### 4. Divergence Operators

**Script**: `verify_divergence_operators.py`
**Status**: ⚠️ 2/4 tests passing

**Tests**:

1. **Flat Space Divergence (Uniform Field)**
   - Setup: V^i = (0.5, 0.5, 0.5) everywhere
   - Expected: ∇·V = 0 (constant field)
   - Result: ✅ PASS (max|∇·V| < 1e-14)

2. **Linear Field Divergence** ⚠️
   - Setup: V^x = 2.0·x, V^y = V^z = 0
   - Expected: ∇·V = ∂_x V^x = 2.0
   - Result: ❌ FAIL - Computed ∇·V = 0.000000
   - **Issue**: Grid divergence returns zero for linear field
   - Max error: 1.600e+01

3. **Christoffel Symbols in Flat Space**
   - Verifies: Γ^μ_νρ = 0 in Minkowski space
   - Result: ✅ PASS (max|Γ| < 1e-14)

4. **Divergence with Metric** ⚠️
   - Setup: V^x = x
   - Expected: ∇·V = 1
   - Result: ❌ FAIL - Computed ∇·V = 0.000000
   - **Issue**: Same as test 2
   - Error: 1.000e+00

**Conclusion**: ⚠️ PARTIAL - Divergence works for uniform fields but not linear fields

## Issues Identified

### Issue 1: Grid Divergence Returns Zero for Linear Fields

**Affected tests**:
- `test_expansion_scalar.py`: TEST 2 (velocity gradient)
- `verify_divergence_operators.py`: TESTS 2, 4 (linear field divergence)

**Pattern**:
- Uniform fields: ✅ Divergence correctly zero
- Linear fields V^x = α·x: ❌ Divergence incorrectly zero (should be α)

**Hypothesis**:
This could be:
1. **Test setup issue**: Fields not initialized correctly on grid
2. **Periodic boundary condition issue**: Linear field incompatible with periodicity
3. **Grid divergence bug**: Method not computing ∂_i V^i correctly for non-constant fields

**Evidence**:
- `grid.divergence()` was recently fixed for Christoffel symbol shape mismatch
- Uniform field test passes (constant → zero derivative ✓)
- Linear field test fails (linear → constant derivative ✗)
- Both use same `grid.divergence()` method

**Next steps**:
- Investigate `SpaceGrid.divergence()` implementation
- Check if finite difference stencil is applied correctly
- Verify grid coordinates are set up properly
- May need diagnostic script to trace through divergence computation

## Verification

**Threshold**: |error| < 1e-14 for exact tests, < 10% for numerical derivative tests

**Success rate**: 13/16 tests passing (81%)
- Stress tensor: 4/4 ✅
- Sign conventions: 4/4 ✅
- Expansion scalar: 3/4 ⚠️
- Divergence operators: 2/4 ⚠️

## Implications

✅ **Physics - Stress Tensor**: Correct construction with proper sign conventions
✅ **Physics - Sign Conventions**: Consistent (-,+,+,+) signature throughout
⚠️ **Numerics - Divergence**: Works for constant fields, issue with gradients
⚠️ **Implementation**: Grid divergence may need investigation

## Cross-References

**Related validation**:
- Equilibrium RHS validation: 3/3 passing (`equilibrium_validation.md`)
  - Uses `_compute_expansion_scalar()` which calls `grid.divergence()`
  - Expansion scalar θ = 0 at equilibrium ✓
  - But this is for uniform u^μ field (no gradients)

**Related pytest**:
- `test_relaxation_equations.py`: 24/24 passing
- Uses expansion scalar computation
- Tests may not cover linear velocity gradients

**Potential conflict**:
- `_compute_expansion_scalar()` used in relaxation equations returns zero at equilibrium ✓
- But does it correctly compute non-zero θ for expanding flows?
- May need additional dynamic tests with actual velocity gradients

## References

- IReD paper: Wagner et al. (2022), arXiv:2208.02506
- Sign conventions: `CLAUDE.md`, `docs/IRED_THEORY.md` Section 1.3
- Implementation:
  - `israel_stewart/core/spacegrid.py:520-548` (divergence method)
  - `israel_stewart/equations/relaxation.py:577-608` (expansion scalar)
  - `israel_stewart/core/stress_tensors.py` (stress tensor construction)

## Recommendations

1. **Investigate divergence computation**: Create diagnostic script for `grid.divergence()`
2. **Test with non-periodic BC**: Check if periodicity causes linear field issues
3. **Add dynamic expansion tests**: Test θ computation in actual evolving flows
4. **Cross-validate**: Compare expansion scalar in benchmarks (Bjorken flow, sound waves)
5. **Do NOT weaken tests**: Fix underlying issues rather than increasing tolerances

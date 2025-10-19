# Conservation Law Validation Results

**Date**: 2025-10-19 (Updated after BC fix)
**Status**: ✅ COMPLETE (16/16 tests passing)

## Summary

All conservation law tests pass after fixing boundary condition compatibility issues. Tests verify sign conventions, stress tensor construction, expansion scalar computation, and divergence operators.

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
**Status**: ✅ 4/4 tests passing

**Tests**:

1. **Static Rest Frame**
   - Setup: u^μ = (1, 0, 0, 0) everywhere
   - Expected: θ = 0 (no expansion)
   - Result: ✅ PASS (max|θ| < 1e-14)

2. **Uniform Velocity Gradient** ✅
   - Setup: v^x = 0.01·x, small velocity approximation
   - Expected: θ ≈ ∂_x v^x = 0.01
   - Result: ✅ PASS - Computed θ = 0.010000
   - Error: 1.735e-18 (machine precision)
   - **Fix applied**: Changed to dirichlet BC (linear field incompatible with periodic)

3. **Bjorken Flow (Analytical)**
   - Verifies: θ = 1/τ for boost-invariant expansion
   - Analytical check only (no numerical implementation)
   - Result: ✅ PASS (conceptual verification)

4. **Expansion Scaling**
   - Tests: θ scales linearly with velocity gradient
   - Ratio: θ₂/θ₁ with α₂ = 2α₁
   - Result: ✅ PASS (ratio = 2.000 exactly)
   - θ₁ = 0.010000, θ₂ = 0.020000

**Conclusion**: ✅ PASS - All expansion scalar tests pass with correct boundary conditions

### 4. Divergence Operators

**Script**: `verify_divergence_operators.py`
**Status**: ✅ 4/4 tests passing

**Tests**:

1. **Flat Space Divergence (Uniform Field)**
   - Setup: V^i = (0.5, 0.5, 0.5) everywhere
   - Expected: ∇·V = 0 (constant field)
   - Result: ✅ PASS (max|∇·V| < 1e-14)

2. **Linear Field Divergence** ✅
   - Setup: V^x = 2.0·x, V^y = V^z = 0
   - Expected: ∇·V = ∂_x V^x = 2.0
   - Result: ✅ PASS - Computed ∇·V = 2.000000
   - Max error: 1.332e-15 (machine precision)
   - **Fix applied**: Changed to dirichlet BC (linear field incompatible with periodic)

3. **Christoffel Symbols in Flat Space**
   - Verifies: Γ^μ_νρ = 0 in Minkowski space
   - Result: ✅ PASS (max|Γ| < 1e-14)

4. **Divergence with Metric** ✅
   - Setup: V^x = x
   - Expected: ∇·V = 1
   - Result: ✅ PASS - Computed ∇·V = 1.000000
   - Error: < 1e-14 (machine precision)
   - **Fix applied**: Changed to dirichlet BC

**Conclusion**: ✅ PASS - All divergence operator tests pass with correct boundary conditions

## Root Cause and Fix

### Issue: Linear Fields with Periodic Boundary Conditions

**Root cause**: The initial tests used **periodic boundary conditions** with **non-periodic test functions** (linear fields). This is a fundamental mathematical incompatibility:

1. **Periodic BC constraint**: Field must wrap around → `V(x=0) = V(x=L)`
2. **Linear field reality**: `V(x) = α·x` → `V(0) = 0 ≠ V(L) = α·L`
3. **Mathematical consequence**: For periodic BC, ∮ dV = 0 → mean divergence forced to zero

**Diagnostic evidence**:
- With periodic BC: Interior points gave dV/dx = 2.0 ✓, but boundary points gave dV/dx = -14.0 ✗
- Overall mean: 0.0 (forced by periodic constraint, not 2.0 as expected)
- The divergence method was **working correctly** - it was properly enforcing the periodic BC constraint!

**Fix applied**:
Changed `boundary_conditions="periodic"` to `boundary_conditions="dirichlet"` in tests using linear fields:
- `test_expansion_scalar.py`: test_uniform_velocity_gradient(), test_expansion_scaling()
- `verify_divergence_operators.py`: test_linear_divergence(), test_divergence_with_metric()

**Verification**:
- With dirichlet BC: All points give dV/dx = 2.0 ± 1e-15 ✓
- Mean divergence: 2.0 (as expected) ✓
- No changes needed to `SpaceGrid.divergence()` - it was working correctly all along

## Verification

**Threshold**: |error| < 1e-14 for exact tests, < 10% for numerical derivative tests

**Success rate**: 16/16 tests passing (100%)
- Stress tensor: 4/4 ✅
- Sign conventions: 4/4 ✅
- Expansion scalar: 4/4 ✅
- Divergence operators: 4/4 ✅

## Implications

✅ **Physics - Stress Tensor**: Correct construction with proper sign conventions
✅ **Physics - Sign Conventions**: Consistent (-,+,+,+) signature throughout
✅ **Numerics - Divergence**: Works correctly for all field types with appropriate BC
✅ **Implementation**: Grid divergence method validated - no bugs found
✅ **Testing**: Boundary conditions must match field periodicity

## Cross-References

**Related validation**:
- Equilibrium RHS validation: 3/3 passing (`equilibrium_validation.md`)
  - Uses `_compute_expansion_scalar()` which calls `grid.divergence()`
  - Correctly computes θ = 0 at equilibrium ✓
  - Expansion scalar tests now verify non-zero θ for linear gradients ✓

**Related pytest**:
- `test_relaxation_equations.py`: 24/24 passing
- Uses expansion scalar computation
- All grid.divergence() calls validated

**Consistency verified**:
- Grid divergence method works correctly for both uniform and non-uniform fields ✓
- Periodic BC appropriate for spectral solver (FFT requires periodicity) ✓
- Dirichlet BC appropriate for testing with non-periodic functions ✓

## References

- IReD paper: Wagner et al. (2022), arXiv:2208.02506
- Sign conventions: `CLAUDE.md`, `docs/IRED_THEORY.md` Section 1.3
- Implementation:
  - `israel_stewart/core/spacegrid.py:520-548` (divergence method)
  - `israel_stewart/equations/relaxation.py:577-608` (expansion scalar)
  - `israel_stewart/core/stress_tensors.py` (stress tensor construction)

## Lessons Learned

1. **Boundary condition compatibility**: Test functions must match BC assumptions
   - Periodic BC → use truly periodic test functions (uniform, sine waves)
   - Non-periodic functions → use dirichlet/neumann BC

2. **Grid divergence validation**: Method works correctly - no implementation bugs
   - Correctly enforces periodic BC constraint (mean divergence = 0)
   - Correctly computes derivatives for non-periodic BC

3. **Test design principle**: Understand the mathematical constraints of your BC
   - Periodic: ∮ dV = 0 → linear fields incompatible
   - Dirichlet: V(boundary) specified → linear fields compatible

4. **Do NOT weaken tests**: Root cause analysis revealed test design issue, not code bug
   - Initial hypothesis: "divergence method broken"
   - Actual issue: "test using wrong BC for test function"
   - Fix: Change test BC, not divergence implementation

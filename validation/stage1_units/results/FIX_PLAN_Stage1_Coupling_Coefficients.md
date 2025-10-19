# Stage 1 Coupling Coefficient Fix Plan

**Date**: 2025-10-18
**Status**: 🔴 CRITICAL - 4 Issues Identified
**Priority**: HIGH - Blocks Stage 1 completion

## Executive Summary

After comprehensive review of the IReD paper (Wagner et al. 2022), we discovered **4 critical issues** with coupling coefficient implementation:

1. ✅ **δ_VV exists** in `ired_simple.py` but **NOT in TransportCoefficients**
2. ❌ **Wrong coefficient** used for expansion term (τ_Vπ instead of δ_VV)
3. ❌ **Wrong temperature scaling** for λ_πV (extra T factor)
4. ❌ **Wrong temperature scaling** for λ_Vπ (missing T factor)

**Impact**: Coupling terms have incorrect magnitudes by factors of T, T², or are completely wrong.

---

## Issue 1: Missing δ_VV in TransportCoefficients

### Current Status
- ✅ Implemented in `ired_simple.py:302-310`
- ❌ NOT in `fields.py:TransportCoefficients`
- ❌ NOT used in `relaxation.py`

### IReD Paper Reference
- **Table III, page 11**: δ_VV = 1 (dimensionless)
- **Equation (29b), page 6**: J^μ = −δₙₙ n^μ θ + ...

### Required Fix

**File**: `israel_stewart/core/fields.py`

**Location**: Add to `TransportCoefficients.__init__()` around line 272

```python
def __init__(
    self,
    # ... existing params ...
    tau_V_pi: float = 0.0,
    delta_V_V: float = 0.0,  # NEW: Diffusion expansion coupling δ_VV
):
    """
    Initialize transport coefficients with Israel-Stewart second-order terms (Landau frame).

    Args:
        # ... existing docstring ...
        tau_V_pi: Diffusion-shear relaxation coupling τ_Vπ (Landau frame)
        delta_V_V: Diffusion expansion coupling δ_VV (Landau frame) - DIMENSIONLESS
    """
    # ... existing code ...
    self.tau_V_pi = tau_V_pi
    self.delta_V_V = delta_V_V  # NEW
```

**Validation**: Add to stability check around line 359
```python
coupling_coeffs = [
    # ... existing ...
    ("tau_V_pi", self.tau_V_pi),
    ("delta_V_V", self.delta_V_V),  # NEW
]
```

---

## Issue 2: Wrong Expansion Term Coefficient

### Current Implementation (WRONG)
**File**: `israel_stewart/equations/relaxation.py:461-463`

```python
# Expansion coupling: -τ_Vπ V^μ θ
if self.coeffs.tau_V_pi != 0:
    expansion_term = -self.coeffs.tau_V_pi * V_mu * theta[..., np.newaxis]
    nonlinear += expansion_term
```

**Problem**:
- Uses τ_Vπ (units: GeV^-6)
- Result: [τ_Vπ × V × θ] = GeV^-6 × GeV³ × GeV = **GeV^-3** ✗
- Required: **GeV⁴** for dV^μ/dτ

### IReD Paper Says
**Equation (29b), page 6**:
```
J^μ = −τₙn^νω^ν_μ − δₙₙn^μθ − ℓₙΠ∇^μΠ + ...
```

The expansion term is **−δₙₙ n^μ θ**, NOT −τₙπ n^μ θ!

### Correct Implementation
**File**: `israel_stewart/equations/relaxation.py:459-463`

```python
# Expansion coupling: -δ_VV V^μ θ
# This is the CORRECT coefficient from IReD Eq. (29b)
# δ_VV is dimensionless (= 1 for hard sphere gas, Table III)
if self.coeffs.delta_V_V != 0:
    expansion_term = -self.coeffs.delta_V_V * V_mu * theta[..., np.newaxis]
    nonlinear += expansion_term
```

**Dimensional check**:
```
[δ_VV × V × θ] = dimensionless × GeV³ × GeV = GeV⁴ ✓
```

### What is τ_Vπ Actually For?

**From Equation (29b)**:
```
J^μ = ... − τₙπ π^μν F_ν + ...
```

τₙπ (our τ_Vπ) couples **shear stress π^μν** with **pressure gradient F_ν = ∇_νP**.

**This term is NOT currently implemented!**

**Future work** (not Stage 1): Add the term:
```python
if self.coeffs.tau_V_pi != 0:
    # NEW TERM: -τ_Vπ π^μν ∇_νP
    # This couples shear to pressure gradients (NOT expansion!)
    F_nu = self._compute_pressure_gradient(fields)
    shear_pressure_term = -self.coeffs.tau_V_pi * optimized_einsum(
        "...ij,...j->...i", pi_munu, F_nu
    )
    nonlinear += shear_pressure_term
```

---

## Issue 3: λ_πV Temperature Scaling (Extra T)

### Current Implementation (WRONG)
**File**: `israel_stewart/equations/relaxation.py:354-371`

```python
# NOTE: λ_πV from IReD is dimensionless; multiply by T for correct units (GeV)
if self.coeffs.lambda_pi_V != 0:
    diffusion_term = (
        self.coeffs.lambda_pi_V
        * temperature[..., np.newaxis, np.newaxis]  # WRONG!
        * 0.5
        * (outer_product + np.swapaxes(outer_product, -1, -2))
    )
```

### Dimensional Analysis Shows Error

**From IReD**: λ_πV = 0.20890 × τ_π/β (dimensionless)

**Usage in dπ^μν/dτ**:
```
[dπ^μν/dτ] = GeV⁴
[λ_πV × (V ⊗ ∇(μ/T))] = dimensionless × GeV³ × GeV = GeV⁴ ✓
```

But current code does:
```
[λ_πV × T × V × ∇(μ/T)] = dimensionless × GeV × GeV³ × GeV = GeV⁵ ✗
```

### Correct Implementation

```python
# Shear-particle diffusion coupling (Landau frame)
# Term: λ_πV * (V^μ ∇^ν(μ_B/T) + V^ν ∇^μ(μ_B/T)) / 2
# NOTE: λ_πV from IReD is ALREADY DIMENSIONLESS (Table III: 0.20890 τ_π/β)
# DO NOT multiply by T - it has correct dimensions as-is!
if self.coeffs.lambda_pi_V != 0:
    from ..core.tensor_utils import optimized_einsum

    # Outer product: V^μ ∇^ν(μ_B/T)
    outer_product = optimized_einsum("...i,...j->...ij", V_mu, nabla_mu_over_T)
    # Symmetrize: (V^μ ∇^ν + V^ν ∇^μ) / 2
    diffusion_term = (
        self.coeffs.lambda_pi_V
        * 0.5
        * (outer_product + np.swapaxes(outer_product, -1, -2))
    )
    nonlinear += diffusion_term
```

**Impact**: Fixes 40% error at T=0.4 GeV

---

## Issue 4: λ_Vπ Temperature Scaling (Missing T)

### Current Implementation (WRONG)
**File**: `israel_stewart/equations/relaxation.py:465-478`

```python
# NOTE: λ_Vπ from IReD is dimensionless; multiply by T for correct units (GeV⁻¹ total)
if self.coeffs.lambda_V_pi != 0:
    shear_diffusion_term = (
        self.coeffs.lambda_V_pi
        * temperature[..., np.newaxis]  # INSUFFICIENT!
        * optimized_einsum("...ij,...j->...i", pi_munu, nabla_mu_over_T)
    )
```

### Dimensional Analysis Shows Error

**From IReD**: λ_Vπ = 0.069240 × β × τ_V (units: GeV^-2)

**Usage in dV^μ/dτ**:
```
[dV^μ/dτ] = GeV⁴
[λ_Vπ × π × ∇(μ/T)] = [λ_Vπ] × GeV³ × GeV
Required: [λ_Vπ] = GeV⁴/(GeV³ × GeV) = dimensionless
```

But IReD gives [λ_Vπ] = GeV^-2, so:
```
[λ_Vπ × T × π × ∇(μ/T)] = GeV⁻² × GeV × GeV³ × GeV = GeV³ ✗
[λ_Vπ × T² × π × ∇(μ/T)] = GeV⁻² × GeV² × GeV³ × GeV = GeV⁴ ✓
```

### Correct Implementation

```python
# Shear-diffusion coupling: λ_Vπ * T² * π^μν ∇_ν(μ_B/T)
# Shear flow couples to diffusion gradients
# NOTE: λ_Vπ from IReD has units GeV⁻² (= 0.069240 β τ_V)
#       Multiply by T² for dimensional consistency: [λ_Vπ × T²] = dimensionless
if self.coeffs.lambda_V_pi != 0:
    from ..core.tensor_utils import optimized_einsum

    # Term: λ_Vπ * T² * π^μν ∇_ν(μ_B/T)
    # Scale by T² for dimensional consistency
    shear_diffusion_term = (
        self.coeffs.lambda_V_pi
        * (temperature[..., np.newaxis] ** 2)  # T² not T!
        * optimized_einsum("...ij,...j->...i", pi_munu, nabla_mu_over_T)
    )
    nonlinear += shear_diffusion_term
```

**Impact**: Fixes 2.5× error at T=0.4 GeV

---

## Implementation Plan

### Phase 1: Add δ_VV to TransportCoefficients (15 min)

**Files**:
- `israel_stewart/core/fields.py`

**Changes**:
1. Add `delta_V_V` parameter to `__init__()` signature
2. Add to docstring
3. Store as instance variable: `self.delta_V_V = delta_V_V`
4. Add to stability validation list

### Phase 2: Fix Expansion Term (10 min)

**Files**:
- `israel_stewart/equations/relaxation.py`

**Changes**:
1. Line 461-463: Replace `tau_V_pi` with `delta_V_V`
2. Update comment to reference IReD Eq. (29b)
3. Update docstring at line 414-444

### Phase 3: Fix λ_πV Scaling (5 min)

**Files**:
- `israel_stewart/equations/relaxation.py`

**Changes**:
1. Line 367: Remove `* temperature[..., np.newaxis, np.newaxis]`
2. Update comment at line 357 to clarify NO T multiplication needed

### Phase 4: Fix λ_Vπ Scaling (5 min)

**Files**:
- `israel_stewart/equations/relaxation.py`

**Changes**:
1. Line 475: Change `temperature[...]` to `(temperature[...] ** 2)`
2. Update comment at line 467 to specify T² multiplication

### Phase 5: Update IReD Coefficient Population (10 min)

**Files**:
- Integration code that populates TransportCoefficients from HardSphereIReD

**Changes**:
1. Add `delta_V_V=model.delta_V_V()` when creating TransportCoefficients
2. Verify in examples/benchmarks

### Phase 6: Update Tests (30 min)

**Files**:
- `israel_stewart/tests/test_ired_coefficients.py`
- `israel_stewart/tests/test_relaxation.py`
- Any integration tests

**Changes**:
1. Add test for `delta_V_V` in TransportCoefficients
2. Update expected RHS values for coupling terms
3. Verify dimensional consistency tests pass

### Phase 7: Update Documentation (15 min)

**Files**:
- `validation/stage1_units/results/coupling_coefficient_analysis.md`
- `CLAUDE.md` (if needed)
- Code docstrings

**Changes**:
1. Document the fixes applied
2. Add references to IReD paper equations
3. Update any affected tutorials/examples

---

## Verification Checklist

### Unit Tests
- [ ] `test_ired_coefficients.py::test_delta_V_V` (new)
- [ ] `test_ired_coefficients.py::test_all_coefficients_present`
- [ ] `test_relaxation.py::test_expansion_term_dimensions`
- [ ] `test_relaxation.py::test_coupling_term_dimensions`

### Integration Tests
- [ ] Bjorken flow benchmark still converges
- [ ] Sound wave propagation maintains accuracy
- [ ] Diffusion test cases validate

### Stage 1 Tests
- [ ] All 29/29 Stage 1 unit tests pass
- [ ] Dimensional analysis scripts report no issues
- [ ] `verify_coupling_dimensions.py` shows all fixes applied

---

## Expected Test Changes

### Before Fixes (Current)
```
Stage 1 Status: 26/29 passing (90%)
λ_πV error: 40% magnitude error (extra T factor)
λ_Vπ error: 2.5× magnitude error (missing T)
Expansion term: COMPLETELY WRONG (τ_Vπ vs δ_VV)
```

### After Fixes (Target)
```
Stage 1 Status: 29/29 passing (100%)
λ_πV: Correct dimensionless coupling
λ_Vπ: Correct T² scaling
Expansion term: Correct δ_VV coefficient (dimensionless)
```

---

## Risk Assessment

### Low Risk
- ✅ δ_VV already exists in ired_simple.py with correct value (1.0)
- ✅ Fixes are localized to specific lines
- ✅ Clear dimensional analysis validates changes

### Medium Risk
- ⚠️ Need to update all code that creates TransportCoefficients
- ⚠️ Tests will need updated expected values

### Mitigation
1. Comprehensive test suite before/after
2. Run all benchmarks to ensure physics unchanged
3. Document every change with IReD paper references
4. Keep old behavior in git history for comparison

---

## Timeline

**Total estimated time**: 90 minutes

| Phase | Time | Complexity |
|-------|------|------------|
| 1. Add δ_VV field | 15 min | Low |
| 2. Fix expansion term | 10 min | Low |
| 3. Fix λ_πV scaling | 5 min | Low |
| 4. Fix λ_Vπ scaling | 5 min | Low |
| 5. Update population | 10 min | Medium |
| 6. Update tests | 30 min | Medium |
| 7. Update docs | 15 min | Low |

---

## References

1. **IReD Paper**: Wagner, Palermo, Ambrus (2022), arXiv:2203.12608v2
   - Equation (29b), page 6: Correct form of J^μ with δₙₙ
   - Table III, page 11: δ_VV = 1 for N₁=4 truncation
   - Appendix B: General formulas for all coefficients

2. **Diagnostic Scripts**:
   - `verify_coupling_dimensions.py`: Automated dimensional checks
   - `investigate_tau_V_pi.py`: τ_Vπ mismatch investigation

3. **Previous Analysis**:
   - `coupling_coefficient_analysis.md`: Comprehensive 300-line analysis

---

## Sign-Off

**Analysis Complete**: 2025-10-18
**Approval Pending**: Code review after implementation
**Stage 1 Completion**: After all fixes verified

---

**CRITICAL NOTE**: Do NOT proceed with implementation until this plan is reviewed and approved. These are fundamental changes to the coupling structure that affect all simulations using IReD coefficients.

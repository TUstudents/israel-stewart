# Coupling Coefficient Dimensional Analysis

**Date**: 2025-10-18
**Status**: 🔴 CRITICAL ISSUES FOUND
**Priority**: HIGH

## Executive Summary

Systematic dimensional analysis of IReD coupling coefficients reveals **three critical inconsistencies** in the current implementation of `relaxation.py`. The temperature scaling factors applied to mixed-unit coefficients (λ_πV, λ_Vπ, τ_Vπ) do not match the dimensions required by the relaxation equations.

## Background

The IReD paper (Wagner et al. 2022) defines transport coefficients as dimensionless ratios involving relaxation times and thermodynamic quantities. When using these coefficients in the Israel-Stewart relaxation equations, temperature scaling factors must be applied to achieve dimensional consistency.

## Findings

### 1. λ_πV (Shear-Diffusion Coupling) - EXTRA T FACTOR

**Location**: `israel_stewart/equations/relaxation.py:354-371`

**IReD Definition** (Table III, N₂=3):
```
λ_πV = 0.20890 × τ_π/β
```
where β = 1/T. This gives **[λ_πV] = dimensionless**.

**Usage in Relaxation Equation**:
```python
# Line 365:
diffusion_term = (
    self.coeffs.lambda_pi_V
    * temperature[..., np.newaxis, np.newaxis]  # Multiplies by T
    * 0.5
    * (outer_product + np.swapaxes(outer_product, -1, -2))
)
```

**Dimensional Analysis**:
- Term contributes to: dπ^μν/dτ → [dπ^μν/dτ] = GeV⁴
- Components: [V^μ] = GeV³, [∇^ν(μ/T)] = GeV
- Required: [λ_πV × T × V × ∇(μ/T)] = GeV⁴
- Therefore: [λ_πV × T] = GeV⁴/(GeV³ × GeV) = **dimensionless**
- So: [λ_πV] = **GeV⁻¹** (must cancel the T factor)

**Problem**:
- IReD gives: dimensionless
- Need: GeV⁻¹
- **Current implementation adds EXTRA T factor!**

**Fix**: **REMOVE** temperature multiplication at line 367
```python
# WRONG (current):
diffusion_term = self.coeffs.lambda_pi_V * temperature[...] * ...

# CORRECT:
diffusion_term = self.coeffs.lambda_pi_V * ...
```

---

### 2. λ_Vπ (Diffusion-Shear Coupling) - MISSING T FACTOR

**Location**: `israel_stewart/equations/relaxation.py:465-478`

**IReD Definition** (Table III, N₁=4):
```
λ_Vπ = 0.069240 × β × τ_V
```
This gives **[λ_Vπ] = GeV⁻¹ × GeV⁻¹ = GeV⁻²**.

**Usage in Relaxation Equation**:
```python
# Line 473:
shear_diffusion_term = (
    self.coeffs.lambda_V_pi
    * temperature[..., np.newaxis]  # Multiplies by T
    * optimized_einsum("...ij,...j->...i", pi_munu, nabla_mu_over_T)
)
```

**Dimensional Analysis**:
- Term contributes to: dV^μ/dτ → [dV^μ/dτ] = GeV⁴
- Components: [π^μν] = GeV³, [∇_ν(μ/T)] = GeV
- Required: [λ_Vπ × T × π × ∇(μ/T)] = GeV⁴
- Therefore: [λ_Vπ × T] = GeV⁴/(GeV³ × GeV) = **dimensionless**
- So: [λ_Vπ] = **GeV⁻¹** (needs T to cancel)

**Problem**:
- IReD gives: GeV⁻²
- Need: GeV⁻¹
- After T multiplication: GeV⁻² × GeV = GeV⁻¹ ✓... wait, this IS GeV⁻¹!
- But we need **dimensionless** after T multiplication
- **Current implementation MISSING ADDITIONAL T factor!**

**Fix**: Multiply by **T²** instead of T at line 475
```python
# WRONG (current):
shear_diffusion_term = self.coeffs.lambda_V_pi * temperature[...] * ...

# CORRECT:
shear_diffusion_term = self.coeffs.lambda_V_pi * (temperature[...]**2) * ...
```

---

### 3. τ_Vπ (Diffusion-Expansion Coupling) - SEVERE MISMATCH

**Location**: `israel_stewart/equations/relaxation.py:459-463`

**IReD Definition** (Table III, N₁=4):
```
τ_Vπ = 0.0071692 × β × τ_V / P
```
where P is pressure. This gives **[τ_Vπ] = GeV⁻¹ × GeV⁻¹ / GeV⁴ = GeV⁻⁶**.

**Usage in Relaxation Equation**:
```python
# Line 462:
expansion_term = -self.coeffs.tau_V_pi * V_mu * theta[..., np.newaxis]
```
(No temperature multiplication!)

**Dimensional Analysis**:
- Term contributes to: dV^μ/dτ → [dV^μ/dτ] = GeV⁴
- Components: [V^μ] = GeV³, [θ] = GeV
- Required: [τ_Vπ × V × θ] = GeV⁴
- Therefore: [τ_Vπ] = GeV⁴/(GeV³ × GeV) = **dimensionless**

**Problem**:
- IReD gives: **GeV⁻⁶**
- Need: **dimensionless**
- **Mismatch of 6 powers of energy!**
- Would need: τ_Vπ × T⁶ ≈ 66.3 × (0.4)⁶ ≈ 0.27

**Critical Observation**:
The symbolic equation at `relaxation.py:175` uses:
```python
diffusion_nonlinear = lambda_V_pi × π × ∇μ - tau_V × V × θ
```
It uses `tau_V` (the relaxation time, GeV⁻¹), NOT `tau_V_pi` (the IReD coefficient, GeV⁻⁶)!

**But even this doesn't work**: [τ_V × V × θ] = GeV⁻¹ × GeV³ × GeV = GeV³, not GeV⁴!

**Possible Resolutions**:
1. **Wrong coefficient**: τ_Vπ from IReD is NOT meant for the expansion term
2. **Missing normalization**: Term should be `-(τ_Vπ/some_quantity) × V × θ`
3. **Different form**: Expansion term has different structure in IReD formulation
4. **Formula error**: IReD formula has typo or missing context

**Status**: **REQUIRES INVESTIGATION** - Check IReD paper Table III context and equation derivations

---

## Numerical Example

Using T = 0.4 GeV, σ = 1.0 fm²:

| Coefficient | IReD Value | IReD Units | Current Scaling | Result Units | Required Units | Status |
|-------------|------------|------------|-----------------|--------------|----------------|--------|
| λ_πV | 0.691 | dimensionless | ×T | GeV | dimensionless | ❌ EXTRA T |
| λ_Vπ | 1.798 | GeV⁻² | ×T | GeV⁻¹ | dimensionless | ❌ NEED T² |
| τ_Vπ | 66.3 | GeV⁻⁶ | (none) | GeV⁻⁶ | dimensionless | ❌ NEED T⁶ |

## Impact Assessment

### Current Code Behavior

The dimensional inconsistencies mean that the magnitudes of coupling terms are **incorrect by powers of temperature**:

1. **λ_πV term**: Off by factor of T = 0.4 (40% error at T=0.4 GeV)
2. **λ_Vπ term**: Off by factor of 1/T = 2.5 (2.5× error)
3. **τ_Vπ term**: Off by factor of T⁶ = 4.1×10⁻³ (240× error!)

### Affected Tests

- **Stage 1 Unit Tests**: 26/29 passing (3 failures likely related)
- **IReD Coefficient Tests**: Unknown impact
- **Spectral Solver Tests**: May have compensating errors
- **Benchmark Validations**: Need re-evaluation after fixes

## Recommended Actions

### Immediate (Critical Priority)

1. ✅ **Verify IReD paper context** for τ_Vπ usage
   - Check Table III notes
   - Review equation derivations
   - Confirm expansion term form

2. **Fix λ_πV scaling** (straightforward)
   - Remove T multiplication at `relaxation.py:367`
   - Update docstring to clarify

3. **Fix λ_Vπ scaling** (straightforward)
   - Change T → T² at `relaxation.py:475`
   - Update docstring

### Short-Term (After Investigation)

4. **Resolve τ_Vπ issue** (requires paper analysis)
   - Determine correct coefficient for expansion term
   - Apply proper normalization
   - Add detailed documentation

5. **Update all affected tests**
   - Recalculate expected values
   - Verify numerical accuracy
   - Check benchmark agreement

6. **Add dimensional analysis tests**
   - Create `test_dimensional_consistency.py`
   - Validate all coupling terms
   - Prevent future regressions

### Long-Term

7. **Document unit conventions**
   - Create comprehensive guide in `docs/`
   - Add inline dimensional analysis
   - Include worked examples

8. **Refactor coefficient handling**
   - Consider dimensionful wrapper classes
   - Automatic unit conversion
   - Type hints for physical quantities

## References

1. Wagner, Palermo, Ambrus (2022), "IReD: Inverse-Reynolds-Dominance approach to relativistic dissipative hydrodynamics", arXiv:2203.12608v2
2. `docs/IReD.pdf` - Original paper with Table III
3. `israel_stewart/equations/ired_simple.py` - Coefficient implementations
4. `israel_stewart/equations/relaxation.py` - Usage in relaxation equations
5. `verify_coupling_dimensions.py` - Diagnostic script (this analysis)
6. `investigate_tau_V_pi.py` - τ_Vπ detailed investigation

## Verification Scripts

Created diagnostic tools:
- `verify_coupling_dimensions.py` - Comprehensive dimensional check
- `investigate_tau_V_pi.py` - Deep dive into τ_Vπ mismatch

Run with:
```bash
uv run python verify_coupling_dimensions.py
uv run python investigate_tau_V_pi.py
```

---

**Prepared by**: Claude (Stage 1 validation)
**Reviewed by**: (Pending)
**Approval**: (Pending IReD paper confirmation)

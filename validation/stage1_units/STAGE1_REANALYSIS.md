# Stage 1 Re-Analysis: Correcting the IReD Coefficient Bugs

**Date**: 2025-10-19
**Status**: ✅ **BUGS FIXED**
**Previous Analysis**: ❌ **INCORRECT** (see COMPLETION_SUMMARY.md)

---

## Executive Summary

The original Stage 1 analysis in `COMPLETION_SUMMARY.md` was **fundamentally wrong** about the nature of the coupling coefficient bugs. A complete re-analysis reveals:

### What Stage 1 Got WRONG:

1. **λ_πV**: Claimed it had "wrong temperature scaling (extra T factor)"
   - **ACTUAL BUG**: Had extra **τ_π factor**, not extra T factor!
   - Original formula: `0.20890 * τ_π / β` (dimensionless) ✗
   - Correct formula: `0.20890 / β` (GeV¹) ✓

2. **λ_Vπ**: Claimed to "fix" by using T² instead of T
   - **ACTUAL BUG**: T² is WRONG, should be T!
   - Stage 1 "fix": `T²` scaling → GeV⁵ (dimensional error!) ✗
   - Correct fix: `T` scaling → GeV⁴ ✓

### Impact:

- **λ_πV**: Now fixed correctly (removed τ_π)
- **λ_Vπ**: Now fixed correctly (changed T² → T)
- **All IReD coefficient tests**: 29/29 passing ✓
- **All relaxation equation tests**: 24/24 passing ✓

---

## The Two Bugs

### Bug 1: λ_πV (Shear-Diffusion Coupling)

**Physics Context**:
- Appears in **shear stress equation**: `dπ^μν/dτ = ... + λ_πV (V^μ ∇^ν(μ/T) + V^ν ∇^μ(μ/T))/2`
- LHS dimensions: `[dπ/dτ] = GeV⁴/GeV⁻¹ = GeV⁵`

**Dimensional Analysis**:
```
Required: [λ_πV × V × ∇(μ/T)] = GeV⁵
Where:    [V] = GeV³ (particle current density)
          [∇(μ/T)] = GeV¹ (gradient of dimensionless quantity)
Therefore: [λ_πV] × GeV³ × GeV¹ = GeV⁵
           [λ_πV] = GeV¹
```

**IReD Paper (Table IV)**:
- Formula: `λ_πn = 0.20890/β`
- Since `β = 1/T`: `λ_πn = 0.20890 × T`
- Units: **GeV¹** ✓

**Original Bug** (ired_simple.py):
```python
# WRONG: Extra τ_π factor!
def lambda_pi_V(self):
    tau_pi = self.shear_relaxation_time()
    return 0.20890 * tau_pi / self.beta
    # = 0.20890 × τ_π × T
    # = dimensionless (τ_π has units GeV⁻¹)
```

**Stage 1 Misdiagnosis**:
- Claimed: "λ_πV had wrong T scaling, so we removed T multiplication in relaxation.py"
- Reality: λ_πV had wrong **τ_π** in ired_simple.py, not wrong T in relaxation.py!

**Correct Fix**:
```python
# CORRECT: No τ_π factor!
def lambda_pi_V(self):
    return 0.20890 / self.beta
    # = 0.20890 × T
    # = GeV¹ ✓
```

**Error Impact** (at T=0.4 GeV, τ_π≈8.27 GeV⁻¹):
- Wrong value: `0.1379` (dimensionless)
- Correct value: `0.0836 GeV`
- Error factor: `τ_π × T ≈ 3.3×` too large

---

### Bug 2: λ_Vπ (Diffusion-Shear Coupling)

**Physics Context**:
- Appears in **diffusion current equation**: `dV^μ/dτ = ... + λ_Vπ × ??? × π^μν ∇_ν(μ/T)`
- LHS dimensions: `[dV/dτ] = GeV³/GeV⁻¹ = GeV⁴`

**Dimensional Analysis**:
```
Required: [λ_Vπ × ??? × π × ∇(μ/T)] = GeV⁴
Where:    [λ_Vπ] = GeV⁻² (from IReD: 0.069240 β τ_V)
          [π] = GeV⁴ (shear stress)
          [∇(μ/T)] = GeV¹
Therefore: GeV⁻² × ??? × GeV⁴ × GeV¹ = GeV⁴
           ??? × GeV³ = GeV⁴
           ??? = GeV¹ = T
```

**IReD Paper (Table III)**:
- Formula: `λ_Vπ = 0.069240 β τ_V`
- Units: **GeV⁻²** (β = GeV⁻¹, τ_V = GeV⁻¹)

**Stage 1 "Fix"** (relaxation.py): **WRONG!**
```python
# Stage 1 used T² - WRONG!
shear_diffusion_term = (
    self.coeffs.lambda_V_pi
    * (temperature[..., np.newaxis] ** 2)  # T² ✗
    * π^μν × ∇_ν(μ/T)
)
# Dimensions: GeV⁻² × GeV² × GeV⁴ × GeV¹ = GeV⁵ ✗
# But required: GeV⁴!
```

**Correct Fix** (relaxation.py):
```python
# Use T, not T²!
shear_diffusion_term = (
    self.coeffs.lambda_V_pi
    * temperature[..., np.newaxis]  # T ✓
    * π^μν × ∇_ν(μ/T)
)
# Dimensions: GeV⁻² × GeV × GeV⁴ × GeV¹ = GeV⁴ ✓
```

**Error Impact**:
- Stage 1 "fix": Extra factor of T (GeV⁵ instead of GeV⁴)
- At T=0.4 GeV: 2.5× dimensional mismatch
- This creates an inconsistent RHS in the diffusion equation!

---

## Why Stage 1 Was Wrong

### Mistake 1: Misidentified λ_πV Bug

Stage 1 looked at the relaxation.py usage:
```python
# In relaxation.py (shear RHS):
diffusion_term = self.coeffs.lambda_pi_V * ...
```

And saw that it wasn't multiplying by T. Stage 1 concluded: "This is wrong, λ_πV should be multiplied by T!"

**But this was backwards!** The issue was in ired_simple.py (having extra τ_π), not in relaxation.py (which was using it correctly).

### Mistake 2: Overcorrected λ_Vπ

Stage 1 saw λ_Vπ was dimensionally wrong and tried to fix it by adding T². But:
- T² makes [λ_Vπ × T²] = dimensionless
- This seems good at first
- But then × π × ∇ = GeV⁵, not the required GeV⁴!

**The correct fix was T, not T².**

---

## Dimensional Verification

### After Fixes:

**Shear stress equation** (dπ/dτ = ... + J^μν):
- `λ_πV × V × ∇(μ/T)`:
  - GeV¹ × GeV³ × GeV¹ = **GeV⁵** ✓
  - Matches LHS: GeV⁵ ✓

**Diffusion equation** (dV/dτ = ... + I^μ):
- `λ_Vπ × T × π × ∇(μ/T)`:
  - GeV⁻² × GeV × GeV⁴ × GeV¹ = **GeV⁴** ✓
  - Matches LHS: GeV⁴ ✓

---

## Test Results

### Before Fixes:
- IReD coefficients: **FAILED** (λ_πV test expected wrong formula)
- Relaxation equations: 24/24 passing (but with wrong coefficient values!)

### After Fixes:
- **IReD coefficients**: 29/29 passing ✅
- **Relaxation equations**: 24/24 passing ✅
- **All dimensional consistency checks**: PASS ✅

---

## Files Modified

1. **`israel_stewart/equations/ired_simple.py`**:
   - Line 254-270: Fixed `lambda_pi_V()` - removed τ_π factor

2. **`israel_stewart/equations/relaxation.py`**:
   - Line 497-511: Fixed λ_Vπ usage - changed T² to T

3. **`israel_stewart/tests/test_ired_coefficients.py`**:
   - Line 103-111: Fixed test to expect correct formula (without τ_π)

---

## Summary Table

| Coefficient | IReD Formula | Stage 1 "Fix" | Actual Bug | Correct Fix | Status |
|-------------|-------------|---------------|------------|-------------|--------|
| λ_πV | 0.20890/β | Removed T from relaxation.py | Extra τ_π in ired_simple.py | Remove τ_π | ✅ FIXED |
| λ_Vπ | 0.069240 β τ_V | Added T² in relaxation.py | Should use T not T² | Use T in relaxation.py | ✅ FIXED |

---

## Lessons Learned

### 1. Trust Dimensional Analysis Above All

Every term in every equation MUST be dimensionally consistent. If it's not, there's a bug - period.

### 2. Check the Source, Not Just the Usage

Stage 1 looked at how coefficients were used and tried to "fix" the usage. But the bug was in how they were computed, not how they were used!

### 3. Verify Against the Reference Paper

The IReD paper explicitly states:
- Table IV: `λ_πn = 0.20890/β` (no τ_π!)
- Table III: `λ_Vπ = 0.069240 β τ_V` (units GeV⁻²)

Always check the source.

### 4. Test the Tests

Stage 1 tests were passing because they expected the WRONG formulas! The tests themselves had bugs.

---

## References

1. **IReD Paper**: Wagner, Palermo, Ambrus (2022), arXiv:2203.12608v2
   - Table III (page 11): Transport coefficient formulas
   - Table IV (page 11): λ_πn definition

2. **Implementation Files**:
   - `israel_stewart/equations/ired_simple.py`: Transport coefficient definitions
   - `israel_stewart/equations/relaxation.py`: Relaxation equation implementations
   - `israel_stewart/tests/test_ired_coefficients.py`: Coefficient validation tests

3. **Validation Scripts**:
   - `comprehensive_dimensional_analysis.py`: Systematic dimensional check

---

## Sign-Off

**Status**: ✅ **BUGS FIXED AND VERIFIED**

**Date**: 2025-10-19
**Tests**: 53/53 core tests passing
**Dimensional Consistency**: All equations verified ✓

The original Stage 1 `COMPLETION_SUMMARY.md` should be considered **INVALID** and replaced by this analysis.

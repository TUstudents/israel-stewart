# Stage 1: Units & Dimensional Analysis

**Status**: 🟡 90% Complete (26/29 tests passing)

**Priority**: HIGH (blocks Stage 3 equation validation)

## Goal

Verify that all physical quantities in the Israel-Stewart implementation have correct dimensions and that unit conversions are mathematically accurate.

## Why This Stage Matters

**"If units are wrong, everything downstream is wrong."**

Dimensional inconsistencies propagate through the entire system:
- Wrong transport coefficients → incorrect physics
- Mixed unit systems → subtle scaling errors
- Missing factors → numerical instability

## Acceptance Criteria

- ✅ All unit conversions accurate to < 10⁻¹⁰
- ⚠️ All coupling coefficients dimensionally consistent
- ✅ Temperature scaling formulas correct
- ✅ Natural units (ℏ=c=k_B=1) implemented correctly

## Current Status

### ✅ Completed

1. **Unit Conversion Infrastructure** (26/26 tests passing)
   - fm/c ↔ GeV⁻¹ conversion: error < 2×10⁻¹⁶
   - GeV⁻¹ ↔ seconds conversion: verified
   - Mean free path calculation: fixed (added (ℏc)³ factor)
   - All time-dependent methods support time_unit parameter

2. **First-Order Coefficients** (dimensional verification complete)
   - η (shear viscosity): [GeV³] ✓
   - ζ (bulk viscosity): [GeV³] ✓
   - D (diffusion coefficient): [GeV²] ✓
   - τ_π, τ_Π, τ_V (relaxation times): [GeV⁻¹] ✓

### ⚠️ In Progress

3. **Mixed-Unit Coupling Coefficients** (needs verification)
   - λ_πV (shear-diffusion coupling): Expected [GeV], actual [dimensionless]
   - λ_Vπ (diffusion-shear coupling): Expected [GeV⁻¹], actual [dimensionless]
   - Temperature scaling added: multiply by T for dimensional consistency
   - **Blocker**: Need to verify against IReD paper Table III definition

### ❌ TODO

4. **IReD Paper Verification**
   - Read `docs/IReD.pdf` Appendix B Table III
   - Check if λ_πV defined as dimensionless with implicit T scaling
   - Verify exact form of coupling terms in relaxation equations
   - Document normalization convention

## Test Scripts

### Existing (from /tmp, to be moved)

- `unit_audit.py` - Comprehensive unit conversion tests
- `check_lambda_pi_V_usage.py` - Dimensional analysis of coupling terms
- `unit_analysis_detailed.py` - Detailed unit breakdown

### To Be Created

- `verify_time_conversions.py` - Test all time unit conversions
- `check_coefficient_dimensions.py` - Validate all transport coefficients
- `verify_temperature_scaling.py` - Test T-dependent formulas

## Key Findings

### 1. Unit Conversion Accuracy ✅

All conversions verified to machine precision:
```python
τ_π (fm/c) / ℏc = τ_π (GeV⁻¹)
1.631643 fm/c / 0.197 GeV·fm = 8.268728 GeV⁻¹

Error: 2.22×10⁻¹⁶ (machine precision) ✓
```

### 2. Mean Free Path Fix ✅

**Before**:
```python
λ_mfp = 1 / (n × σ)  # WRONG: missing (ℏc)³
```

**After**:
```python
HBARC = 0.197  # GeV·fm
λ_mfp = (HBARC**3) / (n × σ)  # Correct: [GeV·fm]³ / ([GeV³] × [fm²])
```

### 3. Coupling Term Dimensional Issue ⚠️

**From relaxation equations** (`relaxation.py:200-212`):
```python
dπ^μν/dτ = ... + λ_πV × (V^μ ∇^ν(μ_B/T) + ...) / 2
```

**Dimensional analysis**:
```
[dπ^μν/dτ] = GeV⁵ (required)
[V^μ] = GeV³ (particle current density)
[∇^ν(μ_B/T)] = GeV (gradient of dimensionless quantity)
[V^μ ∇^ν(μ_B/T)] = GeV⁴

For consistency: [λ_πV] = GeV⁵/GeV⁴ = GeV (EXPECTED)
```

**From IReD formula** (`ired_simple.py:305-308`):
```python
λ_πV = 0.20890 × τ_π / β
[λ_πV] = [τ_π]/[β] = GeV⁻¹ / GeV⁻¹ = dimensionless (ACTUAL)
```

**Resolution** (implemented but needs verification):
```python
# Multiply by temperature for dimensional consistency
λ_πV_physical = λ_πV_IReD × T
[λ_πV_physical] = dimensionless × GeV = GeV ✓
```

## Next Steps

1. **Verify IReD paper convention** (1 day)
   - Read Table III definition of λ_πV
   - Check if coefficients are meant to be dimensionless
   - Document expected usage in solver

2. **Complete dimensional audit** (0.5 days)
   - Verify all second-order couplings
   - Check ℓ_Vπ, τ_Vπ dimensions
   - Validate solver usage matches theory

3. **Document findings** (0.5 days)
   - Write `results/dimensional_analysis.md`
   - Update unit conversion table
   - Add to VALIDATION_ROADMAP

**Total time estimate**: 2 days

## References

- Unit audit summary: `results/unit_audit_summary.md` (to be moved from /tmp)
- IReD paper: `../docs/IReD.pdf` Appendix B Table III
- Implementation: `israel_stewart/equations/ired_simple.py`
- Usage: `israel_stewart/equations/relaxation.py`

## Success Metrics

**Before** (2025-10-18):
- 26/26 conversion tests passing ✓
- Mean free path bug fixed ✓
- Dimensional inconsistency identified ⚠️

**Target**:
- 29/29 all tests passing ✓
- All coefficients dimensionally correct ✓
- IReD paper convention documented ✓

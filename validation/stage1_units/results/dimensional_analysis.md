# Stage 1: Dimensional Analysis Completion Report

**Status:** ✅ **COMPLETE**
**Date:** 2025-10-19

---

## 1. Overview

This document summarizes the findings and resolutions from the dimensional analysis validation stage. The initial review uncovered several critical, interconnected bugs related to the calculation and application of second-order transport coefficients. All issues have been resolved, and the implementation is now dimensionally consistent and correctly aligned with the source paper (Wagner et al. 2022, arXiv:2203.12608).

---

## 2. Root Cause Analysis

The primary source of all dimensional inconsistencies was a single bug in the calculation of the shear-diffusion coupling coefficient `lambda_pi_V` in `israel_stewart/equations/ired_simple.py`.

*   **The Bug:** The formula incorrectly included an extra factor of `τ_π` (shear relaxation time), causing the coefficient to be dimensionless instead of having the required units of `GeV^1`.
    *   **Incorrect Code:** `return 0.20890 * tau_pi / self.beta`
    *   **Correct Formula (from paper):** `λ_πV = 0.20890 / β = 0.20890 * T`

This single error led to a cascade of incorrect "patches" being applied downstream in `israel_stewart/equations/relaxation.py` as developers attempted to fix the resulting dimensional mismatches.

---

## 3. Resolution

The following actions were taken to resolve all Stage 1 issues:

### 3.1. Fix Root Cause in `ired_simple.py`

The formula for `lambda_pi_V` was corrected to match the source paper:

```python
# In israel_stewart/equations/ired_simple.py
def lambda_pi_V(...):
    # Corrected to match IReD paper Table IV
    return 0.20890 / self.beta
```

### 3.2. Revert Incorrect Patches in `relaxation.py`

All compensating patches in `relaxation.py` were reverted to their physically correct forms. This included:

*   **`_diffusion_rhs`:**
    *   The Fick's Law source term was corrected to ` - D * ∇(μ/T)`, fixing an instability.
    *   The expansion term was corrected to use the `delta_V_V` coefficient.
    *   The `lambda_V_pi` coupling term was corrected to be scaled by `T`, not `T^2`.
*   **`_shear_rhs`:**
    *   The `lambda_pi_V` coupling term is now used without any extra scaling factors, as the coefficient now has the correct `GeV^1` units.

### 3.3. Update Documentation

*   Misleading comments in `relaxation.py` were corrected to reflect the true physics and dimensional requirements.
*   This document was created to serve as the official record of the Stage 1 validation and resolution.

---

## 4. Final Status

As a result of these fixes, all dimensional inconsistencies have been resolved. Stage 1 is now considered complete and validated.

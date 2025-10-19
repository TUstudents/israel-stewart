"""
Comprehensive dimensional analysis of ALL IReD transport coefficients.

This script systematically checks every coefficient against:
1. The IReD paper formulas (Table III-IV)
2. The dimensional requirements from relaxation equations
3. The actual implementation in ired_simple.py

Reference: Wagner, Palermo, Ambrus (2022), arXiv:2203.12608v2
"""

import numpy as np

from israel_stewart.equations.ired_simple import HardSphereIReD

print("=" * 80)
print("COMPREHENSIVE DIMENSIONAL ANALYSIS OF IRED COEFFICIENTS")
print("=" * 80)

# Create model at typical RHIC temperature
T = 0.4  # GeV
sigma = 1.0  # fm²
model = HardSphereIReD(temperature=T, cross_section=sigma)

print("\nPhysical parameters:")
print(f"  T = {model.temperature:.3f} GeV")
print(f"  β = 1/T = {model.beta:.3f} GeV⁻¹")
print(f"  P = {model.pressure:.6e} GeV⁴")
print(f"  ε+P = {model.energy_density + model.pressure:.6e} GeV⁴")

# Get relaxation times for reference
tau_pi_nat = model.shear_relaxation_time(time_unit="natural")
tau_V_nat = model.diffusion_relaxation_time(time_unit="natural")

print("\nRelaxation times:")
print(f"  τ_π = {tau_pi_nat:.3f} GeV⁻¹")
print(f"  τ_V = {tau_V_nat:.3f} GeV⁻¹")

print("\n" + "=" * 80)
print("SHEAR STRESS COEFFICIENTS (π^μν equation)")
print("=" * 80)

print("\nRelaxation equation: dπ^μν/dτ = -π^μν/τ_π - 2ησ^μν + J^μν")
print("LHS dimensions: [dπ/dτ] = GeV⁴/GeV⁻¹ = GeV⁵")

shear_coeffs = {
    "τ_ππ": {
        "formula": "1.6944 τ_π",
        "ired_value": model.tau_pi_pi(time_unit="natural"),
        "expected_units": "GeV⁻¹",
        "equation_term": "τ_ππ π^μ_ρ π^ρν / τ_π",
        "term_dimensions": "[τ_ππ π² / τ_π] = GeV⁻¹ × GeV⁸ / GeV⁻¹ = GeV⁸ ≠ GeV⁵",
        "note": "This is R term (quadratic in dissipative), not J term",
    },
    "λ_πV": {
        "formula": "0.20890/β",
        "ired_value": model.lambda_pi_V(),
        "expected_units": "GeV¹",
        "equation_term": "λ_πV (V^μ ∇^ν(μ/T) + V^ν ∇^μ(μ/T))/2",
        "term_dimensions": "[λ_πV V ∇(μ/T)] = GeV¹ × GeV³ × GeV¹ = GeV⁵ ✓",
        "note": "Shear-diffusion coupling (J term)",
    },
    "δ_ππ": {
        "formula": "4/3",
        "ired_value": model.delta_pi_pi(),
        "expected_units": "dimensionless",
        "equation_term": "δ_ππ π^μν θ",
        "term_dimensions": "[δ_ππ π θ] = dimensionless × GeV⁴ × GeV¹ = GeV⁵ ✓",
        "note": "Shear expansion coupling (J term)",
    },
    "ℓ_πV": {
        "formula": "-0.56014/β",
        "ired_value": model.ell_pi_V(),
        "expected_units": "GeV¹",
        "equation_term": "ℓ_πV ∇^μ∇^ν(μ/T)",
        "term_dimensions": "[ℓ_πV ∇∇(μ/T)] = GeV¹ × GeV² = GeV³ ≠ GeV⁵",
        "note": "NOT IMPLEMENTED (needs 2nd derivative infrastructure)",
    },
    "τ_πV": {
        "formula": "-0.56014/(βP)",
        "ired_value": model.tau_pi_V(),
        "expected_units": "GeV⁻³",
        "equation_term": "τ_πV π^μν F_ν",
        "term_dimensions": "[τ_πV π F] = GeV⁻³ × GeV⁴ × GeV⁴ = GeV⁵ ✓",
        "note": "NOT IMPLEMENTED (needs pressure gradient F_ν)",
    },
}

for name, info in shear_coeffs.items():
    print(f"\n{name}:")
    print(f"  IReD formula: {info['formula']}")
    print(f"  Value: {info['ired_value']:.6e}")
    print(f"  Expected units: {info['expected_units']}")
    print(f"  Equation term: {info['equation_term']}")
    print(f"  Dimensions: {info['term_dimensions']}")
    print(f"  Note: {info['note']}")

print("\n" + "=" * 80)
print("DIFFUSION CURRENT COEFFICIENTS (V^μ equation)")
print("=" * 80)

print("\nRelaxation equation: dV^μ/dτ = -V^μ/τ_V - κΔ^μν∇_ν(μ/T) + I^μ")
print("LHS dimensions: [dV/dτ] = GeV³/GeV⁻¹ = GeV⁴")

diffusion_coeffs = {
    "δ_VV": {
        "formula": "1",
        "ired_value": model.delta_V_V(),
        "expected_units": "dimensionless",
        "equation_term": "δ_VV V^μ θ",
        "term_dimensions": "[δ_VV V θ] = dimensionless × GeV³ × GeV¹ = GeV⁴ ✓",
        "note": "Diffusion expansion coupling (FIXED in Stage 1)",
    },
    "λ_VV": {
        "formula": "0.89501 τ_V",
        "ired_value": model.lambda_V_V(time_unit="natural"),
        "expected_units": "GeV⁻¹",
        "equation_term": "λ_VV V_μ V^μ V^ν / (D τ_V)",
        "term_dimensions": "[λ_VV V³ / (D τ_V)] = GeV⁻¹ × GeV⁹ / (GeV³ × GeV⁻¹) = GeV⁴ ✓",
        "note": "Diffusion-diffusion coupling (R term, quadratic)",
    },
    "λ_Vπ": {
        "formula": "0.069240 β τ_V",
        "ired_value": model.lambda_V_pi(time_unit="natural"),
        "expected_units": "GeV⁻²",
        "equation_term": "λ_Vπ T² π^μν ∇_ν(μ/T)",
        "term_dimensions": "[λ_Vπ T² π ∇(μ/T)] = GeV⁻² × GeV² × GeV⁴ × GeV¹ = GeV⁵ ≠ GeV⁴!",
        "note": "BUG FOUND: T² scaling gives GeV⁵ but should be GeV⁴!",
    },
    "ℓ_Vπ": {
        "formula": "0.028677 β τ_V",
        "ired_value": model.ell_V_pi(time_unit="natural"),
        "expected_units": "GeV⁻²",
        "equation_term": "ℓ_Vπ ∇^μ∇^ν(μ/T)",
        "term_dimensions": "NOT IMPLEMENTED",
        "note": "Needs 2nd derivative infrastructure",
    },
    "τ_Vπ": {
        "formula": "0.0071692 β τ_V / P",
        "ired_value": model.tau_V_pi(time_unit="natural"),
        "expected_units": "GeV⁻⁶",
        "equation_term": "τ_Vπ π^μν F_ν",
        "term_dimensions": "NOT IMPLEMENTED",
        "note": "Needs pressure gradient F_ν",
    },
}

for name, info in diffusion_coeffs.items():
    print(f"\n{name}:")
    print(f"  IReD formula: {info['formula']}")
    print(f"  Value: {info['ired_value']:.6e}")
    print(f"  Expected units: {info['expected_units']}")
    print(f"  Equation term: {info['equation_term']}")
    print(f"  Dimensions: {info['term_dimensions']}")
    print(f"  Note: {info['note']}")

print("\n" + "=" * 80)
print("DIMENSIONAL CONSISTENCY CHECK")
print("=" * 80)

issues = []

# Check λ_πV
lambda_pi_V_value = model.lambda_pi_V()
expected_lambda_pi_V = 0.20890 * T
if not np.isclose(lambda_pi_V_value, expected_lambda_pi_V):
    issues.append(
        f"λ_πV: got {lambda_pi_V_value:.6e}, expected {expected_lambda_pi_V:.6e} (0.20890 × T)"
    )

# Check λ_Vπ usage
# The term in relaxation.py is: lambda_V_pi * T² * π * ∇(μ/T)
# This gives: GeV⁻² × GeV² × GeV⁴ × GeV¹ = GeV⁵
# But dV/dτ has units GeV⁴, not GeV⁵!
print("\n❌ CRITICAL BUG FOUND: λ_Vπ temperature scaling")
print("   Current in relaxation.py: λ_Vπ × T² × π × ∇(μ/T)")
print("   Dimensions: GeV⁻² × GeV² × GeV⁴ × GeV¹ = GeV⁵")
print("   Required: GeV⁴")
print("   ERROR: Extra factor of GeV!")
print("   FIX: Should use T instead of T², or λ_Vπ has wrong formula")

issues.append("λ_Vπ: T² scaling gives GeV⁵ but RHS should be GeV⁴")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if issues:
    print(f"\n❌ {len(issues)} DIMENSIONAL ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
else:
    print("\n✅ All coefficients dimensionally consistent!")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("""
1. λ_πV: ✅ FIXED - removed incorrect τ_π factor
2. λ_Vπ: ❌ NEEDS FIX - check IReD paper definition and usage in relaxation.py
3. Stage 1 COMPLETION_SUMMARY.md: ❌ INCORRECT - claimed λ_πV had wrong T scaling
   (actually had wrong τ_π factor, different bug!)
""")

#!/usr/bin/env python3
"""
Check how lambda_pi_V is used in relaxation equations and verify dimensional consistency.
"""

from israel_stewart.equations.ired_simple import HardSphereIReD
import numpy as np

print("=" * 80)
print("DIMENSIONAL ANALYSIS: λ_πV COEFFICIENT")
print("=" * 80)

# Create IReD model
model = HardSphereIReD(temperature=0.4, cross_section=1.0)

# Get transport coefficients
print("\n📊 TRANSPORT COEFFICIENT VALUES:")
print("-" * 80)

tau_pi_fm = model.shear_relaxation_time(time_unit="fm/c")
tau_pi_nat = model.shear_relaxation_time(time_unit="natural")
lambda_pi_V_fm = model.lambda_pi_V(time_unit="fm/c")
lambda_pi_V_nat = model.lambda_pi_V(time_unit="natural")

print(f"τ_π (fm/c) = {tau_pi_fm:.6e} fm/c")
print(f"τ_π (natural) = {tau_pi_nat:.6e} GeV⁻¹")
print(f"λ_πV (fm/c) = {lambda_pi_V_fm:.6e} (claimed units: GeV⁻¹·fm/c)")
print(f"λ_πV (natural) = {lambda_pi_V_nat:.6e} (claimed units: GeV⁻²)")

print(f"\nβ = 1/T = {model.beta:.6e} GeV⁻¹")
print(f"T = {model.temperature:.6e} GeV")

print("\n🔬 CHECKING IReD FORMULA:")
print("-" * 80)
print("From ired_simple.py:")
print("  lambda_pi_V = 0.20890 * tau_pi / beta")

# Verify the formula
coeff_from_table = 0.20890
calculated_fm = coeff_from_table * tau_pi_fm / model.beta
calculated_nat = coeff_from_table * tau_pi_nat / model.beta

print(f"\nUsing fm/c units:")
print(f"  0.20890 × {tau_pi_fm:.6e} / {model.beta:.6e}")
print(f"  = {calculated_fm:.6e}")
print(f"  Matches model.lambda_pi_V(fm/c)? {np.isclose(calculated_fm, lambda_pi_V_fm)}")

print(f"\nUsing natural units:")
print(f"  0.20890 × {tau_pi_nat:.6e} / {model.beta:.6e}")
print(f"  = {calculated_nat:.6e}")
print(f"  Matches model.lambda_pi_V(natural)? {np.isclose(calculated_nat, lambda_pi_V_nat)}")

print("\n⚛️  DIMENSIONAL ANALYSIS:")
print("-" * 80)

print("\nExpected dimensions from relaxation equation:")
print("  dπ^μν/dτ = ... + λ_πV × (V^μ ∇^ν(μ_B/T) + V^ν ∇^μ(μ_B/T))/2")
print()
print("  [dπ^μν/dτ] = [π^μν]/[τ] = GeV⁴/GeV⁻¹ = GeV⁵")
print("  [V^μ] = GeV³ (particle current density)")
print("  [∇^ν(μ_B/T)] = GeV¹ (gradient of dimensionless quantity)")
print("  [V^μ ∇^ν(μ_B/T)] = GeV³ × GeV¹ = GeV⁴")
print()
print("  For dimensional consistency:")
print("  [λ_πV × V^μ ∇^ν] = GeV⁵")
print("  → [λ_πV] = GeV⁵/GeV⁴ = GeV¹")

print("\nActual dimensions from IReD formula:")
print("  λ_πV = 0.20890 × τ_π / β")
print("  [λ_πV] = [τ_π]/[β] = GeV⁻¹ / GeV⁻¹ = dimensionless")

print("\n🚨 DIMENSIONAL MISMATCH DETECTED!")
print("  Expected: [λ_πV] = GeV¹")
print("  From formula: [λ_πV] = dimensionless")
print("  Discrepancy: Factor of GeV¹")

print("\n💡 POSSIBLE RESOLUTIONS:")
print("-" * 80)

print("\n1. **Implicit Temperature Normalization**")
print("   The IReD paper might define λ_πV in units where it's already normalized by T.")
print("   If λ_πV_dimensional = λ_πV_IReD × T, then:")
print("   [λ_πV_dimensional] = dimensionless × GeV = GeV ✓")
print()
print(f"   Testing: λ_πV × T = {lambda_pi_V_nat:.6e} × {model.temperature:.6e}")
print(f"          = {lambda_pi_V_nat * model.temperature:.6e} GeV")

print("\n2. **Different Normalization Convention**")
print("   The IReD coefficient might be defined as λ̃_πV such that:")
print("   λ_πV_physical = λ̃_πV / (some thermodynamic scale)")
print()
print(f"   If normalized by ε+p = {model.energy_density + model.pressure:.6e} GeV⁴:")
print(f"   λ_πV / (ε+p) = {lambda_pi_V_nat:.6e} / {model.energy_density + model.pressure:.6e}")
print(f"                = {lambda_pi_V_nat / (model.energy_density + model.pressure):.6e}")

print("\n3. **Equation Formulation Difference**")
print("   The relaxation equation in the code might have a different form than expected.")
print("   For example, if there's an implicit factor of T⁴ or (ε+p) in the equation.")

print("\n4. **Check IReD Paper Definition**")
print("   Need to check IReD paper Table III to see exact definition of λ_πV.")
print("   The paper might use dimensionless coefficients with implicit scaling.")

print("\n" + "=" * 80)
print("RECOMMENDATION: Examine relaxation.py line 200-212 and IReD paper Table III")
print("to verify the exact form of the λ_πV term and its normalization convention.")
print("=" * 80)

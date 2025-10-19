#!/usr/bin/env -S uv run python
"""
Investigate τ_Vπ usage in IReD paper vs implementation.

The dimensional analysis shows τ_Vπ has units GeV⁻⁶ from the IReD formula,
but it needs to be dimensionless in the expansion term -τ_Vπ V^μ θ.

This script investigates whether:
1. The expansion term should use τ_V (relaxation time) instead of τ_Vπ
2. There's a missing normalization/conversion factor
3. The IReD formula has implicit temperature dependence
"""

from israel_stewart.equations.ired_simple import HardSphereIReD
import numpy as np

def main():
    """Investigate τ_Vπ dimensional mismatch."""
    model = HardSphereIReD(temperature=0.4, cross_section=1.0)

    # Get all relevant quantities
    T = model.temperature
    beta = model.beta
    tau_V = model.diffusion_relaxation_time(time_unit="natural")  # GeV⁻¹
    P = model.pressure  # GeV⁴
    tau_V_pi_formula = 0.0071692 * beta * tau_V / P  # GeV⁻⁶

    print("="*70)
    print("Investigation: τ_Vπ Dimensional Mismatch")
    print("="*70)

    print(f"\n1. Physical Parameters:")
    print(f"   T = {T:.4f} GeV")
    print(f"   β = 1/T = {beta:.4f} GeV⁻¹")
    print(f"   τ_V = {tau_V:.4e} GeV⁻¹")
    print(f"   P = {P:.4e} GeV⁴")

    print(f"\n2. IReD Formula Result:")
    print(f"   τ_Vπ = 0.0071692 × β × τ_V / P")
    print(f"        = 0.0071692 × {beta:.4f} × {tau_V:.4e} / {P:.4e}")
    print(f"        = {tau_V_pi_formula:.4e}")
    print(f"   Units: [τ_Vπ] = GeV⁻¹ × GeV⁻¹ / GeV⁴ = GeV⁻⁶")

    print(f"\n3. Required Dimensions for Expansion Term:")
    print(f"   Term: -τ_Vπ × V^μ × θ (contributes to dV^μ/dτ)")
    print(f"   [dV^μ/dτ] = GeV⁴")
    print(f"   [V^μ] = GeV³")
    print(f"   [θ] = GeV")
    print(f"   Required: [τ_Vπ] = GeV⁴ / (GeV³ × GeV) = GeV⁰ (dimensionless)")

    print(f"\n4. Mismatch Analysis:")
    print(f"   IReD gives: GeV⁻⁶")
    print(f"   Need: GeV⁰")
    print(f"   Missing factor: T⁶ = {T**6:.4e}")

    # Check if T^6 normalization makes sense
    tau_V_pi_normalized = tau_V_pi_formula * (T**6)
    print(f"\n5. With T⁶ Normalization:")
    print(f"   τ_Vπ × T⁶ = {tau_V_pi_formula:.4e} × {T**6:.4e}")
    print(f"            = {tau_V_pi_normalized:.4e} (dimensionless)")

    # Compare with τ_V directly
    print(f"\n6. Alternative: Use τ_V (Relaxation Time) Directly")
    print(f"   From symbolic equation (line 175): -tau_V × V_0 × theta")
    print(f"   This suggests: τ_V (GeV⁻¹), NOT τ_Vπ (GeV⁻⁶)")
    print(f"   ")
    print(f"   But wait! If expansion term is -τ_V × V × θ:")
    print(f"     [τ_V × V × θ] = GeV⁻¹ × GeV³ × GeV = GeV³")
    print(f"   This still doesn't match [dV/dτ] = GeV⁴!")

    print(f"\n7. Correct Interpretation:")
    print(f"   The expansion term might have different normalization.")
    print(f"   In conformal case (P = ε/3), there's often a factor of P in the term.")
    print(f"   ")
    print(f"   Consider: -(τ_Vπ/τ_V) × V × θ")
    print(f"     where τ_Vπ/τ_V = (0.0071692 × β × τ_V / P) / τ_V")
    print(f"                    = 0.0071692 × β / P")
    print(f"                    = {0.0071692 * beta / P:.4e} GeV⁻⁵")
    print(f"   Still doesn't work!")

    print(f"\n8. Possible Resolutions:")
    print(f"   A) τ_Vπ is NOT for the expansion term -τ_Vπ V θ")
    print(f"      → Check IReD paper Table III for actual usage")
    print(f"   ")
    print(f"   B) Expansion coefficient is δ_VV (dimensionless)")
    print(f"      δ_VV = 1.0 from IReD Table III")
    print(f"      Term: -δ_VV × τ_V × V × θ")
    print(f"      But then [δ_VV × τ_V × V × θ] = GeV³, not GeV⁴")
    print(f"   ")
    print(f"   C) Missing enthalpy density h in normalization")
    print(f"      Term: -(τ_Vπ × h) × V × θ where h = ε + P")
    print(f"      [h] = GeV⁴")
    print(f"      [τ_Vπ × h × V × θ] = GeV⁻⁶ × GeV⁴ × GeV³ × GeV = GeV²")
    print(f"      Still wrong!")

    # Check what happens if we use the formula without P
    tau_Vn_alternative = 0.0071692 * beta * tau_V  # Without dividing by P
    print(f"\n9. Alternative Formula (without P division):")
    print(f"   τ_Vπ = 0.0071692 × β × τ_V")
    print(f"        = {tau_Vn_alternative:.4e}")
    print(f"   Units: [τ_Vπ] = GeV⁻¹ × GeV⁻¹ = GeV⁻²")
    print(f"   With T² normalization: {tau_Vn_alternative * T**2:.4e} (dimensionless)")
    print(f"   ⚠️  This matches λ_Vπ pattern! But IReD explicitly has /P in formula.")

    print(f"\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("The IReD formula τ_Vπ = 0.0071692 β τ_V/P cannot be directly used")
    print("in the expansion term -τ_Vπ V θ without additional normalization.")
    print("")
    print("Most likely: τ_Vπ in IReD Table III has a DIFFERENT usage than")
    print("the expansion coupling in the relaxation equations.")
    print("")
    print("ACTION: Check IReD paper carefully to see where τ_Vπ is actually used.")

if __name__ == "__main__":
    main()

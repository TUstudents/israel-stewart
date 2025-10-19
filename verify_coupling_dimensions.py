#!/usr/bin/env -S uv run python
"""
Verify dimensional consistency of IReD coupling coefficients.

This script checks whether the coupling coefficients from ired_simple.py
have the correct dimensions when used in relaxation.py equations.

Focuses on mixed-unit coefficients that couple different dissipative fluxes:
- λ_πV (shear-diffusion): couples π^μν to V^μ
- λ_Vπ (diffusion-shear): couples V^μ to π^μν
- τ_Vπ (diffusion-expansion): couples V^μ to θ
"""

from israel_stewart.equations.ired_simple import HardSphereIReD
import numpy as np

def check_lambda_pi_V():
    """
    Check λ_πV dimensional consistency.

    Usage in relaxation.py line 365:
        diffusion_term = λ_πV * T * (V^μ ∇^ν(μ/T) + V^ν ∇^μ(μ/T)) / 2

    This contributes to dπ^μν/dτ, so:
        [dπ^μν/dτ] = GeV⁴
        [V^μ] = GeV³ (particle current)
        [∇^ν(μ/T)] = GeV (gradient)
        Required: [λ_πV × T × V^μ × ∇^ν(μ/T)] = GeV⁴
        Therefore: [λ_πV × T] = GeV⁴ / (GeV³ × GeV) = GeV⁰
        So: [λ_πV] = GeV⁻¹ (dimensionless after multiplying by T)

    From IReD: λ_πV = 0.20890 τ_π/β
        [τ_π/β] = GeV⁻¹ / GeV⁻¹ = dimensionless

    After multiplying by T:
        [λ_πV × T] = dimensionless × GeV = GeV ✗ MISMATCH!

    Expected: dimensionless
    IReD gives: dimensionless
    After T scaling: GeV

    This suggests λ_πV from IReD should be used directly WITHOUT T scaling!
    """
    model = HardSphereIReD(temperature=0.4, cross_section=1.0)

    lambda_pi_V_fm = model.lambda_pi_V(time_unit="fm/c")
    lambda_pi_V_nat = model.lambda_pi_V(time_unit="natural")

    print("="*70)
    print("λ_πV (Shear-Diffusion Coupling)")
    print("="*70)
    print(f"\nIReD formula: λ_πV = 0.20890 × τ_π/β")
    print(f"  τ_π = {model.shear_relaxation_time():.6f} fm/c")
    print(f"  β = 1/T = {model.beta:.6f} GeV⁻¹")
    print(f"  Result: λ_πV = {lambda_pi_V_fm:.6e} (fm/c units)")
    print(f"  Result: λ_πV = {lambda_pi_V_nat:.6e} (natural units)")

    print(f"\nDimensional Analysis:")
    print(f"  From IReD: [λ_πV] = [τ_π/β] = GeV⁻¹/GeV⁻¹ = dimensionless")
    print(f"  Required in equation: [λ_πV × T] = GeV⁰")
    print(f"  After T scaling: [λ_πV × T] = dimensionless × GeV = GeV")
    print(f"  ❌ MISMATCH: λ_πV from IReD is already dimensionless,")
    print(f"     but equation needs it to become dimensionless AFTER multiplying by T!")
    print(f"  ⚠️  This suggests current implementation has an EXTRA factor of T")

    return {"coefficient": "lambda_pi_V", "status": "dimension_mismatch", "extra_T_factor": True}

def check_lambda_V_pi():
    """
    Check λ_Vπ dimensional consistency.

    Usage in relaxation.py line 473:
        shear_diffusion_term = λ_Vπ * T * π^μν ∇_ν(μ/T)

    This contributes to dV^μ/dτ, so:
        [dV^μ/dτ] = GeV⁴
        [π^μν] = GeV³ (shear stress - recalculated from dπ/dτ = ... + 2η σ)
        [∇_ν(μ/T)] = GeV (gradient)
        Required: [λ_Vπ × T × π^μν × ∇_ν(μ/T)] = GeV⁴
        Therefore: [λ_Vπ × T] = GeV⁴ / (GeV³ × GeV) = GeV⁰
        So: [λ_Vπ] = GeV⁻¹ (dimensionless after multiplying by T)

    From IReD: λ_Vπ = 0.069240 β τ_V
        [β × τ_V] = GeV⁻¹ × GeV⁻¹ = GeV⁻²

    After multiplying by T:
        [λ_Vπ × T] = GeV⁻² × GeV = GeV⁻¹ ✗ MISMATCH!

    Expected: GeV⁻¹ before T, dimensionless after
    IReD gives: GeV⁻²
    After T scaling: GeV⁻¹

    This suggests λ_Vπ needs an ADDITIONAL factor of T!
    """
    model = HardSphereIReD(temperature=0.4, cross_section=1.0)

    lambda_V_pi_fm = model.lambda_V_pi(time_unit="fm/c")
    lambda_V_pi_nat = model.lambda_V_pi(time_unit="natural")

    print("\n" + "="*70)
    print("λ_Vπ (Diffusion-Shear Coupling)")
    print("="*70)
    print(f"\nIReD formula: λ_Vπ = 0.069240 × β × τ_V")
    print(f"  β = 1/T = {model.beta:.6f} GeV⁻¹")
    print(f"  τ_V = {model.diffusion_relaxation_time():.6f} fm/c")
    print(f"  Result: λ_Vπ = {lambda_V_pi_fm:.6e} (fm/c units)")
    print(f"  Result: λ_Vπ = {lambda_V_pi_nat:.6e} (natural units)")

    print(f"\nDimensional Analysis:")
    print(f"  From IReD: [λ_Vπ] = [β × τ_V] = GeV⁻¹ × GeV⁻¹ = GeV⁻²")
    print(f"  Required in equation: [λ_Vπ × T] = GeV⁰")
    print(f"  After T scaling: [λ_Vπ × T] = GeV⁻² × GeV = GeV⁻¹")
    print(f"  ❌ MISMATCH: Need dimensionless, got GeV⁻¹")
    print(f"  ⚠️  This suggests λ_Vπ needs an ADDITIONAL T factor!")
    print(f"  ✅ FIX: Use λ_Vπ × T² instead of λ_Vπ × T")

    return {"coefficient": "lambda_V_pi", "status": "dimension_mismatch", "missing_T_factor": True}

def check_tau_V_pi():
    """
    Check τ_Vπ dimensional consistency.

    Usage in relaxation.py line 462:
        expansion_term = -τ_Vπ × V^μ × θ

    This contributes to dV^μ/dτ, so:
        [dV^μ/dτ] = GeV⁴
        [V^μ] = GeV³ (particle current)
        [θ] = GeV (expansion scalar)
        Required: [τ_Vπ × V^μ × θ] = GeV⁴
        Therefore: [τ_Vπ] = GeV⁴ / (GeV³ × GeV) = GeV⁰ = dimensionless

    From IReD: τ_Vπ = 0.0071692 β τ_V/P
        [β × τ_V / P] = GeV⁻¹ × GeV⁻¹ / GeV⁴ = GeV⁻⁶

    ❌ HUGE MISMATCH!

    Expected: dimensionless
    IReD gives: GeV⁻⁶

    This suggests τ_Vπ needs a factor of T⁶!
    OR the IReD formula is wrong!
    OR there's a different normalization convention!
    """
    model = HardSphereIReD(temperature=0.4, cross_section=1.0)

    tau_V_pi_fm = model.tau_V_pi(time_unit="fm/c")
    tau_V_pi_nat = model.tau_V_pi(time_unit="natural")

    print("\n" + "="*70)
    print("τ_Vπ (Diffusion-Expansion Coupling)")
    print("="*70)
    print(f"\nIReD formula: τ_Vπ = 0.0071692 × β × τ_V / P")
    print(f"  β = 1/T = {model.beta:.6f} GeV⁻¹")
    print(f"  τ_V = {model.diffusion_relaxation_time():.6f} fm/c")
    print(f"  P = {model.pressure:.6e} GeV⁴")
    print(f"  Result: τ_Vπ = {tau_V_pi_fm:.6e} (fm/c units)")
    print(f"  Result: τ_Vπ = {tau_V_pi_nat:.6e} (natural units)")

    print(f"\nDimensional Analysis:")
    print(f"  From IReD: [τ_Vπ] = [β × τ_V / P] = GeV⁻¹ × GeV⁻¹ / GeV⁴ = GeV⁻⁶")
    print(f"  Required in equation: [τ_Vπ] = GeV⁰ (dimensionless)")
    print(f"  ❌ HUGE MISMATCH: GeV⁻⁶ vs dimensionless!")
    print(f"  ⚠️  This suggests τ_Vπ needs a factor of T⁶ = (0.4 GeV)⁶ = 4.1e-3")
    print(f"  OR there's a fundamental error in the IReD formula interpretation!")

    # Check if symbolic equation matches
    print(f"\nNote from symbolic equation (line 175 in relaxation.py):")
    print(f"  diffusion_nonlinear = lambda_V_pi × π × ∇μ - tau_V × V × θ")
    print(f"  ⚠️  Symbolic uses 'tau_V' (relaxation time), not 'tau_V_pi' (coupling)!")
    print(f"  This suggests the expansion term might be: -τ_V × V × θ (not τ_Vπ)")

    return {"coefficient": "tau_V_pi", "status": "severe_dimension_mismatch", "missing_T6_factor": True}

def main():
    """Run all dimensional checks."""
    print("\n" + "="*70)
    print("IReD Coupling Coefficient Dimensional Analysis")
    print("="*70)
    print("\nChecking dimensional consistency of mixed-unit coefficients")
    print("when used in Israel-Stewart relaxation equations.\n")

    results = []
    results.append(check_lambda_pi_V())
    results.append(check_lambda_V_pi())
    results.append(check_tau_V_pi())

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    for r in results:
        status_symbol = "❌" if "mismatch" in r["status"] else "✅"
        print(f"{status_symbol} {r['coefficient']}: {r['status']}")
        if "extra_T_factor" in r and r["extra_T_factor"]:
            print(f"   → Implementation has EXTRA T factor")
        if "missing_T_factor" in r and r["missing_T_factor"]:
            print(f"   → Implementation needs ADDITIONAL T factor")
        if "missing_T6_factor" in r and r["missing_T6_factor"]:
            print(f"   → Severe mismatch: needs T⁶ or formula is wrong")

    print("\n" + "="*70)
    print("Conclusions")
    print("="*70)
    print("1. λ_πV: Current implementation multiplies by T, but IReD gives")
    print("   dimensionless coefficient. This adds an EXTRA T factor!")
    print("   → FIX: Remove T multiplication for λ_πV")
    print()
    print("2. λ_Vπ: Current implementation multiplies by T, but needs T²")
    print("   → FIX: Multiply by T² instead of T")
    print()
    print("3. τ_Vπ: Severe dimensional mismatch (GeV⁻⁶ vs dimensionless)")
    print("   → INVESTIGATE: Check IReD paper formula carefully")
    print("   → Possible: Symbolic equation uses τ_V, not τ_Vπ")

if __name__ == "__main__":
    main()

#!/usr/bin/env -S uv run python
"""
Derive the correct source term form from the dispersion matrix.

The question: should the source be -ζθ or -ζθ/τ_Π?

We'll work backwards from the dispersion matrix to find what time-domain
equation it represents.
"""

import numpy as np

print("=" * 80)
print("DERIVING SOURCE TERM FROM DISPERSION MATRIX")
print("=" * 80)
print()

print("The dispersion matrix (sound_waves.py:426) has row 2:")
print("  (1 - iωτ_Π)·δΠ + iζk·δv_x = 0")
print()

print("This represents a relationship between δΠ and δv_x in Fourier space")
print("for plane waves exp(-iωt + ikx).")
print()

print("=" * 80)
print("DERIVATION 1: Standard Israel-Stewart Form")
print("=" * 80)
print()

print("ASSUME the time-domain equation is:")
print("  τ_Π·∂Π/∂t + Π = -ζθ")
print()

print("Taking Fourier transform (∂/∂t → -iω, ∂/∂x → ik):")
print("  τ_Π·(-iω)·δΠ + δΠ = -ζ·θ")
print()

print("For longitudinal wave, θ = ∇·v = ik·v_x:")
print("  -iωτ_Π·δΠ + δΠ = -ζ·(ik·δv_x)")
print("  δΠ·(-iωτ_Π + 1) = -iζk·δv_x")
print("  δΠ·(1 - iωτ_Π) = -iζk·δv_x")
print()

print("Rearranging to match dispersion matrix form:")
print("  (1 - iωτ_Π)·δΠ + iζk·δv_x = 0")
print()

print("✓ This EXACTLY matches the dispersion matrix!")
print()

print("Therefore, the time-domain equation is:")
print("  τ_Π·∂Π/∂t + Π = -ζθ")
print()

print("Solving for ∂Π/∂t:")
print("  ∂Π/∂t = (-Π - ζθ)/τ_Π")
print("  ∂Π/∂t = -Π/τ_Π - ζθ/τ_Π")
print()

print("So the source term should be: -ζθ/τ_Π")
print()

print("=" * 80)
print("DERIVATION 2: Alternative Form")
print("=" * 80)
print()

print("ASSUME instead the time-domain equation is:")
print("  ∂Π/∂t + Π/τ_Π = -ζθ")
print("  ∂Π/∂t = -Π/τ_Π - ζθ  (no τ_Π division on source)")
print()

print("Taking Fourier transform:")
print("  (-iω)·δΠ + δΠ/τ_Π = -ζ·(ik·δv_x)")
print()

print("Multiply by τ_Π:")
print("  -iωτ_Π·δΠ + δΠ = -ζτ_Π·(ik·δv_x)")
print("  δΠ·(-iωτ_Π + 1) = -iζτ_Π·k·δv_x")
print("  δΠ·(1 - iωτ_Π) = -iζτ_Π·k·δv_x")
print()

print("Rearranging:")
print("  (1 - iωτ_Π)·δΠ + iζτ_Π·k·δv_x = 0")
print()

print("✗ This has an extra τ_Π factor!")
print("✗ Does NOT match the dispersion matrix (which has +iζk, not +iζτ_Πk)")
print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

print("The dispersion matrix encodes:")
print("  (1 - iωτ_Π)·δΠ + iζk·δv_x = 0")
print()

print("This corresponds to the time-domain equation:")
print("  τ_Π·∂Π/∂t + Π = -ζθ")
print()

print("Which gives the RHS:")
print("  dΠ/dt = -Π/τ_Π - ζθ/τ_Π  ← Source has /τ_Π")
print()

print("=" * 80)
print("WHAT WE CHANGED")
print("=" * 80)
print()

print("Original code (correct per dispersion matrix):")
print("  dΠ/dt = -Π/τ_Π - ζθ/τ_Π")
print()

print("Our 'fix' (wrong per dispersion matrix):")
print("  dΠ/dt = -Π/τ_Π - ζθ")
print()

print("But the 'fix' improved frequency accuracy from 33% to 1%!")
print()

print("=" * 80)
print("IMPLICATIONS")
print("=" * 80)
print()

print("Two possibilities:")
print()

print("1. The dispersion matrix coefficient is WRONG")
print("   Should be: +iζτ_Πk (not +iζk)")
print("   This would make our fix correct")
print()

print("2. The dispersion matrix is correct, our fix is wrong")
print("   But then why did frequency improve?")
print("   Perhaps IMEX splitting has an issue with /τ_Π in source?")
print()

print("=" * 80)
print("SAME ANALYSIS FOR SHEAR")
print("=" * 80)
print()

print("Dispersion matrix row 3 (sound_waves.py:431):")
print("  (1 - iωτ_π)·δπ_xx - i·(4/3)ηk·δv_x = 0")
print()

print("Working backwards:")
print("  τ_π·∂π/∂t + π = 2ησ")
print()

print("For longitudinal wave, σ_xx = (2/3)·∂v_x/∂x = (2/3)ik·v_x:")
print("  τ_π·(-iω)·δπ + δπ = 2η·(2/3)ik·δv_x")
print("  -iωτ_π·δπ + δπ = (4/3)iηk·δv_x")
print("  (1 - iωτ_π)·δπ = (4/3)iηk·δv_x")
print("  (1 - iωτ_π)·δπ - (4/3)iηk·δv_x = 0")
print()

print("✓ This matches the dispersion matrix (with minus sign)")
print()

print("So the time-domain equation should be:")
print("  τ_π·∂π/∂t + π = 2ησ")
print("  ∂π/∂t = -π/τ_π + 2ησ/τ_π  ← Source has /τ_π")
print()

print("But we changed it to:")
print("  ∂π/∂t = -π/τ_π + 2ησ  (no /τ_π on source)")
print()

print("=" * 80)
print("RESOLUTION NEEDED")
print("=" * 80)
print()

print("We need to test:")
print("1. Which formulation gives correct phase evolution?")
print("2. Why did removing /τ improve frequency but mess up phase?")
print("3. Is there an issue with how IMEX handles the source terms?")
print()

print("=" * 80)

#!/usr/bin/env python3
"""
Verify the correct signs in the Israel-Stewart dispersion relation.

From first principles:
1. Energy conservation: ∂_t ε + ∂_x[(ε+p)v_x] = 0
2. Momentum conservation: ∂_t[(ε+p)v_x] + ∂_x[p + Π + π_xx] = 0
3. Bulk relaxation: τ_Π*∂_t Π + Π = -ζ*θ where θ = ∂_x v_x
4. Shear relaxation: τ_π*∂_t π_xx + π_xx = (4η/3)*∂_x v_x

For plane wave: δf = f̃ * exp(-iωt + ikx)
Then: ∂_t f → -iω*f̃, ∂_x f → ik*f̃
"""

import numpy as np

print("=" * 80)
print("ISRAEL-STEWART DISPERSION RELATION - SIGN VERIFICATION")
print("=" * 80)
print()

print("Starting from Israel-Stewart equations:")
print()
print("1. Energy: ∂_t ε + ∂_x[(ε+p)v_x] = 0")
print("2. Momentum: ∂_t[(ε+p)v_x] + ∂_x[p + Π + π_xx] = 0")
print("3. Bulk: τ_Π*∂_t Π + Π = -ζ*θ  where θ = ∂_x v_x")
print("4. Shear: τ_π*∂_t π_xx + π_xx = (4η/3)*∂_x v_x")
print()

print("For plane wave: δf = f̃ * exp(-iωt + ikx)")
print("  ∂_t f → -iω*f̃")
print("  ∂_x f → +ik*f̃")
print()

print("-" * 80)
print("Linearization around ε₀, p₀, v_x=0, Π=0, π=0:")
print("-" * 80)
print()

print("Energy conservation:")
print("  ∂_t δε + (ε₀+p₀)*∂_x δv_x = 0")
print("  → -iω*δε + (ε₀+p₀)*ik*δv_x = 0")
print("  → -iω*δε + ik*(ε₀+p₀)*δv_x = 0  ✓")
print()

print("Momentum conservation:")
print("  ∂_t[(ε₀+p₀)*δv_x] + ∂_x[c_s²*δε + δΠ + δπ_xx] = 0")
print("  → (ε₀+p₀)*(-iω)*δv_x + ik*[c_s²*δε + δΠ + δπ_xx] = 0")
print("  → -iω*(ε₀+p₀)*δv_x + ik*c_s²*δε + ik*δΠ + ik*δπ_xx = 0")
print("  → ik*c_s²*δε - iω*(ε₀+p₀)*δv_x + ik*δΠ + ik*δπ_xx = 0  ✓")
print()

print("Bulk relaxation:")
print("  τ_Π*∂_t δΠ + δΠ = -ζ*∂_x δv_x")
print("  → τ_Π*(-iω)*δΠ + δΠ = -ζ*ik*δv_x")
print("  → -iω*τ_Π*δΠ + δΠ = -ζ*ik*δv_x")
print("  → (1 - iω*τ_Π)*δΠ = -ζ*ik*δv_x")
print("  → (1 - iω*τ_Π)*δΠ + ζ*ik*δv_x = 0")
print()
print("  ❌ CODE HAS: (1 - iω*τ_Π)*δΠ + **i**ζk*δv_x = 0")
print("  ✓ SHOULD BE: (1 - iω*τ_Π)*δΠ + **i**ζk*δv_x = 0")
print()
print("  WAIT - the code is CORRECT! Let me recalculate...")
print()
print("  From: (1 - iω*τ_Π)*δΠ = -ζ*ik*δv_x")
print("  Multiply by i: i*(1 - iω*τ_Π)*δΠ = -ζ*i²*k*δv_x")
print("                  i*(1 - iω*τ_Π)*δΠ = -ζ*(-1)*k*δv_x")
print("                  i*(1 - iω*τ_Π)*δΠ = +ζ*k*δv_x")
print()
print("  Actually, keeping as-is:")
print("  (1 - iω*τ_Π)*δΠ = -iζk*δv_x")
print("  (1 - iω*τ_Π)*δΠ + iζk*δv_x = 0  ✓ CODE IS CORRECT")
print()

print("Shear relaxation:")
print("  τ_π*∂_t δπ_xx + δπ_xx = (4η/3)*∂_x δv_x")
print("  → τ_π*(-iω)*δπ_xx + δπ_xx = (4η/3)*ik*δv_x")
print("  → (1 - iω*τ_π)*δπ_xx = i*(4η/3)*k*δv_x")
print("  → (1 - iω*τ_π)*δπ_xx + i*(4η/3)*k*δv_x = 0  ✓ CODE IS CORRECT")
print()

print("=" * 80)
print("CONCLUSION: Dispersion matrix signs are CORRECT")
print("=" * 80)
print()

print("So why negative imaginary frequency?")
print()
print("Possibility 1: Numerical root finder found wrong root")
print("Possibility 2: At high k, Israel-Stewart becomes UNSTABLE")
print("Possibility 3: Sign error elsewhere (relaxation RHS implementation)")
print()

print("Let me check what happens with modified relaxation equations...")
print()
print("User edit removed 'linear' term from relaxation RHS.")
print("This means in the numerical code:")
print("  dΠ/dt = -ζ*θ  (missing -Π/τ_Π term)")
print()
print("But split-step handles -Π/τ_Π separately in advance_linear_terms().")
print("So the effective equation is still:")
print("  dΠ/dt = -Π/τ_Π - ζ*θ  ✓ CORRECT")
print()

print("HYPOTHESIS: The issue is not in dispersion relation.")
print("The negative frequency at k=8 may indicate:")
print("  1. Numerical instability at high k (near grid resolution)")
print("  2. Root finder converged to unphysical solution")
print("  3. Missing physics (second-order terms needed at high k)")

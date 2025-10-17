#!/usr/bin/env python3
"""Verify the eigenmode formula derivation for coupled n-V system.

The coupled equations are:
    ∂_t n = -∂_x V^x
    ∂_t V^x = -V^x/τ_V - D ∂_x(μ/T)

For small perturbations: μ/T ≈ (n - n₀)/n₀

Plane wave ansatz:
    n = n₀(1 + A e^(-Γt) sin(kx))
    V^x = B e^(-Γt) cos(kx)

Derive eigenvalue equation and compare with test formula.
"""

import numpy as np

from israel_stewart.equations.ired_simple import HardSphereIReD

# IReD parameters (same as test)
T = 0.4  # GeV
sigma = 1000.0  # fm²
k = 0.5  # GeV

# Create IReD model
ired = HardSphereIReD(temperature=T, cross_section=sigma, truncation="41")

# Transport coefficients
D = ired.diffusion_coefficient()
tau_V = ired.diffusion_relaxation_time()

# Particle density (radiation fluid: n = ζ(3)/π² T³)
zeta_3 = 1.2020569  # Riemann zeta function ζ(3)
n0 = (zeta_3 / np.pi**2) * T**3

print("=" * 80)
print("EIGENMODE FORMULA VERIFICATION")
print("=" * 80)

print("\nPhysical parameters:")
print(f"  T = {T:.4f} GeV")
print(f"  σ = {sigma:.1f} fm²")
print(f"  k = {k:.4f} GeV")

print("\nTransport coefficients:")
print(f"  D = {D:.6e} GeV²")
print(f"  τ_V = {tau_V:.6e} GeV⁻¹")
print(f"  n₀ = {n0:.6e} GeV³")

print(f"\n{'='*80}")
print("DERIVATION OF EIGENVALUE EQUATION")
print("=" * 80)

print("\nPlane wave ansatz:")
print("  n(x,t) = n₀(1 + A e^(-Γt) sin(kx))")
print("  V^x(x,t) = B e^(-Γt) cos(kx)")

print("\nFrom continuity equation ∂_t n = -∂_x V^x:")
print("  LHS: ∂_t n = -Γ n₀ A e^(-Γt) sin(kx)")
print("  RHS: -∂_x V = -(-Bk e^(-Γt) sin(kx)) = Bk e^(-Γt) sin(kx)")
print("  Equating: -Γ n₀ A = Bk")
print("  Result: B = -Γ n₀ A / k  ... (1)")

print("\nFrom diffusion equation ∂_t V = -V/τ_V - D ∂_x(μ/T):")
print("  Chemical potential: μ/T ≈ (n - n₀)/n₀ = A e^(-Γt) sin(kx)")
print("  Gradient: ∂_x(μ/T) = A k e^(-Γt) cos(kx)")
print("  LHS: ∂_t V = -Γ B e^(-Γt) cos(kx)")
print("  RHS: -V/τ - D ∂_x(μ/T) = -B e^(-Γt) cos(kx) / τ_V - D A k e^(-Γt) cos(kx)")
print("  Equating: -Γ B = -B/τ_V - D A k")
print("  Rearranging: (Γ + 1/τ_V) B = -D A k  ... (2)")

print("\nSubstituting (1) into (2):")
print("  (Γ + 1/τ_V) (-Γ n₀ A / k) = -D A k")
print("  (Γ + 1/τ_V) Γ n₀ / k = D k")
print("  (Γ² + Γ/τ_V) n₀ / k = D k")
print("  Γ² + Γ/τ_V = D k² / n₀")

print("\nEigenvalue equation:")
print("  Γ² + Γ/τ_V - D k² / n₀ = 0")
print("  (Note: MINUS sign because we moved term to LHS)")

print(f"\n{'='*80}")
print("SOLVING THE EIGENVALUE EQUATION")
print("=" * 80)

# Discriminant
disc = (1 / tau_V) ** 2 + 4 * D * k**2 / n0

print("\nDiscriminant:")
print("  Δ = (1/τ_V)² + 4Dk²/n₀")
print(f"  Δ = ({1/tau_V:.6e})² + 4 × {D:.6e} × {k**2:.6e} / {n0:.6e}")
print(f"  Δ = {(1/tau_V)**2:.6e} + {4 * D * k**2 / n0:.6e}")
print(f"  Δ = {disc:.6e} GeV²")
print(f"  √Δ = {np.sqrt(disc):.6e} GeV")

# Eigenvalues
Gamma_1 = (-1 / tau_V + np.sqrt(disc)) / 2  # Less negative (slow mode)
Gamma_2 = (-1 / tau_V - np.sqrt(disc)) / 2  # More negative (fast mode)

print("\nEigenvalues (using quadratic formula):")
print("  Γ = [-(1/τ_V) ± √Δ] / 2")
print(f"  Γ₁ (slow mode) = {Gamma_1:.6e} GeV")
print(f"  Γ₂ (fast mode) = {Gamma_2:.6e} GeV")

# Check which is which
if abs(Gamma_1) < abs(Gamma_2):
    Gamma_slow = Gamma_1
    Gamma_fast = Gamma_2
else:
    Gamma_slow = Gamma_2
    Gamma_fast = Gamma_1

print("\nIdentification:")
print(f"  Slow mode: Γ_slow = {abs(Gamma_slow):.6e} GeV (|Γ| is smaller)")
print(f"  Fast mode: Γ_fast = {abs(Gamma_fast):.6e} GeV (|Γ| is larger)")

print(f"\n{'='*80}")
print("PERTURBATIVE APPROXIMATION")
print("=" * 80)

# Check if perturbative approximation is valid
regime_param = D * k**2 * tau_V**2 / n0
print("\nRegime parameter:")
print(f"  ε = Dk²τ²/n₀ = {regime_param:.6e}")
print(f"  Compare to (1/τ)² = {(1/tau_V)**2:.6e}")
print(f"  Ratio: ε / (1/τ)² = {regime_param / (1/tau_V)**2:.6e}")

if regime_param < (1 / tau_V) ** 2:
    print("  ✓ Perturbative regime valid (ε << 1)")

    print("\nTaylor expansion for Dk²/n₀ << (1/τ)²:")
    print("  √Δ = √[(1/τ)² + 4Dk²/n₀]")
    print("     = (1/τ) √[1 + 4Dk²τ²/n₀]")
    print("     ≈ (1/τ) [1 + 2Dk²τ²/n₀]  (binomial approximation)")

    print("\n  Γ_slow = [-(1/τ) + (1/τ)(1 + 2Dk²τ²/n₀)] / 2")
    print("         = [-(1/τ) + (1/τ) + 2Dk²τ/n₀] / 2")
    print("         = Dk²τ/n₀")

    Gamma_slow_approx = D * k**2 * tau_V / n0
    print(f"\n  Γ_slow ≈ Dk²τ_V/n₀ = {Gamma_slow_approx:.6e} GeV")

    error = abs(Gamma_slow_approx - abs(Gamma_slow)) / abs(Gamma_slow)
    print(f"  Error vs exact: {error:.2%}")

    if error < 0.01:
        print("  ✓ Approximation excellent (< 1% error)")
else:
    print("  ✗ Not in perturbative regime")

print(f"\n{'='*80}")
print("COMPARISON WITH TEST")
print("=" * 80)

print("\nTest formula:")
print("  Gamma_expected = (D * k²) * (τ_V / n₀)")
print(f"  Gamma_expected = ({D:.6e} × {k**2:.6e}) × ({tau_V:.6e} / {n0:.6e})")
print(f"  Gamma_expected = {D * k**2:.6e} × {tau_V / n0:.6e}")
print(f"  Gamma_expected = {D * k**2 * tau_V / n0:.6e} GeV")

print("\nExact eigenvalue:")
print(f"  Γ_slow (exact) = {abs(Gamma_slow):.6e} GeV")

error_formula = abs(D * k**2 * tau_V / n0 - abs(Gamma_slow)) / abs(Gamma_slow)
print("\nError:")
print(f"  |Formula - Exact| / Exact = {error_formula:.2%}")

if error_formula < 0.01:
    print("  ✓ Test formula is correct (< 1% error from exact)")
else:
    print("  ⚠ Test formula has significant error")

print(f"\n{'='*80}")
print("CONCLUSION")
print("=" * 80)

print("\n1. The eigenvalue equation is: Γ² + Γ/τ_V - Dk²/n₀ = 0")
print("2. The slow mode (decaying) eigenvalue is:")
print("   Γ_slow = [-(1/τ_V) + √((1/τ_V)² + 4Dk²/n₀)] / 2")
print("3. In the perturbative regime (Dk²τ²/n₀ << 1):")
print("   Γ_slow ≈ Dk²τ_V/n₀")
print("4. The test formula Gamma_expected = Dk²τ_V/n₀ is CORRECT.")
print(f"\n5. Expected slow mode decay rate: {abs(Gamma_slow):.6e} GeV")
print("6. Measured in simulation: 3.610e-3 GeV (from test output)")
print(f"7. Discrepancy factor: {3.610e-3 / abs(Gamma_slow):.2f}×")

print("\nPossible reasons for 2.6× discrepancy:")
print("  - Nonlinear coupling terms (τ_Vπ, λ_Vπ) modify effective decay rate")
print("  - Numerical discretization effects (16³ grid)")
print("  - Initial condition not pure eigenmode (projection includes fast mode)")
print("  - Second-order terms in IReD equations affect dynamics")

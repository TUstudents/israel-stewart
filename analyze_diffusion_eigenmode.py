#!/usr/bin/env python3
"""Analyze diffusion eigenmode structure for coupled n-V system.

The coupled equations are:
    ∂_t n = -∇·V
    ∂_t V = -V/τ_V - D ∇(μ/T)

For small perturbations: μ/T ≈ (n - n₀)/n₀

Plane wave ansatz: n = n₀(1 + A e^(-Γt) sin(kx)), V = B e^(-Γt) cos(kx)

This gives eigenvalue equation:
    | -Γ      -k   | | A |     | 0 |
    | Dk/n₀  -Γ-1/τ| | B |  =  | 0 |

Eigenvalues: Γ = -(1/τ + √(1/τ² - 4Dk/n₀))/2  (slow mode, diffusion)
             Γ = -(1/τ - √(1/τ² - 4Dk/n₀))/2  (fast mode, relaxation)
"""

import numpy as np

from israel_stewart.equations.ired_simple import HardSphereIReD

# IReD parameters
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
print("DIFFUSION EIGENMODE ANALYSIS")
print("=" * 80)

print("\nTransport coefficients:")
print(f"  D = {D:.6e} GeV²")
print(f"  τ_V = {tau_V:.6e} GeV⁻¹")
print(f"  1/τ_V = {1/tau_V:.6e} GeV")

print("\nBackground state:")
print(f"  n₀ = {n0:.6e} GeV³")
print(f"  T = {T:.4f} GeV")
print(f"  k = {k:.4f} GeV")

print("\nCharacteristic scales:")
print(f"  Dk² = {D * k**2:.6e} GeV (diffusion rate without coupling)")
print(f"  1/τ_V = {1/tau_V:.6e} GeV (relaxation rate)")
print(f"  Dk/n₀ = {D * k / n0:.6e} GeV² (coupling strength)")

# Eigenvalue equation: det(M - ΓI) = 0
# M = [     0        -k     ]
#     [  Dk/n₀   -1/τ_V     ]
#
# Γ² + Γ/τ_V + Dk/n₀ = 0

# Discriminant
disc = (1 / tau_V) ** 2 - 4 * D * k / n0

print("\nEigenvalue equation: Γ² + Γ/τ_V + Dk/n₀ = 0")
print(f"  Discriminant = (1/τ)² - 4Dk/n₀ = {disc:.6e} GeV²")

if disc > 0:
    print(f"  √discriminant = {np.sqrt(disc):.6e} GeV")

    # Two real eigenvalues (overdamped)
    Gamma_slow = (-1 / tau_V + np.sqrt(disc)) / 2
    Gamma_fast = (-1 / tau_V - np.sqrt(disc)) / 2

    print("\n✓ Overdamped system (two real eigenvalues):")
    print(f"  Γ_slow = {Gamma_slow:.6e} GeV (slow diffusion mode)")
    print(f"  Γ_fast = {Gamma_fast:.6e} GeV (fast relaxation mode)")

    # Decay times
    print("\nDecay times:")
    print(f"  τ_slow = 1/|Γ_slow| = {1/abs(Gamma_slow):.6e} GeV⁻¹")
    print(f"  τ_fast = 1/|Γ_fast| = {1/abs(Gamma_fast):.6e} GeV⁻¹")

    # Perturbative analysis for Dk/n₀ << 1/τ²
    if D * k / n0 < (1 / tau_V) ** 2:
        Gamma_slow_approx = -(D * k / n0) / (1 / tau_V)
        Gamma_fast_approx = -1 / tau_V

        print("\nPerturbative approximation (Dk/n₀ << 1/τ²):")
        print(f"  Γ_slow ≈ -Dk²τ_V/n₀ = {Gamma_slow_approx:.6e} GeV")
        print(f"  Γ_fast ≈ -1/τ_V = {Gamma_fast_approx:.6e} GeV")

        print("\nComparison with test expectation:")
        print(f"  Test expects: Γ = Dk² = {D * k**2:.6e} GeV")
        print(f"  Slow mode: Γ_slow = {Gamma_slow:.6e} GeV")
        print(f"  Ratio: Γ_slow / (Dk²) = {Gamma_slow / (D * k**2):.6e}")
        print(f"  Missing factor: n₀ τ_V = {n0 * tau_V:.6e}")

else:
    print("  ✗ Underdamped system (complex eigenvalues) - unusual for diffusion!")

# Eigenvectors
print(f"\n{'='*80}")
print("EIGENVECTOR ANALYSIS")
print("=" * 80)

# For eigenvalue Γ, eigenvector (A, B) satisfies:
#   -Γ A - k B = 0  =>  B = -Γ A / k
#   (Dk/n₀) A + (-Γ - 1/τ_V) B = 0

if disc > 0:
    print(f"\nSlow mode (Γ = {Gamma_slow:.6e} GeV):")
    # B = -Γ A / k
    A_slow = 1.0  # Normalize
    B_slow = -Gamma_slow * A_slow / k
    print(f"  Eigenvector: (n, V) ∝ ({A_slow:.6f}, {B_slow:.6e})")
    print("  Physical interpretation: diffusion-dominated")
    print(f"    For δn ∝ sin(kx), we have V ∝ {B_slow:.6e} cos(kx)")
    print(f"    V / (Dk δn) = {B_slow / (D * k):.6e} (compare to Fick's law: V = -Dk δn)")

    print(f"\nFast mode (Γ = {Gamma_fast:.6e} GeV):")
    A_fast = 1.0
    B_fast = -Gamma_fast * A_fast / k
    print(f"  Eigenvector: (n, V) ∝ ({A_fast:.6f}, {B_fast:.6e})")
    print("  Physical interpretation: relaxation-dominated")
    print("    This mode decays on timescale τ_V, then slow mode remains")

# Initial condition analysis
print(f"\n{'='*80}")
print("INITIAL CONDITION PROJECTION")
print("=" * 80)

print("\nBenchmark initial condition (Fick's law at t=0):")
print("  n(t=0) = n₀(1 + A sin(kx))")
print("  V(t=0) = -D ∂_x(μ/T) ≈ -Dk A cos(kx)  (Fick's law)")

# This initial condition is (A, -DkA)
A_init = 1.0
B_init = -D * k * A_init

print(f"  Initial vector: (A, B) = ({A_init:.6f}, {B_init:.6e})")

# Project onto eigenmodes
# Slow mode: (A_slow, B_slow)
# Fast mode: (A_fast, B_fast)

# Solve: c_slow * (A_slow, B_slow) + c_fast * (A_fast, B_fast) = (A_init, B_init)
# Matrix equation: [A_slow A_fast] [c_slow]  = [A_init]
#                  [B_slow B_fast] [c_fast]    [B_init]

matrix = np.array([[A_slow, A_fast], [B_slow, B_fast]])
rhs = np.array([A_init, B_init])

coeffs = np.linalg.solve(matrix, rhs)
c_slow = coeffs[0]
c_fast = coeffs[1]

print("\nProjection onto eigenmodes:")
print(f"  c_slow = {c_slow:.6e} (slow mode coefficient)")
print(f"  c_fast = {c_fast:.6e} (fast mode coefficient)")

print("\nTime evolution:")
print(f"  A(t) = {c_slow:.6e} exp({Gamma_slow:.6e} t) + {c_fast:.6e} exp({Gamma_fast:.6e} t)")
print(
    f"  B(t) = {c_slow * B_slow:.6e} exp({Gamma_slow:.6e} t) + {c_fast * B_fast:.6e} exp({Gamma_fast:.6e} t)"
)

# What does this mean for measurement?
print(f"\n{'='*80}")
print("IMPLICATIONS FOR DECAY RATE MEASUREMENT")
print("=" * 80)

print(f"\nEarly times (t << τ_V = {tau_V:.6e} GeV⁻¹):")
print("  Both modes present with comparable amplitude")
print("  Effective decay rate: weighted average")
c_total = abs(c_slow) + abs(c_fast)
Gamma_eff_early = (abs(c_slow) * abs(Gamma_slow) + abs(c_fast) * abs(Gamma_fast)) / c_total
print(f"  Γ_eff ≈ {Gamma_eff_early:.6e} GeV")

print("\nLate times (t >> τ_V):")
print("  Fast mode exponentially suppressed: exp(-|Γ_fast| t)")
print(f"  Only slow mode remains: A(t) ≈ {c_slow:.6e} exp({Gamma_slow:.6e} t)")
print(f"  Measured decay rate: Γ_measured = {abs(Gamma_slow):.6e} GeV")

print("\nTime to reach asymptotic regime:")
t_asymptotic = 5 / abs(Gamma_fast)  # 5 e-foldings of fast mode
print(f"  t_asymp ≈ 5/|Γ_fast| = {t_asymptotic:.6e} GeV⁻¹")

print("\nTest evolution time:")
t_test = 100.0  # GeV⁻¹ (from test)
print(f"  t_test = {t_test:.2f} GeV⁻¹")
if t_test > t_asymptotic:
    print(f"  ✓ Long enough to observe slow mode (t_test / t_asymp = {t_test / t_asymptotic:.2f})")
else:
    print(f"  ✗ Too short! Need t >> {t_asymptotic:.2e} GeV⁻¹ to isolate slow mode")
    print("    Will measure mixture of fast and slow modes")

print(f"\n{'='*80}")
print("SUMMARY")
print("=" * 80)

print(f"\n1. Test expects: Γ = Dk² = {D * k**2:.6e} GeV")
print(f"2. Actual slow mode: Γ_slow = {abs(Gamma_slow):.6e} GeV")
print(f"3. Discrepancy factor: {abs(Gamma_slow) / (D * k**2):.6e}")
print(f"4. Early-time effective rate: Γ_eff ≈ {Gamma_eff_early:.6e} GeV")

if abs(Gamma_slow) / (D * k**2) < 0.1:
    print(
        f"\n⚠ CRITICAL: Slow mode is {(D * k**2) / abs(Gamma_slow):.1f}× SLOWER than test expects!"
    )
    print("   Test formula Γ = Dk² is WRONG for coupled n-V system.")
    print("   Correct formula: Γ_slow ≈ Dk² τ_V / n₀ (for Dk/n₀ << 1/τ²)")
elif abs(Gamma_slow) / (D * k**2) > 10.0:
    print(
        f"\n⚠ CRITICAL: Slow mode is {abs(Gamma_slow) / (D * k**2):.1f}× FASTER than test expects!"
    )
    print("   Missing physics or implementation error in coupling.")
else:
    print("\n✓ Slow mode decay rate matches test expectation to within factor of 10")

print("\nRecommendation:")
if t_test < t_asymptotic:
    print(f"  - Increase evolution time to t > {t_asymptotic:.2e} GeV⁻¹ to isolate slow mode")
else:
    print("  - Evolution time is sufficient")

if abs(c_slow) < 0.1 * abs(c_fast):
    print(f"  - Initial condition has small slow mode component ({abs(c_slow):.2e})")
    print("  - Consider initializing with pure slow eigenmode instead of Fick's law")

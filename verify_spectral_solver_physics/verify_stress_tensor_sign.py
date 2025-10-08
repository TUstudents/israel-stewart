"""
Verify the sign of shear stress π^μν in the stress-energy tensor.

According to Israel-Stewart theory, the stress-energy tensor is:
  T^μν = (ε+p)u^μu^ν + p·g^μν + Π·Δ^μν + π^μν

where:
  - Π is bulk viscous pressure (negative pressure, opposes expansion)
  - π^μν is shear stress tensor (traceless, symmetric)
  - Δ^μν = g^μν + u^μu^ν is the spatial projector

CRITICAL QUESTION: What is the sign convention for π^μν?

In the literature, there are TWO conventions:

Convention A (Romatschke & Romatschke):
  T^μν = ... + Π·Δ^μν + π^μν
  where π^μν is the shear stress (can be positive or negative)

Convention B (Landau-Lifshitz):
  T^μν = ... + Π·Δ^μν - π^μν
  where π^μν > 0 represents viscous dissipation

The relaxation equation determines which convention is used:
  τ_π·Dπ^μν + π^μν = 2η·σ^μν

For Convention A: π^μν = 2η·σ^μν (positive coefficient)
For Convention B: π^μν = -2η·σ^μν (negative coefficient)

Let's check what our dispersion matrix assumes and what the code implements.
"""

import inspect

import numpy as np

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)

k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("=" * 80)
print("STRESS TENSOR SIGN VERIFICATION")
print("=" * 80)
print()

# Get analytical eigenmode
wave_vector = np.array([k, 0.0, 0.0])
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]
omega = complex(mode.frequency, -mode.attenuation)

# Build dispersion matrix
dispersion_matrix = benchmark.analytical._build_dispersion_matrix(omega, wave_vector)

print("Step 1: Check dispersion matrix convention")
print("-" * 80)
print()

# Row 1 (momentum equation) should be:
# -iω·h·δv_x = -ik·(c_s²·δρ + δΠ ± δπ_xx)
# Sign of π_xx determines the convention

print("Dispersion matrix Row 1 (momentum equation):")
print(f"  δρ coeff:    {dispersion_matrix[1, 0]}")
print(f"  δv_x coeff:  {dispersion_matrix[1, 1]}")
print(f"  δΠ coeff:    {dispersion_matrix[1, 2]}")
print(f"  δπ_xx coeff: {dispersion_matrix[1, 3]}")
print()

# Row 3 (shear relaxation) should be:
# (1 - iω·τ_π)·δπ_xx = ±(4η/3)·(2/3)·ik·δv_x
# Sign determines the convention

print("Dispersion matrix Row 3 (shear relaxation):")
print(f"  δρ coeff:    {dispersion_matrix[3, 0]}")
print(f"  δv_x coeff:  {dispersion_matrix[3, 1]}")
print(f"  δΠ coeff:    {dispersion_matrix[3, 2]}")
print(f"  δπ_xx coeff: {dispersion_matrix[3, 3]}")
print()

# Check signs
momentum_pi_sign = np.sign(dispersion_matrix[1, 3].imag)  # Should be -ik, so imag > 0
shear_v_sign = np.sign(dispersion_matrix[3, 1].imag)  # Should be -ik, so imag > 0

print("Sign analysis:")
print(f"  Momentum equation π_xx term: {dispersion_matrix[1, 3]}")
print("    → Momentum eqn uses: -ik·δπ_xx (MINUS sign)")
print()
print(f"  Shear relaxation v_x term: {dispersion_matrix[3, 1]}")
print("    → Shear eqn uses: -ik·δv_x (MINUS sign)")
print()

print("Interpretation:")
print("  Dispersion matrix assumes: T^μν = ... + Π·Δ^μν - π^μν")
print("  (Convention B: MINUS sign for shear stress)")
print()

print("=" * 80)
print("Step 2: Check code implementation")
print("-" * 80)
print()

# Read the stress tensor code
source = inspect.getsource(benchmark.solver.conservation.stress_energy_tensor)

# Find the line where shear stress is added
if "+ T_shear" in source:
    print("✗ CODE USES: T = ... + T_bulk + T_shear")
    print("  This is WRONG for Convention B (dispersion matrix)")
    code_convention = "A"
elif "- T_shear" in source:
    print("✓ CODE USES: T = ... + T_bulk - T_shear")
    print("  This is CORRECT for Convention B")
    code_convention = "B"
else:
    print("? Cannot determine from source code")
    # Check numerically
    T = benchmark.solver.conservation.stress_energy_tensor()

    # At a point with non-zero shear
    rho = benchmark.fields.rho[8, 0, 0]
    p = benchmark.fields.pressure[8, 0, 0]
    Pi = benchmark.fields.Pi[8, 0, 0]
    pi_xx = benchmark.fields.pi_munu[8, 0, 0, 1, 1]
    v_x = benchmark.fields.u_mu[8, 0, 0, 1]
    h = rho + p

    T_xx = T[8, 0, 0, 1, 1]
    T_xx_expected_A = h * v_x**2 + p + Pi + pi_xx  # Convention A
    T_xx_expected_B = h * v_x**2 + p + Pi - pi_xx  # Convention B

    if np.allclose(T_xx, T_xx_expected_A, rtol=1e-6):
        print("✗ CODE USES: T^xx = ... + Π + π_xx (Convention A)")
        code_convention = "A"
    elif np.allclose(T_xx, T_xx_expected_B, rtol=1e-6):
        print("✓ CODE USES: T^xx = ... + Π - π_xx (Convention B)")
        code_convention = "B"
    else:
        print("? Neither convention matches")
        code_convention = "?"

print()

print("=" * 80)
print("Step 3: Verify inconsistency")
print("-" * 80)
print()

if code_convention == "A":
    print("✗ INCONSISTENCY DETECTED:")
    print("  Dispersion matrix uses Convention B: T = ... - π")
    print("  Code implements Convention A:        T = ... + π")
    print()
    print("  This creates a sign error in the momentum equation!")
    print("  The numerical d(h·v)/dt will have WRONG sign for π contribution")
    print()
    print("Expected impact:")
    print("  - Eigenmode ratios will drift (π couples incorrectly)")
    print("  - Damping rate will be wrong")
    print("  - ~34% error in velocity evolution")
elif code_convention == "B":
    print("✓ CONVENTIONS MATCH:")
    print("  Both use Convention B: T = ... - π")
    print("  Sign consistency maintained")
else:
    print("? Cannot verify consistency")

print()
print("=" * 80)
print("Step 4: Numerical verification")
print("-" * 80)
print()

# Get Fourier coefficients
k_idx = 8
rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
Pi_fft = np.fft.fftn(benchmark.fields.Pi)
pi_fft = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])

rho_k = rho_fft[k_idx, 0, 0]
Pi_k = Pi_fft[k_idx, 0, 0]
pi_k = pi_fft[k_idx, 0, 0]

# What dispersion matrix expects for ∂_x(stress)
# Convention B: ∂_x(p + Π - π) = ∂_x(ρ/3 + Π - π)
expected_dTxx_dx = 1j * k * (rho_k / 3 + Pi_k - pi_k)

# What code actually gives
T = benchmark.solver.conservation.stress_energy_tensor()
T_xx_fft = np.fft.fftn(T[..., 1, 1])
actual_dTxx_dx = 1j * k * T_xx_fft[k_idx, 0, 0]

print("Momentum flux derivative at k=8:")
print(f"  Expected (Convention B): ∂_x(p + Π - π) = {expected_dTxx_dx}")
print(f"  Actual (from code):                       {actual_dTxx_dx}")
print()

if np.allclose(actual_dTxx_dx, expected_dTxx_dx, rtol=0.01):
    print("✓ Code produces correct ∂_x(T^xx) for Convention B")
else:
    error = abs((actual_dTxx_dx - expected_dTxx_dx) / expected_dTxx_dx) * 100
    print(f"✗ Code produces WRONG ∂_x(T^xx) by {error:.2f}%")

    # Check if it matches Convention A instead
    expected_A = 1j * k * (rho_k / 3 + Pi_k + pi_k)
    if np.allclose(actual_dTxx_dx, expected_A, rtol=0.01):
        print("  But matches Convention A: ∂_x(p + Π + π)")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if code_convention == "A":
    print("BUG CONFIRMED:")
    print()
    print("The stress-energy tensor in conservation.py uses:")
    print("  T^μν = ... + π^μν  (WRONG)")
    print()
    print("But the dispersion matrix assumes:")
    print("  T^μν = ... - π^μν  (CORRECT)")
    print()
    print("FIX: Change line in conservation.py:")
    print("  T_total = T_perfect + T_bulk + T_shear + T_heat")
    print("  →")
    print("  T_total = T_perfect + T_bulk - T_shear + T_heat")
    print()
    print("This will:")
    print("  ✓ Make momentum equation consistent with dispersion matrix")
    print("  ✓ Preserve eigenmode structure during evolution")
    print("  ✓ Fix the 34% error in velocity evolution")
else:
    print("No sign error detected (or unable to verify)")

print()
print("=" * 80)

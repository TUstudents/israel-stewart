"""
Verify dispersion matrix against Israel-Stewart theory.

Literature reference:
- Kovtun, "Lectures on hydrodynamic fluctuations in relativistic theories" (2012)
- Grozdanov & Kaplis, "Constructing higher-order hydrodynamics" (2016)

For plane wave exp(ikx - iωt), the linearized IS equations are:

1. Energy:     -iω·δε + ik·h·δv_x = 0
2. Momentum:   ik·c_s²·δε - iω·h·δv_x + ik·δΠ + ik·δπ_xx = 0
3. Bulk:       (1 - iωτ_Π)·δΠ - iζk·δv_x = 0
4. Shear:      (1 - iωτ_π)·δπ_xx - i(4/3)ηk·δv_x = 0

Key derivation for signs:

Bulk equation: τ_Π DΠ/Dτ + Π = -ζθ
  For plane wave: D/Dτ = -iω, θ = ik·δv_x
  → τ_Π(-iω)δΠ + δΠ = -ζ(ik·δv_x)
  → (1 - iωτ_Π)δΠ = -iζk·δv_x
  → +iζk·δv_x + (1 - iωτ_Π)δΠ = 0  ✓ POSITIVE sign for velocity

Shear equation: τ_π Dπ^μν/Dτ + π^μν = 2ησ^μν
  For 1D wave: σ_xx = (2/3)ik·δv_x
  → τ_π(-iω)δπ_xx + δπ_xx = 2η(2/3)ik·δv_x
  → (1 - iωτ_π)δπ_xx = (4/3)ηik·δv_x
  → -(4/3)ηik·δv_x + (1 - iωτ_π)δπ_xx = 0  ✓ NEGATIVE sign for velocity
"""

import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2*np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=coeffs
)

k = 8.0
wave_vector = np.array([k, 0.0, 0.0])

# Get analytical mode
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]
omega = complex(mode.frequency, -mode.attenuation)

print("="*80)
print("DISPERSION MATRIX VERIFICATION")
print("="*80)
print()

# Build dispersion matrix
matrix = benchmark.analytical._build_dispersion_matrix(omega, wave_vector)

# Extract thermodynamic parameters
epsilon0 = 1.0
p0 = 1.0 / 3.0
h = epsilon0 + p0
c_s2 = p0 / epsilon0

eta = coeffs.shear_viscosity
zeta = coeffs.bulk_viscosity
tau_pi = coeffs.shear_relaxation_time
tau_Pi = coeffs.bulk_relaxation_time

print(f"Parameters:")
print(f"  k = {k:.3f}")
print(f"  ω = {omega.real:.6f} - i·{-omega.imag:.6f}")
print(f"  h = ε₀ + p₀ = {h:.6f}")
print(f"  c_s² = {c_s2:.6f}")
print(f"  η = {eta:.3f}, τ_π = {tau_pi:.3f}")
print(f"  ζ = {zeta:.3f}, τ_Π = {tau_Pi:.3f}")
print()

print("Dispersion matrix M:")
print("  Variables: [δε, δv_x, δΠ, δπ_xx]")
print()

# Expected matrix based on theory
matrix_expected = np.zeros((4, 4), dtype=np.complex128)

# Row 0: Energy conservation
matrix_expected[0, 0] = -1j * omega
matrix_expected[0, 1] = 1j * k * h

# Row 1: Momentum conservation
# Shear stress acts like pressure - pushes back on fluid
matrix_expected[1, 0] = 1j * k * c_s2
matrix_expected[1, 1] = -1j * omega * h
matrix_expected[1, 2] = 1j * k  # Bulk pressure (positive)
matrix_expected[1, 3] = -1j * k  # Shear stress (NEGATIVE!)

# Row 2: Bulk relaxation (POSITIVE sign for velocity)
matrix_expected[2, 1] = 1j * zeta * k  # +iζk
matrix_expected[2, 2] = 1.0 - 1j * omega * tau_Pi

# Row 3: Shear relaxation (NEGATIVE sign for velocity)
matrix_expected[3, 1] = -1j * (4.0 / 3.0) * eta * k  # -(4/3)ηik
matrix_expected[3, 3] = 1.0 - 1j * omega * tau_pi

print("Expected matrix (from theory):")
for i in range(4):
    row_str = "  ["
    for j in range(4):
        val = matrix_expected[i, j]
        if abs(val) < 1e-10:
            row_str += "     0     "
        else:
            row_str += f"{val.real:+6.3f}{val.imag:+6.3f}j"
        if j < 3:
            row_str += ", "
    row_str += "]"
    print(row_str)
print()

print("Actual matrix (from code):")
for i in range(4):
    row_str = "  ["
    for j in range(4):
        val = matrix[i, j]
        if abs(val) < 1e-10:
            row_str += "     0     "
        else:
            row_str += f"{val.real:+6.3f}{val.imag:+6.3f}j"
        if j < 3:
            row_str += ", "
    row_str += "]"
    print(row_str)
print()

# Check for differences
print("Element-by-element comparison:")
errors = []
for i in range(4):
    for j in range(4):
        diff = matrix[i, j] - matrix_expected[i, j]
        if abs(diff) > 1e-10:
            errors.append((i, j, matrix_expected[i, j], matrix[i, j], diff))
            print(f"  M[{i},{j}]: expected {matrix_expected[i,j]:.6f}, got {matrix[i,j]:.6f}  ✗ MISMATCH")

if not errors:
    print("  ✓ All elements match!")
else:
    print()
    print("="*80)
    print("CRITICAL SIGN ERRORS FOUND!")
    print("="*80)
    for i, j, expected, actual, diff in errors:
        print(f"  Row {i}, col {j}: {actual:.6f} should be {expected:.6f}")
        if i == 2 and j == 1:
            print(f"    → Bulk velocity coupling has WRONG SIGN")
        elif i == 3 and j == 1:
            print(f"    → Shear velocity coupling has WRONG SIGN")

print()
print("Physical interpretation:")
print("  Row 2 (bulk): Should have +iζk (expansion drives negative Π)")
print(f"    Code has: {matrix[2, 1]:.6f}")
print()
print("  Row 3 (shear): Should have -(4/3)ηik (shear drives positive π)")
print(f"    Code has: {matrix[3, 1]:.6f}")
print()

print("="*80)

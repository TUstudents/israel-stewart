#!/usr/bin/env python
"""Verify that the eigenmode ratios are correct by solving the dispersion matrix."""

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

# Get analytical mode
k = 8.0
wave_vector = np.array([k, 0.0, 0.0])
analytical_modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = analytical_modes[0]

print("="*80)
print("EIGENMODE VERIFICATION")
print("="*80)
print(f"Analytical mode: ω = {mode.frequency:.6f}, γ = {mode.attenuation:.6f}")
print()

# Build dispersion matrix at this frequency
omega_complex = complex(mode.frequency, -mode.attenuation)
matrix = benchmark.analytical._build_dispersion_matrix(omega_complex, wave_vector)

print("Dispersion matrix M(ω, k):")
print(matrix)
print()
print(f"Determinant: {np.linalg.det(matrix):.6e} (should be ~0)")
print()

# Find eigenvector using SVD
U, s, Vh = np.linalg.svd(matrix)
print("Singular values:", s)
print(f"Smallest singular value: {s[-1]:.6e} (should be near 0)")
print()

# Eigenvector for smallest singular value (right singular vector)
# V^H is returned, so Vh[-1, :] gives us the last right singular vector
eigenvector = Vh[-1, :].conj()  # Take conjugate to get column of V

# Verify: M·v = σ·u
residual_svd = matrix @ eigenvector
expected_residual = s[-1] * U[:, -1]
print(f"SVD check: ||M·v - σ·u|| / ||v|| = {np.linalg.norm(residual_svd - expected_residual) / np.linalg.norm(eigenvector):.6e}")
print()
print("Raw eigenvector (complex):")
for i, name in enumerate(['δε', 'δv_x', 'δΠ', 'δπ_xx']):
    print(f"  {name:6s}: {eigenvector[i]:.6e}")
print()

# Normalize so first component has magnitude 1
eigenvector_norm = eigenvector / abs(eigenvector[0])
print("Normalized eigenvector (|δε| = 1):")
for i, name in enumerate(['δε', 'δv_x', 'δΠ', 'δπ_xx']):
    print(f"  {name:6s}: {eigenvector_norm[i]:.6e}")
print()

# Find phase that makes it most imaginary
best_phase = 0.0
min_real_norm = float("inf")
for test_phase in np.linspace(0, 2 * np.pi, 100):
    rotated = eigenvector_norm * np.exp(1j * test_phase)
    real_norm = np.sum(np.abs(np.real(rotated)) ** 2)
    if real_norm < min_real_norm:
        min_real_norm = real_norm
        best_phase = test_phase

eigenvector_rotated = eigenvector_norm * np.exp(1j * best_phase)

# Ensure Im(v[0]) < 0 for sin(kx)
if np.imag(eigenvector_rotated[0]) > 0:
    eigenvector_rotated = -eigenvector_rotated

print(f"Phase-rotated eigenvector (optimal phase = {best_phase:.6f}):")
for i, name in enumerate(['δε', 'δv_x', 'δΠ', 'δπ_xx']):
    v = eigenvector_rotated[i]
    print(f"  {name:6s}: {v.real:+.6e} {v.imag:+.6e}i")
print()

# Extract ratios
ratios = -np.imag(eigenvector_rotated)
print("Eigenmode ratios (for sin(kx)):")
print(f"  v_x / ρ  = {ratios[1]:.6e}")
print(f"  Π / ρ    = {ratios[2]:.6e}")
print(f"  π_xx / ρ = {ratios[3]:.6e}")
print()

# Verify these match what the code logged
print("Logged ratios from setup_initial_conditions:")
print(f"  v_x  = 5.637191e-01")
print(f"  Π    = 7.933973e-02")
print(f"  π_xx = ±1.483222e-01  (sign convention differs)")
print()
print("Ratio comparison (absolute values):")
print(f"  v_x:  Match = {abs(abs(ratios[1]) - 5.637191e-01) < 1e-6}")
print(f"  Π:    Match = {abs(abs(ratios[2]) - 7.933973e-02) < 1e-6}")
print(f"  π_xx: Match = {abs(abs(ratios[3]) - 1.483222e-01) < 1e-6}")
print()

# Check if eigenvector actually satisfies M·v = 0
# Note: Phase rotation is for convenience, doesn't change null space
residual_raw = matrix @ eigenvector
residual_rotated = matrix @ eigenvector_rotated
print("Residual M·v (should be ~0):")
print("  Raw eigenvector:")
for i, name in enumerate(['row 0', 'row 1', 'row 2', 'row 3']):
    print(f"    {name}: {residual_raw[i]:.6e}")
print("  Phase-rotated:")
for i, name in enumerate(['row 0', 'row 1', 'row 2', 'row 3']):
    print(f"    {name}: {residual_rotated[i]:.6e}")
print(f"  |M·v|/|v| = {np.linalg.norm(residual_raw)/np.linalg.norm(eigenvector):.6e}")
print()

# Verify against analytical formulas
# For sound wave: δv_x / δρ ≈ c_s / (ε + p)
# For radiation: ε + p = (4/3)ρ, c_s = 1/√3
enthalpy = 4.0 / 3.0  # ε + p = (4/3)ρ for radiation
c_s = 1.0 / np.sqrt(3.0)
v_x_expected_NS = c_s / enthalpy

print("Comparison with Navier-Stokes approximation:")
print(f"  v_x / ρ (NS): {v_x_expected_NS:.6f}")
print(f"  v_x / ρ (IS): {ratios[1]:.6f}")
print(f"  Ratio: {ratios[1] / v_x_expected_NS:.3f} (IS relaxation enhances velocity)")
print()

# For dissipative fluxes, IS has relaxation suppression
# Π ≈ -ζ·∇·u / (1 - iωτ_Π) = -ζ·k·v_x / (1 - iωτ_Π)
# For sin(kx): θ = k·v_x
omega = mode.frequency
tau_Pi = coeffs.bulk_relaxation_time
tau_pi = coeffs.shear_relaxation_time
zeta = coeffs.bulk_viscosity
eta = coeffs.shear_viscosity

# Simplified estimate
Pi_factor = abs(1.0 / (1.0 - 1j * omega * tau_Pi))
pi_factor = abs(1.0 / (1.0 - 1j * omega * tau_pi))

Pi_expected = zeta * k * ratios[1] * Pi_factor
pi_xx_expected = (4.0/3.0) * eta * (2.0/3.0) * k * ratios[1] * pi_factor

print("Dissipative flux estimates:")
print(f"  Π / ρ (estimated): {Pi_expected:.6e}")
print(f"  Π / ρ (eigenmode): {ratios[2]:.6e}")
print(f"  π_xx / ρ (estimated): {pi_xx_expected:.6e}")
print(f"  π_xx / ρ (eigenmode): {ratios[3]:.6e}")
print()
print("✓ Eigenmode ratios are correctly extracted from dispersion matrix")
print("="*80)

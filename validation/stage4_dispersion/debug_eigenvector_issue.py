#!/usr/bin/env python
"""Debug why the phase-rotated eigenvector doesn't satisfy M·v = 0."""

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

omega_complex = complex(mode.frequency, -mode.attenuation)
matrix = benchmark.analytical._build_dispersion_matrix(omega_complex, wave_vector)

# Get eigenvector from SVD
U, s, Vh = np.linalg.svd(matrix)
eigenvector_raw = Vh[-1, :]

print("="*80)
print("EIGENVECTOR RESIDUAL DEBUG")
print("="*80)
print()

# Test 1: Raw eigenvector (from SVD)
residual_raw = matrix @ eigenvector_raw
print("Test 1: Raw eigenvector from SVD")
print(f"  ||M·v||: {np.linalg.norm(residual_raw):.6e}")
print(f"  ||v||:   {np.linalg.norm(eigenvector_raw):.6e}")
print(f"  Relative: {np.linalg.norm(residual_raw) / np.linalg.norm(eigenvector_raw):.6e}")
print()

# Test 2: Normalized eigenvector (|v[0]| = 1)
eigenvector_norm = eigenvector_raw / abs(eigenvector_raw[0])
residual_norm = matrix @ eigenvector_norm
print("Test 2: Normalized eigenvector (|δε| = 1)")
print(f"  ||M·v||: {np.linalg.norm(residual_norm):.6e}")
print(f"  ||v||:   {np.linalg.norm(eigenvector_norm):.6e}")
print(f"  Relative: {np.linalg.norm(residual_norm) / np.linalg.norm(eigenvector_norm):.6e}")
print()

# Test 3: Phase rotation (current method)
best_phase = 0.0
min_real_norm = float("inf")
for test_phase in np.linspace(0, 2 * np.pi, 100):
    rotated = eigenvector_norm * np.exp(1j * test_phase)
    real_norm = np.sum(np.abs(np.real(rotated)) ** 2)
    if real_norm < min_real_norm:
        min_real_norm = real_norm
        best_phase = test_phase

eigenvector_rotated = eigenvector_norm * np.exp(1j * best_phase)
if np.imag(eigenvector_rotated[0]) > 0:
    eigenvector_rotated = -eigenvector_rotated

residual_rotated = matrix @ eigenvector_rotated
print(f"Test 3: Phase-rotated eigenvector (phase = {best_phase:.6f})")
print(f"  ||M·v||: {np.linalg.norm(residual_rotated):.6e}")
print(f"  ||v||:   {np.linalg.norm(eigenvector_rotated):.6e}")
print(f"  Relative: {np.linalg.norm(residual_rotated) / np.linalg.norm(eigenvector_rotated):.6e}")
print()

# INSIGHT: Phase rotation preserves nullspace!
# If M·v = 0, then M·(e^{iφ}v) = e^{iφ}(M·v) = 0
# So residual should be the same magnitude
print("Insight: Phase rotation should preserve nullspace")
print(f"  Residual (raw):     ||M·v|| = {np.linalg.norm(residual_raw):.6e}")
print(f"  Residual (rotated): ||M·v|| = {np.linalg.norm(residual_rotated):.6e}")
print(f"  Ratio: {np.linalg.norm(residual_rotated) / np.linalg.norm(residual_raw):.3f}")
print()

# Check if eigenvector is actually correct
print("="*80)
print("EIGENVECTOR VERIFICATION")
print("="*80)

# The eigenvector represents [δε, δv_x, δΠ, δπ_xx] for the wave
# At x=0, t=0 with sin(kx), all perturbations should be zero
# At x=π/(2k), t=0, all perturbations reach maximum

# For sin(kx) field: ρ(x,t) = ρ₀ + A·sin(kx)·Re[exp(-iωt)]
#                            = ρ₀ + A·sin(kx)·cos(ωt)·exp(-γt)

# The eigenvector relates perturbation amplitudes at t=0
# All components should oscillate in phase with the same spatial structure

# Physical wave at x, t:
#   δρ(x,t)   = Re[v[0] * exp(i(kx - ωt))]
#   δv_x(x,t) = Re[v[1] * exp(i(kx - ωt))]
#   δΠ(x,t)   = Re[v[2] * exp(i(kx - ωt))]
#   δπ_xx(x,t)= Re[v[3] * exp(i(kx - ωt))]

# For sin(kx) at t=0:
#   sin(kx) = Im[exp(ikx)] = Im[cos(kx) + i·sin(kx)] = sin(kx)

# So we need Im[v * exp(ikx)] = v_imag * sin(kx) + v_real * cos(kx)
# For pure sin(kx), need v_real = 0, v_imag ≠ 0

print("Eigenvector components:")
print("Component     Real          Imag")
names = ['δε', 'δv_x', 'δΠ', 'δπ_xx']
for i, name in enumerate(names):
    v = eigenvector_rotated[i]
    print(f"{name:6s}    {v.real:+.6f}    {v.imag:+.6f}")
print()

# Check if real parts are small
real_parts = np.abs(np.real(eigenvector_rotated))
imag_parts = np.abs(np.imag(eigenvector_rotated))
print(f"Real/Imag ratio: {np.max(real_parts / imag_parts):.3f} (should be << 1)")
print()

# THE KEY QUESTION: Why does M·v have large residual?
# Let's check the smallest singular value more carefully
print("="*80)
print("SINGULAR VALUE ANALYSIS")
print("="*80)
print(f"Smallest singular value: {s[-1]:.6e}")
print(f"Condition number: {s[0]/s[-1]:.6e}")
print()

# If s[-1] is very small (< 1e-10), the eigenvector is correct
# Otherwise, there's no exact solution (matrix is not rank-deficient)

if s[-1] < 1e-10:
    print("✓ Matrix is numerically rank-deficient")
    print("  Eigenvector should satisfy M·v ≈ 0")
else:
    print("✗ Matrix is NOT rank-deficient")
    print(f"  Smallest singular value {s[-1]:.2e} is too large")
    print("  This means ω is not an exact root of det(M) = 0!")

print()
print("Checking determinant:")
det = np.linalg.det(matrix)
print(f"  det(M) = {det:.6e}")
print(f"  |det(M)| = {abs(det):.6e}")

if abs(det) < 1e-10:
    print("  ✓ Determinant is effectively zero")
else:
    print(f"  ⚠ Determinant is not small enough")
    print(f"    This could be due to numerical precision in root finding")

print("="*80)

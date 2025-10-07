#!/usr/bin/env python
"""Verify that dissipative fields are initialized to correct eigenmode values."""

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

# Initialize sound wave
k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("="*80)
print("EIGENMODE INITIALIZATION VERIFICATION")
print("="*80)
print()

# Get analytical eigenmode ratios
wave_vector = np.array([k, 0.0, 0.0])
analytical_modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = analytical_modes[0]

omega_complex = complex(mode.frequency, -mode.attenuation)
dispersion_matrix = benchmark.analytical._build_dispersion_matrix(omega_complex, wave_vector)

eigenvalues, eigenvectors = np.linalg.eig(dispersion_matrix)
idx_min = np.argmin(np.abs(eigenvalues))
eigenvector = eigenvectors[:, idx_min]

# Normalize and extract ratios (see setup_initial_conditions)
eigenvector = eigenvector / abs(eigenvector[0])

# Find phase rotation
best_phase = 0.0
min_real_norm = float("inf")
for test_phase in np.linspace(0, 2 * np.pi, 100):
    rotated = eigenvector * np.exp(1j * test_phase)
    real_norm = np.sum(np.abs(np.real(rotated)) ** 2)
    if real_norm < min_real_norm:
        min_real_norm = real_norm
        best_phase = test_phase

eigenvector = eigenvector * np.exp(1j * best_phase)

if np.imag(eigenvector[0]) > 0:
    eigenvector = -eigenvector

# Expected ratios
v_x_ratio_expected = -np.imag(eigenvector[1])
Pi_ratio_expected = -np.imag(eigenvector[2])
pi_xx_ratio_expected = -np.imag(eigenvector[3])

print("Expected eigenmode ratios (analytical):")
print(f"  δv_x / δρ  = {v_x_ratio_expected:.6f}")
print(f"  δΠ / δρ    = {Pi_ratio_expected:.6f}")
print(f"  δπ_xx / δρ = {pi_xx_ratio_expected:.6f}")
print()

# Measure actual ratios from Fourier transform
fields = benchmark.fields
amplitude = 0.01  # Default amplitude

# FFT of fields
rho_fft = np.fft.fftn(fields.rho - 1.0)  # Subtract background
v_x_fft = np.fft.fftn(fields.u_mu[..., 1])
Pi_fft = np.fft.fftn(fields.Pi)
pi_xx_fft = np.fft.fftn(fields.pi_munu[..., 1, 1])

# Extract mode k=8
k_idx = 8
rho_k = rho_fft[k_idx, 0, 0]
v_x_k = v_x_fft[k_idx, 0, 0]
Pi_k = Pi_fft[k_idx, 0, 0]
pi_xx_k = pi_xx_fft[k_idx, 0, 0]

print(f"Fourier amplitudes at k={k_idx}:")
print(f"  |δρ_k|    = {np.abs(rho_k):.6e}")
print(f"  |δv_x_k|  = {np.abs(v_x_k):.6e}")
print(f"  |δΠ_k|    = {np.abs(Pi_k):.6e}")
print(f"  |δπ_xx_k| = {np.abs(pi_xx_k):.6e}")
print()

# Compute actual ratios (compare absolute values)
if np.abs(rho_k) > 1e-10:
    v_x_ratio_actual = np.abs(v_x_k) / np.abs(rho_k)
    Pi_ratio_actual = np.abs(Pi_k) / np.abs(rho_k)
    pi_xx_ratio_actual = np.abs(pi_xx_k) / np.abs(rho_k)

    # Expected ratios (absolute values for comparison)
    v_x_ratio_expected_abs = abs(v_x_ratio_expected)
    Pi_ratio_expected_abs = abs(Pi_ratio_expected)
    pi_xx_ratio_expected_abs = abs(pi_xx_ratio_expected)

    print("Measured eigenmode ratios (from fields):")
    print(f"  |δv_x| / |δρ|  = {v_x_ratio_actual:.6f}")
    print(f"  |δΠ| / |δρ|    = {Pi_ratio_actual:.6f}")
    print(f"  |δπ_xx| / |δρ| = {pi_xx_ratio_actual:.6f}")
    print()

    print("Comparison (measured / expected):")
    print(f"  v_x:   {v_x_ratio_actual / v_x_ratio_expected_abs:.6f}  ", end="")
    if abs(v_x_ratio_actual / v_x_ratio_expected_abs - 1.0) < 0.01:
        print("✓ Match")
    else:
        print(f"✗ Error: {abs(v_x_ratio_actual / v_x_ratio_expected_abs - 1.0) * 100:.1f}%")

    print(f"  Π:     {Pi_ratio_actual / Pi_ratio_expected_abs:.6f}  ", end="")
    if abs(Pi_ratio_actual / Pi_ratio_expected_abs - 1.0) < 0.01:
        print("✓ Match")
    else:
        print(f"✗ Error: {abs(Pi_ratio_actual / Pi_ratio_expected_abs - 1.0) * 100:.1f}%")

    print(f"  π_xx:  {pi_xx_ratio_actual / pi_xx_ratio_expected_abs:.6f}  ", end="")
    if abs(pi_xx_ratio_actual / pi_xx_ratio_expected_abs - 1.0) < 0.01:
        print("✓ Match")
    else:
        print(f"✗ Error: {abs(pi_xx_ratio_actual / pi_xx_ratio_expected_abs - 1.0) * 100:.1f}%")
else:
    print("✗ No density perturbation found!")

print()
print("="*80)

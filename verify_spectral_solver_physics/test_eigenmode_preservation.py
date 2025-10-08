#!/usr/bin/env python
"""
Quick test of eigenmode ratio preservation after the momentum conversion fix.

Tests that complex eigenmode ratios are preserved during short-time evolution
with the linearized momentum-to-velocity conversion.
"""

import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup with zero second-order terms for linear analysis
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
    lambda_pi_pi=0.0,
    lambda_pi_Pi=0.0,
    xi_1=0.0,
    xi_2=0.0,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2*np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=coeffs
)

k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("="*80)
print("EIGENMODE RATIO PRESERVATION TEST (with fix)")
print("="*80)
print()

# Get analytical eigenmode ratios
wave_vector = np.array([k, 0.0, 0.0])
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]

omega_complex = complex(mode.frequency, -mode.attenuation)
dispersion_matrix = benchmark.analytical._build_dispersion_matrix(omega_complex, wave_vector)

U, s, Vh = np.linalg.svd(dispersion_matrix)
eigenvector = Vh[-1, :].conj()

# Normalize to density component
eigenvector = eigenvector / eigenvector[0]

v_x_ratio_expected = eigenvector[1]
Pi_ratio_expected = eigenvector[2]
pi_xx_ratio_expected = eigenvector[3]

print(f"Analytical Complex Eigenmode Ratios:")
print(f"  δv_x/δρ  = {v_x_ratio_expected.real:.4f} + {v_x_ratio_expected.imag:.4f}j")
print(f"  δΠ/δρ    = {Pi_ratio_expected.real:.4f} + {Pi_ratio_expected.imag:.4f}j")
print(f"  δπ_xx/δρ = {pi_xx_ratio_expected.real:.4f} + {pi_xx_ratio_expected.imag:.4f}j")
print()

# Measure initial ratios
k_idx = 8
rho_fft_0 = np.fft.fftn(benchmark.fields.rho - 1.0)
v_fft_0 = np.fft.fftn(benchmark.fields.u_mu[..., 1])
Pi_fft_0 = np.fft.fftn(benchmark.fields.Pi)
pi_fft_0 = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])

rho_k_0 = rho_fft_0[k_idx, 0, 0]
v_k_0 = v_fft_0[k_idx, 0, 0]
Pi_k_0 = Pi_fft_0[k_idx, 0, 0]
pi_k_0 = pi_fft_0[k_idx, 0, 0]

v_ratio_0 = v_k_0 / rho_k_0
Pi_ratio_0 = Pi_k_0 / rho_k_0
pi_ratio_0 = pi_k_0 / rho_k_0

print(f"Initial ratios (t=0):")
print(f"  v_x/ρ  = {v_ratio_0.real:.4f} + {v_ratio_0.imag:.4f}j")
print(f"  Π/ρ    = {Pi_ratio_0.real:.4f} + {Pi_ratio_0.imag:.4f}j")
print(f"  π_xx/ρ = {pi_ratio_0.real:.4f} + {pi_ratio_0.imag:.4f}j")
print()

# Evolve for short time
dx = benchmark.grid.spatial_spacing[0]
dt = min(
    0.5 * dx / max(mode.sound_speed, 0.1),
    0.05 * min(coeffs.bulk_relaxation_time, coeffs.shear_relaxation_time)
)

t_final = 0.5  # Short time test
n_steps = int(t_final / dt)

print(f"Evolving with RK4, dt={dt:.6f}, {n_steps} steps to t={t_final}...")
for _ in range(n_steps):
    benchmark.solver.time_step(dt, method="rk4")

# Measure final ratios
rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
v_fft = np.fft.fftn(benchmark.fields.u_mu[..., 1])
Pi_fft = np.fft.fftn(benchmark.fields.Pi)
pi_fft = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])

rho_k = rho_fft[k_idx, 0, 0]
v_k = v_fft[k_idx, 0, 0]
Pi_k = Pi_fft[k_idx, 0, 0]
pi_k = pi_fft[k_idx, 0, 0]

v_ratio = v_k / rho_k
Pi_ratio = Pi_k / rho_k
pi_ratio = pi_k / rho_k

print()
print(f"Final ratios (t={t_final}):")
print(f"  v_x/ρ  = {v_ratio.real:.4f} + {v_ratio.imag:.4f}j")
print(f"  Π/ρ    = {Pi_ratio.real:.4f} + {Pi_ratio.imag:.4f}j")
print(f"  π_xx/ρ = {pi_ratio.real:.4f} + {pi_ratio.imag:.4f}j")
print()

# Compute percentage changes
v_real_change = abs((v_ratio.real - v_ratio_0.real) / v_ratio_0.real) * 100
v_imag_change = abs((v_ratio.imag - v_ratio_0.imag) / abs(v_ratio_0.imag)) * 100 if abs(v_ratio_0.imag) > 1e-10 else 0
Pi_real_change = abs((Pi_ratio.real - Pi_ratio_0.real) / Pi_ratio_0.real) * 100
Pi_imag_change = abs((Pi_ratio.imag - Pi_ratio_0.imag) / abs(Pi_ratio_0.imag)) * 100 if abs(Pi_ratio_0.imag) > 1e-10 else 0
pi_real_change = abs((pi_ratio.real - pi_ratio_0.real) / pi_ratio_0.real) * 100
pi_imag_change = abs((pi_ratio.imag - pi_ratio_0.imag) / abs(pi_ratio_0.imag)) * 100 if abs(pi_ratio_0.imag) > 1e-10 else 0

print(f"Ratio changes:")
print(f"  v_x  real: {v_real_change:.2f}%  imag: {v_imag_change:.2f}%")
print(f"  Π    real: {Pi_real_change:.2f}%  imag: {Pi_imag_change:.2f}%")
print(f"  π_xx real: {pi_real_change:.2f}%  imag: {pi_imag_change:.2f}%")
print()

# Overall assessment
max_change = max(v_real_change, v_imag_change, Pi_real_change, Pi_imag_change, pi_real_change, pi_imag_change)

print("="*80)
print("ASSESSMENT")
print("="*80)
print()

if max_change < 1.0:
    print("✓ EXCELLENT: Eigenmode ratios preserved to <1%")
    print(f"  Maximum change: {max_change:.2f}%")
    print("  → Linearized momentum conversion fixes the drift!")
elif max_change < 5.0:
    print("✓ GOOD: Eigenmode ratios preserved to <5%")
    print(f"  Maximum change: {max_change:.2f}%")
    print("  → Significant improvement over 30% drift without fix")
elif max_change < 10.0:
    print("⚠ MODERATE: Eigenmode ratios drift by 5-10%")
    print(f"  Maximum change: {max_change:.2f}%")
    print("  → Some improvement but still has issues")
else:
    print("✗ LARGE: Eigenmode ratios drift by >10%")
    print(f"  Maximum change: {max_change:.2f}%")
    print("  → Fix did not resolve the drift")

print()
print("="*80)

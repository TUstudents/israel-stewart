"""
Track how eigenmode ratios drift during evolution.

For a perfect eigenmode, the ratios (v/ρ, Π/ρ, π/ρ) should remain constant.
"""

import numpy as np

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup
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

k = 8.0
benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)
benchmark.setup_initial_conditions(wave_number=k)

# Get analytical eigenmode
wave_vector = np.array([k, 0.0, 0.0])
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]
omega = complex(mode.frequency, -mode.attenuation)

print("=" * 80)
print("EIGENMODE STRUCTURE PRESERVATION")
print("=" * 80)
print()
print(f"Testing k={k}, ω={omega}")
print()

# Get initial ratios
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

print("Initial eigenmode ratios:")
print(f"  v/ρ:  {v_ratio_0}")
print(f"  Π/ρ:  {Pi_ratio_0}")
print(f"  π/ρ:  {pi_ratio_0}")
print()

# Evolve and check ratios
times = [0.01, 0.05, 0.1]
dt = 0.01

for t in times:
    # Evolve to time t
    n_steps = int(t / dt)
    benchmark = NumericalSoundWaveBenchmark(
        domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
    )
    benchmark.setup_initial_conditions(wave_number=k)

    for _ in range(n_steps):
        benchmark.solver.time_step(dt, method="rk4")

    # Get Fourier coefficients
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

    print(f"t = {t:.3f}:")
    print(f"  v/ρ:  {v_ratio}  (error: {abs(v_ratio - v_ratio_0)/abs(v_ratio_0)*100:.2f}%)")
    print(f"  Π/ρ:  {Pi_ratio}  (error: {abs(Pi_ratio - Pi_ratio_0)/abs(Pi_ratio_0)*100:.2f}%)")
    print(f"  π/ρ:  {pi_ratio}  (error: {abs(pi_ratio - pi_ratio_0)/abs(pi_ratio_0)*100:.2f}%)")
    print()

    # Also check if individual fields decay correctly
    expected_rho = rho_k_0 * np.exp(-1j * omega * t)
    expected_v = v_k_0 * np.exp(-1j * omega * t)
    expected_Pi = Pi_k_0 * np.exp(-1j * omega * t)
    expected_pi = pi_k_0 * np.exp(-1j * omega * t)

    rho_decay_error = abs(rho_k - expected_rho) / abs(expected_rho) * 100
    v_decay_error = abs(v_k - expected_v) / abs(expected_v) * 100
    Pi_decay_error = abs(Pi_k - expected_Pi) / abs(expected_Pi) * 100
    pi_decay_error = abs(pi_k - expected_pi) / abs(expected_pi) * 100

    print("  Individual field decay errors:")
    print(f"    ρ:  {rho_decay_error:.2f}%")
    print(f"    v:  {v_decay_error:.2f}%")
    print(f"    Π:  {Pi_decay_error:.2f}%")
    print(f"    π:  {pi_decay_error:.2f}%")
    print()
    print("-" * 80)
    print()

print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
print("If eigenmode structure is preserved:")
print("  - Ratios should remain constant (errors < 1%)")
print("  - Individual fields decay as exp(-iωt)")
print()
print("If eigenmode drifts:")
print("  - Ratios change over time")
print("  - Individual fields don't follow exp(-iωt)")
print("  - This causes RHS to become inaccurate")
print()
print("=" * 80)

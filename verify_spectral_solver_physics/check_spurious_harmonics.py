"""
Check if spurious harmonics are being excited during evolution.

The fundamental is k=8. If spurious modes (k=16, k=24, etc.) grow,
they can interfere with the eigenmode and cause drift.
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

print("=" * 80)
print("SPURIOUS HARMONIC CHECK")
print("=" * 80)
print()
print(f"Fundamental: k={k}")
print()

# Check harmonics at multiple times
times = [0.0, 0.01, 0.05, 0.1]
dt = 0.01

for t in times:
    if t > 0:
        # Evolve to time t
        n_steps = int(t / dt)
        benchmark = NumericalSoundWaveBenchmark(
            domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
        )
        benchmark.setup_initial_conditions(wave_number=k)

        for _ in range(n_steps):
            benchmark.solver.time_step(dt, method="rk4")

    # Get FFTs
    rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
    v_fft = np.fft.fftn(benchmark.fields.u_mu[..., 1])
    Pi_fft = np.fft.fftn(benchmark.fields.Pi)
    pi_fft = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])

    # Get amplitudes at k=8, k=16, k=24
    k8_rho = abs(rho_fft[8, 0, 0])
    k16_rho = abs(rho_fft[16, 0, 0])
    k24_rho = abs(rho_fft[24, 0, 0])

    k8_v = abs(v_fft[8, 0, 0])
    k16_v = abs(v_fft[16, 0, 0])
    k24_v = abs(v_fft[24, 0, 0])

    k8_Pi = abs(Pi_fft[8, 0, 0])
    k16_Pi = abs(Pi_fft[16, 0, 0])
    k24_Pi = abs(Pi_fft[24, 0, 0])

    k8_pi = abs(pi_fft[8, 0, 0])
    k16_pi = abs(pi_fft[16, 0, 0])
    k24_pi = abs(pi_fft[24, 0, 0])

    print(f"t = {t:.3f}:")
    print(
        f"  ρ:  k=8: {k8_rho:.2e}  k=16: {k16_rho:.2e}  k=24: {k24_rho:.2e}  (ratio: {k16_rho/k8_rho:.2e}, {k24_rho/k8_rho:.2e})"
    )
    print(
        f"  v:  k=8: {k8_v:.2e}  k=16: {k16_v:.2e}  k=24: {k24_v:.2e}  (ratio: {k16_v/k8_v:.2e}, {k24_v/k8_v:.2e})"
    )
    print(
        f"  Π:  k=8: {k8_Pi:.2e}  k=16: {k16_Pi:.2e}  k=24: {k24_Pi:.2e}  (ratio: {k16_Pi/k8_Pi:.2e}, {k24_Pi/k8_Pi:.2e})"
    )
    print(
        f"  π:  k=8: {k8_pi:.2e}  k=16: {k16_pi:.2e}  k=24: {k24_pi:.2e}  (ratio: {k16_pi/k8_pi:.2e}, {k24_pi/k8_pi:.2e})"
    )
    print()

    # Check if spurious modes are growing relative to fundamental
    if t > 0:
        threshold = 1e-3  # 0.1% of fundamental
        if k16_rho / k8_rho > threshold or k16_v / k8_v > threshold:
            print(f"  ⚠ WARNING: k=16 mode is significant (>{threshold*100:.1f}% of fundamental)")
        if k24_rho / k8_rho > threshold or k24_v / k8_v > threshold:
            print(f"  ⚠ WARNING: k=24 mode is significant (>{threshold*100:.1f}% of fundamental)")

    print("-" * 80)
    print()

print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
print("Spurious harmonics can arise from:")
print("  1. Nonlinear terms (even if small)")
print("  2. Numerical aliasing")
print("  3. Discretization errors")
print()
print("If k=16 or k=24 modes grow significantly:")
print("  - They interfere with k=8 fundamental")
print("  - Eigenmode structure is corrupted")
print("  - RHS becomes inaccurate")
print()
print("=" * 80)

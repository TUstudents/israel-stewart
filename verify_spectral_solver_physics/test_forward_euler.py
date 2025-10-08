#!/usr/bin/env python
"""
Test eigenmode preservation with simple forward Euler.

If the problem is in RK4's intermediate stages, forward Euler (which only
evaluates RHS once per step) might work better.
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
    domain_size=2*np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=coeffs
)
benchmark.setup_initial_conditions(wave_number=k)

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

print("="*80)
print("FORWARD EULER TEST")
print("="*80)
print()
print(f"Initial eigenmode ratios:")
print(f"  v/ρ:  {v_ratio_0}")
print(f"  Π/ρ:  {Pi_ratio_0}")
print(f"  π/ρ:  {pi_ratio_0}")
print()

# Manual forward Euler evolution
dt = 0.01
t_final = 0.1
n_steps = int(t_final / dt)

print(f"Evolving with forward Euler: dt={dt}, n_steps={n_steps}")
print()

for step in range(n_steps):
    # Get RHS
    rhs = benchmark.solver._compute_full_coupled_rhs(benchmark.fields)

    # Forward Euler update
    benchmark.fields.rho += dt * rhs["drho_dt"]
    benchmark.fields.u_mu[..., 1:4] += dt * rhs["du_dt"]
    benchmark.fields.Pi += dt * rhs["dPi_dt"]
    benchmark.fields.pi_munu += dt * rhs["dpi_munu_dt"]

    # Update pressure (for radiation: p = rho/3)
    benchmark.fields.pressure[:] = benchmark.fields.rho / 3.0

    if (step + 1) % 10 == 0 or step == 0:
        # Check eigenmode ratios
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

        t = (step + 1) * dt
        print(f"t = {t:.3f}:")
        print(f"  v/ρ  error: {abs(v_ratio - v_ratio_0)/abs(v_ratio_0)*100:.2f}%")
        print(f"  Π/ρ  error: {abs(Pi_ratio - Pi_ratio_0)/abs(Pi_ratio_0)*100:.2f}%")
        print(f"  π/ρ  error: {abs(pi_ratio - pi_ratio_0)/abs(pi_ratio_0)*100:.2f}%")
        print()

print("="*80)
print("INTERPRETATION")
print("="*80)
print()
print("If forward Euler works better than RK4:")
print("  - Problem is in RK4's intermediate stages")
print("  - Substep evaluations might use inconsistent field states")
print()
print("If forward Euler has same drift:")
print("  - Problem is more fundamental")
print("  - Could be in RHS computation or eigenmode itself")
print()
print("="*80)

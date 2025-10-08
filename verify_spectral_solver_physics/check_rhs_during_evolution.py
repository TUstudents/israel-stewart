#!/usr/bin/env python
"""
Check if RHS remains correct throughout evolution.

The RHS was verified to be correct at t=0, but what about at later times?
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

# Get analytical eigenmode
wave_vector = np.array([k, 0.0, 0.0])
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]
omega = complex(mode.frequency, -mode.attenuation)

print("="*80)
print("RHS VERIFICATION DURING EVOLUTION")
print("="*80)
print()
print(f"Testing k={k}, ω={omega}")
print()

# Check at multiple time points
times = [0.0, 0.01, 0.05, 0.1]
dt = 0.01

for t in times:
    if t > 0:
        # Evolve to time t
        n_steps = int(t / dt)
        benchmark = NumericalSoundWaveBenchmark(
            domain_size=2*np.pi,
            grid_points=(32, 32, 16),
            transport_coeffs=coeffs
        )
        benchmark.setup_initial_conditions(wave_number=k)

        for _ in range(n_steps):
            benchmark.solver.time_step(dt, method="rk4")

    # Get Fourier coefficients
    k_idx = 8
    rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
    v_fft = np.fft.fftn(benchmark.fields.u_mu[..., 1])
    Pi_fft = np.fft.fftn(benchmark.fields.Pi)
    pi_fft = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])

    rho_k = rho_fft[k_idx, 0, 0]
    v_k = v_fft[k_idx, 0, 0]
    Pi_k = Pi_fft[k_idx, 0, 0]
    pi_k = pi_fft[k_idx, 0, 0]

    # Expected eigenmode time evolution: field(t) = field(0) * exp(-iωt)
    phase_factor = np.exp(-1j * omega * t)

    # Get numerical RHS
    rhs = benchmark.solver._compute_full_coupled_rhs(benchmark.fields)

    drho_dt = rhs["drho_dt"]
    dv_dt = rhs["du_dt"][..., 0]  # x-component
    dPi_dt = rhs["dPi_dt"]
    dpi_xx_dt = rhs["dpi_munu_dt"][..., 1, 1]

    drho_dt_fft = np.fft.fftn(drho_dt)
    dv_dt_fft = np.fft.fftn(dv_dt)
    dPi_dt_fft = np.fft.fftn(dPi_dt)
    dpi_xx_dt_fft = np.fft.fftn(dpi_xx_dt)

    drho_k_dt = drho_dt_fft[k_idx, 0, 0]
    dv_k_dt = dv_dt_fft[k_idx, 0, 0]
    dPi_k_dt = dPi_dt_fft[k_idx, 0, 0]
    dpi_xx_k_dt = dpi_xx_dt_fft[k_idx, 0, 0]

    # Expected from eigenmode: d(field)/dt = -iω * field
    expected_drho_dt = -1j * omega * rho_k
    expected_dv_dt = -1j * omega * v_k
    expected_dPi_dt = -1j * omega * Pi_k
    expected_dpi_dt = -1j * omega * pi_k

    # Compute errors
    rho_error = abs(drho_k_dt - expected_drho_dt) / abs(expected_drho_dt) if abs(expected_drho_dt) > 1e-14 else 0
    v_error = abs(dv_k_dt - expected_dv_dt) / abs(expected_dv_dt) if abs(expected_dv_dt) > 1e-14 else 0
    Pi_error = abs(dPi_k_dt - expected_dPi_dt) / abs(expected_dPi_dt) if abs(expected_dPi_dt) > 1e-14 else 0
    pi_error = abs(dpi_xx_k_dt - expected_dpi_dt) / abs(expected_dpi_dt) if abs(expected_dpi_dt) > 1e-14 else 0

    print(f"t = {t:.3f}:")
    print(f"  Field values:")
    print(f"    ρ(k):    {rho_k}")
    print(f"    v(k):    {v_k}")
    print(f"    Π(k):    {Pi_k}")
    print(f"    π_xx(k): {pi_k}")
    print()
    print(f"  RHS errors:")
    print(f"    dρ/dt:    {rho_error*100:6.2f}%")
    print(f"    dv/dt:    {v_error*100:6.2f}%")
    print(f"    dΠ/dt:    {Pi_error*100:6.2f}%")
    print(f"    dπ_xx/dt: {pi_error*100:6.2f}%")
    print()
    print("-"*80)
    print()

print("="*80)

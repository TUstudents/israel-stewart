#!/usr/bin/env python
"""Test with longer simulation time to allow transients to decay."""

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

# Get analytical solution
k = 8.0
analytical_modes = benchmark.analytical.analyze_dispersion_relation(
    wave_vector=np.array([k, 0.0, 0.0])
)
analytical = analytical_modes[0]

print(f"Analytical: ω = {analytical.frequency:.6f}, γ = {analytical.attenuation:.6f}")
print(f"Wave period: T = {2*np.pi/analytical.frequency:.3f}")
print()

# Calculate timestep
dx = benchmark.grid.spatial_spacing[0]
dt = min(
    0.5 * dx / analytical.sound_speed,
    0.01 * min(coeffs.bulk_relaxation_time, coeffs.shear_relaxation_time)
)
print(f"Using dt = {dt:.6f}")
print()

# Test different simulation times
test_times = [1.0, 2.0, 4.0, 8.0, 16.0]

print("Testing damping measurement vs simulation time:")
print()

for t_final in test_times:
    # Reset
    benchmark.setup_initial_conditions(wave_number=k)

    # Track mode energy
    time_points = []
    energy_list = []

    nx = benchmark.grid_points[0]
    k_index = 8

    def track_energy(t, fields):
        rho_fft = np.fft.fftn(fields.rho - 1.0)
        v_fft = np.fft.fftn(fields.u_mu[..., 1])
        rho_k = np.abs(rho_fft[k_index, 0, 0])
        v_k = np.abs(v_fft[k_index, 0, 0])
        energy = rho_k**2 + (4/3) * v_k**2
        time_points.append(t)
        energy_list.append(energy)

    track_energy(0.0, benchmark.fields)

    # Evolve
    benchmark.solver.evolve(
        t_final=t_final,
        dt=dt,
        method='split_step',
        callback=track_energy
    )

    # Measure damping from energy decay
    time_array = np.array(time_points)
    energy_array = np.array(energy_list)

    # Fit only the second half to avoid initial transients
    mid_idx = len(time_array) // 2
    time_fit = time_array[mid_idx:]
    energy_fit = energy_array[mid_idx:]

    if len(time_fit) > 10:
        log_energy = np.log(energy_fit)
        coeffs_fit = np.polyfit(time_fit, log_energy, 1)
        gamma_second_half = -coeffs_fit[0] / 2  # Energy decays as e^{-2γt}
    else:
        gamma_second_half = 0.0

    # Also fit full time
    log_energy_full = np.log(energy_array)
    coeffs_full = np.polyfit(time_array, log_energy_full, 1)
    gamma_full = -coeffs_full[0] / 2

    print(f"t_final = {t_final:4.1f}:")
    print(f"  γ (full):        {gamma_full:+.6f}")
    print(f"  γ (2nd half):    {gamma_second_half:+.6f}")
    print(f"  γ (analytical):  {analytical.attenuation:+.6f}")
    print(f"  Error (2nd half): {abs((gamma_second_half - analytical.attenuation) / analytical.attenuation * 100):.1f}%")
    print()

print("="*60)
print("If error decreases with longer t_final, transients are the issue")
print("="*60)

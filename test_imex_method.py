#!/usr/bin/env python
"""Test IMEX method vs split-step for damping accuracy."""

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

print("="*80)
print("SPLIT-STEP vs IMEX COMPARISON")
print("="*80)
print(f"Analytical: ω = {analytical.frequency:.6f}, γ = {analytical.attenuation:.6f}")
print()

# Calculate timestep
dx = benchmark.grid.spatial_spacing[0]
dt = min(
    0.5 * dx / analytical.sound_speed,
    0.01 * min(coeffs.bulk_relaxation_time, coeffs.shear_relaxation_time)
)
print(f"Using dt = {dt:.6f}")
print()

# Test both methods
methods = ['split_step', 'spectral_imex']
results = {}

for method in methods:
    print(f"Testing method: {method}")
    print("-" * 80)

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
    t_final = 3.2
    benchmark.solver.evolve(
        t_final=t_final,
        dt=dt,
        method=method,
        callback=track_energy
    )

    # Measure damping
    time_array = np.array(time_points)
    energy_array = np.array(energy_list)

    log_energy = np.log(energy_array)
    coeffs_fit = np.polyfit(time_array, log_energy, 1)
    gamma_measured = -coeffs_fit[0] / 2

    results[method] = {
        'gamma': gamma_measured,
        'time': time_array,
        'energy': energy_array
    }

    print(f"  γ_measured = {gamma_measured:.6f}")
    print(f"  γ_analytical = {analytical.attenuation:.6f}")
    print(f"  Error: {abs((gamma_measured - analytical.attenuation) / analytical.attenuation * 100):.1f}%")
    print()

print("="*80)
print("SUMMARY")
print("="*80)
print(f"Analytical γ:          {analytical.attenuation:.6f}")
print()
print(f"split_step γ:          {results['split_step']['gamma']:.6f}  (error: {abs((results['split_step']['gamma'] - analytical.attenuation) / analytical.attenuation * 100):.1f}%)")
print(f"spectral_imex γ:       {results['spectral_imex']['gamma']:.6f}  (error: {abs((results['spectral_imex']['gamma'] - analytical.attenuation) / analytical.attenuation * 100):.1f}%)")
print()

if abs(results['spectral_imex']['gamma'] - analytical.attenuation) < abs(results['split_step']['gamma'] - analytical.attenuation):
    improvement = abs(results['split_step']['gamma'] - analytical.attenuation) / abs(results['spectral_imex']['gamma'] - analytical.attenuation)
    print(f"✓ IMEX is {improvement:.1f}× more accurate than split-step")
else:
    print("⚠ Split-step is more accurate (unexpected)")

print("="*80)

#!/usr/bin/env -S uv run python
"""
Test stability at lower wavenumber k=1.

Check if the instability is specific to high k or affects all wavenumbers.
"""

import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark, SoundWaveAnalysis
from israel_stewart.core.fields import TransportCoefficients
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.metrics import MinkowskiMetric

k = 1.0  # Low wavenumber
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
)

# Analytical prediction
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(32, 32, 16),
    boundary_conditions="periodic"
)
metric = MinkowskiMetric()
analytical = SoundWaveAnalysis(grid, metric, coeffs)
wave_vector = np.array([k, 0.0, 0.0])
modes = analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]

print("=" * 80)
print(f"STABILITY TEST AT k={k}")
print("=" * 80)
print()
print(f"Analytical predictions:")
print(f"  ω = {mode.frequency:.6f}")
print(f"  γ = {mode.attenuation:.6f}")
print()

# Test both IMEX and RK4
for method_name, method_code in [("IMEX", "spectral_imex"), ("RK4", "rk4")]:
    print(f"Testing {method_name}...")

    benchmark = NumericalSoundWaveBenchmark(
        domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
    )
    benchmark.setup_initial_conditions(wave_number=k)

    k_idx = int(k)
    rho_fft_0 = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]

    # Evolve
    dt = 0.01
    n_steps = 100  # t=1.0
    times = [0.0]
    amplitudes = [abs(rho_fft_0)]

    for i in range(n_steps):
        benchmark.solver.time_step(dt, method=method_code)
        if (i + 1) % 25 == 0:
            t = (i + 1) * dt
            rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
            times.append(t)
            amplitudes.append(abs(rho_fft))

    times = np.array(times)
    amplitudes = np.array(amplitudes)

    # Measure damping
    log_amp = np.log(amplitudes)
    gamma_measured = -(log_amp[-1] - log_amp[0]) / (times[-1] - times[0])

    # Measure frequency
    rho_fft_f = np.fft.fftn(benchmark.fields.rho - 1.0)[k_idx, 0, 0]
    phase = np.angle(rho_fft_f) - np.angle(rho_fft_0)
    if phase < -np.pi:
        phase += 2*np.pi
    elif phase > np.pi:
        phase -= 2*np.pi
    omega_measured = phase / times[-1]

    omega_error = abs(omega_measured - mode.frequency) / mode.frequency * 100

    print(f"  {method_name} Results:")
    print(f"    Frequency: ω = {omega_measured:.6f} (error: {omega_error:.2f}%)")
    print(f"    Damping:   γ = {gamma_measured:+.6f} (analytical: {mode.attenuation:.6f})")

    if gamma_measured > 0:
        damping_error = abs(gamma_measured - mode.attenuation) / mode.attenuation * 100
        print(f"    ✓ Stable (damping error: {damping_error:.1f}%)")
    else:
        print(f"    ✗ UNSTABLE (negative damping)")

    print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
print("If k=1 is stable but k=8 is unstable:")
print("  → High-k numerical instability (likely IMEX splitting issue)")
print()
print("If both are unstable:")
print("  → Fundamental problem with source term formulation")
print()
print("=" * 80)

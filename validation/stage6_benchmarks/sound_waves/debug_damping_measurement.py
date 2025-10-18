#!/usr/bin/env python
"""Debug damping measurement by plotting Fourier mode amplitude."""

import numpy as np
import matplotlib.pyplot as plt
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
print()

# Run simulation with manually tracked Fourier mode
benchmark.setup_initial_conditions(wave_number=k)

# Calculate timestep
dx = benchmark.grid.spatial_spacing[0]
dt = min(
    0.5 * dx / analytical.sound_speed,
    0.01 * min(coeffs.bulk_relaxation_time, coeffs.shear_relaxation_time)
)
print(f"Using dt = {dt:.6f}")

# Evolve and track
time_points = []
rho_k_amplitudes = []

nx = benchmark.grid_points[0]
L_x = 2 * np.pi
k_index = int(round(k * L_x / (2 * np.pi)))
print(f"Tracking Fourier mode at k_index = {k_index}")
print()

# Initial amplitude
rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
rho_k_0 = rho_fft[k_index, 0, 0]
amp_0 = np.abs(rho_k_0)
time_points.append(0.0)
rho_k_amplitudes.append(amp_0)

print(f"Initial amplitude: |ρ_k(0)| = {amp_0:.6e}")

# Callback
def track_mode(t, fields):
    rho_fft = np.fft.fftn(fields.rho - 1.0)
    amp = np.abs(rho_fft[k_index, 0, 0])
    time_points.append(t)
    rho_k_amplitudes.append(amp)

    if len(time_points) % 200 == 0:
        gamma_inst = -np.log(amp / amp_0) / t if t > 0 else 0
        print(f"t={t:.3f}: |ρ_k|={amp:.6e}, γ_inst={gamma_inst:.6f}")

# Evolve
t_final = 3.2
print(f"\nEvolving to t={t_final}...")
benchmark.solver.evolve(
    t_final=t_final,
    dt=dt,
    method='split_step',
    callback=track_mode
)

# Analysis
time_array = np.array(time_points)
amp_array = np.array(rho_k_amplitudes)

# Fit exponential decay
log_amp = np.log(amp_array)
coeffs_fit = np.polyfit(time_array, log_amp, 1)
gamma_measured = -coeffs_fit[0]

print()
print(f"Final amplitude: |ρ_k({t_final:.1f})| = {amp_array[-1]:.6e}")
print(f"Measured γ (linear fit): {gamma_measured:.6f}")
print(f"Analytical γ: {analytical.attenuation:.6f}")
print(f"Error: {abs(gamma_measured - analytical.attenuation) / analytical.attenuation * 100:.1f}%")
print()

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Amplitude vs time
ax1.semilogy(time_array, amp_array, 'o-', label='Numerical', markersize=2)
amp_analytical = amp_0 * np.exp(-analytical.attenuation * time_array)
ax1.semilogy(time_array, amp_analytical, '--', label=f'Analytical (γ={analytical.attenuation:.3f})', linewidth=2)
ax1.set_xlabel('Time')
ax1.set_ylabel('|ρ_k|')
ax1.set_title(f'Fourier Mode Amplitude (k={k})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Log amplitude vs time (to check linearity)
ax2.plot(time_array, log_amp, 'o-', label='Numerical', markersize=2)
fit_line = coeffs_fit[0] * time_array + coeffs_fit[1]
ax2.plot(time_array, fit_line, '--', label=f'Fit (γ={gamma_measured:.3f})', linewidth=2)
log_amp_analytical = np.log(amp_0) - analytical.attenuation * time_array
ax2.plot(time_array, log_amp_analytical, ':', label=f'Analytical (γ={analytical.attenuation:.3f})', linewidth=2)
ax2.set_xlabel('Time')
ax2.set_ylabel('ln|ρ_k|')
ax2.set_title('Log Amplitude (should be linear)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/damping_measurement_debug.png', dpi=150)
print(f"Saved plot to /tmp/damping_measurement_debug.png")

#!/usr/bin/env python
"""Track total mode energy instead of just density amplitude."""

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

# Run simulation
benchmark.setup_initial_conditions(wave_number=k)

# Calculate timestep
dx = benchmark.grid.spatial_spacing[0]
dt = min(
    0.5 * dx / analytical.sound_speed,
    0.01 * min(coeffs.bulk_relaxation_time, coeffs.shear_relaxation_time)
)
print(f"Using dt = {dt:.6f}")

# Track multiple quantities
time_points = []
rho_k_amps = []
v_k_amps = []
energy_k = []

nx = benchmark.grid_points[0]
k_index = 8

# Callback
def track_energy(t, fields):
    # FFTs of perturbations
    rho_fft = np.fft.fftn(fields.rho - 1.0)
    v_fft = np.fft.fftn(fields.u_mu[..., 1])  # v^x

    # Mode amplitudes
    rho_k = np.abs(rho_fft[k_index, 0, 0])
    v_k = np.abs(v_fft[k_index, 0, 0])

    # Total mode energy (ρ² + (ε+p)v²)
    # For radiation: ε+p = 4/3
    energy = rho_k**2 + (4/3) * v_k**2

    time_points.append(t)
    rho_k_amps.append(rho_k)
    v_k_amps.append(v_k)
    energy_k.append(energy)

# Initial state
track_energy(0.0, benchmark.fields)

# Evolve
t_final = 3.2
print(f"\nEvolving to t={t_final}...\n")
benchmark.solver.evolve(
    t_final=t_final,
    dt=dt,
    method='split_step',
    callback=track_energy
)

# Convert to arrays
time_array = np.array(time_points)
rho_k_array = np.array(rho_k_amps)
v_k_array = np.array(v_k_amps)
energy_array = np.array(energy_k)

# Fit exponential to energy (should be cleaner)
log_energy = np.log(energy_array)
coeffs_fit = np.polyfit(time_array, log_energy, 1)
gamma_from_energy = -coeffs_fit[0] / 2  # Energy decays as e^{-2γt}

# Fit to density alone
log_rho = np.log(rho_k_array)
coeffs_rho = np.polyfit(time_array, log_rho, 1)
gamma_from_rho = -coeffs_rho[0]

print(f"Measured γ (from density): {gamma_from_rho:.6f}")
print(f"Measured γ (from energy): {gamma_from_energy:.6f}")
print(f"Analytical γ: {analytical.attenuation:.6f}")
print()

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Density amplitude
axes[0].semilogy(time_array, rho_k_array, 'o-', label='|ρ_k|', markersize=2)
axes[0].set_ylabel('|ρ_k|')
axes[0].set_title('Density Mode Amplitude (oscillates)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Velocity amplitude
axes[1].semilogy(time_array, v_k_array, 'o-', label='|v_k|', markersize=2, color='green')
axes[1].set_ylabel('|v_k|')
axes[1].set_title('Velocity Mode Amplitude (oscillates)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Total energy
axes[2].semilogy(time_array, energy_array, 'o-', label='|ρ_k|² + (4/3)|v_k|²', markersize=2, color='red')
energy_analytical = energy_array[0] * np.exp(-2 * analytical.attenuation * time_array)
axes[2].semilogy(time_array, energy_analytical, '--', label=f'Analytical (γ={analytical.attenuation:.3f})', linewidth=2)
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Mode Energy')
axes[2].set_title(f'Total Mode Energy (should decay smoothly, γ_fit={gamma_from_energy:.3f})')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/energy_tracking_debug.png', dpi=150)
print(f"Saved plot to /tmp/energy_tracking_debug.png")

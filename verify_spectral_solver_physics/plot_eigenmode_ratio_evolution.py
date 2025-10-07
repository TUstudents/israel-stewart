#!/usr/bin/env python
"""
Verify eigenmode ratio preservation during evolution with visualization.

Tracks how eigenmode ratios (v/ρ, Π/ρ, π/ρ) evolve over time and plots results.
This version performs a full complex-valued analysis.
"""

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

k = 8.0
# NOTE: Using the reverted, original initialization logic as requested by user
benchmark.setup_initial_conditions(wave_number=k)

print("="*80)
print("COMPLEX EIGENMODE RATIO EVOLUTION ANALYSIS")
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

# Normalize to the density component
if abs(eigenvector[0]) > 1e-12:
    eigenvector = eigenvector / eigenvector[0]

# These are the expected COMPLEX ratios relative to δρ
v_x_ratio_complex = eigenvector[1]
Pi_ratio_complex = eigenvector[2]
pi_xx_ratio_complex = eigenvector[3]

print(f"Analytical Complex Eigenmode Ratios (relative to δρ):")
print(f"  δv_x/δρ  = {v_x_ratio_complex.real:.4f} + {v_x_ratio_complex.imag:.4f}j")
print(f"  δΠ/δρ    = {Pi_ratio_complex.real:.4f} + {Pi_ratio_complex.imag:.4f}j")
print(f"  δπ_xx/δρ = {pi_xx_ratio_complex.real:.4f} + {pi_xx_ratio_complex.imag:.4f}j")
print()

# Track field evolution
time_points = []
rho_k_list = []
v_k_list = []
Pi_k_list = []
pi_k_list = []

k_idx = 8

def track_fields(t, fields):
    # Store the full complex Fourier component, not just the magnitude
    rho_fft = np.fft.fftn(fields.rho - 1.0)
    v_fft = np.fft.fftn(fields.u_mu[..., 1])
    Pi_fft = np.fft.fftn(fields.Pi)
    pi_fft = np.fft.fftn(fields.pi_munu[..., 1, 1])

    time_points.append(t)
    rho_k_list.append(rho_fft[k_idx, 0, 0])
    v_k_list.append(v_fft[k_idx, 0, 0])
    Pi_k_list.append(Pi_fft[k_idx, 0, 0])
    pi_k_list.append(pi_fft[k_idx, 0, 0])

track_fields(0.0, benchmark.fields)

# Evolve
dx = benchmark.grid.spatial_spacing[0]
dt = min(
    0.5 * dx / max(mode.sound_speed, 0.1),
    0.05 * min(coeffs.bulk_relaxation_time, coeffs.shear_relaxation_time)
)

t_final = 10.0 # Shorter time for quicker analysis

print(f"Evolving with 'spectral_imex', dt={dt:.6f}...")
benchmark.solver.evolve(
    t_final=t_final,
    dt=dt,
    method="rk4",
    callback=track_fields
)
print()

# Analyze evolution of complex ratios
time_array = np.array(time_points)
rho_k = np.array(rho_k_list)
v_k = np.array(v_k_list)
Pi_k = np.array(Pi_k_list)
pi_k = np.array(pi_k_list)

# Calculate complex ratios
v_ratio_t = v_k / rho_k
Pi_ratio_t = Pi_k / rho_k
pi_ratio_t = pi_k / rho_k

print("Evolution of Complex Ratios:")
print("Time   | v_x/ρ (real) | v_x/ρ (imag) | Π/ρ (real)   | Π/ρ (imag)   | π_xx/ρ (real)| π_xx/ρ (imag)|")
print("-"*95)

# Print table at different times
indices_to_print = [0, len(time_array)//4, len(time_array)//2, 3*len(time_array)//4, -1]
for i in indices_to_print:
    t = time_array[i]
    print(f"{t:6.3f} | {v_ratio_t[i].real:12.4f} | {v_ratio_t[i].imag:12.4f} | {Pi_ratio_t[i].real:12.4f} | {Pi_ratio_t[i].imag:12.4f} | {pi_ratio_t[i].real:12.4f} | {pi_ratio_t[i].imag:12.4f} |")

print("-"*95)
print(f"Target | {v_x_ratio_complex.real:12.4f} | {v_x_ratio_complex.imag:12.4f} | {Pi_ratio_complex.real:12.4f} | {Pi_ratio_complex.imag:12.4f} | {pi_xx_ratio_complex.real:12.4f} | {pi_xx_ratio_complex.imag:12.4f} |")
print()

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Evolution of Complex Eigenmode Ratios', fontsize=16)

# Velocity Ratio
axes[0, 0].plot(time_array, v_ratio_t.real, 'g-', label='Real(v/ρ)')
axes[0, 0].plot(time_array, v_ratio_t.imag, 'g--', label='Imag(v/ρ)')
axes[0, 0].axhline(v_x_ratio_complex.real, color='g', linestyle=':', label='Expected Real')
axes[0, 0].axhline(v_x_ratio_complex.imag, color='g', linestyle='-.', label='Expected Imag')
axes[0, 0].set_title('Velocity Ratio')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Bulk Pressure Ratio
axes[0, 1].plot(time_array, Pi_ratio_t.real, 'r-', label='Real(Π/ρ)')
axes[0, 1].plot(time_array, Pi_ratio_t.imag, 'r--', label='Imag(Π/ρ)')
axes[0, 1].axhline(Pi_ratio_complex.real, color='r', linestyle=':', label='Expected Real')
axes[0, 1].axhline(Pi_ratio_complex.imag, color='r', linestyle='-.', label='Expected Imag')
axes[0, 1].set_title('Bulk Pressure Ratio')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Shear Stress Ratio
axes[0, 2].plot(time_array, pi_ratio_t.real, 'm-', label='Real(π/ρ)')
axes[0, 2].plot(time_array, pi_ratio_t.imag, 'm--', label='Imag(π/ρ)')
axes[0, 2].axhline(pi_xx_ratio_complex.real, color='m', linestyle=':', label='Expected Real')
axes[0, 2].axhline(pi_xx_ratio_complex.imag, color='m', linestyle='-.', label='Expected Imag')
axes[0, 2].set_title('Shear Stress Ratio')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Amplitudes
axes[1, 0].semilogy(time_array, np.abs(rho_k), label='|δρ|')
axes[1, 0].semilogy(time_array, np.abs(v_k), label='|δv|')
axes[1, 0].set_title('Amplitudes')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Phase of Ratios
axes[1, 1].plot(time_array, np.angle(v_ratio_t), label='∠(v/ρ)')
axes[1, 1].axhline(np.angle(v_x_ratio_complex), linestyle='--', label='Expected')
axes[1, 1].set_title('Phase of Velocity Ratio')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Hide empty subplot
axes[1, 2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = 'verify_spectral_solver_physics/plot_eigenmode_ratio_evolution.png'
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
print()
print("="*80)

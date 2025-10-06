#!/usr/bin/env python
"""Check if initialization excites only the intended k=8 mode or multiple modes."""

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

# Initialize with k=8 eigenmode
k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("="*80)
print("MODE PURITY CHECK")
print("="*80)
print()

# Check FFT spectrum of initialized fields
fields_list = [
    ('ρ', benchmark.fields.rho - 1.0),
    ('v_x', benchmark.fields.u_mu[..., 1]),
    ('Π', benchmark.fields.Pi),
    ('π_xx', benchmark.fields.pi_munu[..., 1, 1])
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, field) in enumerate(fields_list):
    # 3D FFT
    field_fft = np.fft.fftn(field)

    # Get x-direction spectrum (k_x, k_y=0, k_z=0)
    spectrum_x = np.abs(field_fft[:, 0, 0])

    # k values
    nx = benchmark.grid_points[0]
    k_vals = np.fft.fftfreq(nx, d=benchmark.grid.spatial_spacing[0]) * 2 * np.pi

    # Plot spectrum
    axes[idx].semilogy(np.arange(nx), spectrum_x, 'o-', markersize=4)
    axes[idx].axvline(k, color='red', linestyle='--', label=f'Target k={k}')
    axes[idx].set_xlabel('k-index')
    axes[idx].set_ylabel(f'|{name}_k|')
    axes[idx].set_title(f'{name} Fourier Spectrum')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

    # Print dominant modes
    dominant_indices = np.where(spectrum_x > 0.01 * np.max(spectrum_x))[0]
    print(f"{name} field:")
    print(f"  Max amplitude: {np.max(spectrum_x):.6e} at k-index {np.argmax(spectrum_x)}")
    print(f"  Dominant k-indices (>1% of max): {dominant_indices}")

    # Check mode purity: ratio of k=8 to total
    k_index = int(k)
    total_power = np.sum(spectrum_x**2)
    target_power = spectrum_x[k_index]**2 + spectrum_x[nx - k_index]**2  # k and -k
    purity = target_power / total_power
    print(f"  Mode purity (k=±{k}): {purity*100:.1f}%")

    # Check for unexpected modes
    other_modes = []
    for i in range(nx):
        if i != k_index and i != (nx - k_index) and spectrum_x[i] > 0.01 * spectrum_x[k_index]:
            other_modes.append((i, spectrum_x[i] / spectrum_x[k_index]))

    if other_modes:
        print(f"  ⚠ Other significant modes detected:")
        for i, ratio in other_modes:
            print(f"    k-index {i}: {ratio*100:.1f}% of target amplitude")
    else:
        print(f"  ✓ Clean single-mode initialization")
    print()

plt.tight_layout()
plt.savefig('/tmp/mode_purity.png', dpi=150)
print(f"Saved spectrum plot to /tmp/mode_purity.png")
print()

# Now check if the ratios between fields match the eigenmode at k=8
print("="*80)
print("EIGENMODE RATIO VERIFICATION")
print("="*80)

k_index = int(k)
rho_k = benchmark.fields.rho - 1.0
v_k = benchmark.fields.u_mu[..., 1]
Pi_k = benchmark.fields.Pi
pi_k = benchmark.fields.pi_munu[..., 1, 1]

# FFT
rho_fft = np.fft.fftn(rho_k)
v_fft = np.fft.fftn(v_k)
Pi_fft = np.fft.fftn(Pi_k)
pi_fft = np.fft.fftn(pi_k)

# Get k=8 mode
rho_k8 = rho_fft[k_index, 0, 0]
v_k8 = v_fft[k_index, 0, 0]
Pi_k8 = Pi_fft[k_index, 0, 0]
pi_k8 = pi_fft[k_index, 0, 0]

print(f"Fourier coefficients at k={k}:")
print(f"  ρ_k  = {rho_k8:.6e}")
print(f"  v_k  = {v_k8:.6e}")
print(f"  Π_k  = {Pi_k8:.6e}")
print(f"  π_k  = {pi_k8:.6e}")
print()

# Compute ratios (should match eigenmode)
# Note: for sin(kx), the FFT coefficient is -i*A*N/2
# So ratios of coefficients = ratios of amplitudes
print("Ratios (from Fourier coefficients):")
print(f"  v / ρ    = {v_k8 / rho_k8}")
print(f"  Π / ρ    = {Pi_k8 / rho_k8}")
print(f"  π / ρ    = {pi_k8 / rho_k8}")
print()

# These should be real ratios (both numerator and denominator are purely imaginary)
v_ratio_fft = abs(v_k8 / rho_k8)
Pi_ratio_fft = abs(Pi_k8 / rho_k8)
pi_ratio_fft = abs(pi_k8 / rho_k8)

print("Ratio magnitudes:")
print(f"  |v / ρ|   = {v_ratio_fft:.6f}")
print(f"  |Π / ρ|   = {Pi_ratio_fft:.6f}")
print(f"  |π / ρ|   = {pi_ratio_fft:.6f}")
print()

print("Expected from eigenmode:")
print(f"  v / ρ    = 5.637191e-01")
print(f"  Π / ρ    = 7.933973e-02")
print(f"  π / ρ    = 1.483222e-01")
print()

# Check match
print("Verification:")
print(f"  v ratio: {v_ratio_fft:.6f} vs 0.563719 → Error: {abs(v_ratio_fft - 0.563719)/0.563719*100:.2f}%")
print(f"  Π ratio: {Pi_ratio_fft:.6f} vs 0.079340 → Error: {abs(Pi_ratio_fft - 0.079340)/0.079340*100:.2f}%")
print(f"  π ratio: {pi_ratio_fft:.6f} vs 0.148322 → Error: {abs(pi_ratio_fft - 0.148322)/0.148322*100:.2f}%")
print()

print("="*80)
print("CONCLUSION")
print("="*80)

# Summary
all_pure = all([
    purity > 0.99 for _, field in fields_list
    for purity in [np.sum((np.abs(np.fft.fftn(field)[:, 0, 0])[[k_index, 32-k_index]])**2) /
                   np.sum(np.abs(np.fft.fftn(field)[:, 0, 0])**2)]
])

ratios_match = all([
    abs(v_ratio_fft - 0.563719)/0.563719 < 0.01,
    abs(Pi_ratio_fft - 0.079340)/0.079340 < 0.01,
    abs(pi_ratio_fft - 0.148322)/0.148322 < 0.01
])

if all_pure and ratios_match:
    print("✓ Initialization is a pure k=8 eigenmode")
    print("  Problem must be in the time evolution, not initialization")
else:
    if not all_pure:
        print("✗ Multiple k-modes present in initialization")
        print("  This will cause beating and oscillations")
    if not ratios_match:
        print("✗ Field ratios don't match eigenmode")
        print("  Check eigenmode extraction or field definitions")

print("="*80)

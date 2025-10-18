#!/usr/bin/env python
"""Debug FFT mode extraction for sound waves."""

import numpy as np

# Test case: pure sine wave sin(kx)
k = 8.0
L = 2 * np.pi
nx = 32

x = np.linspace(0, L, nx, endpoint=False)
A = 0.01

# Test field: ρ = 1 + A sin(kx)
rho = 1.0 + A * np.sin(k * x)
delta_rho = rho - 1.0

# FFT
rho_fft = np.fft.fft(delta_rho)

# Expected k-index
k_index = int(round(k * L / (2 * np.pi)))

print(f"Test: sine wave with k={k}, A={A}")
print(f"Grid: nx={nx}, L={L}, dx={L/nx:.4f}")
print(f"k_index = {k_index}")
print()

print("FFT coefficients:")
for i in range(min(12, nx)):
    print(f"  rho_fft[{i}] = {rho_fft[i]:.6e}")

print()
print(f"Mode at k_index={k_index}:")
print(f"  rho_fft[{k_index}] = {rho_fft[k_index]:.6e}")
print(f"  |rho_fft[{k_index}]| = {np.abs(rho_fft[k_index]):.6e}")
print()

# For sin(kx), FFT should give:
# FFT[sin(kx)] = (1/(2i)) * [δ(k-k₀) - δ(k+k₀)]
# So rho_fft[k] = -i*N*A/2 and rho_fft[-k] = +i*N*A/2

expected_amp = nx * A / 2.0
print(f"Expected FFT amplitude: {expected_amp:.6e}")
print(f"Expected rho_fft[{k_index}] ≈ -i × {expected_amp:.4f} = {-1j * expected_amp:.6e}")
print(f"Expected rho_fft[{nx-k_index}] ≈ +i × {expected_amp:.4f} = {1j * expected_amp:.6e}")
print()

# Check negative frequency mode
print(f"Negative frequency mode:")
print(f"  rho_fft[{nx-k_index}] = {rho_fft[nx-k_index]:.6e}")
print(f"  |rho_fft[{nx-k_index}]| = {np.abs(rho_fft[nx-k_index]):.6e}")
print()

# Correct way to get amplitude: average of |rho_fft[k]| and |rho_fft[-k]|
# Or use rFFT which only gives positive frequencies
amp_plus = np.abs(rho_fft[k_index])
amp_minus = np.abs(rho_fft[nx - k_index])

print(f"Amplitudes:")
print(f"  |rho_fft[+k]| = {amp_plus:.6e}")
print(f"  |rho_fft[-k]| = {amp_minus:.6e}")
print(f"  Average: {(amp_plus + amp_minus) / 2:.6e}")
print(f"  Expected: {expected_amp:.6e}")
print()

# Test with rfft (real FFT, only positive frequencies)
rho_rfft = np.fft.rfft(delta_rho)
print(f"Using rfft (real-to-complex):")
print(f"  rho_rfft[{k_index}] = {rho_rfft[k_index]:.6e}")
print(f"  |rho_rfft[{k_index}]| = {np.abs(rho_rfft[k_index]):.6e}")
print()

# 3D case
print("="*60)
print("3D FFT test:")
rho_3d = np.zeros((nx, nx, 16))
X = x[:, None, None]
rho_3d[:] = 1.0 + A * np.sin(k * X)
delta_rho_3d = rho_3d - 1.0

rho_fft_3d = np.fft.fftn(delta_rho_3d)

print(f"  rho_fft_3d[{k_index}, 0, 0] = {rho_fft_3d[k_index, 0, 0]:.6e}")
print(f"  |rho_fft_3d[{k_index}, 0, 0]| = {np.abs(rho_fft_3d[k_index, 0, 0]):.6e}")
print(f"  Expected: {nx * nx * 16 * A / 2:.6e}")
print()

# Recommendation
print("="*60)
print("RECOMMENDATION:")
print("For real-valued fields, FFT gives conjugate pairs.")
print("For sin(kx), the amplitude oscillates if you only track one component.")
print()
print("Solutions:")
print("  1. Use |rho_fft[k]| + |rho_fft[-k]| (sum both components)")
print("  2. Use rfft for real fields")
print("  3. Track the real-space envelope, not Fourier amplitude")

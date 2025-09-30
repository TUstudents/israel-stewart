#!/usr/bin/env python3
"""
Minimal FFT derivative test to identify scaling issue.
"""

import numpy as np

print("=" * 70)
print("MINIMAL FFT DERIVATIVE TEST")
print("=" * 70)

# Test parameters
nx = 16
L = 2 * np.pi
dx = L / nx

print(f"\nGrid: nx={nx}, L={L:.4f}, dx={dx:.4f}")

# Create test field: f(x) = sin(x), df/dx = cos(x)
x = np.linspace(0, L, nx, endpoint=False)
f = np.sin(x)
expected_df = np.cos(x)

print("Test field: f(x) = sin(x)")
print("Expected derivative: df/dx = cos(x)")

# Method 1: NumPy fftfreq with default normalization
print("\n" + "-" * 70)
print("Method 1: fftfreq with manual k-vector")
print("-" * 70)

k = np.fft.fftfreq(nx, dx) * 2 * np.pi
print(f"k-vector from fftfreq(nx, dx) * 2π: {k}")
print("Expected k for sin(x): k=1 (3rd element)")

f_k = np.fft.fft(f)
print(f"\nFFT(sin(x)) magnitudes: {np.abs(f_k)[:nx//2+1]}")
print(f"Peak at k=1: {np.abs(f_k[1]):.3f} (should be ~{nx/2:.1f})")

df_k = 1j * k * f_k
df = np.fft.ifft(df_k).real

error = np.max(np.abs(df - expected_df))
print(f"\nDerivative amplitude: [{np.min(df):.3f}, {np.max(df):.3f}]")
print(f"Expected amplitude: [{np.min(expected_df):.3f}, {np.max(expected_df):.3f}]")
print(f"Max error: {error:.2e}")
print(f"{'✓ PASS' if error < 1e-10 else '✗ FAIL'}")

# Method 2: fftfreq without extra 2π
print("\n" + "-" * 70)
print("Method 2: fftfreq WITHOUT extra 2π")
print("-" * 70)

k2 = np.fft.fftfreq(nx, dx)
print(f"k-vector from fftfreq(nx, dx): {k2}")

df_k2 = 1j * k2 * f_k
df2 = np.fft.ifft(df_k2).real

error2 = np.max(np.abs(df2 - expected_df))
print(f"Derivative amplitude: [{np.min(df2):.3f}, {np.max(df2):.3f}]")
print(f"Max error: {error2:.2e}")
print(f"{'✓ PASS' if error2 < 1e-10 else '✗ FAIL'}")

# Method 3: Manual k-vector
print("\n" + "-" * 70)
print("Method 3: Manual k-vector k = 2π/L * [0, 1, ..., n/2, -n/2, ..., -1]")
print("-" * 70)

k3 = 2 * np.pi / L * np.fft.fftfreq(nx) * nx
print(f"k-vector: {k3}")

df_k3 = 1j * k3 * f_k
df3 = np.fft.ifft(df_k3).real

error3 = np.max(np.abs(df3 - expected_df))
print(f"Derivative amplitude: [{np.min(df3):.3f}, {np.max(df3):.3f}]")
print(f"Max error: {error3:.2e}")
print(f"{'✓ PASS' if error3 < 1e-10 else '✗ FAIL'}")

# Check what the code is doing
print("\n" + "=" * 70)
print("SPECTRAL.PY ANALYSIS")
print("=" * 70)

print("\nCode at line 144-145:")
print("  kx = np.fft.fftfreq(self.nx, self.dx) * 2 * np.pi")
print(f"\nWith nx={nx}, dx={dx:.4f}:")
code_kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
print(f"  kx = {code_kx}")
print("\nThis is CORRECT for derivatives!")

# But let me check the dx calculation
print("\n" + "-" * 70)
print("Checking dx calculation in SpacetimeGrid")
print("-" * 70)

# From SpacetimeGrid:
# spatial_ranges = [(0.0, 2π), (0.0, 2π), (0.0, 2π)]
# grid_points = (10, 16, 16, 16)
# self.dx = (x_max - x_min) / (nx - 1)  # OR is it / nx ?

dx_v1 = L / (nx - 1)  # Including endpoint
dx_v2 = L / nx  # Excluding endpoint

print(f"dx with endpoint=True (/ (nx-1)): {dx_v1:.4f}")
print(f"dx with endpoint=False (/ nx): {dx_v2:.4f}")
print(f"Used dx = {dx:.4f}")

print("\nIf grid uses endpoint=False (periodic), then dx = L/nx is correct")
print("But if SpacetimeGrid calculates dx = L/(nx-1), that's the bug!")

# Test with wrong dx
print("\n" + "-" * 70)
print("TEST: Effect of using dx_wrong = L/(nx-1)")
print("-" * 70)

dx_wrong = L / (nx - 1)
k_wrong = np.fft.fftfreq(nx, dx_wrong) * 2 * np.pi
print(f"k_wrong = fftfreq(nx, {dx_wrong:.4f}) * 2π")
print(f"k_wrong[1] = {k_wrong[1]:.4f} (should be 1.0 for sin(x))")

df_k_wrong = 1j * k_wrong * f_k
df_wrong = np.fft.ifft(df_k_wrong).real

error_wrong = np.max(np.abs(df_wrong - expected_df))
print(f"\nDerivative amplitude: [{np.min(df_wrong):.3f}, {np.max(df_wrong):.3f}]")
print("Expected amplitude: [-1.000, 1.000]")
print(f"Ratio: {np.max(df_wrong) / np.max(expected_df):.4f}")
print(f"Max error: {error_wrong:.2e}")

ratio_wrong = np.max(df_wrong) / np.max(expected_df)
print(f"\nExpected ratio if dx_wrong: {nx/(nx-1):.4f} = 1.0667")
print(f"Observed ratio from diagnostic: 0.938 (inverse: {1/0.938:.4f} = 1.0661)")
print(f"Match? {np.abs(ratio_wrong - 0.938) < 0.01 or np.abs(1/ratio_wrong - 1/0.938) < 0.01}")

print("\n" + "=" * 70)

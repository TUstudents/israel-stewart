"""
Debug the stress tensor divergence computation.

Check if ∂_i(T^ix) is being computed correctly.
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
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)

k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("=" * 80)
print("STRESS TENSOR DIVERGENCE DEBUG")
print("=" * 80)
print()

# Compute stress tensor
T = benchmark.solver.conservation.stress_energy_tensor()

print(f"Stress tensor shape: {T.shape}")
print("  Expected: (32, 32, 16, 4, 4)")
print()

# Extract momentum flux for x-momentum (j=1)
# We need T^ix for i=1,2,3
T_flux_x = T[..., 1:4, 1]  # (T^1x, T^2x, T^3x) = (T^xx, T^yx, T^zx)

print(f"Momentum flux shape: {T_flux_x.shape}")
print("  Expected: (32, 32, 16, 3)")
print()

print("Components at (8, 0, 0):")
print(f"  T^xx: {T[8, 0, 0, 1, 1]:.6e}")
print(f"  T^yx: {T[8, 0, 0, 2, 1]:.6e}")
print(f"  T^zx: {T[8, 0, 0, 3, 1]:.6e}")
print()

# The flux vector should be (T^xx, T^yx, T^zx) at each point
print("Flux vector at (8, 0, 0):")
print(f"  T_flux_x[8, 0, 0, :] = {T_flux_x[8, 0, 0, :]}")
print()

# Compute divergence manually and with solver
div_manual = np.zeros((32, 32, 16))
dx = benchmark.grid.spatial_spacing[0]
dy = benchmark.grid.spatial_spacing[1]
dz = benchmark.grid.spatial_spacing[2]

# ∂_x(T^xx) using spectral derivative
T_xx = T[..., 1, 1]
dT_xx_dx = benchmark.solver.spectral.spatial_derivative(T_xx, direction=0)

# ∂_y(T^yx) using spectral derivative
T_yx = T[..., 2, 1]
dT_yx_dy = benchmark.solver.spectral.spatial_derivative(T_yx, direction=1)

# ∂_z(T^zx) using spectral derivative
T_zx = T[..., 3, 1]
dT_zx_dz = benchmark.solver.spectral.spatial_derivative(T_zx, direction=2)

div_manual = dT_xx_dx + dT_yx_dy + dT_zx_dz

# Using solver's spatial_divergence
div_solver = benchmark.solver.spectral.spatial_divergence(T_flux_x)

print("Divergence at (8, 0, 0):")
print(f"  Manual (sum of derivatives): {div_manual[8, 0, 0]:.6e}")
print(f"  Solver (spatial_divergence): {div_solver[8, 0, 0]:.6e}")
print(f"  Match: {np.allclose(div_manual, div_solver, rtol=1e-10)}")
print()

# FFT to check k=8 mode
T_xx_fft = np.fft.fftn(T_xx)
div_manual_fft = np.fft.fftn(div_manual)
div_solver_fft = np.fft.fftn(div_solver)

k_idx = 8
print(f"Fourier mode k={k}:")
print(f"  T^xx(k):            {T_xx_fft[k_idx, 0, 0]}")
print(f"  ∂_i(T^ix)(k) manual: {div_manual_fft[k_idx, 0, 0]}")
print(f"  ∂_i(T^ix)(k) solver: {div_solver_fft[k_idx, 0, 0]}")
print()

# Expected from analytical
wave_vector = np.array([k, 0.0, 0.0])
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]
omega = complex(mode.frequency, -mode.attenuation)

# For a plane wave propagating in x-direction:
# T^xx ~ cos(kx), T^yx = T^zx = 0
# ∂_i(T^ix) = ∂_x(T^xx) = -k·sin(kx)
#
# In Fourier space: T^xx(k) = T^xx_amplitude
# ∂_x(T^xx) → ik·T^xx(k)

expected_div_fft = 1j * k * T_xx_fft[k_idx, 0, 0]
print(f"Expected ∂_x(T^xx)(k) = ik·T^xx(k): {expected_div_fft}")
print(f"Actual ∂_i(T^ix)(k):                {div_solver_fft[k_idx, 0, 0]}")
print(
    f"Ratio: {div_solver_fft[k_idx, 0, 0] / expected_div_fft if abs(expected_div_fft) > 1e-14 else 'N/A'}"
)
print()

print("=" * 80)

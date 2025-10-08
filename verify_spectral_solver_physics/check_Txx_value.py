#!/usr/bin/env python
"""
Check if T^xx is computed correctly according to the eigenmode.

For a linearized sound wave, T^xx should include contributions from:
1. Pressure perturbation δp
2. Bulk viscosity δΠ
3. Shear stress δπ_xx
4. Kinetic energy term h·u^x² (but this is O(v²), negligible for linear theory)

The Fourier mode k=8 should match the analytical prediction from the eigenmode.
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
    domain_size=2*np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=coeffs
)

k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("="*80)
print("T^xx VALUE CHECK")
print("="*80)
print()

# Compute stress tensor
T = benchmark.solver.conservation.stress_energy_tensor()

# Extract components
rho = benchmark.fields.rho
p = benchmark.fields.pressure
Pi = benchmark.fields.Pi
pi_xx = benchmark.fields.pi_munu[..., 1, 1]
v_x = benchmark.fields.u_mu[..., 1]
h = rho + p

# T^xx components
T_xx = T[..., 1, 1]

# Manual computation
T_xx_manual = h * v_x**2 + p + Pi + pi_xx

print(f"At grid point (8, 0, 0):")
print(f"  h·v_x²: {(h * v_x**2)[8, 0, 0]:.6e}  (kinetic, O(v²))")
print(f"  p:      {p[8, 0, 0]:.6e}  (pressure)")
print(f"  Π:      {Pi[8, 0, 0]:.6e}  (bulk)")
print(f"  π_xx:   {pi_xx[8, 0, 0]:.6e}  (shear)")
print(f"  Sum:    {T_xx_manual[8, 0, 0]:.6e}")
print(f"  Actual: {T_xx[8, 0, 0]:.6e}")
print(f"  Match:  {np.allclose(T_xx_manual[8, 0, 0], T_xx[8, 0, 0])}")
print()

# FFT to get k=8 mode
T_xx_fft = np.fft.fftn(T_xx)
T_xx_manual_fft = np.fft.fftn(T_xx_manual)

p_fft = np.fft.fftn(p)
Pi_fft = np.fft.fftn(Pi)
pi_xx_fft = np.fft.fftn(pi_xx)
kinetic_fft = np.fft.fftn(h * v_x**2)

k_idx = 8
print(f"Fourier mode k={k}:")
print(f"  Pressure(k):  {p_fft[k_idx, 0, 0]}")
print(f"  Bulk(k):      {Pi_fft[k_idx, 0, 0]}")
print(f"  Shear(k):     {pi_xx_fft[k_idx, 0, 0]}")
print(f"  Kinetic(k):   {kinetic_fft[k_idx, 0, 0]}  (O(v²), should be small)")
print(f"  T^xx(k):      {T_xx_fft[k_idx, 0, 0]}")
print(f"  Manual(k):    {T_xx_manual_fft[k_idx, 0, 0]}")
print()

# Now check what the analytical eigenmode predicts
wave_vector = np.array([k, 0.0, 0.0])
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]
omega = complex(mode.frequency, -mode.attenuation)

# Get eigenvector
dispersion_matrix = benchmark.analytical._build_dispersion_matrix(omega, wave_vector)
U, s, Vh = np.linalg.svd(dispersion_matrix)
eigenvector = Vh[-1, :].conj() / Vh[-1, 0].conj()

# Initial Fourier coefficients
rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
v_fft = np.fft.fftn(benchmark.fields.u_mu[..., 1])

rho_k = rho_fft[k_idx, 0, 0]
v_k = v_fft[k_idx, 0, 0]
Pi_k = Pi_fft[k_idx, 0, 0]
pi_k = pi_xx_fft[k_idx, 0, 0]

print(f"Initial Fourier coefficients:")
print(f"  ρ_k:   {rho_k}")
print(f"  v_k:   {v_k}")
print(f"  Π_k:   {Pi_k}")
print(f"  π_k:   {pi_k}")
print()

# For radiation fluid: p = ρ/3, so δp = δρ/3
p_k = rho_k / 3.0

# T^xx in Fourier space (linearized):
# T^xx = p₀ + δp + δΠ + δπ_xx + h₀·(v_x)²
# The k=8 mode only has oscillating parts (δp, δΠ, δπ_xx)
# Kinetic term h₀·v² is second order, gives k=16 mode

T_xx_k_linear = p_k + Pi_k + pi_k

print(f"Analytical prediction (linear theory):")
print(f"  δp(k) = δρ(k)/3:  {p_k}")
print(f"  δΠ(k):            {Pi_k}")
print(f"  δπ_xx(k):         {pi_k}")
print(f"  T^xx(k) expected: {T_xx_k_linear}")
print(f"  T^xx(k) actual:   {T_xx_fft[k_idx, 0, 0]}")
print()

ratio = T_xx_fft[k_idx, 0, 0] / T_xx_k_linear if abs(T_xx_k_linear) > 1e-14 else np.nan
print(f"Ratio (actual/expected): {ratio}")

if np.allclose(T_xx_fft[k_idx, 0, 0], T_xx_k_linear, rtol=0.01):
    print("✓ T^xx(k) matches linear theory prediction")
else:
    error = abs((T_xx_fft[k_idx, 0, 0] - T_xx_k_linear) / T_xx_k_linear) * 100
    print(f"✗ T^xx(k) is WRONG by {error:.2f}%")

print()
print("="*80)

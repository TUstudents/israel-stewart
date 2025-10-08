#!/usr/bin/env python
"""
Compare dispersion matrix prediction for ∂_x(T^xx) with numerical value.

The dispersion matrix row 1 is the momentum equation:
  -iω·δv_x = -ik·(c_s²·δρ + δΠ - δπ_xx)

This should match what we get from -∂_x(T^xx).
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
print("DISPERSION MATRIX VS NUMERICAL")
print("="*80)
print()

# Get analytical eigenmode
wave_vector = np.array([k, 0.0, 0.0])
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]
omega = complex(mode.frequency, -mode.attenuation)

# Build dispersion matrix
dispersion_matrix = benchmark.analytical._build_dispersion_matrix(omega, wave_vector)

print("Dispersion matrix (linearized Israel-Stewart):")
print(dispersion_matrix)
print()

# Row 1 is the momentum equation
print("Row 1 (momentum equation):")
for i, name in enumerate(['δρ', 'δv_x', 'δΠ', 'δπ_xx']):
    print(f"  Coeff of {name:6s}: {dispersion_matrix[1, i]}")
print()

# The momentum equation from the dispersion matrix is:
# -iω·(ε₀+p₀)·δv_x = -ik·(c_s²·δρ + δΠ - δπ_xx)
#
# For radiation: ε₀+p₀ = 4/3, c_s² = 1/3
# So: -iω·(4/3)·δv_x = -ik·(1/3·δρ + δΠ - δπ_xx)
#
# Rearranging: -iω·δv_x = -ik·(1/4·δρ + 3/4·δΠ - 3/4·δπ_xx)

# Get Fourier coefficients
k_idx = 8
rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
v_fft = np.fft.fftn(benchmark.fields.u_mu[..., 1])
Pi_fft = np.fft.fftn(benchmark.fields.Pi)
pi_fft = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])

rho_k = rho_fft[k_idx, 0, 0]
v_k = v_fft[k_idx, 0, 0]
Pi_k = Pi_fft[k_idx, 0, 0]
pi_k = pi_fft[k_idx, 0, 0]

print(f"Initial Fourier coefficients:")
print(f"  δρ(k):    {rho_k}")
print(f"  δv_x(k):  {v_k}")
print(f"  δΠ(k):    {Pi_k}")
print(f"  δπ_xx(k): {pi_k}")
print()

# From dispersion matrix row 1:
# -iω·(4/3)·δv = -ik·(1/3·δρ + δΠ - δπ_xx)
c_s_sq = 1.0 / 3.0
h_0 = 4.0 / 3.0

# RHS of momentum equation from dispersion matrix
rhs_dispersion = -1j * k * (c_s_sq * rho_k + Pi_k - pi_k)

# LHS of momentum equation
lhs_dispersion = -1j * omega * h_0 * v_k

print(f"Dispersion matrix momentum equation:")
print(f"  LHS: -iω·h₀·δv = {lhs_dispersion}")
print(f"  RHS: -ik·(c_s²·δρ + δΠ - δπ_xx) = {rhs_dispersion}")
print(f"  Balance check: {np.allclose(lhs_dispersion, rhs_dispersion, rtol=1e-6)}")
print()

# Now what does the numerical simulation give for ∂_t(h·v)?
T = benchmark.solver.conservation.stress_energy_tensor()
T_flux_x = T[..., 1:4, 1]
div_T = benchmark.solver.spectral.spatial_divergence(T_flux_x)
div_T_fft = np.fft.fftn(div_T)

d_hv_dt_numerical = -div_T_fft[k_idx, 0, 0]

print(f"Numerical simulation:")
print(f"  ∂_t(h·v)(k) = -∂_i(T^ix)(k) = {d_hv_dt_numerical}")
print()

# Compare with dispersion matrix
print(f"Comparison:")
print(f"  Dispersion matrix predicts: ∂_t(h·v) = -ik·(c_s²·δρ + δΠ - δπ_xx) = {rhs_dispersion}")
print(f"  Numerical simulation gives: ∂_t(h·v) = -∂_i(T^ix)            = {d_hv_dt_numerical}")
print(f"  Ratio: {d_hv_dt_numerical / rhs_dispersion if abs(rhs_dispersion) > 1e-14 else 'N/A'}")
print()

# Now let's check if T^xx in the code matches what the dispersion matrix expects
# T^xx = p + Π + π_xx (ignoring kinetic term for linear theory)
# For radiation: p = ρ/3, so δp = δρ/3
# So: δT^xx = δρ/3 + δΠ + δπ_xx

T_xx = T[..., 1, 1]
T_xx_fft = np.fft.fftn(T_xx)
T_xx_k = T_xx_fft[k_idx, 0, 0]

T_xx_expected = rho_k / 3.0 + Pi_k + pi_k

print(f"T^xx components:")
print(f"  δρ/3:       {rho_k / 3.0}")
print(f"  δΠ:         {Pi_k}")
print(f"  δπ_xx:      {pi_k}")
print(f"  Sum:        {T_xx_expected}")
print(f"  Actual:     {T_xx_k}")
print(f"  Match:      {np.allclose(T_xx_k, T_xx_expected, rtol=0.01)}")
print()

# And check ∂_x(T^xx) = ik·T^xx
dT_xx_dx = 1j * k * T_xx_k
print(f"Spatial derivative:")
print(f"  ∂_x(T^xx)(k) = ik·T^xx(k) = {dT_xx_dx}")
print(f"  From dispersion: -ik·(c_s²·δρ + δΠ - δπ_xx) = {rhs_dispersion}")
print()

# These should match!
if np.allclose(dT_xx_dx, rhs_dispersion, rtol=0.01):
    print("✓ ∂_x(T^xx) matches dispersion matrix prediction")
else:
    print("✗ ∂_x(T^xx) DOES NOT match dispersion matrix!")
    print(f"  Error: {abs((dT_xx_dx - rhs_dispersion) / rhs_dispersion) * 100:.2f}%")

print()
print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

# The key question: does ∂_x(T^xx) = ∂_x(p + Π + π) match ∂_x(c_s²·ρ + Π - π)?
# With c_s² = 1/3 and p = ρ/3:
# ∂_x(p + Π + π) = ∂_x(ρ/3 + Π + π)
# ∂_x(c_s²·ρ + Π - π) = ∂_x(ρ/3 + Π - π)
#
# These differ by the SIGN of π!

print("Expected from theory:")
print(f"  ∂_x(T^xx) = ∂_x(p + Π + π_xx)")
print(f"            = ik·(δρ/3 + δΠ + δπ_xx)")
print(f"            = {1j * k * (rho_k/3 + Pi_k + pi_k)}")
print()
print("Dispersion matrix uses:")
print(f"  ∂_x(...) = -ik·(c_s²·δρ + δΠ - δπ_xx)")
print(f"           = -ik·(δρ/3 + δΠ - δπ_xx)")
print(f"           = {-1j * k * (rho_k/3 + Pi_k - pi_k)}")
print()

if not np.allclose(1j * k * (rho_k/3 + Pi_k + pi_k), -1j * k * (rho_k/3 + Pi_k - pi_k), rtol=0.01):
    print("✗ SIGN ERROR: T^xx uses +π_xx but dispersion matrix uses -π_xx!")
    print("  This is the source of the 34% error")
else:
    print("✓ Signs match")

print()
print("="*80)

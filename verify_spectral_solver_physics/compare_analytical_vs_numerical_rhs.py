#!/usr/bin/env python
"""
Compare analytical RHS (from dispersion matrix) vs numerical RHS (from solver).

For a perfect eigenmode, the numerical RHS should match the analytical prediction:
  dφ/dt = -iω·φ

where φ = (δρ, δv, δΠ, δπ) and ω is the complex eigenfrequency.

If they don't match, we can see which component is wrong and diagnose the issue.
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
    xi_1=0.0,
    xi_2=0.0,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2*np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=coeffs
)

k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("="*80)
print("ANALYTICAL VS NUMERICAL RHS COMPARISON")
print("="*80)
print()

# Get analytical eigenmode
wave_vector = np.array([k, 0.0, 0.0])
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]

omega = complex(mode.frequency, -mode.attenuation)
print(f"Eigenmode: ω = {omega.real:.6f} - i·{-omega.imag:.6f}")
print(f"  (frequency = {omega.real:.6f}, damping = {-omega.imag:.6f})")
print()

# Get eigenvector
dispersion_matrix = benchmark.analytical._build_dispersion_matrix(omega, wave_vector)
U, s, Vh = np.linalg.svd(dispersion_matrix)
eigenvector = Vh[-1, :].conj() / Vh[-1, 0].conj()

print("Eigenvector components:")
print(f"  δε:    {eigenvector[0]}")
print(f"  δv_x:  {eigenvector[1]}")
print(f"  δΠ:    {eigenvector[2]}")
print(f"  δπ_xx: {eigenvector[3]}")
print()

# Extract perturbation amplitudes from fields at k=8
k_idx = 8
rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
v_fft = np.fft.fftn(benchmark.fields.u_mu[..., 1])
Pi_fft = np.fft.fftn(benchmark.fields.Pi)
pi_fft = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])

# Fourier coefficients
rho_k = rho_fft[k_idx, 0, 0]
v_k = v_fft[k_idx, 0, 0]
Pi_k = Pi_fft[k_idx, 0, 0]
pi_k = pi_fft[k_idx, 0, 0]

print("Initial Fourier coefficients:")
print(f"  ρ_k:   {rho_k}")
print(f"  v_k:   {v_k}")
print(f"  Π_k:   {Pi_k}")
print(f"  π_k:   {pi_k}")
print()

# Analytical RHS: dφ/dt = -iω·φ
drho_dt_analytical = -1j * omega * rho_k
dv_dt_analytical = -1j * omega * v_k
dPi_dt_analytical = -1j * omega * Pi_k
dpi_dt_analytical = -1j * omega * pi_k

# Numerical RHS from solver
rhs = benchmark.solver._compute_full_coupled_rhs(benchmark.fields)

drho_dt_numerical_real = rhs["drho_dt"]
dv_dt_numerical_real = rhs["du_dt"][..., 0]  # x-component
dPi_dt_numerical_real = rhs["dPi_dt"]
dpi_dt_numerical_real = rhs["dpi_munu_dt"][..., 1, 1]

# FFT of numerical RHS
drho_dt_numerical_fft = np.fft.fftn(drho_dt_numerical_real)
dv_dt_numerical_fft = np.fft.fftn(dv_dt_numerical_real)
dPi_dt_numerical_fft = np.fft.fftn(dPi_dt_numerical_real)
dpi_dt_numerical_fft = np.fft.fftn(dpi_dt_numerical_real)

# Extract k=8 mode
drho_dt_numerical = drho_dt_numerical_fft[k_idx, 0, 0]
dv_dt_numerical = dv_dt_numerical_fft[k_idx, 0, 0]
dPi_dt_numerical = dPi_dt_numerical_fft[k_idx, 0, 0]
dpi_dt_numerical = dpi_dt_numerical_fft[k_idx, 0, 0]

print("="*80)
print("RHS COMPARISON AT k=8")
print("="*80)
print()

print("dρ/dt:")
print(f"  Analytical: {drho_dt_analytical}")
print(f"  Numerical:  {drho_dt_numerical}")
print(f"  Ratio:      {drho_dt_numerical / drho_dt_analytical if abs(drho_dt_analytical) > 1e-14 else 'N/A'}")
print()

print("dv_x/dt:")
print(f"  Analytical: {dv_dt_analytical}")
print(f"  Numerical:  {dv_dt_numerical}")
print(f"  Ratio:      {dv_dt_numerical / dv_dt_analytical if abs(dv_dt_analytical) > 1e-14 else 'N/A'}")
print()

print("dΠ/dt:")
print(f"  Analytical: {dPi_dt_analytical}")
print(f"  Numerical:  {dPi_dt_numerical}")
print(f"  Ratio:      {dPi_dt_numerical / dPi_dt_analytical if abs(dPi_dt_analytical) > 1e-14 else 'N/A'}")
print()

print("dπ_xx/dt:")
print(f"  Analytical: {dpi_dt_analytical}")
print(f"  Numerical:  {dpi_dt_numerical}")
print(f"  Ratio:      {dpi_dt_numerical / dpi_dt_analytical if abs(dpi_dt_analytical) > 1e-14 else 'N/A'}")
print()

# Check if ratios match
print("="*80)
print("RATIO ANALYSIS")
print("="*80)
print()

# If eigenmode is preserved, all d(φ)/dt should have same ratio to φ
# i.e., dρ/dt / ρ = dv/dt / v = dΠ/dt / Π = dπ/dt / π = -iω

ratio_rho = drho_dt_numerical / rho_k if abs(rho_k) > 1e-14 else 0
ratio_v = dv_dt_numerical / v_k if abs(v_k) > 1e-14 else 0
ratio_Pi = dPi_dt_numerical / Pi_k if abs(Pi_k) > 1e-14 else 0
ratio_pi = dpi_dt_numerical / pi_k if abs(pi_k) > 1e-14 else 0

print("d(field)/dt / field (should all equal -iω):")
print(f"  dρ/dt / ρ:     {ratio_rho}")
print(f"  dv/dt / v:     {ratio_v}")
print(f"  dΠ/dt / Π:     {ratio_Pi}")
print(f"  dπ/dt / π:     {ratio_pi}")
print(f"  Expected (-iω): {-1j * omega}")
print()

# Check consistency
ratios_match = (
    np.allclose(ratio_rho, -1j * omega, rtol=0.01) and
    np.allclose(ratio_v, -1j * omega, rtol=0.01) and
    np.allclose(ratio_Pi, -1j * omega, rtol=0.01) and
    np.allclose(ratio_pi, -1j * omega, rtol=0.01)
)

if ratios_match:
    print("✓ All ratios match -iω to within 1%")
    print("  → Eigenmode will be preserved during evolution")
else:
    print("✗ Ratios DO NOT match -iω")
    print("  → Eigenmode will drift during evolution")
    print()
    print("Discrepancies:")
    print(f"  ρ:  {abs((ratio_rho - (-1j * omega)) / (-1j * omega)) * 100:.2f}%")
    print(f"  v:  {abs((ratio_v - (-1j * omega)) / (-1j * omega)) * 100:.2f}%")
    print(f"  Π:  {abs((ratio_Pi - (-1j * omega)) / (-1j * omega)) * 100:.2f}%")
    print(f"  π:  {abs((ratio_pi - (-1j * omega)) / (-1j * omega)) * 100:.2f}%")

print()
print("="*80)

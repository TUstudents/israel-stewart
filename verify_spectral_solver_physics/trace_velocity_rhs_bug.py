#!/usr/bin/env python
"""
Trace the velocity RHS bug step by step.

We know:
- dρ/dt is correct
- dΠ/dt is correct
- dπ/dt is correct
- dv/dt is WRONG (34% error)

This script traces through the momentum equation to find where the bug is.
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
print("VELOCITY RHS BUG TRACE")
print("="*80)
print()

# Get analytical eigenmode
wave_vector = np.array([k, 0.0, 0.0])
modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
mode = modes[0]
omega = complex(mode.frequency, -mode.attenuation)

# Fourier coefficients at k=8
k_idx = 8
rho_fft = np.fft.fftn(benchmark.fields.rho - 1.0)
v_fft = np.fft.fftn(benchmark.fields.u_mu[..., 1])

rho_k = rho_fft[k_idx, 0, 0]
v_k = v_fft[k_idx, 0, 0]

print(f"Initial conditions at k={k}:")
print(f"  ρ_k = {rho_k}")
print(f"  v_k = {v_k}")
print()

# Step 1: Conservation equations give d(momentum_density)/dt
conservation_rhs = benchmark.solver.conservation.evolution_equations()
dmom_dt = conservation_rhs["dmom_dt"]

# FFT of momentum RHS
dmom_dt_fft = np.fft.fftn(dmom_dt[..., 0])
dmom_dt_k = dmom_dt_fft[k_idx, 0, 0]

print(f"Step 1: Conservation equations")
print(f"  d(h·v_x)/dt at k={k}: {dmom_dt_k}")
print()

# For a linearized eigenmode: d(h·v)/dt = h₀·dv/dt (since h ≈ h₀ = 4/3)
# So: dv/dt = d(h·v)/dt / h₀
h_background = 4.0 / 3.0
dv_dt_expected = dmom_dt_k / h_background

print(f"Step 2: Convert momentum to velocity (linear regime)")
print(f"  h₀ = {h_background}")
print(f"  dv_x/dt (expected) = d(h·v)/dt / h₀ = {dv_dt_expected}")
print()

# Analytical expectation
dv_dt_analytical = -1j * omega * v_k
print(f"Step 3: Compare with analytical")
print(f"  dv/dt (analytical) = -iω·v = {dv_dt_analytical}")
print(f"  dv/dt (from d(mom)/dt) = {dv_dt_expected}")
print(f"  Match: {np.allclose(dv_dt_expected, dv_dt_analytical, rtol=0.01)}")
print()

# What does the solver actually give?
rhs = benchmark.solver._compute_full_coupled_rhs(benchmark.fields)
dv_dt_numerical = rhs["du_dt"][..., 0]
dv_dt_numerical_fft = np.fft.fftn(dv_dt_numerical)
dv_dt_numerical_k = dv_dt_numerical_fft[k_idx, 0, 0]

print(f"Step 4: What solver returns")
print(f"  dv/dt (solver) = {dv_dt_numerical_k}")
print()

print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

# Check each step
step1_ok = np.allclose(dmom_dt_k, dv_dt_analytical * h_background, rtol=0.01)
step2_ok = np.allclose(dv_dt_expected, dv_dt_analytical, rtol=0.01)
step3_ok = np.allclose(dv_dt_numerical_k, dv_dt_analytical, rtol=0.01)

print(f"Step 1 (Conservation gives correct d(mom)/dt): {step1_ok}")
if not step1_ok:
    print(f"  Error: {abs((dmom_dt_k - dv_dt_analytical * h_background) / (dv_dt_analytical * h_background)) * 100:.2f}%")

print(f"Step 2 (Conversion d(mom)/dt → dv/dt correct): {step2_ok}")
if not step2_ok:
    print(f"  Error: {abs((dv_dt_expected - dv_dt_analytical) / dv_dt_analytical) * 100:.2f}%")

print(f"Step 3 (Solver returns correct dv/dt): {step3_ok}")
if not step3_ok:
    print(f"  Error: {abs((dv_dt_numerical_k - dv_dt_analytical) / dv_dt_analytical) * 100:.2f}%")

print()

# Additional check: are we actually using the linearized conversion?
print("Conversion check:")
print(f"  dv/dt (manual linear):  {dv_dt_expected}")
print(f"  dv/dt (solver):         {dv_dt_numerical_k}")
print(f"  Match: {np.allclose(dv_dt_expected, dv_dt_numerical_k, rtol=1e-10)}")

if not np.allclose(dv_dt_expected, dv_dt_numerical_k, rtol=1e-10):
    print()
    print("✗ Solver is NOT using the simple d(mom)/dt / h₀ conversion")
    print("  Something else is modifying dv/dt!")
else:
    print()
    print("✓ Solver IS using linearized conversion")
    print("  → Bug must be in conservation equations (d(mom)/dt)")

print()
print("="*80)

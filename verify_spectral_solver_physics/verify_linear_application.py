#!/usr/bin/env python
"""Verify that linear relaxation is applied exactly once, not multiple times."""

import numpy as np
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
benchmark.setup_initial_conditions(wave_number=k)

print("="*80)
print("LINEAR RELAXATION APPLICATION CHECK")
print("="*80)
print()

# Get initial Π and π at mode k
k_idx = 8
Pi_fft_0 = np.fft.fftn(benchmark.fields.Pi)
pi_fft_0 = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])

Pi_k_0 = Pi_fft_0[k_idx, 0, 0]
pi_k_0 = pi_fft_0[k_idx, 0, 0]

print(f"Initial values at k={k}:")
print(f"  Π_k  = {Pi_k_0:.6e}")
print(f"  π_k  = {pi_k_0:.6e}")
print()

# Expected decay from PURE linear relaxation
dt = 0.01
tau_Pi = coeffs.bulk_relaxation_time
tau_pi = coeffs.shear_relaxation_time

exp_factor_Pi = np.exp(-dt / tau_Pi)
exp_factor_pi = np.exp(-dt / tau_pi)

Pi_k_expected_linear = Pi_k_0 * exp_factor_Pi
pi_k_expected_linear = pi_k_0 * exp_factor_pi

print(f"Expected from PURE linear relaxation (exp(-dt/τ)):")
print(f"  Π_k  = {Pi_k_expected_linear:.6e}  (factor: {exp_factor_Pi:.6f})")
print(f"  π_k  = {pi_k_expected_linear:.6e}  (factor: {exp_factor_pi:.6f})")
print()

# Apply split_step with ONE timestep
benchmark.solver.time_step(dt, method="split_step")

# Get final values
Pi_fft_1 = np.fft.fftn(benchmark.fields.Pi)
pi_fft_1 = np.fft.fftn(benchmark.fields.pi_munu[..., 1, 1])

Pi_k_1 = Pi_fft_1[k_idx, 0, 0]
pi_k_1 = pi_fft_1[k_idx, 0, 0]

print(f"Actual after split_step:")
print(f"  Π_k  = {Pi_k_1:.6e}")
print(f"  π_k  = {pi_k_1:.6e}")
print()

# Compare
ratio_Pi = abs(Pi_k_1) / abs(Pi_k_0)
ratio_pi = abs(pi_k_1) / abs(pi_k_0)

print(f"Decay ratios:")
print(f"  Bulk:  {ratio_Pi:.6f}  (expected: {exp_factor_Pi:.6f})")
print(f"  Shear: {ratio_pi:.6f}  (expected: {exp_factor_pi:.6f})")
print()

# split_step applies linear term in 3 steps:
# Step 1: exp(-dt/(2τ))
# Step 3: source terms (should NOT include -Π/τ after fix)
# Step 4: exp(-dt/(2τ))
# Total: exp(-dt/(2τ)) * exp(-dt/(2τ)) = exp(-dt/τ)

expected_total = exp_factor_Pi

if abs(ratio_Pi - expected_total) < 0.001:
    print("✓ Bulk linear relaxation applied correctly (once)")
else:
    print(f"✗ Bulk linear relaxation WRONG!")
    print(f"  Ratio: {ratio_Pi:.6f}")
    print(f"  Expected (1×): {exp_factor_Pi:.6f}")
    print(f"  If double-counted: {exp_factor_Pi**2:.6f}")
    print(f"  If triple-counted: {exp_factor_Pi**3:.6f}")

if abs(ratio_pi - exp_factor_pi) < 0.001:
    print("✓ Shear linear relaxation applied correctly (once)")
else:
    print(f"✗ Shear linear relaxation WRONG!")
    print(f"  Ratio: {ratio_pi:.6f}")
    print(f"  Expected (1×): {exp_factor_pi:.6f}")
    print(f"  If double-counted: {exp_factor_pi**2:.6f}")
    print(f"  If triple-counted: {exp_factor_pi**3:.6f}")

print()
print("="*80)

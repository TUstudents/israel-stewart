#!/usr/bin/env python3
"""
Debug script to check shear tensor and source terms for viscous fluxes.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Create benchmark
transport_coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=transport_coeffs,
)

# Setup initial conditions
wave_number = 1.0
benchmark.setup_initial_conditions(wave_number=wave_number, amplitude=0.01)

print("=" * 80)
print("SHEAR TENSOR AND SOURCE TERMS DEBUG")
print("=" * 80)
print()

# Get initial velocity field
u_mu = benchmark.fields.u_mu
print("Velocity field:")
print(f"  u^t mean: {np.mean(u_mu[..., 0]):.6f}")
print(f"  u^x mean: {np.mean(u_mu[..., 1]):.6e}")
print(f"  u^x std:  {np.std(u_mu[..., 1]):.6e}")
print(f"  u^x max:  {np.max(np.abs(u_mu[..., 1])):.6e}")
print()

# Expected amplitude for k=1, amplitude=0.01 sound wave:
# δu_x ~ A * c_s ~ 0.01 * 0.58 ~ 0.006
expected_ux = 0.01 * 0.589
print(f"Expected u^x amplitude: ~{expected_ux:.6e}")
print(f"Actual u^x amplitude:   {np.max(np.abs(u_mu[..., 1])):.6e}")
print(f"Ratio: {np.max(np.abs(u_mu[..., 1])) / expected_ux:.2f}")
print()

# Compute shear tensor using relaxation equations
if benchmark.solver.relaxation is not None:
    print("Computing shear tensor via relaxation equations...")

    # Access the private method
    shear_tensor = benchmark.solver.relaxation._compute_shear_tensor(u_mu)

    print(f"Shear tensor σ^μν:")
    print(f"  Shape: {shear_tensor.shape}")
    print(f"  Mean:  {np.mean(shear_tensor):.6e}")
    print(f"  Std:   {np.std(shear_tensor):.6e}")
    print(f"  Max:   {np.max(np.abs(shear_tensor)):.6e}")
    print()

    # Expected shear for longitudinal wave: σ_xx ~ ∂u_x/∂x ~ k * A ~ 1.0 * 0.006 ~ 0.006
    expected_shear = wave_number * expected_ux
    print(f"Expected shear magnitude: ~{expected_shear:.6e}")
    print(f"Actual shear magnitude:   {np.max(np.abs(shear_tensor)):.6e}")
    print(f"Ratio: {np.max(np.abs(shear_tensor)) / expected_shear:.2f}")
    print()

    # Compute expected viscous stress: π ~ 2η*σ ~ 2 * 0.08 * 0.006 ~ 0.001
    expected_pi = 2 * transport_coeffs.shear_viscosity * expected_shear
    print(f"Expected viscous stress π ~ 2η*σ: ~{expected_pi:.6e}")
    print()

# Compute expansion scalar
expansion_scalar = benchmark.solver.relaxation._compute_expansion_scalar(u_mu)
print(f"Expansion scalar θ:")
print(f"  Mean:  {np.mean(expansion_scalar):.6e}")
print(f"  Std:   {np.std(expansion_scalar):.6e}")
print(f"  Max:   {np.max(np.abs(expansion_scalar)):.6e}")
print()

# Expected bulk pressure: Π ~ ζ*θ ~ 0.04 * 0.006 ~ 0.0002
expected_Pi = transport_coeffs.bulk_viscosity * expected_shear
print(f"Expected bulk pressure Π ~ ζ*θ: ~{expected_Pi:.6e}")
print()

# Now compute the actual RHS
print("=" * 80)
print("RELAXATION RHS CHECK")
print("=" * 80)
print()

rhs = benchmark.solver.relaxation.compute_relaxation_rhs(benchmark.fields)

# Unpack RHS (Pi, pi_munu, q_mu)
nx, ny, nz = benchmark.grid_points
n_Pi = nx * ny * nz
n_pi = nx * ny * nz * 4 * 4  # 4x4 tensor
n_q = nx * ny * nz * 4  # 4-vector

dPi_dt = rhs[:n_Pi].reshape(nx, ny, nz)
dpi_dt = rhs[n_Pi : n_Pi + n_pi].reshape(nx, ny, nz, 4, 4)

print(f"dΠ/dt:")
print(f"  Mean:  {np.mean(dPi_dt):.6e}")
print(f"  Std:   {np.std(dPi_dt):.6e}")
print(f"  Max:   {np.max(np.abs(dPi_dt)):.6e}")
print()

print(f"dπ^μν/dt:")
print(f"  Mean:  {np.mean(dpi_dt):.6e}")
print(f"  Std:   {np.std(dpi_dt):.6e}")
print(f"  Max:   {np.max(np.abs(dpi_dt)):.6e}")
print()

# Expected RHS magnitude: dπ/dt ~ 2η*σ/τ_π ~ 2*0.08*0.006/0.5 ~ 0.002
expected_dpi_dt = 2 * transport_coeffs.shear_viscosity * expected_shear / transport_coeffs.shear_relaxation_time
print(f"Expected dπ/dt magnitude: ~{expected_dpi_dt:.6e}")
print(f"Actual dπ/dt magnitude:   {np.max(np.abs(dpi_dt)):.6e}")
print(f"Ratio: {np.max(np.abs(dpi_dt)) / expected_dpi_dt:.2f}")

print("=" * 80)

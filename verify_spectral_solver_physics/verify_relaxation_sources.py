#!/usr/bin/env python
"""Verify relaxation equation source terms have correct signs and magnitudes."""

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

# Initialize sound wave
k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("="*80)
print("RELAXATION SOURCE TERM VERIFICATION")
print("="*80)
print()

fields = benchmark.fields
relaxation = benchmark.solver.relaxation

# Compute expansion scalar and shear tensor
expansion = relaxation._compute_expansion_scalar(fields.u_mu)
shear = relaxation._compute_shear_tensor(fields.u_mu)

print("Kinematic quantities:")
print(f"  max(|θ|)     = {np.max(np.abs(expansion)):.6e}")
print(f"  max(|σ_xx|)  = {np.max(np.abs(shear[..., 1, 1])):.6e}")
print(f"  max(|σ_xy|)  = {np.max(np.abs(shear[..., 1, 2])):.6e}")
print()

# Check current dissipative fields
print("Dissipative fields (initial):")
print(f"  max(|Π|)     = {np.max(np.abs(fields.Pi)):.6e}")
print(f"  max(|π_xx|)  = {np.max(np.abs(fields.pi_munu[..., 1, 1])):.6e}")
print()

# Compute RHS components
rhs_flat = relaxation.compute_relaxation_rhs(fields)

# Unpack
Pi_size = fields.Pi.size
pi_munu_size = fields.pi_munu.size

dPi_dt = rhs_flat[:Pi_size].reshape(fields.Pi.shape)
dpi_munu_dt = rhs_flat[Pi_size : Pi_size + pi_munu_size].reshape(fields.pi_munu.shape)

print("Relaxation RHS (total):")
print(f"  max(|dΠ/dt|)      = {np.max(np.abs(dPi_dt)):.6e}")
print(f"  max(|dπ_xx/dt|)   = {np.max(np.abs(dpi_munu_dt[..., 1, 1])):.6e}")
print()

# Decompose bulk RHS
Pi = fields.Pi
theta = expansion

linear_bulk = -Pi / coeffs.bulk_relaxation_time
source_bulk = -coeffs.bulk_viscosity * theta

print("Bulk pressure RHS decomposition:")
print(f"  Linear term (-Π/τ_Π):  max = {np.max(np.abs(linear_bulk)):.6e}")
print(f"  Source term (-ζθ):     max = {np.max(np.abs(source_bulk)):.6e}")
print(f"  Total dΠ/dt:           max = {np.max(np.abs(dPi_dt)):.6e}")
print()

# Check if source dominates
if np.max(np.abs(source_bulk)) > np.max(np.abs(linear_bulk)):
    print(f"  Source term is {np.max(np.abs(source_bulk)) / np.max(np.abs(linear_bulk)):.1f}× larger than linear")
else:
    print(f"  Linear term is {np.max(np.abs(linear_bulk)) / np.max(np.abs(source_bulk)):.1f}× larger than source")
print()

# Decompose shear RHS
pi_munu = fields.pi_munu
sigma_munu = shear

linear_shear = -pi_munu / coeffs.shear_relaxation_time
source_shear = 2.0 * coeffs.shear_viscosity * sigma_munu

print("Shear stress RHS decomposition (xx component):")
print(f"  Linear term (-π/τ_π):  max = {np.max(np.abs(linear_shear[..., 1, 1])):.6e}")
print(f"  Source term (2ησ):     max = {np.max(np.abs(source_shear[..., 1, 1])):.6e}")
print(f"  Total dπ_xx/dt:        max = {np.max(np.abs(dpi_munu_dt[..., 1, 1])):.6e}")
print()

# Check if source dominates
if np.max(np.abs(source_shear[..., 1, 1])) > np.max(np.abs(linear_shear[..., 1, 1])):
    ratio = np.max(np.abs(source_shear[..., 1, 1])) / np.max(np.abs(linear_shear[..., 1, 1]))
    print(f"  Source term is {ratio:.1f}× larger than linear")
else:
    ratio = np.max(np.abs(linear_shear[..., 1, 1])) / np.max(np.abs(source_shear[..., 1, 1]))
    print(f"  Linear term is {ratio:.1f}× larger than source")
print()

# Check signs at peak
k_idx = 8
print(f"Values at Fourier mode k={k_idx}:")
print(f"  θ(k)         = {expansion[k_idx, 0, 0]:.6e}")
print(f"  Π(k)         = {fields.Pi[k_idx, 0, 0]:.6e}")
print(f"  π_xx(k)      = {fields.pi_munu[k_idx, 0, 0, 1, 1]:.6e}")
print(f"  σ_xx(k)      = {shear[k_idx, 0, 0, 1, 1]:.6e}")
print()
print(f"  -Π/τ_Π       = {linear_bulk[k_idx, 0, 0]:.6e}")
print(f"  -ζθ          = {source_bulk[k_idx, 0, 0]:.6e}")
print(f"  dΠ/dt        = {dPi_dt[k_idx, 0, 0]:.6e}")
print()
print(f"  -π_xx/τ_π    = {linear_shear[k_idx, 0, 0, 1, 1]:.6e}")
print(f"  2ησ_xx       = {source_shear[k_idx, 0, 0, 1, 1]:.6e}")
print(f"  dπ_xx/dt     = {dpi_munu_dt[k_idx, 0, 0, 1, 1]:.6e}")
print()

# Expected from dispersion relation
print("Expected behavior:")
print("  - For sound wave compression (θ > 0): Π should grow (dΠ/dt > 0)")
print("  - Source -ζθ < 0 drives Π negative (compression reduces pressure)")
print("  - Linear -Π/τ drives Π back to zero")

print("="*80)

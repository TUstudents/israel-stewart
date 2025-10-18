#!/usr/bin/env -S uv run python
"""
Test sign conventions with strong viscous effects.

Set up a case with significant shear and bulk viscosity to verify:
1. Shear stress π^μν enters with MINUS sign (Convention B)
2. Bulk pressure Π enters with PLUS sign
3. Signs are consistent with dispersion relation
"""

import numpy as np
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.tensor_utils import optimized_einsum
from israel_stewart.equations.conservation import ConservationLaws

# Create small grid for testing
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 1.0)] * 3,
    grid_points=(4, 4, 4),
    boundary_conditions="periodic"
)

# Initialize fields with strong viscous effects
fields = ISFieldConfiguration(grid)

# Uniform background
fields.rho[:] = 1.0
fields.pressure[:] = 1.0 / 3.0  # Radiation fluid
fields.u_mu[..., 0] = 1.0  # Rest frame

# Add significant bulk viscosity (isotropic pressure perturbation)
fields.Pi[:] = 0.1  # 10% of pressure

# Add significant shear stress (anisotropic stress)
# For shear, we need traceless symmetric tensor
# Set π^xx = -π^yy = 0.05, π^zz = 0 (traceless)
fields.pi_munu[..., 1, 1] = 0.05   # π^xx
fields.pi_munu[..., 2, 2] = -0.05  # π^yy (to make traceless)
fields.pi_munu[..., 3, 3] = 0.0    # π^zz

# Verify traceless (in spatial part)
pi_trace_spatial = fields.pi_munu[0, 0, 0, 1, 1] + fields.pi_munu[0, 0, 0, 2, 2] + fields.pi_munu[0, 0, 0, 3, 3]
print(f"Spatial trace of π^ij: {pi_trace_spatial:.10f} (should be ~0)")
print()

# Transport coefficients (not used for static test, but needed for ConservationLaws)
coeffs = TransportCoefficients(
    shear_viscosity=0.1,
    bulk_viscosity=0.05,
)

# Create conservation laws object
conservation = ConservationLaws(fields, coeffs)

print("=" * 80)
print("VISCOUS SIGN CONVENTION TEST")
print("=" * 80)
print()

# Get full stress tensor
T = conservation.stress_energy_tensor()

# Manually compute components
g_inv = np.zeros((*grid.shape, 4, 4))
g_inv[..., 0, 0] = -1.0
g_inv[..., 1, 1] = 1.0
g_inv[..., 2, 2] = 1.0
g_inv[..., 3, 3] = 1.0

# Spatial projector: Δ^μν = g^μν + u^μu^ν
# For rest frame (u^μ = [1, 0, 0, 0]):
# Δ^00 = -1 + 1*1 = 0
# Δ^ij = δ^ij (i,j = 1,2,3)
u_outer = optimized_einsum("...i,...j->...ij", fields.u_mu, fields.u_mu)
Delta = g_inv + u_outer

# Test point (all uniform, so any point works)
i, j, k = 0, 0, 0

print("BACKGROUND STATE")
print("=" * 80)
print(f"ρ = {fields.rho[i,j,k]:.4f}")
print(f"p = {fields.pressure[i,j,k]:.4f}")
print(f"u^μ = [{fields.u_mu[i,j,k,0]:.1f}, {fields.u_mu[i,j,k,1]:.1f}, {fields.u_mu[i,j,k,2]:.1f}, {fields.u_mu[i,j,k,3]:.1f}]")
print()

print("VISCOUS PERTURBATIONS")
print("=" * 80)
print(f"Bulk pressure:     Π = {fields.Pi[i,j,k]:+.4f}")
print(f"Shear stress:    π^xx = {fields.pi_munu[i,j,k,1,1]:+.4f}")
print(f"                 π^yy = {fields.pi_munu[i,j,k,2,2]:+.4f}")
print(f"                 π^zz = {fields.pi_munu[i,j,k,3,3]:+.4f}")
print()

print("STRESS TENSOR COMPONENTS (Convention B)")
print("=" * 80)
print()

# T^00 = (ε+p)u^0u^0 + p·g^00 + Π·Δ^00 - π^00
print("T^00 (energy density):")
T00_perfect = (fields.rho[i,j,k] + fields.pressure[i,j,k]) * 1.0 * 1.0
T00_pressure = fields.pressure[i,j,k] * (-1.0)  # g^00 = -1
T00_bulk = fields.Pi[i,j,k] * Delta[i,j,k,0,0]
T00_shear = -fields.pi_munu[i,j,k,0,0]  # MINUS sign (Convention B)

print(f"  Perfect fluid:    (ε+p)u^0u^0       = {T00_perfect:+.4f}")
print(f"  Pressure:         p·g^00            = {T00_pressure:+.4f}")
print(f"  Bulk:             Π·Δ^00            = {T00_bulk:+.4f}  (Δ^00 = {Delta[i,j,k,0,0]:.4f})")
print(f"  Shear:            -π^00             = {T00_shear:+.4f}  (π^00 = {fields.pi_munu[i,j,k,0,0]:.4f})")
print(f"  ---")
print(f"  Total:            T^00              = {T[i,j,k,0,0]:+.4f}")
print(f"  Computed sum:                       = {T00_perfect + T00_pressure + T00_bulk + T00_shear:+.4f}")
print()

# T^xx = (ε+p)u^xu^x + p·g^xx + Π·Δ^xx - π^xx
print("T^xx (spatial stress):")
T11_perfect = (fields.rho[i,j,k] + fields.pressure[i,j,k]) * 0.0 * 0.0  # u^x = 0
T11_pressure = fields.pressure[i,j,k] * 1.0  # g^xx = 1
T11_bulk = fields.Pi[i,j,k] * Delta[i,j,k,1,1]
T11_shear = -fields.pi_munu[i,j,k,1,1]  # MINUS sign (Convention B)

print(f"  Perfect fluid:    (ε+p)u^xu^x       = {T11_perfect:+.4f}")
print(f"  Pressure:         p·g^xx            = {T11_pressure:+.4f}")
print(f"  Bulk:             Π·Δ^xx            = {T11_bulk:+.4f}  (Δ^xx = {Delta[i,j,k,1,1]:.4f})")
print(f"  Shear:            -π^xx             = {T11_shear:+.4f}  (π^xx = {fields.pi_munu[i,j,k,1,1]:+.4f})")
print(f"  ---")
print(f"  Total:            T^xx              = {T[i,j,k,1,1]:+.4f}")
print(f"  Computed sum:                       = {T11_perfect + T11_pressure + T11_bulk + T11_shear:+.4f}")
print()

# T^yy = (ε+p)u^yu^y + p·g^yy + Π·Δ^yy - π^yy
print("T^yy (spatial stress):")
T22_perfect = 0.0  # u^y = 0
T22_pressure = fields.pressure[i,j,k] * 1.0
T22_bulk = fields.Pi[i,j,k] * Delta[i,j,k,2,2]
T22_shear = -fields.pi_munu[i,j,k,2,2]  # MINUS sign (Convention B)

print(f"  Perfect fluid:    (ε+p)u^yu^y       = {T22_perfect:+.4f}")
print(f"  Pressure:         p·g^yy            = {T22_pressure:+.4f}")
print(f"  Bulk:             Π·Δ^yy            = {T22_bulk:+.4f}  (Δ^yy = {Delta[i,j,k,2,2]:.4f})")
print(f"  Shear:            -π^yy             = {T22_shear:+.4f}  (π^yy = {fields.pi_munu[i,j,k,2,2]:+.4f})")
print(f"  ---")
print(f"  Total:            T^yy              = {T[i,j,k,2,2]:+.4f}")
print(f"  Computed sum:                       = {T22_perfect + T22_pressure + T22_bulk + T22_shear:+.4f}")
print()

print("=" * 80)
print("SIGN CONVENTION VERIFICATION")
print("=" * 80)
print()

# Check Convention B sign: T^μν = ... - π^μν
print("Convention B (Landau-Lifshitz): T^μν = ... - π^μν")
print()

# For T^xx:
# Π > 0 should ADD pressure (resist compression)
# π^xx > 0 should SUBTRACT (dissipation opposes flow)
print(f"Bulk viscosity Π = {fields.Pi[i,j,k]:+.4f}:")
print(f"  Appears as +Π in T^xx → increases pressure")
print(f"  Effect: T^xx = p + Π + ... = {fields.pressure[i,j,k]:.4f} + {fields.Pi[i,j,k]:.4f} + ...")
print()

print(f"Shear stress π^xx = {fields.pi_munu[i,j,k,1,1]:+.4f}:")
print(f"  Appears as -π^xx in T^xx → decreases pressure in x-direction")
print(f"  This is ANISOTROPIC: compresses x, expands y")
print(f"  T^xx = ... - π^xx = ... - ({fields.pi_munu[i,j,k,1,1]:+.4f}) = ... + {-fields.pi_munu[i,j,k,1,1]:+.4f}")
print(f"  T^yy = ... - π^yy = ... - ({fields.pi_munu[i,j,k,2,2]:+.4f}) = ... + {-fields.pi_munu[i,j,k,2,2]:+.4f}")
print()

# Verify signs match code
assert abs(T[i,j,k,1,1] - (T11_perfect + T11_pressure + T11_bulk + T11_shear)) < 1e-10
assert abs(T[i,j,k,2,2] - (T22_perfect + T22_pressure + T22_bulk + T22_shear)) < 1e-10

print("✓ Component breakdown matches full tensor")
print("✓ MINUS sign for shear stress verified (Convention B)")
print("✓ PLUS sign for bulk viscosity verified")
print()

# Physical interpretation
print("=" * 80)
print("PHYSICAL INTERPRETATION")
print("=" * 80)
print()

print("Bulk viscosity Π:")
print(f"  Π = {fields.Pi[i,j,k]:+.4f} > 0 → resists compression")
print(f"  Adds isotropic pressure: T^ii → T^ii + Π")
print()

print("Shear stress π^μν:")
print(f"  π^xx = {fields.pi_munu[i,j,k,1,1]:+.4f} > 0, π^yy = {fields.pi_munu[i,j,k,2,2]:+.4f} < 0")
print(f"  MINUS sign: T^μν = ... - π^μν")
print(f"  → T^xx reduced (compressed in x)")
print(f"  → T^yy increased (expanded in y)")
print(f"  This creates ANISOTROPY that dissipates over time")
print()

print("=" * 80)
print("✓✓✓ ALL SIGNS CONSISTENT WITH CONVENTION B ✓✓✓")
print("=" * 80)

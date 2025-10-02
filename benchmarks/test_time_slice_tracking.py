#!/usr/bin/env python3
"""
Detailed test: Track which time slices are modified by time_step()

This test checks whether time_step() operates as:
A) 4D spacetime refinement (modifies all time slices)
B) 3+1D evolution (modifies only current/latest slice)
"""

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics

# Create grid with multiple time slices
nt, nx, ny, nz = 10, 8, 8, 8
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),
    spatial_ranges=[(0.0, 1.0)] * 3,
    grid_points=(nt, nx, ny, nz),
    boundary_conditions="periodic",
)

print(f"Grid shape: {grid.shape}")
print(f"Time coordinates: {grid.coordinates['t']}")
print()

# Create fields and initialize with unique pattern at each time slice
fields = ISFieldConfiguration(grid)

# Get spatial coordinates
x = grid.coordinates["x"]
y = grid.coordinates["y"]
z = grid.coordinates["z"]
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Create spatial density perturbation (Gaussian)
r_squared = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
gaussian = np.exp(-r_squared / (2 * 0.1**2))

# Initialize each time slice with spatial structure + unique offset
for t_idx in range(nt):
    # Each time slice gets unique offset + same spatial structure
    unique_offset = 0.1 * t_idx
    fields.rho[t_idx, :, :, :] = 1.0 + unique_offset + 0.01 * gaussian
    fields.pressure[t_idx, :, :, :] = fields.rho[t_idx, :, :, :] / 3.0

    # Velocity field proportional to gradient (should cause evolution)
    u_x = 0.01 * gaussian
    fields.u_mu[t_idx, :, :, :, 0] = np.sqrt(1.0 + u_x**2)
    fields.u_mu[t_idx, :, :, :, 1] = u_x
    fields.u_mu[t_idx, :, :, :, 2] = 0.0
    fields.u_mu[t_idx, :, :, :, 3] = 0.0

print("Initial state - each time slice has unique value:")
print("At corner (0,0,0):")
for t_idx in range(nt):
    print(f"  Slice {t_idx}: rho = {fields.rho[t_idx, 0, 0, 0]:.4f}")

# Also check at center where Gaussian peaks
mid_idx = nx // 2
print(f"\nAt center ({mid_idx},{mid_idx},{mid_idx}):")
for t_idx in range(nt):
    print(f"  Slice {t_idx}: rho = {fields.rho[t_idx, mid_idx, mid_idx, mid_idx]:.4f}")
print()

# Store initial values
rho_initial = fields.rho.copy()

# Create solver and take ONE time step
coeffs = TransportCoefficients(shear_viscosity=0.0, bulk_viscosity=0.0)
hydro = SpectralISHydrodynamics(grid, fields, coeffs)

dt = 0.01
print(f"Calling time_step(dt={dt})...")
hydro.time_step(dt)
print("✅ time_step() completed")
print()

# Check which time slices changed
print("After time_step - checking for changes:")
print("At corner (0,0,0):")
changes_corner = []
for t_idx in range(nt):
    rho_before = rho_initial[t_idx, 0, 0, 0]
    rho_after = fields.rho[t_idx, 0, 0, 0]
    delta = abs(rho_after - rho_before)
    changed = delta > 1e-10
    changes_corner.append(changed)

    status = "CHANGED" if changed else "unchanged"
    print(f"  Slice {t_idx}: rho = {rho_after:.6f} (was {rho_before:.6f}) - {status}")

print(f"\nAt center ({mid_idx},{mid_idx},{mid_idx}):")
changes_center = []
for t_idx in range(nt):
    rho_before = rho_initial[t_idx, mid_idx, mid_idx, mid_idx]
    rho_after = fields.rho[t_idx, mid_idx, mid_idx, mid_idx]
    delta = abs(rho_after - rho_before)
    changed = delta > 1e-10
    changes_center.append(changed)

    status = "CHANGED" if changed else "unchanged"
    print(
        f"  Slice {t_idx}: rho = {rho_after:.6f} (was {rho_before:.6f}, Δ={delta:.2e}) - {status}"
    )

# Use center changes for analysis
changes = changes_center

print()
print("=" * 70)
print("ANALYSIS:")
print("=" * 70)

if all(changes):
    print("✅ ALL time slices were modified")
    print("   → This suggests 4D spacetime refinement behavior")
    print("   → OR buggy code that updates all slices incorrectly")
elif changes[-1] and not any(changes[:-1]):
    print("✅ ONLY the LAST time slice was modified")
    print("   → This confirms 3+1D time evolution behavior")
    print("   → Latest slice stores current state")
elif any(changes):
    print(f"⚠️  SOME time slices were modified: {sum(changes)}/{nt}")
    print(f"   → Modified slices: {[i for i, c in enumerate(changes) if c]}")
    print("   → Unexpected behavior - needs investigation")
else:
    print("❌ NO time slices were modified")
    print("   → time_step() had no effect!")

print()
print("Additional check - are all modified slices identical?")
modified_slices = [fields.rho[i, :, :, :] for i, c in enumerate(changes) if c]
if len(modified_slices) > 1:
    all_same = all(np.allclose(modified_slices[0], s) for s in modified_slices[1:])
    if all_same:
        print("   → YES, all modified slices have identical values")
        print("   → This suggests they were ALL updated with the SAME formula")
        print("   → Likely bug: treating 4D array as single state")
    else:
        print("   → NO, modified slices have different values")
        print("   → Each slice was updated differently (expected for refinement)")

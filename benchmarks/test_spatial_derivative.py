#!/usr/bin/env python3
"""
Test spatial derivative behavior on 4D fields
"""

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.spectral import SpectralISolver

# Create grid
nt, nx, ny, nz = 5, 8, 8, 8
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),
    spatial_ranges=[(0.0, 1.0)] * 3,
    grid_points=(nt, nx, ny, nz),
    boundary_conditions="periodic",
)

# Create fields with LINEAR variation in x, DIFFERENT at each time slice
fields = ISFieldConfiguration(grid)
x = grid.coordinates["x"]

for t_idx in range(nt):
    time_offset = t_idx * 0.5
    # Linear in x: f(x) = time_offset + 2*x
    for i in range(nx):
        fields.rho[t_idx, i, :, :] = time_offset + 2.0 * x[i]

print("Field values along x-axis:")
print("x values:", x[:4])
for t_idx in range(nt):
    print(f"t_slice {t_idx}: rho(x) = {fields.rho[t_idx, :4, 0, 0]}")
print()

# Create spectral solver
spectral = SpectralISolver(grid, fields, None)

# Compute spatial derivative
print("Computing spatial derivative in x-direction...")
deriv_x = spectral.spatial_derivative(fields.rho, direction=0)

print(f"\nDerivative shape: {deriv_x.shape}")
print("\nExpected: ∂f/∂x = 2.0 everywhere (constant derivative)")

# Check which time slice was actually used
print("\n" + "=" * 70)
if len(deriv_x.shape) == 3:
    print("✅ CORRECT: spatial_derivative returned 3D result (single time slice)")
    print(f"   Shape: {deriv_x.shape}")
    print("\n Actual derivatives along x-axis (from LATEST time slice):")
    print(f"   ∂ρ/∂x = {deriv_x[:4, 0, 0]}")
    print(f"\n   Latest time slice had rho(x) = {fields.rho[-1, :4, 0, 0]}")
    print(f"   → Derivative correctly computed from time slice {nt-1}")
elif deriv_x.shape[0] == nt:
    print("⚠️  UNEXPECTED: spatial_derivative returned 4D result (all time slices)")
    print(f"   Shape: {deriv_x.shape}")
    print("\nActual derivatives along x-axis:")
    for t_idx in range(nt):
        print(f"t_slice {t_idx}: ∂ρ/∂x = {deriv_x[t_idx, :4, 0, 0]}")
    # Check if all time slices have same derivative
    all_same = np.allclose(deriv_x[0], deriv_x[1:])
    if all_same:
        print("   All time slices have IDENTICAL derivatives")
        print("   → Likely used only latest slice, then broadcast to all slices")
    else:
        print("   Time slices have DIFFERENT derivatives")
        print("   → Computed derivative at each time slice separately")

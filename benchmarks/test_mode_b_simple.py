#!/usr/bin/env python3
"""
Simple Mode B Test: Manual time stepping

Tests if we can do traditional time evolution by manually calling time_step()
"""

import numpy as np

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics

# Create grid with enough time slices
nt, nx, ny, nz = 20, 16, 16, 16
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 2.0),
    spatial_ranges=[(0.0, 2 * np.pi)] * 3,
    grid_points=(nt, nx, ny, nz),
    boundary_conditions="periodic",
)

print(f"Grid created: {grid.shape}")
print(f"Time points: {grid.coordinates['t'][:5]}...")

# Create fields
fields = ISFieldConfiguration(grid)

# Initialize at t=0 with Gaussian wave packet
c_s = np.sqrt(1.0 / 3.0)
rho_0 = 1.0
amplitude = 0.01
sigma = 0.5
x0, y0, z0 = np.pi, np.pi, np.pi

# Get spatial coordinates
x = grid.coordinates["x"]
y = grid.coordinates["y"]
z = grid.coordinates["z"]
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Gaussian at all time slices (initialize entire spacetime)
r_squared = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2
gaussian = np.exp(-r_squared / (2 * sigma**2))

for t_idx in range(nt):
    fields.rho[t_idx, :, :, :] = rho_0 + amplitude * gaussian
    fields.pressure[t_idx, :, :, :] = fields.rho[t_idx, :, :, :] / 3.0

    u_x = (c_s / rho_0) * amplitude * gaussian
    fields.u_mu[t_idx, :, :, :, 1] = u_x
    fields.u_mu[t_idx, :, :, :, 2] = 0.0
    fields.u_mu[t_idx, :, :, :, 3] = 0.0
    fields.u_mu[t_idx, :, :, :, 0] = np.sqrt(1.0 + u_x**2)

print("Fields initialized")

# Create solver
coeffs = TransportCoefficients(shear_viscosity=0.0, bulk_viscosity=0.0)
hydro = SpectralISHydrodynamics(grid, fields, coeffs)

print("Solver created")

# Try calling time_step() directly
dt = 0.1
print(f"\nCalling time_step(dt={dt})...")
try:
    hydro.time_step(dt)
    print("✅ time_step() succeeded")
    print(f"Field rho shape after: {fields.rho.shape}")
    print(f"Field rho range: [{np.min(fields.rho):.6f}, {np.max(fields.rho):.6f}]")
except Exception as e:
    print(f"❌ time_step() failed: {e}")
    import traceback

    traceback.print_exc()

print("\nConclusion:")
print("time_step() refines the entire 4D spacetime grid.")
print("It does NOT step forward in time sequentially.")
print("evolve() calls time_step() repeatedly, refining the same 4D grid.")

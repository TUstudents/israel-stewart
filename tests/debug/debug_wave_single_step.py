"""Debug: Single timestep check."""

import numpy as np

from israel_stewart.core import ISFieldConfiguration, SpacetimeGrid, TransportCoefficients
from israel_stewart.solvers.spectral import SpectralISHydrodynamics

grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),
    spatial_ranges=[(0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
    grid_points=(10, 8, 8, 8),  # Smaller for speed
    boundary_conditions="periodic",
)

fields = ISFieldConfiguration(grid)

c_s = np.sqrt(1.0 / 3.0)
k = 1.0
x = np.linspace(0, 2 * np.pi, 8, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

amplitude = 0.01
rho_0 = 1.0
fields.rho[:] = np.broadcast_to(rho_0 + amplitude * np.sin(k * X), (*grid.shape,)).copy()
fields.pressure[:] = fields.rho / 3.0
fields.u_mu[..., 0] = 1.0

coeffs = TransportCoefficients(shear_viscosity=0.0, bulk_viscosity=0.0)
hydro = SpectralISHydrodynamics(grid, fields, coeffs)

print("Initial rho (1D slice):", fields.rho[-1, :, 0, 0])
print("Taking one timestep...")
hydro.time_step(0.05)
print("Final rho (1D slice):", fields.rho[-1, :, 0, 0])
print("Changed?", not np.allclose(fields.rho[-1, :, 0, 0], rho_0 + amplitude * np.sin(k * x)))

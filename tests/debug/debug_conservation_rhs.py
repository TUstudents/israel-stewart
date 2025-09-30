"""Debug: Check conservation RHS magnitudes."""

import numpy as np

from israel_stewart.core import ISFieldConfiguration, SpacetimeGrid, TransportCoefficients
from israel_stewart.solvers.spectral import SpectralISHydrodynamics

grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),
    spatial_ranges=[(0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
    grid_points=(10, 16, 16, 16),
    boundary_conditions="periodic",
)

fields = ISFieldConfiguration(grid)

c_s = np.sqrt(1.0 / 3.0)
k = 1.0
x = np.linspace(0, 2 * np.pi, 16, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

amplitude = 0.01
rho_0 = 1.0
fields.rho[:] = np.broadcast_to(rho_0 + amplitude * np.sin(k * X), (*grid.shape,)).copy()
fields.pressure[:] = fields.rho / 3.0
fields.u_mu[..., 0] = 1.0  # Rest frame initially

coeffs = TransportCoefficients(shear_viscosity=0.0, bulk_viscosity=0.0)
hydro = SpectralISHydrodynamics(grid, fields, coeffs)

# Get RHS
if hydro.conservation is not None:
    rhs = hydro.conservation.evolution_equations()
    print("Conservation RHS:")
    for key, val in rhs.items():
        print(f"  {key}:")
        print(f"    Shape: {val.shape}")
        print(f"    Min: {np.min(val):.3e}, Max: {np.max(val):.3e}")
        print(f"    Mean abs: {np.mean(np.abs(val)):.3e}")
        print(f"    All zero? {np.allclose(val, 0)}")
else:
    print("No conservation module")

print("\nInitial velocity u_mu (sample at [0,0,0,0]):")
print(f"  {fields.u_mu[0, 0, 0, 0, :]}")
print(f"\nVelocity is rest frame? {np.allclose(fields.u_mu[..., 1:4], 0)}")

"""Debug script to identify why conservation evolution fails."""

import numpy as np

from israel_stewart.core import ISFieldConfiguration, SpacetimeGrid, TransportCoefficients
from israel_stewart.solvers.spectral import SpectralISHydrodynamics

# Setup grid
grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),
    spatial_ranges=[(0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
    grid_points=(10, 32, 32, 32),
    boundary_conditions="periodic",
)

fields = ISFieldConfiguration(grid)

# Initial wave
c_s = np.sqrt(1.0 / 3.0)
k = 1.0
x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

amplitude = 0.01
rho_0 = 1.0
fields.rho = np.broadcast_to(rho_0 + amplitude * np.sin(k * X), (*grid.shape,))
fields.pressure = fields.rho / 3.0
fields.u_mu[..., 0] = 1.0

# Zero viscosity
coeffs = TransportCoefficients(
    shear_viscosity=0.0,
    bulk_viscosity=0.0,
)

hydro = SpectralISHydrodynamics(grid, fields, coeffs)

# Try calling conservation.evolution_equations() directly
print("Testing conservation.evolution_equations()...")
try:
    if hydro.conservation is not None:
        evolution_rhs = hydro.conservation.evolution_equations()
        print(f"SUCCESS! Got RHS with keys: {evolution_rhs.keys()}")
        for key, val in evolution_rhs.items():
            print(f"  {key}: shape={val.shape}, finite={np.all(np.isfinite(val))}")
    else:
        print("ERROR: conservation module is None")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

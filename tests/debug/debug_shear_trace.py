"""Debug: Check if shear tensor is traceless."""

import numpy as np

from israel_stewart.core import ISFieldConfiguration, SpacetimeGrid, TransportCoefficients
from israel_stewart.solvers.spectral import SpectralISHydrodynamics

grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),
    spatial_ranges=[(0.0, 2 * np.pi)] * 3,
    grid_points=(10, 16, 16, 16),
    boundary_conditions="periodic",
)

fields = ISFieldConfiguration(grid)
fields.rho.fill(1.0)
fields.pressure.fill(0.33)
fields.u_mu[..., 0] = 1.0
fields.Pi.fill(0.01)
fields.pi_munu.fill(0.005)  # NOT traceless!

coeffs = TransportCoefficients(
    shear_viscosity=0.1,
    bulk_viscosity=0.05,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3,
)

hydro = SpectralISHydrodynamics(grid, fields, coeffs)

if hydro.conservation is not None:
    T_munu = hydro.conservation.stress_energy_tensor()

    sample_idx = (0, 0, 0, 0)
    print(f"At point {sample_idx}:")
    print(f"  pi_munu trace (should be 0): {np.trace(fields.pi_munu[sample_idx])}")
    print(f"  pi_munu[0,0]: {fields.pi_munu[sample_idx + (0,0)]}")
    print(f"  pi_munu[1,1]: {fields.pi_munu[sample_idx + (1,1)]}")
    print(f"  pi_munu[2,2]: {fields.pi_munu[sample_idx + (2,2)]}")
    print(f"  pi_munu[3,3]: {fields.pi_munu[sample_idx + (3,3)]}")

    trace_computed = (
        -T_munu[sample_idx + (0, 0)]
        + T_munu[sample_idx + (1, 1)]
        + T_munu[sample_idx + (2, 2)]
        + T_munu[sample_idx + (3, 3)]
    )
    expected_trace = -1.0 + 3 * 0.34

    print(f"\n  Computed trace: {trace_computed}")
    print(f"  Expected (no shear): {expected_trace}")
    print(f"  Difference: {trace_computed - expected_trace}")

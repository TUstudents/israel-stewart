"""Debug: Check stress-energy tensor trace structure."""

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
fields.Pi.fill(0.01)
fields.u_mu[..., 0] = 1.0

coeffs = TransportCoefficients(shear_viscosity=0.1)
hydro = SpectralISHydrodynamics(grid, fields, coeffs)

if hydro.conservation is None:
    print("No conservation module")
else:
    T_munu = hydro.conservation.stress_energy_tensor()
    print(f"T_munu shape: {T_munu.shape}")

    # Compute trace (corrected for metric signature)
    trace_wrong = np.trace(T_munu, axis1=-2, axis2=-1)
    trace_correct = -T_munu[..., 0, 0] + T_munu[..., 1, 1] + T_munu[..., 2, 2] + T_munu[..., 3, 3]
    print(f"Trace shape: {trace_correct.shape}")

    # Sample values at one point
    sample_idx = (0, 0, 0, 0)
    print(f"\nAt point {sample_idx}:")
    print(f"  ρ = {fields.rho[sample_idx]}")
    print(f"  P = {fields.pressure[sample_idx]}")
    print(f"  Π = {fields.Pi[sample_idx]}")
    expected = fields.rho[sample_idx] - 3 * (fields.pressure[sample_idx] + fields.Pi[sample_idx])
    print(f"  Expected trace (ρ - 3(P + Π)): {expected}")
    print(f"  Computed trace (wrong): {trace_wrong[sample_idx]}")
    print(f"  Computed trace (correct): {trace_correct[sample_idx]}")
    print(
        f"  Error (correct): {abs(trace_correct[sample_idx] - expected) / abs(expected) * 100:.1f}%"
    )

    # Check diagonal components
    print(f"\n  T^00 = {T_munu[sample_idx + (0, 0)]}")
    print(f"  T^11 = {T_munu[sample_idx + (1, 1)]}")
    print(f"  T^22 = {T_munu[sample_idx + (2, 2)]}")
    print(f"  T^33 = {T_munu[sample_idx + (3, 3)]}")
    print(
        f"  Sum (trace) = {T_munu[sample_idx + (0, 0)] + T_munu[sample_idx + (1, 1)] + T_munu[sample_idx + (2, 2)] + T_munu[sample_idx + (3, 3)]}"
    )

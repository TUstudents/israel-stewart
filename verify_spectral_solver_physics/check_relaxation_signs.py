"""
Check if relaxation equations use consistent sign convention with stress tensor.

If stress tensor uses T = ... - π, then the relaxation equation should match:
  τ_π·Dπ^μν + π^μν = 2η·σ^μν

NOT:
  τ_π·Dπ^μν - π^μν = 2η·σ^μν

Let's verify the code uses the correct sign.
"""

import numpy as np

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)

k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("=" * 80)
print("RELAXATION EQUATION SIGN CHECK")
print("=" * 80)
print()

# Get relaxation RHS
rhs_vector = benchmark.solver.relaxation.compute_relaxation_rhs(benchmark.fields)

# Unpack
nx, ny, nz = benchmark.grid_points
Pi_size = nx * ny * nz
pi_size = nx * ny * nz * 4 * 4

dPi_dt = rhs_vector[:Pi_size].reshape((nx, ny, nz))
dpi_dt = rhs_vector[Pi_size : Pi_size + pi_size].reshape((nx, ny, nz, 4, 4))

# Get field values
Pi = benchmark.fields.Pi
pi = benchmark.fields.pi_munu

# Check at a grid point
ix, iy, iz = 8, 0, 0

print(f"At grid point ({ix}, {iy}, {iz}):")
print(f"  Π  = {Pi[ix, iy, iz]:.6e}")
print(f"  π_xx = {pi[ix, iy, iz, 1, 1]:.6e}")
print()

print("Time derivatives:")
print(f"  dΠ/dt  = {dPi_dt[ix, iy, iz]:.6e}")
print(f"  dπ_xx/dt = {dpi_dt[ix, iy, iz, 1, 1]:.6e}")
print()

# The RHS should include:
# dΠ/dt = -Π/τ_Π - ζ·θ/τ_Π + nonlinear
# dπ/dt = -π/τ_π + 2η·σ/τ_π + nonlinear

# Check if linear term has correct sign
# We removed the linear term from source (for IMEX), but let's check the source term

# Compute θ and σ
u_mu = benchmark.fields.u_mu
theta = benchmark.solver.relaxation._compute_expansion_scalar(u_mu)
sigma = benchmark.solver.relaxation._compute_shear_tensor(u_mu)

print("Kinematic quantities:")
print(f"  θ    = {theta[ix, iy, iz]:.6e}")
print(f"  σ_xx = {sigma[ix, iy, iz, 1, 1]:.6e}")
print()

# Expected source terms (without linear -Π/τ and -π/τ)
bulk_source_expected = -coeffs.bulk_viscosity * theta[ix, iy, iz] / coeffs.bulk_relaxation_time
shear_source_expected = (
    2.0 * coeffs.shear_viscosity * sigma[ix, iy, iz, 1, 1] / coeffs.shear_relaxation_time
)

print("Expected source terms (first-order only):")
print(f"  Bulk:  -ζ·θ/τ_Π  = {bulk_source_expected:.6e}")
print(f"  Shear: 2η·σ/τ_π  = {shear_source_expected:.6e}")
print()

# The actual RHS includes the linear term since we're using RK4 (not IMEX)
linear_bulk = -Pi[ix, iy, iz] / coeffs.bulk_relaxation_time
linear_shear = -pi[ix, iy, iz, 1, 1] / coeffs.shear_relaxation_time

expected_total_bulk = linear_bulk + bulk_source_expected
expected_total_shear = linear_shear + shear_source_expected

print("Expected total RHS (with linear terms):")
print(f"  dΠ/dt  = {expected_total_bulk:.6e}")
print(f"  dπ_xx/dt = {expected_total_shear:.6e}")
print()

print("Actual RHS from code:")
print(f"  dΠ/dt  = {dPi_dt[ix, iy, iz]:.6e}")
print(f"  dπ_xx/dt = {dpi_dt[ix, iy, iz, 1, 1]:.6e}")
print()

# Check if they match
bulk_match = np.allclose(dPi_dt[ix, iy, iz], expected_total_bulk, rtol=0.01)
shear_match = np.allclose(dpi_dt[ix, iy, iz, 1, 1], expected_total_shear, rtol=0.01)

print("Sign consistency check:")
if bulk_match:
    print("  ✓ Bulk relaxation has correct signs")
else:
    print(
        f"  ✗ Bulk relaxation sign error: {abs((dPi_dt[ix, iy, iz] - expected_total_bulk) / expected_total_bulk) * 100:.2f}%"
    )

if shear_match:
    print("  ✓ Shear relaxation has correct signs")
else:
    print(
        f"  ✗ Shear relaxation sign error: {abs((dpi_dt[ix, iy, iz, 1, 1] - expected_total_shear) / expected_total_shear) * 100:.2f}%"
    )

print()
print("=" * 80)

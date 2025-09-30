"""Debug: Check if wave is actually propagating."""

import numpy as np

from israel_stewart.core import ISFieldConfiguration, SpacetimeGrid, TransportCoefficients
from israel_stewart.solvers.spectral import SpectralISHydrodynamics

grid = SpacetimeGrid(
    coordinate_system="cartesian",
    time_range=(0.0, 1.0),
    spatial_ranges=[(0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
    grid_points=(10, 32, 32, 32),
    boundary_conditions="periodic",
)

fields = ISFieldConfiguration(grid)

c_s = np.sqrt(1.0 / 3.0)
k = 1.0
x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

amplitude = 0.01
rho_0 = 1.0
fields.rho[:] = np.broadcast_to(rho_0 + amplitude * np.sin(k * X), (*grid.shape,)).copy()
fields.pressure[:] = fields.rho / 3.0
fields.u_mu[..., 0] = 1.0

coeffs = TransportCoefficients(shear_viscosity=0.0, bulk_viscosity=0.0)
hydro = SpectralISHydrodynamics(grid, fields, coeffs)

# Check initial wave
rho_initial = fields.rho[-1, :, 0, 0]  # 1D slice along x
print("Initial wave:")
print(f"  Min: {rho_initial.min():.6f}, Max: {rho_initial.max():.6f}")
print(f"  Amplitude: {(rho_initial.max() - rho_initial.min())/2:.6f}")
print(f"  Peak at index: {np.argmax(rho_initial)}")

# Evolve
dt = 0.05
n_steps = 10
total_time = dt * n_steps

for step in range(n_steps):
    hydro.time_step(dt)

# Check final wave
rho_final = fields.rho[-1, :, 0, 0]
expected_shift = c_s * total_time
expected_peak_shift = expected_shift / (2 * np.pi) * 32  # in grid points

print(f"\nAfter {total_time:.2f} time:")
print(f"  Min: {rho_final.min():.6f}, Max: {rho_final.max():.6f}")
print(f"  Amplitude: {(rho_final.max() - rho_final.min())/2:.6f}")
print(f"  Peak at index: {np.argmax(rho_final)}")
print(f"  Expected shift: {expected_peak_shift:.2f} grid points")

# Check if wave moved
initial_peak = np.argmax(rho_initial)
final_peak = np.argmax(rho_final)
actual_shift = final_peak - initial_peak
print(f"\nActual peak shift: {actual_shift} grid points")
print(f"Expected peak shift: {expected_peak_shift:.2f} grid points")
print(f"Wave moved? {abs(actual_shift) > 0}")

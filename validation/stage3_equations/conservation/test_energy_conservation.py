#!/usr/bin/env -S uv run python
"""
Test total energy conservation during time evolution.

For a closed system with periodic boundary conditions and no external sources,
total energy E = ∫ T^00 dV should be conserved.
"""

import numpy as np
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.solvers.spectral import SpectralISHydrodynamics
from israel_stewart.equations.conservation import ConservationLaws

# Create grid with periodic BC (closed system)
grid = SpaceGrid(
    coordinate_system="cartesian",
    spatial_ranges=[(0.0, 2*np.pi)] * 3,
    grid_points=(32, 32, 16),
    boundary_conditions="periodic"
)

# Transport coefficients
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
)

# Initialize with sound wave perturbation
fields = ISFieldConfiguration(grid)
X, Y, Z = grid.meshgrid()

k = 1.0
rho_0 = 1.0
amplitude = 0.01

# Sound wave in x-direction
fields.rho[:] = rho_0 + amplitude * np.cos(k * X)
fields.pressure[:] = fields.rho / 3.0  # Radiation fluid
fields.u_mu[..., 0] = 1.0  # Rest frame initially
fields.u_mu[..., 1] = amplitude * 0.45 * np.cos(k * X)  # Small velocity

# Create solver
solver = SpectralISHydrodynamics(grid=grid, fields=fields, coeffs=coeffs)

# Create conservation laws object
conservation = ConservationLaws(fields, coeffs, spectral_solver=solver)

# Compute cell volume
dx, dy, dz = grid.dx, grid.dy, grid.dz
dV = dx * dy * dz

def compute_total_energy():
    """Compute total energy E = ∫ T^00 dV."""
    T = conservation.stress_energy_tensor()
    T00 = T[..., 0, 0]  # Energy density component
    return np.sum(T00) * dV

def compute_total_momentum():
    """Compute total momentum P^i = ∫ T^0i dV."""
    T = conservation.stress_energy_tensor()
    Px = np.sum(T[..., 0, 1]) * dV
    Py = np.sum(T[..., 0, 2]) * dV
    Pz = np.sum(T[..., 0, 3]) * dV
    return np.array([Px, Py, Pz])

print("=" * 80)
print("ENERGY CONSERVATION TEST")
print("=" * 80)
print()
print(f"Grid: {grid.shape}")
print(f"Cell volume: dV = {dV:.6e}")
print(f"Total volume: V = {dx*grid.shape[0] * dy*grid.shape[1] * dz*grid.shape[2]:.6f}")
print()

# Initial values
E_initial = compute_total_energy()
P_initial = compute_total_momentum()

print(f"Initial total energy:    E = {E_initial:.10f}")
print(f"Initial total momentum:  P = ({P_initial[0]:.6e}, {P_initial[1]:.6e}, {P_initial[2]:.6e})")
print()

# Evolve system
dt = 0.01
n_steps = 100
times = [0.0]
energies = [E_initial]
momenta = [P_initial]

print("Evolving system...")
for i in range(n_steps):
    solver.time_step(dt)

    if (i + 1) % 20 == 0:
        t = (i + 1) * dt
        E = compute_total_energy()
        P = compute_total_momentum()

        times.append(t)
        energies.append(E)
        momenta.append(P)

        dE = E - E_initial
        dE_rel = dE / E_initial * 100

        print(f"  t = {t:.2f}: E = {E:.10f}, ΔE = {dE:+.6e} ({dE_rel:+.6f}%)")

times = np.array(times)
energies = np.array(energies)
momenta = np.array(momenta)

# Final values
E_final = energies[-1]
P_final = momenta[-1]

dE_total = E_final - E_initial
dE_rel = abs(dE_total) / E_initial * 100

dP_total = P_final - P_initial
dP_mag = np.linalg.norm(dP_total)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print(f"Initial energy: E₀ = {E_initial:.10f}")
print(f"Final energy:   E  = {E_final:.10f}")
print(f"Change:         ΔE = {dE_total:+.6e} ({dE_rel:+.6f}%)")
print()
print(f"Initial momentum: P₀ = ({P_initial[0]:.6e}, {P_initial[1]:.6e}, {P_initial[2]:.6e})")
print(f"Final momentum:   P  = ({P_final[0]:.6e}, {P_final[1]:.6e}, {P_final[2]:.6e})")
print(f"Change:           ΔP = {dP_mag:.6e}")
print()

# Check conservation
energy_tolerance = 0.01  # 0.01% = 1e-4 relative tolerance
momentum_tolerance = 1e-4  # Absolute tolerance

energy_conserved = dE_rel < energy_tolerance
momentum_conserved = dP_mag < momentum_tolerance

print("=" * 80)
print("CONSERVATION CHECK")
print("=" * 80)
print()

if energy_conserved:
    print(f"✓ Energy conserved within {energy_tolerance}% tolerance")
else:
    print(f"✗ Energy NOT conserved: {dE_rel:.6f}% change exceeds {energy_tolerance}% tolerance")

if momentum_conserved:
    print(f"✓ Momentum conserved within {momentum_tolerance:.0e} tolerance")
else:
    print(f"✗ Momentum NOT conserved: |ΔP| = {dP_mag:.6e} exceeds {momentum_tolerance:.0e} tolerance")

print()

if energy_conserved and momentum_conserved:
    print("✓✓✓ ALL CONSERVATION LAWS SATISFIED ✓✓✓")
else:
    print("⚠️  Some conservation laws violated - check numerical scheme")
    print()
    print("Note: Small violations (<1%) are acceptable due to:")
    print("  - Numerical discretization errors")
    print("  - Time integration truncation")
    print("  - Spectral aliasing")

print()
print("=" * 80)

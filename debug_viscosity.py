#!/usr/bin/env python3
"""
Debug script to check if viscous dissipative fluxes are evolving during simulation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients

# Create benchmark
transport_coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=transport_coeffs,
)

# Setup initial conditions
wave_number = 1.0
benchmark.setup_initial_conditions(wave_number=wave_number)

print("=" * 80)
print("VISCOUS FLUX EVOLUTION DEBUG")
print("=" * 80)
print()

# Check initial state
print("Initial State:")
print(f"  rho mean:     {np.mean(benchmark.fields.rho):.6f}")
print(f"  rho std:      {np.std(benchmark.fields.rho):.6f}")
print(f"  Pi mean:      {np.mean(benchmark.fields.Pi):.6e}")
print(f"  Pi std:       {np.std(benchmark.fields.Pi):.6e}")
print(f"  Pi max:       {np.max(np.abs(benchmark.fields.Pi)):.6e}")
print(f"  pi_munu max:  {np.max(np.abs(benchmark.fields.pi_munu)):.6e}")
print()

# Manually evolve a few steps and monitor dissipative fluxes
dt = 0.033  # Same as in simulation
n_steps = 10

print(f"Evolving {n_steps} steps with dt={dt:.3f}:")
print()

for step in range(n_steps):
    t = step * dt

    # Take one timestep using split_step method
    benchmark.solver.time_step(dt, method="split_step")

    # Monitor dissipative fluxes
    Pi_max = np.max(np.abs(benchmark.fields.Pi))
    pi_max = np.max(np.abs(benchmark.fields.pi_munu))
    rho_std = np.std(benchmark.fields.rho)

    print(f"Step {step:3d} (t={t:6.3f}):  Pi_max={Pi_max:.6e}  pi_max={pi_max:.6e}  rho_std={rho_std:.6e}")

print()
print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)

final_Pi_max = np.max(np.abs(benchmark.fields.Pi))
final_pi_max = np.max(np.abs(benchmark.fields.pi_munu))

print(f"Final Pi_max:     {final_Pi_max:.6e}")
print(f"Final pi_max:     {final_pi_max:.6e}")
print()

if final_Pi_max < 1e-10 and final_pi_max < 1e-10:
    print("❌ PROBLEM: Dissipative fluxes are essentially zero!")
    print("   Viscosity is NOT being activated in the simulation.")
    print()
    print("Possible causes:")
    print("1. Relaxation equations not computing correct source terms")
    print("2. Expansion scalar θ or shear tensor σ^μν is zero")
    print("3. Viscous fluxes not being used in stress-energy tensor")
elif final_Pi_max < 1e-6 and final_pi_max < 1e-6:
    print("⚠️  WARNING: Dissipative fluxes are very small")
    print(f"   Expected magnitude ~ η*k² ~ {transport_coeffs.shear_viscosity * wave_number**2:.6e}")
else:
    print("✓ Dissipative fluxes are active")
    print(f"   Bulk viscous pressure: {final_Pi_max:.6e}")
    print(f"   Shear stress:          {final_pi_max:.6e}")

print("=" * 80)

# Now check what the expansion scalar and shear tensor actually are
print()
print("KINEMATIC QUANTITIES CHECK")
print("=" * 80)

# Compute expansion scalar manually
u_mu = benchmark.fields.u_mu
velocity_spatial = u_mu[..., 1:4]  # Spatial components

# Compute divergence
dx, dy, dz = benchmark.grid.spatial_spacing

div_ux = np.gradient(velocity_spatial[..., 0], dx, axis=0)
div_uy = np.gradient(velocity_spatial[..., 1], dy, axis=1)
div_uz = np.gradient(velocity_spatial[..., 2], dz, axis=2)

expansion = div_ux + div_uy + div_uz

print(f"Expansion scalar θ:")
print(f"  Mean:     {np.mean(expansion):.6e}")
print(f"  Std:      {np.std(expansion):.6e}")
print(f"  Max:      {np.max(np.abs(expansion)):.6e}")
print()

if np.max(np.abs(expansion)) < 1e-10:
    print("❌ PROBLEM: Expansion scalar is zero!")
    print("   This means ∇·u = 0, so no bulk viscous pressure is sourced.")
else:
    print("✓ Expansion scalar is non-zero")

print("=" * 80)

#!/usr/bin/env python3
"""Trace how momentum evolves to see if dissipative terms contribute."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

transport_coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi,
    grid_points=(16, 16, 8),  # Small for speed
    transport_coeffs=transport_coeffs,
)

wave_number = 1.0
benchmark.setup_initial_conditions(wave_number=wave_number)

print("=" * 80)
print("MOMENTUM EVOLUTION TRACE")
print("=" * 80)
print()

# Get initial state
T_initial = benchmark.solver.conservation.stress_energy_tensor()

print("Initial stress-energy tensor components at x=π/2:")
ix = 8
iy = 8
iz = 4

print(f"  T^00 (energy density):     {T_initial[ix, iy, iz, 0, 0]:.6e}")
print(f"  T^0x (energy flux):        {T_initial[ix, iy, iz, 0, 1]:.6e}")
print(f"  T^x0 (momentum density):   {T_initial[ix, iy, iz, 1, 0]:.6e}")
print(f"  T^xx (xx stress):          {T_initial[ix, iy, iz, 1, 1]:.6e}")
print()

# Decompose T^xx into parts
rho = benchmark.fields.rho[ix, iy, iz]
pressure = benchmark.fields.pressure[ix, iy, iz]
Pi = benchmark.fields.Pi[ix, iy, iz]
pi_xx = benchmark.fields.pi_munu[ix, iy, iz, 1, 1]
u_x = benchmark.fields.u_mu[ix, iy, iz, 1]

print("Components contributing to T^xx:")
print(f"  ρ*u^x*u^x:  {rho * u_x * u_x:.6e}")
print(f"  p:          {pressure:.6e}")
print(f"  Π:          {Pi:.6e}")
print(f"  π_xx:       {pi_xx:.6e}")
print(f"  Total:      {T_initial[ix, iy, iz, 1, 1]:.6e}")
print()

# Get evolution equations
evolution_rhs = benchmark.solver.conservation.evolution_equations()

drho_dt = evolution_rhs["drho_dt"]
dmom_dt = evolution_rhs["dmom_dt"]

print("Time derivatives from conservation laws:")
print(f"  dρ/dt at (ix, iy, iz):           {drho_dt[ix, iy, iz]:.6e}")
print(f"  d(ρu^x)/dt at (ix, iy, iz):      {dmom_dt[ix, iy, iz, 0]:.6e}")
print()

# Manually compute what dmom/dt should be from stress divergence
# ∂_t (ρu^x) = -∂_i T^ix

dx = benchmark.grid.spatial_spacing[0]

# Compute ∂_x T^xx using finite difference
T_xx_forward = T_initial[(ix+1) % 16, iy, iz, 1, 1]
T_xx_backward = T_initial[(ix-1) % 16, iy, iz, 1, 1]
dT_xx_dx = (T_xx_forward - T_xx_backward) / (2 * dx)

print("Manual calculation:")
print(f"  T^xx at ix-1: {T_xx_backward:.6e}")
print(f"  T^xx at ix:   {T_initial[ix, iy, iz, 1, 1]:.6e}")
print(f"  T^xx at ix+1: {T_xx_forward:.6e}")
print(f"  ∂T^xx/∂x:     {dT_xx_dx:.6e}")
print(f"  Expected d(ρu^x)/dt ≈ -∂T^xx/∂x = {-dT_xx_dx:.6e}")
print()

# Compare with actual
actual_dmom = dmom_dt[ix, iy, iz, 0]
expected_dmom = -dT_xx_dx

print(f"Comparison:")
print(f"  From evolution_equations(): {actual_dmom:.6e}")
print(f"  From manual ∂T/∂x:          {expected_dmom:.6e}")
print(f"  Ratio:                      {actual_dmom / expected_dmom if abs(expected_dmom) > 1e-10 else np.nan:.4f}")
print()

if abs(actual_dmom - expected_dmom) / max(abs(expected_dmom), 1e-10) < 0.1:
    print("✓ Stress-energy tensor IS being used in momentum evolution")
else:
    print("❌ Stress-energy tensor NOT being used correctly!")

print()
print("Now check contribution from dissipative terms...")
print()

# Recompute T^xx without dissipative terms
T_xx_ideal = rho * u_x * u_x + pressure

# With dissipative terms
T_xx_full = T_xx_ideal + Pi + pi_xx

print(f"T^xx breakdown:")
print(f"  Ideal fluid:        {T_xx_ideal:.6e}")
print(f"  + Bulk (Π):         {Pi:.6e} → {T_xx_ideal + Pi:.6e}")
print(f"  + Shear (π_xx):     {pi_xx:.6e} → {T_xx_full:.6e}")
print()

print(f"Dissipative contribution to T^xx:")
print(f"  Fraction: {(Pi + pi_xx) / T_xx_ideal * 100:.2f}%")
print()

if abs(Pi + pi_xx) / abs(T_xx_ideal) < 0.001:
    print("⚠️  Dissipative terms are < 0.1% of ideal fluid stress")
    print("   This is too small to measurably affect dynamics!")
else:
    print("✓ Dissipative terms are significant")

print("=" * 80)

#!/usr/bin/env python3
"""
Trace how momentum evolves to verify dissipative terms contribute correctly.

Checks:
1. Momentum evolution uses stress-energy tensor correctly
2. Dissipative terms (Π, π) contribute to stress tensor
3. Contribution is significant at wave peaks
"""

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
    grid_points=(32, 32, 16),
    transport_coeffs=transport_coeffs,
)

wave_number = 8.0  # High k for significant dissipative effects
benchmark.setup_initial_conditions(wave_number=wave_number)

print("=" * 80)
print("MOMENTUM EVOLUTION TRACE")
print("=" * 80)
print()
print(f"Wave number k = {wave_number}")
print(f"Domain size = {benchmark.domain_size}")
print(f"Wavelength λ = {2*np.pi/wave_number:.6f}")
print()

# Get initial state
T_initial = benchmark.solver.conservation.stress_energy_tensor()

# Check at point with both amplitude and gradient
# For sin(kx) mode at t=0:
#   - Amplitude max: x=π/(2k) ≈ 0.196 (ix=1 for k=8)
#   - Gradient max: x=0 (ix=0)
# Use ix=3 to avoid both extrema
dx_grid = 2 * np.pi / 32
ix = 3  # Point with both non-zero amplitude and gradient
iy = 0
iz = 0
actual_x = ix * dx_grid

print(f"Checking at x ≈ {actual_x:.6f}:")
print(f"  Grid index: ix={ix}, iy={iy}, iz={iz}")
print(f"  sin(kx) = {np.sin(wave_number * actual_x):.3f} (amplitude factor)")
print(f"  cos(kx) = {np.cos(wave_number * actual_x):.3f} (gradient factor)")
print()

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
nx = benchmark.grid_points[0]

# Compute ∂_x T^xx using finite difference
T_xx_forward = T_initial[(ix+1) % nx, iy, iz, 1, 1]
T_xx_backward = T_initial[(ix-1) % nx, iy, iz, 1, 1]
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

relative_error = abs(actual_dmom - expected_dmom) / max(abs(expected_dmom), 1e-10)
if relative_error < 0.1:
    print("✓ Stress-energy tensor IS being used in momentum evolution")
    print(f"  Relative error: {relative_error*100:.2f}%")
else:
    print("✗ Stress-energy tensor NOT being used correctly!")
    print(f"  Relative error: {relative_error*100:.2f}%")

print()
print("=" * 80)
print("DISSIPATIVE CONTRIBUTION TO STRESS TENSOR")
print("=" * 80)
print()

# Recompute T^xx without dissipative terms
T_xx_ideal = rho * u_x * u_x + pressure

# With dissipative terms
T_xx_full = T_xx_ideal + Pi + pi_xx

print(f"T^xx breakdown at wave peak:")
print(f"  Ideal fluid (ρu²+p):  {T_xx_ideal:.6e}")
print(f"  + Bulk (Π):           {Pi:.6e}")
print(f"  + Shear (π_xx):       {pi_xx:.6e}")
print(f"  Total:                {T_xx_full:.6e}")
print()

dissipative_fraction = abs((Pi + pi_xx) / T_xx_ideal) * 100
print(f"Dissipative contribution:")
print(f"  |Π + π_xx| / |ideal| = {dissipative_fraction:.2f}%")
print()

if dissipative_fraction < 0.1:
    print("✗ PROBLEM: Dissipative terms < 0.1% of ideal fluid stress")
    print("  → Too small to measurably affect dynamics")
    print("  → Check: Are we at wave peak? Is k high enough?")
elif dissipative_fraction < 1.0:
    print("⚠ SMALL: Dissipative terms < 1% of ideal fluid stress")
    print("  → Dissipative effects present but weak")
elif dissipative_fraction < 10.0:
    print("✓ GOOD: Dissipative terms are 1-10% of ideal fluid stress")
    print("  → Significant contribution to momentum evolution")
    print("  → Expected for high-k waves in Israel-Stewart theory")
else:
    print("⚠ LARGE: Dissipative terms > 10% of ideal fluid stress")
    print("  → Very strong dissipative effects")
    print("  → May indicate initialization or relaxation time issues")

print()
print("Physical context:")
print(f"  At k={wave_number}, dissipative fields contribute ~{dissipative_fraction:.1f}% to stress")
print(f"  This drives damping rate γ ≈ {dissipative_fraction/100 * wave_number:.3f} (order of magnitude)")
print("  Actual γ from dispersion relation includes relaxation time effects")
print()
print("=" * 80)

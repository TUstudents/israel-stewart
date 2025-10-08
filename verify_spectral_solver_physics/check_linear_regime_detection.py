"""
Check if linear regime detection is working correctly in the fixed code.
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
    lambda_pi_pi=0.0,
    lambda_pi_Pi=0.0,
    xi_1=0.0,
    xi_2=0.0,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)

k = 8.0
benchmark.setup_initial_conditions(wave_number=k)

print("=" * 80)
print("LINEAR REGIME DETECTION CHECK")
print("=" * 80)
print()

# Check perturbation amplitudes
rho = benchmark.fields.rho
u_spatial = benchmark.fields.u_mu[..., 1:4]

max_rho_perturbation = np.max(np.abs(rho - 1.0))
max_velocity = np.max(np.abs(u_spatial))

print("Perturbation amplitudes:")
print(f"  max |δρ| = {max_rho_perturbation:.6f}")
print(f"  max |v|  = {max_velocity:.6f}")
print()

is_linear = (max_rho_perturbation < 0.1) and (max_velocity < 0.1)
print(f"Linear regime detected: {is_linear}")
print("  (threshold: |δρ| < 0.1 and |v| < 0.1)")
print()

if is_linear:
    print("✓ Should use linearized conversion: du/dt = d(h·u)/dt / h₀")
else:
    print("✗ Should use nonlinear conversion: du/dt = [d(h·u)/dt - u·dh/dt] / h")

print()

# Manually compute what conversion should give
conservation_rhs = benchmark.solver.conservation.evolution_equations()
dmom_dt = conservation_rhs["dmom_dt"]

# Background enthalpy
h_background = 4.0 / 3.0

# Linear conversion
du_dt_linear = dmom_dt / h_background

# What does the actual solver give?
# We need to do one timestep and compare
print("Testing one RHS evaluation...")
rhs = benchmark.solver._compute_full_coupled_rhs(benchmark.fields)
du_dt_actual = rhs["du_dt"]

print("du^x/dt at (8, 0, 0):")
print(f"  Linear formula:  {du_dt_linear[8, 0, 0, 0]:.6e}")
print(f"  Solver returns:  {du_dt_actual[8, 0, 0, 0]:.6e}")
print(f"  Match:           {np.allclose(du_dt_linear[8, 0, 0, 0], du_dt_actual[8, 0, 0, 0])}")
print()

if np.allclose(du_dt_linear, du_dt_actual):
    print("✓ Solver IS using linearized conversion")
else:
    print("✗ Solver is NOT using linearized conversion")
    print(f"  Difference: {np.max(np.abs(du_dt_linear - du_dt_actual)):.6e}")

print()
print("=" * 80)

#!/usr/bin/env python
"""Quick test of timestep fix for k=8 wave."""

import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup: same parameters as investigation
coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=0.5,
    bulk_relaxation_time=0.3
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2*np.pi,
    grid_points=(32, 32, 16),
    transport_coeffs=coeffs
)

# Test with k=8
k = 8.0
amplitude = 0.01
simulation_time = 3.2  # ~3 wave periods for k=8
method = 'split_step'
dt_factor = 0.5

print("="*80)
print("TIMESTEP FIX VALIDATION")
print("="*80)
print(f"Transport coefficients:")
print(f"  τ_Π = {coeffs.bulk_relaxation_time}")
print(f"  τ_π = {coeffs.shear_relaxation_time}")
print(f"  τ_min = {min(coeffs.bulk_relaxation_time, coeffs.shear_relaxation_time)}")
print()

# Get analytical solution
analytical_modes = benchmark.analytical.analyze_dispersion_relation(
    wave_vector=np.array([k, 0.0, 0.0])
)
analytical = analytical_modes[0]
print(f"Analytical dispersion (k={k}):")
print(f"  ω = {analytical.frequency:.6f} /time")
print(f"  γ = {analytical.attenuation:.6f} /time")
print(f"  c_s = {analytical.sound_speed:.6f}")
print()

# Calculate expected timesteps
dx = benchmark.grid.spatial_spacing[0]
dt_wave = dt_factor * dx / max(analytical.sound_speed, 0.1)
dt_relax = 0.01 * min(coeffs.bulk_relaxation_time, coeffs.shear_relaxation_time)

print(f"Timestep calculation:")
print(f"  dx = {dx:.6f}")
print(f"  dt_wave (CFL) = {dt_wave:.6f}")
print(f"  dt_relax (stability) = {dt_relax:.6f}")
print(f"  dt_final = min(dt_wave, dt_relax) = {min(dt_wave, dt_relax):.6f}")
print()

# Run simulation
print(f"Running simulation for t_final = {simulation_time}...")
print(f"(This will take ~{int(simulation_time / min(dt_wave, dt_relax))} timesteps)")
result = benchmark.run_simulation(
    wave_number=k,
    simulation_time=simulation_time,
    method=method,
    dt_factor=dt_factor,
    n_periods=0  # Override automatic time extension
)

print()
print("="*80)
print("RESULTS")
print("="*80)
print(f"Numerical measurement:")
print(f"  ω_measured = {result.measured_frequency:.6f} /time")
print(f"  γ_measured = {result.measured_damping_rate:.6f} /time")
print()
print(f"Comparison:")
print(f"  ω_error = {result.frequency_error:.2f}%")
print(f"  γ_error = {result.damping_error:.2f}%")
print()

# Check stability
if result.measured_damping_rate > 0:
    print("✓ STABLE: Mode is decaying (γ > 0)")
else:
    print("✗ UNSTABLE: Mode is growing (γ < 0)")
print()

# Check accuracy
gamma_measured = result.measured_damping_rate
error_percent = abs(result.damping_error)

if error_percent < 20 and gamma_measured > 0:
    print(f"✓ ACCURATE: Damping error {error_percent:.1f}% < 20%")
elif gamma_measured > 0:
    print(f"⚠ STABLE but INACCURATE: Damping error {error_percent:.1f}% > 20%")
    print("  (May need smaller timestep or longer simulation)")
else:
    print(f"✗ UNSTABLE: Cannot assess accuracy")

print("="*80)

#!/usr/bin/env python3
"""Check what initial flux values are actually set."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

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
benchmark.setup_initial_conditions(wave_number=wave_number, amplitude=0.01)

print("=" * 80)
print("INITIAL FLUX VALUES")
print("=" * 80)
print()

print("From eigenmode analysis, we expect:")
print("  δv_x:   4.028e-04 (imaginary part)")
print("  δΠ:     1.769e-04 (imaginary part)")
print("  δπ_xx:  4.559e-04 (imaginary part)")
print()

print("Actual initial values:")
print(f"  u^x max:     {np.max(np.abs(benchmark.fields.u_mu[..., 1])):.6e}")
print(f"  Π max:       {np.max(np.abs(benchmark.fields.Pi)):.6e}")
print(f"  π_xx max:    {np.max(np.abs(benchmark.fields.pi_munu[..., 1, 1])):.6e}")
print()

# Check signs by looking at spatial structure
X, Y, Z = benchmark.grid.meshgrid()
ix_max = np.argmax(np.abs(np.sin(wave_number * X[:, 0, 0])))

print(f"At antinode (ix={ix_max}):")
print(f"  sin(kx):  {np.sin(wave_number * X[ix_max, 0, 0]):.6f}")
print(f"  δρ:       {benchmark.fields.rho[ix_max, 0, 0] - 1.0:.6e}")
print(f"  δv_x:     {benchmark.fields.u_mu[ix_max, 0, 0, 1]:.6e}")
print(f"  Π:        {benchmark.fields.Pi[ix_max, 0, 0]:.6e}")
print(f"  π_xx:     {benchmark.fields.pi_munu[ix_max, 0, 0, 1, 1]:.6e}")
print()

print("=" * 80)

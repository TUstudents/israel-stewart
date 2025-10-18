#!/usr/bin/env python3
"""Check actual Π values after initialization."""

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

wave_number = 1.0
amplitude = 0.01
benchmark.setup_initial_conditions(wave_number=wave_number, amplitude=amplitude)

X, Y, Z = benchmark.grid.meshgrid()

print("=" * 80)
print("INITIAL Π FIELD VALUES")
print("=" * 80)
print()

print(f"Amplitude: {amplitude}")
print(f"Wave number: {wave_number}")
print()

# Expected maximum from eigenmode
Pi_ratio = 1.768843e-02  # From eigenvector analysis
expected_max = amplitude * Pi_ratio
print(f"Expected |Π|_max from eigenmode: {expected_max:.6e}")
print()

# Actual values
Pi = benchmark.fields.Pi
print(f"Actual Π_max: {np.max(np.abs(Pi)):.6e}")
print(f"Actual Π_min: {np.min(Pi):.6e}")
print()

# Sample locations
print("Sample Π values:")
ix = 16  # x = π
iy = 16
iz = 8
print(f"  Π at x=0 (ix=0):    {Pi[0, iy, iz]:.6e} (expect ~0)")
print(f"  Π at x=π/2 (ix=8):  {Pi[8, iy, iz]:.6e} (expect ~{expected_max:.6e})")
print(f"  Π at x=π (ix=16):   {Pi[16, iy, iz]:.6e} (expect ~0)")
print(f"  Π at x=3π/2 (ix=24): {Pi[24, iy, iz]:.6e} (expect ~{-expected_max:.6e})")
print()

# Also check x positions
print("Grid x positions:")
print(f"  X[0, {iy}, {iz}] = {X[0, iy, iz]:.6f}")
print(f"  X[8, {iy}, {iz}] = {X[8, iy, iz]:.6f} (expect π/2 = {np.pi/2:.6f})")
print(f"  X[16, {iy}, {iz}] = {X[16, iy, iz]:.6f} (expect π = {np.pi:.6f})")
print(f"  X[24, {iy}, {iz}] = {X[24, iy, iz]:.6f} (expect 3π/2 = {3*np.pi/2:.6f})")
print()

print("sin(k*X) values:")
print(f"  sin(k*X)[0, {iy}, {iz}] = {np.sin(wave_number * X[0, iy, iz]):.6f}")
print(f"  sin(k*X)[8, {iy}, {iz}] = {np.sin(wave_number * X[8, iy, iz]):.6f} (expect 1)")
print(f"  sin(k*X)[16, {iy}, {iz}] = {np.sin(wave_number * X[16, iy, iz]):.6f} (expect 0)")
print(f"  sin(k*X)[24, {iy}, {iz}] = {np.sin(wave_number * X[24, iy, iz]):.6f} (expect -1)")

print("=" * 80)

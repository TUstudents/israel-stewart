#!/usr/bin/env python3
"""Diagnose why IMEX method doesn't work."""

import sys
import time
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
    grid_points=(8, 8, 4),  # Very small grid for speed
    transport_coeffs=transport_coeffs,
)

wave_number = 1.0
benchmark.setup_initial_conditions(wave_number=wave_number)

print("=" * 80)
print("IMEX METHOD DIAGNOSTIC")
print("=" * 80)
print()

# Test single step with each method
dt = 0.01

print("Initial state:")
print(f"  rho std: {np.std(benchmark.fields.rho):.6e}")
print()

# Split-step (should work)
print("Testing split_step method...")
benchmark.setup_initial_conditions(wave_number=wave_number)
rho_initial = np.std(benchmark.fields.rho)

start = time.time()
try:
    for i in range(5):
        benchmark.solver.time_step(dt, method="split_step")
    elapsed_split = time.time() - start
    rho_final = np.std(benchmark.fields.rho)

    print(f"  ✓ Completed 5 steps in {elapsed_split:.3f}s")
    print(f"  rho: {rho_initial:.6e} → {rho_final:.6e}")
    print(f"  Ratio: {rho_final/rho_initial:.6f}")
    print()
except Exception as e:
    print(f"  ❌ Failed: {e}")
    print()

# IMEX (probably broken)
print("Testing spectral_imex method...")
benchmark.setup_initial_conditions(wave_number=wave_number)
rho_initial = np.std(benchmark.fields.rho)

start = time.time()
try:
    for i in range(5):
        print(f"  Step {i+1}/5...", end='\r')
        benchmark.solver.time_step(dt, method="spectral_imex")
        if time.time() - start > 10:
            print(f"\n  ⏱️  Timeout after {i+1} steps")
            break
    elapsed_imex = time.time() - start
    rho_final = np.std(benchmark.fields.rho)

    print(f"\n  ✓ Completed in {elapsed_imex:.3f}s")
    print(f"  rho: {rho_initial:.6e} → {rho_final:.6e}")
    print(f"  Ratio: {rho_final/rho_initial:.6f}")

    if abs(rho_final - rho_initial) < 1e-10:
        print(f"  ❌ PROBLEM: No evolution detected!")

    if elapsed_imex > 0 and elapsed_split > 0:
        print(f"  Speed: {elapsed_imex/elapsed_split:.1f}x slower than split_step")
    print()
except Exception as e:
    print(f"\n  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    print()

print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if elapsed_imex > 10 * elapsed_split:
    print("⚠️  IMEX is >10x slower than split_step")
    print("   Likely issue: Implicit solver is expensive or iterating too much")
elif abs(rho_final - rho_initial) < 1e-10:
    print("⚠️  IMEX produces no evolution")
    print("   Likely issue: _compute_explicit_rhs or _solve_implicit_stage returns zero")
else:
    print("✓ IMEX appears functional, just slower")

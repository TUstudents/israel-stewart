#!/usr/bin/env python3
"""Analyze dispersion relation at different k values to find where instability appears."""

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

print("=" * 80)
print("DISPERSION RELATION ANALYSIS - STABILITY VS WAVE NUMBER")
print("=" * 80)
print()

# Test range of k values
k_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

print(f"{'k':>6s}  {'ω_r':>10s}  {'γ':>10s}  {'ω*τ_π':>10s}  {'Status':>15s}")
print("-" * 80)

for k in k_values:
    wave_vector = np.array([k, 0.0, 0.0])

    try:
        modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)

        if modes:
            mode = modes[0]
            omega_r = mode.frequency
            gamma = mode.attenuation  # Negative means growth!

            # Dimensionless parameter
            omega_tau = omega_r * transport_coeffs.shear_relaxation_time

            if gamma > 0:
                status = "✓ Stable (damped)"
            elif gamma < -1e-6:
                status = "❌ UNSTABLE!"
            else:
                status = "~ Marginally stable"

            print(f"{k:6.1f}  {omega_r:10.4f}  {gamma:10.4f}  {omega_tau:10.4f}  {status:>15s}")
        else:
            print(f"{k:6.1f}  {'FAILED':>10s}  {'':>10s}  {'':>10s}  {'No mode found':>15s}")

    except Exception as e:
        print(f"{k:6.1f}  {'ERROR':>10s}  {'':>10s}  {'':>10s}  {str(e)[:15]:>15s}")

print()
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

print("Expected behavior:")
print("  - At low k: Stable damped modes (γ > 0)")
print("  - At high k: Damping increases with k² (Navier-Stokes limit)")
print()

print("Observed:")
print("  - If γ < 0 at high k: INSTABILITY - indicates physics or numerical error")
print("  - If ω*τ > 1: Fast oscillations, relaxation lags behind")
print()

print("Possible causes of instability:")
print("  1. ❌ Wrong sign in dispersion relation (RULED OUT - verified correct)")
print("  2. ❌ Wrong sign in numerical implementation (RULED OUT - verified correct)")
print("  3. ⚠️  Missing second-order terms in dispersion relation")
print("  4. ⚠️  Numerical root finder converged to unphysical root")
print("  5. ⚠️  Israel-Stewart formulation itself unstable at high k/τ")
print()

print("=" * 80)

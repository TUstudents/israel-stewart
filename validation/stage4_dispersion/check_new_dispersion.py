#!/usr/bin/env python
"""Check dispersion relation after sign fix."""

import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

# Setup
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

k = 8.0
wave_vector = np.array([k, 0.0, 0.0])

print("="*80)
print("DISPERSION RELATION AFTER SIGN FIX")
print("="*80)
print()

# Get all roots
print("Finding roots of dispersion relation...")
try:
    modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
    print(f"Found {len(modes)} modes:")
    print()

    for i, mode in enumerate(modes):
        omega_complex = complex(mode.frequency, -mode.attenuation)
        print(f"Mode {i+1}:")
        print(f"  ω = {mode.frequency:.6f} - i·{mode.attenuation:.6f}")
        print(f"  Type: {mode.mode_type}")
        print(f"  c_s = {mode.sound_speed:.6f}")

        # Check residual
        matrix = benchmark.analytical._build_dispersion_matrix(omega_complex, wave_vector)
        det = np.linalg.det(matrix)
        print(f"  det(M) = {abs(det):.6e}  {'✓' if abs(det) < 1e-6 else '✗'}")

        # Classify
        if mode.attenuation > 0:
            print(f"  → STABLE (damped)")
        elif mode.attenuation < -1e-6:
            print(f"  → UNSTABLE (growing)!")
        else:
            print(f"  → NEUTRAL (undamped)")
        print()

except Exception as e:
    print(f"Error finding modes: {e}")
    import traceback
    traceback.print_exc()

# Try manual search for sound mode
print("Manual search for sound mode near ω ≈ c_s·k...")
c_s = 1.0 / np.sqrt(3.0)
omega_guess = c_s * k

# Search in complex plane
print()
for gamma in [0.0, 0.1, 0.5, 1.0, 2.0]:
    omega = complex(omega_guess, -gamma)
    matrix = benchmark.analytical._build_dispersion_matrix(omega, wave_vector)
    det = np.linalg.det(matrix)
    print(f"  ω = {omega.real:.3f} - i·{gamma:.3f}:  det = {abs(det):.6e}")

print()
print("="*80)

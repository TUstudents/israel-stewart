#!/usr/bin/env -S uv run python
"""Simple RK4 test to check if basic evolution works."""

import numpy as np
from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients

coeffs = TransportCoefficients(
    shear_viscosity=0.08,
    bulk_viscosity=0.04,
    shear_relaxation_time=1.0,
    bulk_relaxation_time=0.5,
)

benchmark = NumericalSoundWaveBenchmark(
    domain_size=2 * np.pi, grid_points=(32, 32, 16), transport_coeffs=coeffs
)
benchmark.setup_initial_conditions(wave_number=1.0)

print("Testing RK4 evolution...")
print(f"Initial density: mean = {np.mean(benchmark.fields.rho):.6f}")

try:
    # Try just one step
    print("Attempting single RK4 step with dt=0.01...")
    benchmark.solver.time_step(0.01, method="rk4")
    print(f"After 1 step: mean = {np.mean(benchmark.fields.rho):.6f}")
    print("✓ RK4 step successful")

    # Try a few more
    print("\nTrying 10 more steps...")
    for i in range(10):
        benchmark.solver.time_step(0.01, method="rk4")
        if (i + 1) % 5 == 0:
            print(f"  After {i+1} steps: mean = {np.mean(benchmark.fields.rho):.6f}")

    print("\n✓ RK4 evolution working")

except Exception as e:
    print(f"\n✗ RK4 failed: {e}")
    import traceback
    traceback.print_exc()

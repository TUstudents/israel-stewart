#!/usr/bin/env python3
"""
Compare explicit RK4 vs implicit IMEX performance.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients


def test_method(method: str, t_final: float = 0.1):
    """Test performance of integration method."""
    print(f"\nTesting {method}...")

    transport_coeffs = TransportCoefficients(
        shear_viscosity=0.08,
        bulk_viscosity=0.04,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
        lambda_pi_pi=0.1,
        lambda_pi_Pi=0.05,
        xi_1=0.2,
        xi_2=0.1,
    )

    benchmark = NumericalSoundWaveBenchmark(
        domain_size=2 * np.pi,
        grid_points=(32, 32, 16),
        transport_coeffs=transport_coeffs,
    )

    benchmark.setup_initial_conditions(
        wave_number=8.0,
        amplitude=0.01,
        background_density=1.0,
    )

    step_count = [0]

    def count(t, fields):
        step_count[0] += 1

    start = time.time()
    try:
        benchmark.solver.evolve(
            t_final=t_final,
            dt=None,  # Adaptive
            method=method,
            callback=count,
        )
        elapsed = time.time() - start
        n_steps = step_count[0]

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Steps: {n_steps}")
        print(f"  Time/step: {elapsed/max(n_steps,1):.4f}s")
        print(f"  Speedup vs IMEX: {3.14/elapsed:.2f}x")

        return elapsed, n_steps

    except Exception as e:
        print(f"  FAILED: {e}")
        return None, None


if __name__ == "__main__":
    print("=" * 60)
    print("EXPLICIT (RK4) vs IMPLICIT (IMEX) PERFORMANCE TEST")
    print("=" * 60)

    # Test IMEX (current default)
    imex_time, imex_steps = test_method("spectral_imex", t_final=0.1)

    # Test explicit RK4
    rk4_time, rk4_steps = test_method("rk4", t_final=0.1)

    # Test split-step
    split_time, split_steps = test_method("split_step", t_final=0.1)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if all(t is not None for t in [imex_time, rk4_time, split_time]):
        methods = [
            ("IMEX", imex_time, imex_steps),
            ("RK4", rk4_time, rk4_steps),
            ("Split-step", split_time, split_steps),
        ]
        methods.sort(key=lambda x: x[1])  # Sort by time

        print("Ranking (fastest first):")
        for i, (name, t, steps) in enumerate(methods, 1):
            print(f"  {i}. {name:12s}: {t:.2f}s ({steps} steps)")

#!/usr/bin/env python3
"""
Diagnostic to analyze IMEX Newton-Krylov solver performance.

Identifies if NK iterations are converging slowly.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients


def diagnose_nk_solver():
    """Monitor Newton-Krylov iterations."""
    print("Setting up minimal benchmark...")

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

    # Monkey-patch Newton-Krylov to track iterations
    original_solve = benchmark.solver._newton_krylov_solve

    nk_calls = [0]
    nk_iterations = []

    def tracked_solve(stiff_rhs_func, y_explicit):
        nk_calls[0] += 1
        # Call original and track iterations
        result = original_solve(stiff_rhs_func, y_explicit)
        # Try to extract iteration count (if available in scipy output)
        nk_iterations.append(nk_calls[0])  # Placeholder
        return result

    benchmark.solver._newton_krylov_solve = tracked_solve

    # Run one timestep
    print("Running single timestep...")
    try:
        benchmark.solver.time_step(dt=0.01, method="spectral_imex")
        print(f"\nNewton-Krylov solver called: {nk_calls[0]} times in 1 timestep")
        print("(IMEX RK2 scheme has 2 implicit stages, so expect 2 calls per step)")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    diagnose_nk_solver()

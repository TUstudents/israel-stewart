#!/usr/bin/env python3
"""
Diagnostic script to profile sound wave benchmark performance.

Identifies where time is spent during numerical simulation.
"""

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.fields import TransportCoefficients


def run_short_simulation(k: float = 8.0, t_final: float = 0.1):
    """Run minimal simulation for performance testing."""
    print(f"Setting up benchmark for k={k}, t_final={t_final}")

    # Create minimal benchmark
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
        grid_points=(32, 32, 16),  # Smaller grid for faster test
        transport_coeffs=transport_coeffs,
    )

    # Setup initial conditions
    benchmark.setup_initial_conditions(
        wave_number=k,
        amplitude=0.01,
        background_density=1.0,
    )

    # Count timesteps
    step_count = [0]

    def count_callback(t, fields):
        step_count[0] += 1

    print("Running simulation...")
    start = time.time()

    # Run short simulation with profiling
    try:
        benchmark.solver.evolve(
            t_final=t_final,
            dt=None,  # Use adaptive
            method="spectral_imex",
            callback=count_callback,
        )
        elapsed = time.time() - start
        n_steps = step_count[0]

        print("\nResults:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Steps: {n_steps}")
        print(f"  Time per step: {elapsed/max(n_steps, 1):.4f}s")
        print(f"  Projected time for 1.0 time units: {elapsed*10:.1f}s")

        return elapsed, n_steps

    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def profile_simulation(k: float = 8.0, t_final: float = 0.1):
    """Run with cProfile to identify bottlenecks."""
    print("=" * 80)
    print("PROFILING SOUND WAVE SIMULATION")
    print("=" * 80)
    print()

    profiler = cProfile.Profile()
    profiler.enable()

    elapsed, n_steps = run_short_simulation(k, t_final)

    profiler.disable()

    if elapsed is not None:
        print("\n" + "=" * 80)
        print("TOP 20 TIME-CONSUMING FUNCTIONS")
        print("=" * 80)
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(20)

        print("\n" + "=" * 80)
        print("TOP 20 BY SELF TIME")
        print("=" * 80)
        stats.sort_stats("tottime")
        stats.print_stats(20)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile sound wave benchmark")
    parser.add_argument("--k", type=float, default=8.0, help="Wave number")
    parser.add_argument("--time", type=float, default=0.1, help="Simulation time")
    parser.add_argument("--no-profile", action="store_true", help="Skip profiling")

    args = parser.parse_args()

    if args.no_profile:
        run_short_simulation(args.k, args.time)
    else:
        profile_simulation(args.k, args.time)

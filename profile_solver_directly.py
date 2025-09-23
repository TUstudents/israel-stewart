#!/usr/bin/env python3
"""
Direct Profiling of Sound Wave Solver

Uses Python's cProfile to identify actual hotspots in the solver.
"""

import cProfile
import pstats
import time
import numpy as np

from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


def run_solver_test():
    """Run solver test that will be profiled."""
    # Create a 16^3 test case
    grid = SpacetimeGrid(
        coordinate_system='cartesian',
        time_range=(0.0, 1.0),
        spatial_ranges=[(0.0, 2*np.pi), (0.0, 2*np.pi), (0.0, 2*np.pi)],
        grid_points=(4, 16, 16, 16),
        boundary_conditions='periodic'
    )

    fields = ISFieldConfiguration(grid)
    fields.rho.fill(1.0)
    fields.pressure.fill(0.33)

    coeffs = TransportCoefficients(
        shear_viscosity=0.1,
        bulk_viscosity=0.05,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3
    )

    print('Creating solver...')
    solver = SpectralISHydrodynamics(grid, fields, coeffs)

    print('Running timesteps for profiling...')

    # Run several timesteps
    dt = 0.01
    for i in range(5):
        solver.time_step(dt, method='split_step')

    print('Profiling complete')


def main():
    """Profile the solver and analyze results."""
    print("🔍 PROFILING SOLVER DIRECTLY")
    print("=" * 40)

    # Profile the execution
    profiler = cProfile.Profile()

    start_time = time.time()
    profiler.enable()

    run_solver_test()

    profiler.disable()
    elapsed = time.time() - start_time

    print(f"\nTotal execution time: {elapsed:.2f}s")

    # Analyze the results
    stats = pstats.Stats(profiler)

    print("\n🕐 TOP FUNCTIONS BY CUMULATIVE TIME:")
    print("-" * 80)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    print("\n🔥 TOP FUNCTIONS BY SELF TIME:")
    print("-" * 80)
    stats.sort_stats('time')
    stats.print_stats(20)  # Top 20 functions by self time

    # Look specifically for our modules
    print("\n🎯 ISRAEL-STEWART MODULE FUNCTIONS:")
    print("-" * 80)
    stats.print_stats('israel_stewart', 30)

    # Look for numpy/scipy functions
    print("\n📊 NUMPY/SCIPY FUNCTIONS:")
    print("-" * 80)
    stats.print_stats('numpy|scipy', 15)

    # Look for FFT functions specifically
    print("\n🌊 FFT FUNCTIONS:")
    print("-" * 80)
    stats.print_stats('fft', 10)

    # Save detailed stats to file
    stats.dump_stats('profiler/solver_profile.prof')
    print(f"\n✅ Detailed profile saved to: profiler/solver_profile.prof")
    print("   Use 'python -m pstats profiler/solver_profile.prof' to analyze interactively")


if __name__ == "__main__":
    main()
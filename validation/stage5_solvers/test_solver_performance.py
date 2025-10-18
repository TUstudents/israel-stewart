#!/usr/bin/env python3
"""
Test Solver Performance to Understand Bottlenecks

This script tests the solver performance step by step to identify bottlenecks.
"""

import time

import numpy as np

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark


def test_single_timestep():
    """Test performance of a single timestep."""
    print("🔧 TESTING SOLVER PERFORMANCE")
    print("=" * 40)

    # Ultra-minimal grid
    benchmark = NumericalSoundWaveBenchmark(
        domain_size=2 * np.pi,
        grid_points=(8, 8, 4, 4),  # Extremely small
    )

    print(f"Grid: {benchmark.grid_points}")
    print(f"Total grid points: {np.prod(benchmark.grid_points)}")

    # Setup minimal initial conditions
    benchmark.setup_initial_conditions(wave_number=1.0, amplitude=0.01)

    print(
        f"Initial density range: [{benchmark.fields.rho.min():.6f}, {benchmark.fields.rho.max():.6f}]"
    )

    # Test single timestep
    print("\\nTesting single timestep...")
    dt = 0.1

    try:
        start_time = time.time()
        benchmark.solver.evolve(dt)
        elapsed = time.time() - start_time

        print(f"✅ Single timestep completed in {elapsed:.2f} seconds")

        # Check if fields changed
        new_rho_range = [benchmark.fields.rho.min(), benchmark.fields.rho.max()]
        print(f"New density range: [{new_rho_range[0]:.6f}, {new_rho_range[1]:.6f}]")

        if abs(new_rho_range[0] - 0.99) > 1e-10 or abs(new_rho_range[1] - 1.01) > 1e-10:
            print("✅ Fields evolved (densities changed)")
        else:
            print("⚠️  Fields may not have evolved significantly")

        return True, elapsed

    except Exception as e:
        print(f"❌ Single timestep failed: {e}")
        import traceback

        traceback.print_exc()
        return False, 0


def estimate_full_simulation_time(single_step_time, n_steps):
    """Estimate how long a full simulation would take."""
    total_time = single_step_time * n_steps

    print("\\n📊 TIME ESTIMATION:")
    print(f"   Single timestep: {single_step_time:.2f} seconds")
    print(f"   For {n_steps} steps: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    if total_time > 300:  # > 5 minutes
        print("   🐌 VERY SLOW - not practical for routine testing")
    elif total_time > 60:  # > 1 minute
        print("   🐌 SLOW - acceptable for limited testing")
    else:
        print("   ⚡ FAST - good for routine use")


def main():
    """Test solver performance."""
    print("Testing Israel-Stewart spectral solver performance...")
    print("This will help understand why simulations are slow.")
    print()

    # Test single step
    success, step_time = test_single_timestep()

    if success:
        # Estimate different simulation scenarios
        scenarios = [
            (50, "Minimal simulation (50 steps)"),
            (200, "Short simulation (200 steps)"),
            (1000, "Full simulation (1000 steps)"),
        ]

        for n_steps, description in scenarios:
            print(f"\\n{description}:")
            estimate_full_simulation_time(step_time, n_steps)

        print("\\n🎯 RECOMMENDATIONS:")
        if step_time < 0.1:
            print("   ✅ Performance is good - can run full benchmarks")
        elif step_time < 1.0:
            print("   ⚠️  Performance is moderate - use small grids and short simulations")
        else:
            print("   ❌ Performance is poor - solver needs optimization before benchmarking")
            print("   Consider:")
            print("     • Smaller grids (4x4x4x4)")
            print("     • Larger timesteps")
            print("     • Fewer evolution steps")
            print("     • Solver algorithm optimization")

    else:
        print("\\n❌ Cannot run performance tests - solver has fundamental issues")

    return success


if __name__ == "__main__":
    main()

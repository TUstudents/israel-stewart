#!/usr/bin/env python3
"""
Minimal Sound Wave Benchmark - Quick but Complete Test

This script runs a minimal but complete numerical simulation to validate
the Israel-Stewart solver against analytical predictions.

Usage:
    python run_minimal_sound_benchmark.py
"""

import time

import numpy as np

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark


def run_minimal_benchmark():
    """Run a single, minimal but complete benchmark."""
    print("🔊 MINIMAL COMPLETE SOUND WAVE BENCHMARK")
    print("=" * 50)
    print("Running ONE complete simulation with minimal parameters...")
    print()

    # Minimal grid for speed
    print("Initializing benchmark...")
    benchmark = NumericalSoundWaveBenchmark(
        domain_size=2 * np.pi,
        grid_points=(16, 16, 8, 8),  # Very small grid for speed
        # grid_points=(32, 32, 16, 16),
    )
    print(f"✅ Grid: {benchmark.grid_points} (minimal for speed)")
    print()

    # Single test case
    k = 1.0  # Simple wave number
    sim_time = 3.0  # Short simulation

    print(f"Running simulation: k={k}, sim_time={sim_time}")
    print("Parameters chosen for speed, not accuracy...")
    print()

    try:
        start_time = time.time()

        # This is the crucial call that was missing before
        result = benchmark.run_simulation(
            wave_number=k,
            simulation_time=sim_time,
            n_periods=3,  # Few periods for speed
            dt_factor=0.2,  # Larger timestep for speed
        )

        elapsed_time = time.time() - start_time

        print("✅ SIMULATION COMPLETED!")
        print(f"   Runtime: {elapsed_time:.1f} seconds")
        print()

        # Real results from actual numerical vs analytical comparison
        print("📊 NUMERICAL vs ANALYTICAL RESULTS:")
        print(f"   Wave number: k = {result.wave_number}")
        print(f"   Analytical frequency: ω = {result.analytical_frequency:.4f}")
        print(f"   Numerical frequency:  ω = {result.measured_frequency:.4f}")
        print(f"   Frequency error: {result.frequency_error*100:.1f}%")
        print()
        print(f"   Analytical damping: γ = {result.analytical_damping_rate:.4f}")
        print(f"   Numerical damping:  γ = {result.measured_damping_rate:.4f}")
        print(f"   Damping error: {result.damping_error*100:.1f}%")
        print()
        print(f'   Convergence achieved: {"✅ YES" if result.convergence_achieved else "❌ NO"}')
        print(f"   Grid resolution: {result.grid_resolution}")
        print(f"   Simulation time: {result.simulation_time:.1f} time units")

        # Analyze the time series data
        time_series = result.time_series_data
        if "density" in time_series and len(time_series["density"]) > 0:
            density = time_series["density"]
            time_array = time_series["time"]

            print()
            print("📈 TIME SERIES ANALYSIS:")
            print(f"   Data points: {len(density)}")
            print(f"   Density range: [{density.min():.6f}, {density.max():.6f}]")
            print(f"   Oscillation amplitude: {(density.max() - density.min())/2:.6f}")
            print(f"   Time span: {time_array[-1] - time_array[0]:.2f} time units")

            # Check if oscillations are present
            density_variation = np.ptp(density)
            if density_variation > 1e-6:
                print("   ✅ Oscillations detected in time series")
            else:
                print("   ❌ No significant oscillations detected")

        # Assessment
        print()
        print("🎯 ASSESSMENT:")

        # Frequency accuracy
        if result.frequency_error < 0.1:
            freq_status = "✅ EXCELLENT"
        elif result.frequency_error < 0.2:
            freq_status = "✅ GOOD"
        elif result.frequency_error < 0.5:
            freq_status = "⚠️  FAIR"
        else:
            freq_status = "❌ POOR"
        print(f"   Frequency accuracy: {freq_status} ({result.frequency_error*100:.1f}% error)")

        # Damping accuracy (more tolerant)
        if result.damping_error < 0.3:
            damp_status = "✅ GOOD"
        elif result.damping_error < 0.7:
            damp_status = "⚠️  FAIR"
        else:
            damp_status = "❌ POOR"
        print(f"   Damping accuracy: {damp_status} ({result.damping_error*100:.1f}% error)")

        # Overall
        if result.convergence_achieved and result.frequency_error < 0.3:
            overall = "✅ BENCHMARK PASSED"
        elif result.frequency_error < 0.5:
            overall = "⚠️  BENCHMARK MARGINAL"
        else:
            overall = "❌ BENCHMARK FAILED"
        print(f"   Overall result: {overall}")

        return True, result

    except Exception as e:
        print(f"❌ SIMULATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def main():
    """Main execution."""
    print("Starting minimal complete sound wave benchmark...")
    print("This runs an ACTUAL numerical simulation with time evolution.")
    print("Expected runtime: 1-5 minutes.")
    print()

    try:
        success, result = run_minimal_benchmark()

        print()
        print("🏁 FINAL RESULT:")
        if success:
            print("✅ Minimal benchmark completed successfully!")
            print("This demonstrates that the complete numerical vs analytical")
            print("comparison is working. The benchmark framework is functional.")
            print()
            print("For production use:")
            print("• Increase grid resolution for higher accuracy")
            print("• Run longer simulations for better statistics")
            print("• Test multiple wave numbers for comprehensive validation")
        else:
            print("❌ Minimal benchmark failed!")
            print("There are issues with the solver that need to be addressed.")

    except KeyboardInterrupt:
        print("\\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\\n❌ Benchmark failed: {e}")


if __name__ == "__main__":
    main()

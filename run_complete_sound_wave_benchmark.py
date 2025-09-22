#!/usr/bin/env python3
"""
Complete Sound Wave Benchmark - Full Numerical vs Analytical Comparison

This script runs the ACTUAL numerical simulation and compares results to analytical
predictions. This is the real benchmark that was missing from the previous script.

Usage:
    python run_complete_sound_wave_benchmark.py
    # or
    uv run python run_complete_sound_wave_benchmark.py
"""

import time

import numpy as np

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark


def run_full_benchmark():
    """Run the complete numerical vs analytical benchmark."""
    print("🔊 COMPLETE SOUND WAVE BENCHMARK")
    print("=" * 50)
    print("This runs the ACTUAL numerical simulation and compares to analytical results.")
    print()

    # Initialize benchmark with reasonable resolution
    print("Initializing benchmark...")
    benchmark = NumericalSoundWaveBenchmark(
        domain_size=2 * np.pi,
        grid_points=(32, 32, 16, 16),  # Conservative grid for stability
    )
    print(f"✅ Grid: {benchmark.grid_points}")
    print(f"✅ Domain: {benchmark.domain_size}")
    print(
        f"✅ Transport coeffs: η={benchmark.transport_coeffs.shear_viscosity}, ζ={benchmark.transport_coeffs.bulk_viscosity}"
    )
    print()

    # Test parameters
    test_cases = [
        {"k": 0.5, "sim_time": 8.0, "name": "Low frequency"},
        {"k": 1.0, "sim_time": 6.0, "name": "Moderate frequency"},
        {"k": 2.0, "sim_time": 4.0, "name": "High frequency"},
    ]

    results = []

    print("Running numerical simulations...")
    print("=" * 40)

    for i, case in enumerate(test_cases):
        k = case["k"]
        sim_time = case["sim_time"]
        name = case["name"]

        print(f"\\nTest {i+1}/3: {name} (k={k})")
        print("-" * 30)

        try:
            # Run the actual simulation
            start_time = time.time()

            result = benchmark.run_simulation(
                wave_number=k,
                simulation_time=sim_time,
                n_periods=5,
                dt_factor=0.1,  # Conservative timestep
            )

            elapsed_time = time.time() - start_time

            # Display results
            print(f"✅ Simulation completed in {elapsed_time:.1f} seconds")
            print(
                f"   Analytical: ω = {result.analytical_frequency:.4f}, γ = {result.analytical_damping_rate:.4f}"
            )
            print(
                f"   Numerical:  ω = {result.measured_frequency:.4f}, γ = {result.measured_damping_rate:.4f}"
            )
            print(
                f"   Errors: freq = {result.frequency_error*100:.1f}%, damp = {result.damping_error*100:.1f}%"
            )
            print(f'   Convergence: {"✅ YES" if result.convergence_achieved else "❌ NO"}')

            # Check if simulation data looks reasonable
            time_series = result.time_series_data
            if "density" in time_series:
                density_variation = np.ptp(time_series["density"])  # peak-to-peak
                print(f"   Density variation: {density_variation:.6f}")

                # Basic sanity check
                if density_variation < 1e-6:
                    print("   ⚠️  WARNING: Very small density variations detected")
                elif density_variation > 0.1:
                    print("   ⚠️  WARNING: Very large density variations - may be unstable")

            results.append(
                {"case": case, "result": result, "simulation_time": elapsed_time, "success": True}
            )

        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            results.append(
                {
                    "case": case,
                    "result": None,
                    "simulation_time": 0,
                    "success": False,
                    "error": str(e),
                }
            )

    # Summary analysis
    print("\\n🎯 BENCHMARK RESULTS SUMMARY")
    print("=" * 40)

    successful_runs = [r for r in results if r["success"]]
    failed_runs = [r for r in results if not r["success"]]

    print(f"Successful simulations: {len(successful_runs)}/{len(results)}")

    if failed_runs:
        print("\\nFailed simulations:")
        for fail in failed_runs:
            print(f'   ❌ k={fail["case"]["k"]}: {fail.get("error", "Unknown error")}')

    if successful_runs:
        print("\\nSuccessful simulations:")

        freq_errors = []
        damp_errors = []
        convergence_count = 0

        for run in successful_runs:
            result = run["result"]
            case = run["case"]

            print(f'\\n   k = {case["k"]} ({case["name"]}):')
            print(f"      Frequency error: {result.frequency_error*100:.1f}%")
            print(f"      Damping error: {result.damping_error*100:.1f}%")
            print(f'      Simulation time: {run["simulation_time"]:.1f}s')
            print(f'      Converged: {"Yes" if result.convergence_achieved else "No"}')

            freq_errors.append(result.frequency_error)
            damp_errors.append(result.damping_error)
            if result.convergence_achieved:
                convergence_count += 1

        # Overall statistics
        print("\\n📊 OVERALL STATISTICS:")
        print(f"   Average frequency error: {np.mean(freq_errors)*100:.1f}%")
        print(f"   Average damping error: {np.mean(damp_errors)*100:.1f}%")
        print(
            f"   Convergence rate: {convergence_count}/{len(successful_runs)} ({convergence_count/len(successful_runs)*100:.0f}%)"
        )

        # Performance
        total_sim_time = sum(run["simulation_time"] for run in successful_runs)
        print(f"   Total simulation time: {total_sim_time:.1f}s")
        print(f"   Average time per run: {total_sim_time/len(successful_runs):.1f}s")

        # Assessment
        print("\\n🏆 BENCHMARK ASSESSMENT:")
        avg_freq_error = np.mean(freq_errors)
        avg_damp_error = np.mean(damp_errors)
        convergence_rate = convergence_count / len(successful_runs)

        if avg_freq_error < 0.1 and convergence_rate > 0.7:
            print("   ✅ EXCELLENT: High accuracy and convergence")
        elif avg_freq_error < 0.2 and convergence_rate > 0.5:
            print("   ✅ GOOD: Acceptable accuracy and convergence")
        elif avg_freq_error < 0.3:
            print("   ⚠️  FAIR: Moderate accuracy, needs improvement")
        else:
            print("   ❌ POOR: Low accuracy, significant issues detected")

        # Technical assessment
        print("\\n🔧 TECHNICAL ASSESSMENT:")
        if total_sim_time / len(successful_runs) > 300:  # > 5 minutes per run
            print("   ⚠️  Performance: Slow (consider optimization)")
        elif total_sim_time / len(successful_runs) > 60:  # > 1 minute per run
            print("   ⚠️  Performance: Moderate (acceptable for testing)")
        else:
            print("   ✅ Performance: Fast (good for routine use)")

        if convergence_rate < 0.5:
            print("   ❌ Stability: Poor convergence rate")
        elif convergence_rate < 0.8:
            print("   ⚠️  Stability: Moderate convergence rate")
        else:
            print("   ✅ Stability: Excellent convergence rate")

    else:
        print("\\n❌ NO SUCCESSFUL SIMULATIONS")
        print("The benchmark cannot be assessed. Check solver implementation.")

    return results


def main():
    """Main benchmark execution."""
    print("Starting complete sound wave benchmark...")
    print("This will run actual numerical simulations and compare to analytical results.")
    print("Expected runtime: 5-15 minutes depending on system performance.")
    print()

    try:
        results = run_full_benchmark()

        print("\\n🎯 FINAL CONCLUSION:")
        successful = sum(1 for r in results if r["success"])
        if successful > 0:
            print(f"✅ Benchmark completed: {successful}/{len(results)} simulations successful")
            print("Results show actual numerical vs analytical comparison.")
        else:
            print("❌ Benchmark failed: No successful simulations")
            print("Solver may have implementation issues requiring investigation.")

    except KeyboardInterrupt:
        print("\\n\\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\\n\\n❌ Benchmark failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

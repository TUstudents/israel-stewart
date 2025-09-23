#!/usr/bin/env python3
"""
Profiled Sound Wave Benchmark - Baseline Performance Analysis

Runs the sound wave benchmark with detailed profiling to establish
baseline performance metrics for optimization.
"""

import time
import psutil
import os
import numpy as np

from israel_stewart.benchmarks.sound_waves import NumericalSoundWaveBenchmark
from israel_stewart.core.performance import (
    profile_operation,
    detailed_performance_report,
    reset_performance_stats
)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_profiled_benchmark():
    """Run benchmark with detailed profiling."""
    print("🔊 PROFILED SOUND WAVE BENCHMARK - BASELINE ANALYSIS")
    print("=" * 60)
    print("Target: 16×16×16 grid performance analysis")
    print()

    # Reset profiling state
    reset_performance_stats()

    # Record initial memory
    initial_memory = get_memory_usage()
    peak_memory = initial_memory

    print(f"Initial memory usage: {initial_memory:.1f} MB")
    print()

    # Create benchmark with 16^3 grid
    print("Initializing benchmark...")
    with profile_operation("benchmark_initialization"):
        benchmark = NumericalSoundWaveBenchmark(
            domain_size=2 * np.pi,
            grid_points=(8, 16, 16, 16),  # 16×16×16 spatial grid, 8 time points
        )

    current_memory = get_memory_usage()
    peak_memory = max(peak_memory, current_memory)
    print(f"✅ Grid: {benchmark.grid_points}")
    print(f"Memory after initialization: {current_memory:.1f} MB (+{current_memory-initial_memory:.1f} MB)")
    print()

    # Benchmark parameters
    k = 1.0  # Wave number
    sim_time = 1.0  # Short simulation for profiling

    print(f"Running simulation: k={k}, sim_time={sim_time}")
    print("Focus: Profile performance, not accuracy")
    print()

    try:
        overall_start = time.time()

        # Run simulation with profiling
        with profile_operation("complete_simulation", {"grid_size": (16, 16, 16), "sim_time": sim_time}):
            result = benchmark.run_simulation(
                wave_number=k,
                simulation_time=sim_time,
                n_periods=2,  # Few periods for speed
                dt_factor=0.3,  # Reasonable timestep
            )

        overall_time = time.time() - overall_start
        final_memory = get_memory_usage()
        peak_memory = max(peak_memory, final_memory)

        print("✅ SIMULATION COMPLETED!")
        print(f"   Total runtime: {overall_time:.1f} seconds")
        print(f"   Final memory: {final_memory:.1f} MB")
        print(f"   Peak memory: {peak_memory:.1f} MB")
        print(f"   Memory increase: {peak_memory - initial_memory:.1f} MB")
        print()

        # Generate detailed performance report
        print("📊 GENERATING DETAILED PERFORMANCE REPORT...")
        performance_report = detailed_performance_report()

        # Save full report to file
        report_file = "profiler/baseline_16x16x16_performance.json"
        with open(report_file, 'w') as f:
            import json
            json.dump(performance_report, f, indent=2)
        print(f"✅ Full report saved to: {report_file}")
        print()

        # Print top time consumers
        print("🕐 TOP TIME CONSUMERS:")
        operations = performance_report.get("operations", {})
        if operations:
            # Sort by total time
            sorted_ops = sorted(operations.items(),
                              key=lambda x: x[1].get("total_time", 0),
                              reverse=True)

            for i, (op_name, stats) in enumerate(sorted_ops[:10]):
                total_time = stats.get("total_time", 0)
                call_count = stats.get("call_count", 0)
                avg_time = total_time / max(call_count, 1)
                percent = (total_time / overall_time) * 100

                print(f"   {i+1:2d}. {op_name:30s} "
                      f"{total_time:6.2f}s ({percent:4.1f}%) "
                      f"[{call_count:3d} calls, {avg_time*1000:6.1f}ms avg]")

        print()

        # Print memory allocations
        print("💾 MEMORY ALLOCATION ANALYSIS:")
        allocations = performance_report.get("large_allocations", [])
        if allocations:
            print("   Large allocations (>10MB):")
            total_allocated = 0
            for alloc in allocations[:10]:
                size_mb = alloc.get("size_mb", 0)
                operation = alloc.get("operation", "unknown")
                shape = alloc.get("shape", "unknown")
                total_allocated += size_mb
                print(f"   • {size_mb:6.1f} MB - {operation:20s} {shape}")

            print(f"   Total large allocations: {total_allocated:.1f} MB")
        else:
            print("   No large allocations detected")

        print()

        # Identify suspects for optimization
        print("🎯 OPTIMIZATION TARGETS IDENTIFIED:")
        print()

        # Analyze FFT operations
        fft_ops = [op for op in operations.keys() if 'fft' in op.lower()]
        if fft_ops:
            fft_time = sum(operations[op].get("total_time", 0) for op in fft_ops)
            fft_percent = (fft_time / overall_time) * 100
            print(f"   FFT operations: {fft_time:.2f}s ({fft_percent:.1f}% of total)")
            for op in fft_ops:
                stats = operations[op]
                print(f"     • {op}: {stats.get('total_time', 0):.2f}s, {stats.get('call_count', 0)} calls")

        # Analyze tensor operations
        tensor_ops = [op for op in operations.keys()
                     if any(word in op.lower() for word in ['tensor', 'einsum', 'contraction', 'stress'])]
        if tensor_ops:
            tensor_time = sum(operations[op].get("total_time", 0) for op in tensor_ops)
            tensor_percent = (tensor_time / overall_time) * 100
            print(f"   Tensor operations: {tensor_time:.2f}s ({tensor_percent:.1f}% of total)")
            for op in tensor_ops:
                stats = operations[op]
                print(f"     • {op}: {stats.get('total_time', 0):.2f}s, {stats.get('call_count', 0)} calls")

        # Analyze derivative operations
        deriv_ops = [op for op in operations.keys()
                    if any(word in op.lower() for word in ['derivative', 'divergence', 'gradient'])]
        if deriv_ops:
            deriv_time = sum(operations[op].get("total_time", 0) for op in deriv_ops)
            deriv_percent = (deriv_time / overall_time) * 100
            print(f"   Derivative operations: {deriv_time:.2f}s ({deriv_percent:.1f}% of total)")
            for op in deriv_ops:
                stats = operations[op]
                print(f"     • {op}: {stats.get('total_time', 0):.2f}s, {stats.get('call_count', 0)} calls")

        print()

        # Summary of findings
        print("📋 BASELINE SUMMARY:")
        print(f"   Total runtime: {overall_time:.1f}s")
        print(f"   Peak memory: {peak_memory:.1f} MB")
        print(f"   Grid size: 16×16×16 = {16**3:,} points")
        print(f"   Performance per grid point: {overall_time/(16**3)*1000:.2f} ms/point")
        print(f"   Memory per grid point: {peak_memory/(16**3)*1024:.1f} KB/point")

        print()
        print("🎯 OPTIMIZATION OPPORTUNITIES:")
        print("   1. FFT operations (likely biggest impact)")
        print("   2. Large memory allocations")
        print("   3. Repeated tensor operations")
        print("   4. Derivative computation patterns")

        return True, result, {
            'runtime': overall_time,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': peak_memory - initial_memory,
            'report_file': report_file
        }

    except Exception as e:
        print(f"❌ PROFILED SIMULATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def main():
    """Main execution with profiling."""
    print("Starting profiled sound wave benchmark for baseline analysis...")
    print("This will establish performance baselines for optimization.")
    print()

    try:
        success, result, metrics = run_profiled_benchmark()

        print()
        print("🏁 PROFILING COMPLETE:")
        if success:
            print("✅ Baseline profiling completed successfully!")
            print(f"   Runtime: {metrics['runtime']:.1f}s")
            print(f"   Peak memory: {metrics['peak_memory_mb']:.1f} MB")
            print(f"   Detailed report: {metrics['report_file']}")
            print()
            print("Next steps:")
            print("• Analyze profiling data to identify bottlenecks")
            print("• Implement targeted optimizations")
            print("• Re-run to measure improvements")
        else:
            print("❌ Baseline profiling failed!")
            print("Need to fix fundamental issues before optimization.")

    except KeyboardInterrupt:
        print("\n⚠️  Profiling interrupted by user")
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")


if __name__ == "__main__":
    main()
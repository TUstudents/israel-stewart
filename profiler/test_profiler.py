#!/usr/bin/env python3
"""
Test script for the enhanced performance profiler.

This script demonstrates the new hierarchical profiling capabilities
and tests FFT profiling on a minimal Israel-Stewart solver setup.
"""

import numpy as np
import time
import json

from israel_stewart.core.performance import (
    profile_operation,
    detailed_performance_report,
    get_fft_profiler,
    fft_efficiency_report,
    reset_performance_stats
)


def simulate_fft_operations():
    """Simulate FFT operations to test FFT profiler."""
    fft_profiler = get_fft_profiler()

    # Test various FFT sizes
    test_sizes = [
        (64, 64, 32),     # Small grid
        (128, 128, 64),   # Medium grid
        (256, 256, 128),  # Large grid
    ]

    for size in test_sizes:
        print(f"Testing FFT size: {size}")

        # Create test data
        real_data = np.random.rand(*size)
        complex_data = np.random.rand(*size) + 1j * np.random.rand(*size)

        # Test forward FFT
        fft_result = fft_profiler.profile_fft(real_data, axes=(0, 1, 2))

        # Test inverse FFT
        ifft_result = fft_profiler.profile_ifft(fft_result, axes=(0, 1, 2))

        # Test real FFT
        rfft_result = fft_profiler.profile_rfft(real_data, axis=0)

        # Test inverse real FFT
        irfft_result = fft_profiler.profile_irfft(rfft_result, axis=0)


def simulate_spectral_operations():
    """Simulate spectral operations to test hierarchical profiling."""

    grid_size = (64, 64, 32, 32)

    with profile_operation("spectral_timestep", {"grid_size": grid_size}):

        # Simulate stress tensor computation
        with profile_operation("stress_energy_tensor"):
            # Simulate heavy computation
            stress_tensor = np.random.rand(*grid_size, 4, 4)
            time.sleep(0.1)  # Simulate computation time

            # Simulate tensor contractions
            with profile_operation("tensor_contractions"):
                contracted = np.einsum('ijklmn,ijklnp->ijklmp', stress_tensor, stress_tensor)
                time.sleep(0.05)

        # Simulate divergence computation
        with profile_operation("divergence_T"):
            # Simulate spatial derivatives
            with profile_operation("spatial_derivatives"):
                grad_x = np.gradient(stress_tensor, axis=0)
                grad_y = np.gradient(stress_tensor, axis=1)
                grad_z = np.gradient(stress_tensor, axis=2)
                time.sleep(0.08)

            # Simulate divergence assembly
            with profile_operation("divergence_assembly"):
                divergence = grad_x + grad_y + grad_z
                time.sleep(0.03)

        # Simulate viscous operator
        for i in range(5):  # Multiple calls to test frequency tracking
            with profile_operation("viscous_operator"):
                viscous_term = np.random.rand(*grid_size[:3])
                time.sleep(0.02)


def simulate_large_memory_allocation():
    """Simulate operations with large memory allocations."""

    with profile_operation("large_allocation_test"):
        # Allocate large arrays (should trigger memory tracking)
        large_array1 = np.zeros((512, 512, 256))  # ~500MB
        time.sleep(0.01)

        large_array2 = np.ones((512, 512, 256))   # Another ~500MB
        time.sleep(0.01)

        # Simulate computation
        result = large_array1 + large_array2
        time.sleep(0.02)

        # Clean up
        del large_array1, large_array2, result


def run_profiling_test():
    """Run comprehensive profiling test."""

    print("🔍 Testing Enhanced Performance Profiler")
    print("=" * 50)

    # Reset stats
    reset_performance_stats()

    # Test 1: Hierarchical operation profiling
    print("1. Testing hierarchical operation profiling...")
    simulate_spectral_operations()

    # Test 2: FFT profiling
    print("2. Testing FFT profiling...")
    simulate_fft_operations()

    # Test 3: Memory allocation tracking
    print("3. Testing memory allocation tracking...")
    simulate_large_memory_allocation()

    # Generate reports
    print("\n📊 Generating Performance Reports...")

    # Detailed hierarchical report
    detailed_report = detailed_performance_report()

    # FFT efficiency report
    fft_report = fft_efficiency_report()

    # Print summary
    print("\n📈 PERFORMANCE SUMMARY:")
    print("-" * 30)

    summary = detailed_report["summary"]
    print(f"Total operations: {summary['total_operations']}")
    print(f"Total time: {summary['total_time']:.3f}s")
    print(f"Unique operations: {summary['unique_operations']}")
    print(f"Large allocations: {summary['large_allocations_count']}")

    print("\n🐌 SLOWEST OPERATIONS:")
    for op, total_time in summary["slowest_operations"][:3]:
        print(f"  {op}: {total_time:.3f}s")

    print("\n🔄 MOST FREQUENT OPERATIONS:")
    for op, count in summary["most_frequent_operations"][:3]:
        print(f"  {op}: {count} calls")

    # Hierarchical timing analysis
    print("\n🏗️ HIERARCHICAL TIMING:")
    hierarchy = detailed_report["hierarchical_timing"]
    for parent, data in list(hierarchy.items())[:3]:
        print(f"\n  {parent} ({data['parent_total_time']:.3f}s total):")
        for child, child_data in list(data["children"].items())[:2]:
            print(f"    └─ {child}: {child_data['percentage']:.1f}% ({child_data['total_time']:.3f}s)")

    # Memory analysis
    print("\n💾 MEMORY HOTSPOTS:")
    memory_hotspots = detailed_report["memory_analysis"]["memory_hotspots"]
    for op, mem_data in list(memory_hotspots.items())[:3]:
        print(f"  {op}: {mem_data['max_peak_mb']:.1f}MB peak")

    # FFT analysis
    print("\n⚡ FFT PERFORMANCE:")
    fft_ops = fft_report["fft_operations"]
    for fft_type, fft_data in fft_ops.items():
        print(f"  {fft_type}: {fft_data['total_operations']} ops, {fft_data['avg_time']:.4f}s avg")

    # Optimization targets
    print("\n🎯 TOP OPTIMIZATION TARGETS:")
    targets = detailed_report["optimization_targets"][:3]
    for target in targets:
        print(f"  {target['operation']} ({target['type']}): {target['recommendation']}")

    # Save detailed reports to files
    print("\n💾 Saving detailed reports...")

    with open("detailed_performance_report.json", "w") as f:
        json.dump(detailed_report, f, indent=2, default=str)

    with open("fft_efficiency_report.json", "w") as f:
        json.dump(fft_report, f, indent=2, default=str)

    print("Reports saved to:")
    print("  - detailed_performance_report.json")
    print("  - fft_efficiency_report.json")

    print("\n✅ Profiling test completed successfully!")


if __name__ == "__main__":
    run_profiling_test()
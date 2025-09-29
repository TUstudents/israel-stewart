#!/usr/bin/env python3
"""
Test suite for ISFieldConfiguration copy optimization.

This script benchmarks memory usage and performance of field copying operations
before and after optimization.
"""

import gc
import sys
import time
import tracemalloc
from typing import Optional

import numpy as np
import psutil

# Add project root to path
sys.path.insert(0, "/home/feynman/projects/israel-stewart")

from israel_stewart.core.fields import ISFieldConfiguration
from israel_stewart.core.spacetime_grid import SpacetimeGrid


class MemoryProfiler:
    """Profile memory usage for field operations."""

    def __init__(self):
        self.process = psutil.Process()
        tracemalloc.start()

    def get_memory_info(self):
        """Get current memory usage information."""
        memory_info = self.process.memory_info()
        current, peak = tracemalloc.get_traced_memory()
        return {
            "rss_mb": memory_info.rss / 1024**2,
            "traced_current_mb": current / 1024**2,
            "traced_peak_mb": peak / 1024**2,
        }

    def reset_peak(self):
        """Reset peak memory tracking."""
        tracemalloc.stop()
        tracemalloc.start()

    def stop(self):
        """Stop memory profiling."""
        tracemalloc.stop()


def create_test_field_config(grid_size: int = 16) -> ISFieldConfiguration:
    """Create test field configuration with specified grid size."""
    grid = SpacetimeGrid(
        coordinate_system="cartesian",
        time_range=(0.0, 1.0),
        spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        grid_points=(10, grid_size, grid_size, grid_size),
    )

    fields = ISFieldConfiguration(grid)

    # Fill with non-trivial data
    fields.rho.fill(1.5)
    fields.n.fill(0.8)
    fields.pressure.fill(0.5)
    fields.temperature.fill(2.0)

    # Set some patterns in four-velocity
    fields.u_mu[..., 0] = 1.2
    fields.u_mu[..., 1:4] = 0.1

    # Set dissipative fluxes
    fields.Pi.fill(0.05)
    fields.pi_munu[..., 0, 0] = 0.02
    fields.q_mu[..., 0] = 0.03

    return fields


def benchmark_copy_operation(fields: ISFieldConfiguration, num_iterations: int = 10) -> dict:
    """Benchmark the current copy operation."""
    profiler = MemoryProfiler()

    # Warm up
    for _ in range(3):
        copy = fields.copy()
        del copy

    gc.collect()
    profiler.reset_peak()

    # Baseline memory
    baseline_memory = profiler.get_memory_info()

    # Time the copy operations
    start_time = time.time()
    copies = []

    for i in range(num_iterations):
        copy = fields.copy()
        copies.append(copy)

        if i == 0:  # Measure memory after first copy
            first_copy_memory = profiler.get_memory_info()

    end_time = time.time()

    # Final memory measurement
    final_memory = profiler.get_memory_info()

    # Clean up
    del copies
    gc.collect()

    return {
        "total_time": end_time - start_time,
        "avg_time_per_copy": (end_time - start_time) / num_iterations,
        "baseline_memory_mb": baseline_memory["rss_mb"],
        "first_copy_memory_mb": first_copy_memory["rss_mb"],
        "final_memory_mb": final_memory["rss_mb"],
        "memory_per_copy_mb": (final_memory["rss_mb"] - baseline_memory["rss_mb"]) / num_iterations,
        "peak_traced_mb": final_memory["traced_peak_mb"],
        "num_iterations": num_iterations,
    }


def test_selective_copy_concept(fields: ISFieldConfiguration) -> dict:
    """Test concept for selective copying."""
    profiler = MemoryProfiler()

    # Get list of all field names
    field_names = [
        "rho",
        "n",
        "u_mu",
        "Pi",
        "pi_munu",
        "q_mu",
        "rho_tilde",
        "u_mu_tilde",
        "pressure",
        "temperature",
        "eta",
        "zeta",
        "kappa",
    ]

    # Test copying only essential fields
    essential_fields = ["rho", "pressure", "u_mu", "Pi"]

    gc.collect()
    profiler.reset_peak()
    baseline_memory = profiler.get_memory_info()

    # Manual selective copy simulation
    start_time = time.time()
    new_config = ISFieldConfiguration(fields.grid)

    for field_name in essential_fields:
        field_data = getattr(fields, field_name)
        setattr(new_config, field_name, field_data.copy())

    end_time = time.time()
    selective_memory = profiler.get_memory_info()

    # Compare with full copy
    full_copy_start = time.time()
    full_copy = fields.copy()
    full_copy_end = time.time()
    full_copy_memory = profiler.get_memory_info()

    del new_config, full_copy
    gc.collect()

    return {
        "selective_copy_time": end_time - start_time,
        "full_copy_time": full_copy_end - full_copy_start,
        "selective_memory_mb": selective_memory["rss_mb"] - baseline_memory["rss_mb"],
        "full_copy_memory_mb": full_copy_memory["rss_mb"] - selective_memory["rss_mb"],
        "speedup_ratio": (full_copy_end - full_copy_start) / (end_time - start_time),
        "memory_ratio": (full_copy_memory["rss_mb"] - selective_memory["rss_mb"])
        / (selective_memory["rss_mb"] - baseline_memory["rss_mb"]),
        "num_fields_copied": len(essential_fields),
        "total_fields": len(field_names),
    }


def run_comprehensive_benchmark():
    """Run comprehensive benchmark of field copying."""
    print("=" * 70)
    print("ISFIELD CONFIGURATION COPY OPTIMIZATION BENCHMARK")
    print("=" * 70)

    # Test with different grid sizes
    grid_sizes = [16, 24]

    for grid_size in grid_sizes:
        print(f"\n📊 Testing with {grid_size}³ grid...")

        # Create test configuration
        fields = create_test_field_config(grid_size)

        print(f"Grid points: {np.prod(fields.grid.shape):,}")
        print(f"Fields memory estimate: {estimate_field_memory_mb(fields):.1f} MB")

        # Benchmark current copy operation
        print("\n⏱️  Current copy() method benchmark:")
        copy_results = benchmark_copy_operation(fields, num_iterations=5)

        print(f"  Average time per copy: {copy_results['avg_time_per_copy']:.4f}s")
        print(f"  Memory per copy: {copy_results['memory_per_copy_mb']:.1f} MB")
        print(f"  Peak traced memory: {copy_results['peak_traced_mb']:.1f} MB")

        # Test selective copying concept
        print("\n🎯 Selective copy concept test:")
        selective_results = test_selective_copy_concept(fields)

        print(f"  Selective copy time: {selective_results['selective_copy_time']:.4f}s")
        print(f"  Full copy time: {selective_results['full_copy_time']:.4f}s")
        print(f"  Speedup ratio: {selective_results['speedup_ratio']:.1f}x")
        print(f"  Memory reduction: {selective_results['memory_ratio']:.1f}x less")
        print(
            f"  Fields copied: {selective_results['num_fields_copied']}/{selective_results['total_fields']}"
        )

        del fields
        gc.collect()


def estimate_field_memory_mb(fields: ISFieldConfiguration) -> float:
    """Estimate total memory usage of all fields."""
    total_bytes = 0

    # Count array sizes
    for attr_name in dir(fields):
        attr = getattr(fields, attr_name)
        if isinstance(attr, np.ndarray):
            total_bytes += attr.nbytes

    return total_bytes / 1024**2


if __name__ == "__main__":
    print("Starting ISFieldConfiguration copy optimization tests...")

    try:
        run_comprehensive_benchmark()
        print("\n✅ Benchmark completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

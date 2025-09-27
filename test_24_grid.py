#!/usr/bin/env python3
"""
Test 24³ grid to validate scaling predictions.
"""

import os
import sys
import time
import tracemalloc

import numpy as np
import psutil

# Add the project root to path
sys.path.insert(0, "/home/feynman/projects/israel-stewart")

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


class MemoryMonitor:
    """Monitor memory usage throughout the test."""

    def __init__(self, max_memory_gb=6.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.measurements = []

    def check_memory(self, label=""):
        """Check current memory usage and abort if approaching limit."""
        memory_info = self.process.memory_info()
        current_memory = memory_info.rss

        self.peak_memory = max(self.peak_memory, current_memory)
        self.measurements.append((time.time(), current_memory, label))

        memory_gb = current_memory / 1024**3
        print(f"Memory: {memory_gb:.2f} GB {label}")

        if current_memory > self.max_memory_bytes:
            print(
                f"ERROR: Memory usage ({memory_gb:.2f} GB) exceeds limit ({self.max_memory_bytes/1024**3:.1f} GB)"
            )
            print("Aborting to prevent system breakdown!")
            sys.exit(1)

        return current_memory

    def get_peak_memory_gb(self):
        """Get peak memory usage in GB."""
        return self.peak_memory / 1024**3


def main():
    print("Testing 24³ grid scaling...")

    monitor = MemoryMonitor(max_memory_gb=6.0)
    tracemalloc.start()

    monitor.check_memory("Initial")

    try:
        # Create 24³ grid
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(10, 24, 24, 24),  # 138,240 total points
        )

        fields = ISFieldConfiguration(grid)
        fields.rho.fill(1.0)
        fields.pressure.fill(0.33)
        fields.Pi.fill(0.01)
        fields.pi_munu.fill(0.005)
        fields.u_mu[..., 0] = 1.0
        fields.u_mu[..., 1:4] = 0.01

        monitor.check_memory("After grid/fields creation")

        coeffs = TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05,
            shear_relaxation_time=0.5,
            bulk_relaxation_time=0.3,
        )

        solver = SpectralISHydrodynamics(grid, fields, coeffs)
        monitor.check_memory("After solver initialization")

        # Run 5 steps to test scaling
        dt = 0.01
        times = []

        for step in range(5):
            step_start = time.time()
            solver.time_step(dt, method="spectral_imex")
            step_time = time.time() - step_start
            times.append(step_time)

            print(f"Step {step+1}/5: {step_time:.4f}s")
            monitor.check_memory(f"After step {step+1}")

        # Results
        print("\n" + "=" * 50)
        print("24³ GRID RESULTS")
        print("=" * 50)
        print(f"Grid points: {np.prod(grid.grid_points):,}")
        print(f"Average time per step: {np.mean(times):.4f}s")
        print(f"Peak memory: {monitor.get_peak_memory_gb():.2f} GB")

        # Compare to 16³ baseline
        baseline_time = 1.55  # from previous test
        baseline_memory = 0.97  # GB
        scaling_factor = 24**3 / 16**3  # 3.375

        print("\nScaling analysis (24³ vs 16³):")
        print(f"  Grid points ratio: {scaling_factor:.1f}x")
        print(f"  Time ratio: {np.mean(times)/baseline_time:.1f}x")
        print(f"  Memory ratio: {monitor.get_peak_memory_gb()/baseline_memory:.1f}x")

        print("\n✅ 24³ test completed successfully!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        monitor.check_memory("Error state")

    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Tracemalloc peak: {peak / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()

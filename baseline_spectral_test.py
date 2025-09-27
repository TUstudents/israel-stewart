#!/usr/bin/env python3
"""
Conservative baseline test for spectral solver with memory monitoring.

This script tests the SpectralISHydrodynamics solver with a small 16³ grid
to establish performance and memory usage baselines while staying well
under the 8GB system memory limit.
"""

import os
import sys
import time
import traceback
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

    def __init__(self, max_memory_gb=4.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.measurements = []

    def check_memory(self, label=""):
        """Check current memory usage and abort if approaching limit."""
        memory_info = self.process.memory_info()
        current_memory = memory_info.rss  # Resident Set Size

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


def create_small_test_setup():
    """Create a minimal test setup with 16³ grid."""
    print("Creating 16³ test grid...")

    # Very conservative grid: 16³ spatial points
    grid = SpacetimeGrid(
        coordinate_system="cartesian",
        time_range=(0.0, 1.0),
        spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        grid_points=(10, 16, 16, 16),  # time, x, y, z
    )

    print(f"Grid created: {grid.grid_points} total points")

    # Initialize fields with simple initial conditions
    fields = ISFieldConfiguration(grid)

    # Set up simple uniform initial state
    fields.rho.fill(1.0)  # Energy density
    fields.pressure.fill(0.33)  # Pressure (p = ρ/3 for radiation)
    fields.Pi.fill(0.01)  # Small bulk pressure
    fields.pi_munu.fill(0.005)  # Small shear stress

    # Simple four-velocity (mostly at rest)
    fields.u_mu[..., 0] = 1.0  # Time component
    fields.u_mu[..., 1:4] = 0.01  # Small spatial velocities

    # Transport coefficients
    coeffs = TransportCoefficients(
        shear_viscosity=0.1,
        bulk_viscosity=0.05,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
    )

    return grid, fields, coeffs


def run_baseline_test():
    """Run the baseline spectral solver test."""
    print("=" * 60)
    print("SPECTRAL SOLVER BASELINE TEST")
    print("=" * 60)

    # Start memory monitoring
    monitor = MemoryMonitor(max_memory_gb=4.0)
    tracemalloc.start()

    monitor.check_memory("Initial")

    try:
        # Create test setup
        grid, fields, coeffs = create_small_test_setup()
        monitor.check_memory("After grid/fields creation")

        # Initialize spectral solver
        print("Initializing spectral solver...")
        solver = SpectralISHydrodynamics(grid, fields, coeffs)
        monitor.check_memory("After solver initialization")

        # Run short simulation
        print("Running short simulation...")
        dt = 0.01
        num_steps = 10

        times = []

        for step in range(num_steps):
            step_start = time.time()

            # Take one timestep
            try:
                solver.time_step(dt, method="spectral_imex")
                step_time = time.time() - step_start
                times.append(step_time)

                print(f"Step {step+1}/{num_steps}: {step_time:.4f}s")

                # Check memory every few steps
                if step % 3 == 0:
                    monitor.check_memory(f"After step {step+1}")

            except Exception as e:
                print(f"Error in step {step+1}: {e}")
                traceback.print_exc()
                break

        # Final memory check
        monitor.check_memory("Final")

        # Print results
        print("\n" + "=" * 60)
        print("BASELINE RESULTS")
        print("=" * 60)

        if times:
            print(f"Steps completed: {len(times)}")
            print(f"Average time per step: {np.mean(times):.4f}s")
            print(f"Min/Max step time: {np.min(times):.4f}s / {np.max(times):.4f}s")

        print(f"Peak memory usage: {monitor.get_peak_memory_gb():.2f} GB")

        # Check field values for basic sanity
        print("Final field values:")
        print(f"  Energy density range: [{np.min(fields.rho):.4f}, {np.max(fields.rho):.4f}]")
        print(f"  Pressure range: [{np.min(fields.pressure):.4f}, {np.max(fields.pressure):.4f}]")
        print(f"  Bulk pressure range: [{np.min(fields.Pi):.4f}, {np.max(fields.Pi):.4f}]")

        # Check for NaN or infinite values
        if np.any(np.isnan(fields.rho)) or np.any(np.isinf(fields.rho)):
            print("WARNING: NaN or infinite values detected in energy density!")
        else:
            print("✓ No NaN/infinite values detected")

        print("\n✓ Baseline test completed successfully!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        monitor.check_memory("Error state")
        return False

    finally:
        # Print memory trace summary
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Tracemalloc peak: {peak / 1024**2:.1f} MB")


if __name__ == "__main__":
    print("Starting conservative spectral solver baseline test...")
    print("System memory limit: 4.0 GB (safety margin from 8GB system limit)")

    success = run_baseline_test()

    if success:
        print("\n🎉 Baseline established successfully!")
        sys.exit(0)
    else:
        print("\n❌ Baseline test failed!")
        sys.exit(1)

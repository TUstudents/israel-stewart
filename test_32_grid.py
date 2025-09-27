#!/usr/bin/env python3
"""
Test 32³ grid - final scaling validation before approaching limits.
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
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.spectral import SpectralISHydrodynamics


def main():
    print("Testing 32³ grid - final safe scaling test...")

    process = psutil.Process(os.getpid())
    tracemalloc.start()

    def check_memory(label=""):
        memory_gb = process.memory_info().rss / 1024**3
        print(f"Memory: {memory_gb:.2f} GB {label}")
        if memory_gb > 7.0:  # Conservative abort at 7GB
            print("ERROR: Approaching 8GB limit - aborting!")
            sys.exit(1)
        return memory_gb

    check_memory("Initial")

    try:
        # Create 32³ grid - 327,680 total points
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(10, 32, 32, 32),
        )

        fields = ISFieldConfiguration(grid)
        fields.rho.fill(1.0)
        fields.pressure.fill(0.33)
        fields.Pi.fill(0.01)
        fields.pi_munu.fill(0.005)
        fields.u_mu[..., 0] = 1.0
        fields.u_mu[..., 1:4] = 0.01

        check_memory("After grid/fields")

        coeffs = TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05,
            shear_relaxation_time=0.5,
            bulk_relaxation_time=0.3,
        )

        solver = SpectralISHydrodynamics(grid, fields, coeffs)
        memory_after_init = check_memory("After solver init")

        # Run just 3 steps for safety
        times = []
        for step in range(3):
            step_start = time.time()
            solver.time_step(0.01, method="spectral_imex")
            step_time = time.time() - step_start
            times.append(step_time)

            memory_after_step = check_memory(f"After step {step+1}")
            print(f"Step {step+1}/3: {step_time:.4f}s")

        # Results
        print("\n" + "=" * 50)
        print("32³ GRID RESULTS")
        print("=" * 50)
        print(f"Grid points: {np.prod(grid.grid_points):,}")
        print(f"Average time per step: {np.mean(times):.4f}s")
        print(f"Peak memory: {memory_after_step:.2f} GB")

        # Scaling comparison
        ratios = {
            "16³": {"points": 40960, "time": 1.55, "memory": 0.97},
            "24³": {"points": 138240, "time": 5.09, "memory": 1.64},
            "32³": {
                "points": np.prod(grid.grid_points),
                "time": np.mean(times),
                "memory": memory_after_step,
            },
        }

        print("\nComplete scaling analysis:")
        for size, data in ratios.items():
            print(
                f"  {size}: {data['points']:,} points, {data['time']:.2f}s/step, {data['memory']:.2f}GB"
            )

        print(f"\n✅ 32³ test completed - system stable at {memory_after_step:.2f} GB")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Tracemalloc peak: {peak / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()

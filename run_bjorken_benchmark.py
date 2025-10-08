#!/usr/bin/env -S uv run python
"""
Bjorken Flow Benchmark Runner

This executable script runs comprehensive Bjorken flow validation against
analytical solutions, demonstrating the spectral solver's accuracy for
1D boost-invariant relativistic hydrodynamic expansion.

Usage:
    python run_bjorken_benchmark.py [--resolution RESOLUTION] [--output OUTPUT]

Examples:
    python run_bjorken_benchmark.py                    # Standard resolution
    python run_bjorken_benchmark.py --resolution high  # High resolution
    python run_bjorken_benchmark.py --output bjorken_results.h5
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.bjorken_flow import (
    BjorkenBenchmark,
    BjorkenFlowSolution,
    create_standard_bjorken_benchmark,
)
from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_benchmark(resolution: str = "standard") -> BjorkenBenchmark:
    """Create Bjorken benchmark with specified resolution.

    Args:
        resolution: One of "low", "standard", "high"

    Returns:
        Configured BjorkenBenchmark instance
    """
    resolution_configs = {
        "low": {"grid_points": (16, 16, 16), "tau0": 0.5, "temperature0": 0.3},
        "standard": {"grid_points": (32, 32, 32), "tau0": 0.5, "temperature0": 0.3},
        "high": {"grid_points": (64, 64, 64), "tau0": 0.5, "temperature0": 0.3},
    }

    if resolution not in resolution_configs:
        raise ValueError(
            f"Unknown resolution: {resolution}. Choose from {list(resolution_configs.keys())}"
        )

    config = resolution_configs[resolution]

    # Create grid
    grid = SpaceGrid(
        coordinate_system="cartesian",
        spatial_ranges=[(0.0, 2 * np.pi)] * 3,
        grid_points=config["grid_points"],
        boundary_conditions="periodic",
    )

    # Transport coefficients (typical QGP values)
    transport_coeffs = TransportCoefficients(
        shear_viscosity=0.08,  # η/s ~ 1/(4π)
        bulk_viscosity=0.04,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
        # Second-order coefficients
        lambda_pi_pi=0.1,
        lambda_pi_Pi=0.05,
        xi_1=0.2,
        xi_2=0.1,
    )

    # Analytical solution
    analytical = BjorkenFlowSolution(
        tau0=config["tau0"],
        temperature0=config["temperature0"],
        transport_coefficients=transport_coeffs,
    )

    benchmark = BjorkenBenchmark(
        grid=grid,
        coefficients=transport_coeffs,
        analytical_solution=analytical,
    )

    logger.info(f"Created Bjorken benchmark with {resolution} resolution: {config['grid_points']}")

    return benchmark


def run_benchmark(
    benchmark: BjorkenBenchmark,
    final_time: float = 10.0,
    timestep: float | None = None,
    method: str = "spectral_imex",
) -> dict:
    """Run Bjorken flow benchmark and compute errors.

    Args:
        benchmark: Configured BjorkenBenchmark instance
        final_time: Final proper time τ
        timestep: Time step size (None for auto)
        method: Integration method

    Returns:
        Dictionary with time series and error metrics
    """
    logger.info(f"Running Bjorken benchmark to τ = {final_time}")
    logger.info(f"Integration method: {method}")

    start_time = time.time()

    # Run simulation
    results = benchmark.run_numerical_simulation(
        final_time=final_time,
        timestep=timestep,
        method=method,
    )

    elapsed = time.time() - start_time
    logger.info(f"Simulation completed in {elapsed:.2f}s")

    # Compute errors
    errors = benchmark.compute_errors(results)

    # Log key metrics
    logger.info(f"Temperature L2 error: {errors['temperature_l2_error']:.6e}")
    logger.info(f"Shear stress L2 error: {errors['shear_l2_error']:.6e}")
    logger.info(f"Bulk pressure L2 error: {errors['bulk_l2_error']:.6e}")

    return {"results": results, "errors": errors, "elapsed_time": elapsed}


def plot_results(results: dict, output_path: Path | None = None) -> None:
    """Create validation plots comparing numerical and analytical solutions.

    Args:
        results: Dictionary from run_benchmark()
        output_path: Optional path to save figure
    """
    time_series = results["results"]
    errors = results["errors"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Temperature evolution
    ax = axes[0, 0]
    ax.plot(
        time_series["times"],
        time_series["temperature_numerical"],
        "o-",
        label="Numerical",
        alpha=0.7,
    )
    ax.plot(
        time_series["times"],
        time_series["temperature_analytical"],
        "--",
        label="Analytical",
        linewidth=2,
    )
    ax.set_xlabel("Proper time τ")
    ax.set_ylabel("Temperature T")
    ax.set_title(f"Temperature Evolution (L2 error: {errors['temperature_l2_error']:.2e})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shear stress evolution
    ax = axes[0, 1]
    ax.plot(
        time_series["times"], time_series["shear_numerical"], "o-", label="Numerical", alpha=0.7
    )
    ax.plot(
        time_series["times"], time_series["shear_analytical"], "--", label="Analytical", linewidth=2
    )
    ax.set_xlabel("Proper time τ")
    ax.set_ylabel("Shear stress magnitude |π|")
    ax.set_title(f"Shear Stress Evolution (L2 error: {errors['shear_l2_error']:.2e})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bulk pressure evolution
    ax = axes[1, 0]
    ax.plot(time_series["times"], time_series["bulk_numerical"], "o-", label="Numerical", alpha=0.7)
    ax.plot(
        time_series["times"], time_series["bulk_analytical"], "--", label="Analytical", linewidth=2
    )
    ax.set_xlabel("Proper time τ")
    ax.set_ylabel("Bulk pressure Π")
    ax.set_title(f"Bulk Pressure Evolution (L2 error: {errors['bulk_l2_error']:.2e})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative errors over time
    ax = axes[1, 1]
    temp_rel_error = np.abs(
        np.array(time_series["temperature_numerical"])
        - np.array(time_series["temperature_analytical"])
    ) / np.array(time_series["temperature_analytical"])
    ax.semilogy(time_series["times"], temp_rel_error, "o-", label="Temperature")
    ax.set_xlabel("Proper time τ")
    ax.set_ylabel("Relative error")
    ax.set_title("Relative Errors vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    """Main entry point for Bjorken benchmark."""
    parser = argparse.ArgumentParser(
        description="Run Bjorken flow benchmark with spectral solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                          # Standard resolution
    %(prog)s --resolution high        # High resolution test
    %(prog)s --output results.png     # Save plot to file
    %(prog)s --final-time 15.0        # Run to τ = 15
        """,
    )

    parser.add_argument(
        "--resolution",
        choices=["low", "standard", "high"],
        default="standard",
        help="Grid resolution (default: standard)",
    )

    parser.add_argument(
        "--final-time",
        type=float,
        default=10.0,
        help="Final proper time τ (default: 10.0)",
    )

    parser.add_argument(
        "--timestep",
        type=float,
        default=None,
        help="Time step size (default: auto)",
    )

    parser.add_argument(
        "--method",
        choices=["spectral_imex", "rk4"],
        default="spectral_imex",
        help="Integration method (default: spectral_imex)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for plot (default: show plot)",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (just show metrics)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("BJORKEN FLOW BENCHMARK - SPECTRAL SOLVER VALIDATION")
    print("=" * 80)
    print()

    try:
        # Create benchmark
        benchmark = create_benchmark(resolution=args.resolution)

        # Run simulation
        results = run_benchmark(
            benchmark,
            final_time=args.final_time,
            timestep=args.timestep,
            method=args.method,
        )

        # Display summary
        print()
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Resolution:          {args.resolution}")
        print(f"Grid points:         {benchmark.grid.grid_points}")
        print(f"Final time:          τ = {args.final_time}")
        print(f"Integration method:  {args.method}")
        print(f"Elapsed time:        {results['elapsed_time']:.2f}s")
        print()
        print("L2 Errors:")
        print(f"  Temperature:       {results['errors']['temperature_l2_error']:.6e}")
        print(f"  Shear stress:      {results['errors']['shear_l2_error']:.6e}")
        print(f"  Bulk pressure:     {results['errors']['bulk_l2_error']:.6e}")
        print()

        # Validation criteria (relaxed for initial validation)
        temp_threshold = 1e-2
        shear_threshold = 1e-2
        bulk_threshold = 1e-2

        temp_pass = results["errors"]["temperature_l2_error"] < temp_threshold
        shear_pass = results["errors"]["shear_l2_error"] < shear_threshold
        bulk_pass = results["errors"]["bulk_l2_error"] < bulk_threshold

        all_pass = temp_pass and shear_pass and bulk_pass

        print("Validation Status:")
        print(
            f"  Temperature:       {'✓ PASS' if temp_pass else '✗ FAIL'} (threshold: {temp_threshold:.0e})"
        )
        print(
            f"  Shear stress:      {'✓ PASS' if shear_pass else '✗ FAIL'} (threshold: {shear_threshold:.0e})"
        )
        print(
            f"  Bulk pressure:     {'✓ PASS' if bulk_pass else '✗ FAIL'} (threshold: {bulk_threshold:.0e})"
        )
        print()

        if all_pass:
            print("✅ BJORKEN FLOW BENCHMARK PASSED")
        else:
            print("⚠️  BJORKEN FLOW BENCHMARK - SOME METRICS OUTSIDE THRESHOLD")
            print("   (This may indicate implementation is still in progress)")

        print("=" * 80)

        # Plot results
        if not args.no_plot:
            plot_results(results, output_path=args.output)

        return 0 if all_pass else 1

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        print(f"\n❌ Benchmark failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

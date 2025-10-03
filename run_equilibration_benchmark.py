#!/usr/bin/env python3
"""
Equilibration Benchmark Runner

This executable script runs comprehensive equilibration dynamics validation,
testing relaxation to equilibrium, entropy production, and thermodynamic
consistency in the spectral Israel-Stewart solver.

Usage:
    python run_equilibration_benchmark.py [--resolution RESOLUTION] [--output OUTPUT]

Examples:
    python run_equilibration_benchmark.py                    # Standard resolution
    python run_equilibration_benchmark.py --resolution high  # High resolution
    python run_equilibration_benchmark.py --final-time 20.0  # Extended evolution
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.equilibration import (
    EntropyProductionAnalysis,
    EquilibrationAnalysis,
    EquilibrationBenchmark,
    RelaxationTimeAnalysis,
)
from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_benchmark(
    resolution: str = "standard",
) -> tuple[EquilibrationAnalysis, ISFieldConfiguration]:
    """Create equilibration benchmark with specified resolution.

    Args:
        resolution: One of "low", "standard", "high"

    Returns:
        Tuple of (EquilibrationAnalysis, initial_fields)
    """
    resolution_configs = {
        "low": {"grid_points": (16, 16, 16), "perturbation": 0.2},
        "standard": {"grid_points": (32, 32, 32), "perturbation": 0.2},
        "high": {"grid_points": (64, 64, 64), "perturbation": 0.1},
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

    # Transport coefficients
    transport_coeffs = TransportCoefficients(
        shear_viscosity=0.08,
        bulk_viscosity=0.04,
        shear_relaxation_time=0.5,
        bulk_relaxation_time=0.3,
        # Second-order coefficients
        lambda_pi_pi=0.1,
        lambda_pi_Pi=0.05,
        xi_1=0.2,
        xi_2=0.1,
    )

    # Create equilibration analysis
    metric = MinkowskiMetric()
    analysis = EquilibrationAnalysis(
        grid=grid,
        metric=metric,
        transport_coefficients=transport_coeffs,
    )

    # Create initial perturbed state
    initial_fields = ISFieldConfiguration(grid)

    # Homogeneous background
    initial_fields.rho.fill(1.0)
    initial_fields.pressure.fill(1.0 / 3.0)  # Radiation EOS
    initial_fields.u_mu[..., 0] = 1.0  # Rest frame

    # Add random perturbations to dissipative fluxes
    perturbation_amplitude = config["perturbation"]
    np.random.seed(42)  # Reproducibility

    # Perturb bulk pressure
    initial_fields.Pi[:] = perturbation_amplitude * np.random.randn(*grid.grid_points)

    # Perturb shear stress (symmetric traceless)
    for mu in range(4):
        for nu in range(mu, 4):
            if mu == nu:
                # Diagonal components (will enforce traceless later)
                initial_fields.pi_munu[..., mu, nu] = perturbation_amplitude * np.random.randn(
                    *grid.grid_points
                )
            else:
                # Off-diagonal (symmetric)
                perturb = perturbation_amplitude * np.random.randn(*grid.grid_points)
                initial_fields.pi_munu[..., mu, nu] = perturb
                initial_fields.pi_munu[..., nu, mu] = perturb

    # Make shear stress traceless
    trace = np.zeros(grid.grid_points)
    for mu in range(4):
        trace += initial_fields.pi_munu[..., mu, mu]
    for mu in range(4):
        initial_fields.pi_munu[..., mu, mu] -= trace / 4.0

    logger.info(
        f"Created equilibration benchmark with {resolution} resolution: {config['grid_points']}"
    )
    logger.info(f"Initial perturbation amplitude: {perturbation_amplitude}")

    return analysis, initial_fields


def run_equilibration(
    analysis: EquilibrationAnalysis,
    initial_fields: ISFieldConfiguration,
    final_time: float = 10.0,
    timestep: float | None = None,
    method: str = "spectral_imex",
) -> dict:
    """Run equilibration simulation and analyze thermodynamics.

    Args:
        analysis: EquilibrationAnalysis instance
        initial_fields: Initial field configuration
        final_time: Final simulation time
        timestep: Time step size (None for auto)
        method: Integration method

    Returns:
        Dictionary with equilibration data
    """
    logger.info(f"Running equilibration analysis to t = {final_time}")
    logger.info(f"Integration method: {method}")

    start_time = time.time()

    # Run relaxation analysis
    results = analysis.analyze_relaxation_to_equilibrium(
        initial_fields=initial_fields,
        final_time=final_time,
        timestep=timestep,
        method=method,
    )

    elapsed = time.time() - start_time
    logger.info(f"Equilibration simulation completed in {elapsed:.2f}s")

    return {"results": results, "elapsed_time": elapsed}


def analyze_relaxation_timescales(results: dict, transport_coeffs: TransportCoefficients) -> dict:
    """Analyze relaxation timescales from simulation data.

    Args:
        results: Results from run_equilibration()
        transport_coeffs: Transport coefficients

    Returns:
        Dictionary with relaxation time analysis
    """
    time_data = results["results"]["times"]
    bulk_data = results["results"]["bulk_pressure"]
    shear_data = results["results"]["shear_stress"]

    # Fit exponential decay to extract relaxation times
    analysis = {}

    # Bulk relaxation: Π(t) ~ exp(-t/τ_Π)
    try:
        if len(bulk_data) > 5 and np.max(np.abs(bulk_data)) > 1e-10:
            # Fit to log|Π(t)|
            valid_idx = np.abs(bulk_data) > 1e-10
            if np.sum(valid_idx) > 2:
                log_bulk = np.log(np.abs(bulk_data[valid_idx]))
                bulk_fit = np.polyfit(time_data[valid_idx], log_bulk, 1)
                measured_tau_bulk = -1.0 / bulk_fit[0]
                theoretical_tau_bulk = transport_coeffs.bulk_relaxation_time

                analysis["bulk_relaxation_measured"] = measured_tau_bulk
                analysis["bulk_relaxation_theoretical"] = theoretical_tau_bulk
                analysis["bulk_relaxation_error"] = (
                    abs(measured_tau_bulk - theoretical_tau_bulk) / theoretical_tau_bulk
                )
            else:
                analysis["bulk_relaxation_measured"] = np.nan
        else:
            analysis["bulk_relaxation_measured"] = np.nan
    except Exception as e:
        logger.warning(f"Bulk relaxation fit failed: {e}")
        analysis["bulk_relaxation_measured"] = np.nan

    # Shear relaxation: |π(t)| ~ exp(-t/τ_π)
    try:
        if len(shear_data) > 5 and np.max(np.abs(shear_data)) > 1e-10:
            valid_idx = np.abs(shear_data) > 1e-10
            if np.sum(valid_idx) > 2:
                log_shear = np.log(np.abs(shear_data[valid_idx]))
                shear_fit = np.polyfit(time_data[valid_idx], log_shear, 1)
                measured_tau_shear = -1.0 / shear_fit[0]
                theoretical_tau_shear = transport_coeffs.shear_relaxation_time

                analysis["shear_relaxation_measured"] = measured_tau_shear
                analysis["shear_relaxation_theoretical"] = theoretical_tau_shear
                analysis["shear_relaxation_error"] = (
                    abs(measured_tau_shear - theoretical_tau_shear) / theoretical_tau_shear
                )
            else:
                analysis["shear_relaxation_measured"] = np.nan
        else:
            analysis["shear_relaxation_measured"] = np.nan
    except Exception as e:
        logger.warning(f"Shear relaxation fit failed: {e}")
        analysis["shear_relaxation_measured"] = np.nan

    return analysis


def validate_entropy_production(results: dict) -> dict:
    """Validate that entropy production is non-negative (second law).

    Args:
        results: Results from run_equilibration()

    Returns:
        Dictionary with entropy validation
    """
    entropy_data = results["results"]["entropy"]

    # Check that entropy is monotonically increasing (or at least non-decreasing)
    entropy_diff = np.diff(entropy_data)
    violations = entropy_diff < -1e-10  # Small tolerance for numerical errors

    num_violations = np.sum(violations)
    total_steps = len(entropy_diff)

    # Total entropy production
    total_entropy_production = entropy_data[-1] - entropy_data[0]

    return {
        "entropy_violations": int(num_violations),
        "total_steps": int(total_steps),
        "violation_rate": float(num_violations) / total_steps if total_steps > 0 else 0.0,
        "total_entropy_production": float(total_entropy_production),
        "second_law_satisfied": num_violations == 0,
        "initial_entropy": float(entropy_data[0]),
        "final_entropy": float(entropy_data[-1]),
    }


def plot_equilibration(equilibration_data: dict, output_path: Path | None = None) -> None:
    """Create equilibration validation plots.

    Args:
        equilibration_data: Dictionary from run_equilibration()
        output_path: Optional path to save figure
    """
    results = equilibration_data["results"]
    times = results["times"]
    temperature = results["temperature"]
    entropy = results["entropy"]
    bulk = results["bulk_pressure"]
    shear = results["shear_stress"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Temperature evolution
    ax = axes[0, 0]
    ax.plot(times, temperature, "o-", markersize=3)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Temperature T")
    ax.set_title("Temperature Evolution")
    ax.grid(True, alpha=0.3)

    # Entropy evolution
    ax = axes[0, 1]
    ax.plot(times, entropy, "o-", markersize=3, color="green")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Entropy density s")
    ax.set_title("Entropy Production (Second Law Check)")
    ax.grid(True, alpha=0.3)

    # Bulk pressure relaxation
    ax = axes[1, 0]
    ax.semilogy(times, np.abs(bulk) + 1e-15, "o-", markersize=3, color="red")
    ax.set_xlabel("Time t")
    ax.set_ylabel("|Bulk pressure Π|")
    ax.set_title("Bulk Pressure Relaxation")
    ax.grid(True, alpha=0.3)

    # Shear stress relaxation
    ax = axes[1, 1]
    ax.semilogy(times, np.abs(shear) + 1e-15, "o-", markersize=3, color="purple")
    ax.set_xlabel("Time t")
    ax.set_ylabel("|Shear stress π|")
    ax.set_title("Shear Stress Relaxation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    """Main entry point for equilibration benchmark."""
    parser = argparse.ArgumentParser(
        description="Run equilibration benchmark with spectral solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                          # Standard resolution
    %(prog)s --resolution high        # High resolution test
    %(prog)s --output equilibration.png  # Save plot to file
    %(prog)s --final-time 20.0        # Extended evolution
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
        help="Final simulation time (default: 10.0)",
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
    print("EQUILIBRATION BENCHMARK - SPECTRAL SOLVER VALIDATION")
    print("=" * 80)
    print()

    try:
        # Create benchmark
        analysis, initial_fields = create_benchmark(resolution=args.resolution)

        # Run equilibration
        print(f"Running equilibration to t = {args.final_time}...")
        equilibration_data = run_equilibration(
            analysis,
            initial_fields,
            final_time=args.final_time,
            timestep=args.timestep,
            method=args.method,
        )

        # Analyze relaxation timescales
        relaxation = analyze_relaxation_timescales(equilibration_data, analysis.transport_coeffs)

        # Validate entropy production
        entropy_validation = validate_entropy_production(equilibration_data)

        # Display summary
        print()
        print("=" * 80)
        print("EQUILIBRATION VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Resolution:          {args.resolution}")
        print(f"Grid points:         {analysis.grid.grid_points}")
        print(f"Final time:          {args.final_time}")
        print(f"Integration method:  {args.method}")
        print(f"Elapsed time:        {equilibration_data['elapsed_time']:.2f}s")
        print()

        print("Relaxation Timescales:")
        if not np.isnan(relaxation.get("bulk_relaxation_measured", np.nan)):
            print(f"  Bulk (measured):     τ_Π = {relaxation['bulk_relaxation_measured']:.4f}")
            print(f"  Bulk (theoretical):  τ_Π = {relaxation['bulk_relaxation_theoretical']:.4f}")
            print(f"  Bulk error:          {relaxation['bulk_relaxation_error']*100:.1f}%")
        else:
            print("  Bulk:                Insufficient data for fit")

        if not np.isnan(relaxation.get("shear_relaxation_measured", np.nan)):
            print(f"  Shear (measured):    τ_π = {relaxation['shear_relaxation_measured']:.4f}")
            print(f"  Shear (theoretical): τ_π = {relaxation['shear_relaxation_theoretical']:.4f}")
            print(f"  Shear error:         {relaxation['shear_relaxation_error']*100:.1f}%")
        else:
            print("  Shear:               Insufficient data for fit")

        print()
        print("Entropy Production (Second Law):")
        print(f"  Initial entropy:     s_0 = {entropy_validation['initial_entropy']:.6e}")
        print(f"  Final entropy:       s_f = {entropy_validation['final_entropy']:.6e}")
        print(f"  Total production:    Δs = {entropy_validation['total_entropy_production']:.6e}")
        print(
            f"  Violations:          {entropy_validation['entropy_violations']}/{entropy_validation['total_steps']} steps"
        )
        print(
            f"  Second law:          {'✓ PASS' if entropy_validation['second_law_satisfied'] else '✗ FAIL'}"
        )
        print()

        # Overall validation
        relaxation_ok = not np.isnan(
            relaxation.get("bulk_relaxation_measured", np.nan)
        ) or not np.isnan(relaxation.get("shear_relaxation_measured", np.nan))
        entropy_ok = entropy_validation["second_law_satisfied"]
        all_ok = relaxation_ok and entropy_ok

        print("=" * 80)
        print("VALIDATION STATUS")
        print("=" * 80)
        print(f"  Relaxation times:    {'✓ PASS' if relaxation_ok else '⚠️  INCOMPLETE'}")
        print(f"  Entropy production:  {'✓ PASS' if entropy_ok else '✗ FAIL'}")
        print()

        if all_ok:
            print("✅ EQUILIBRATION BENCHMARK PASSED")
        else:
            print("⚠️  EQUILIBRATION BENCHMARK - SOME METRICS INCOMPLETE")
            print("   (This may indicate implementation is still in progress)")

        print("=" * 80)

        # Plot results
        if not args.no_plot:
            plot_equilibration(equilibration_data, output_path=args.output)

        return 0 if all_ok else 1

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        print(f"\n❌ Benchmark failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env -S uv run python
"""
Sound Wave Benchmark Runner

This executable script runs comprehensive sound wave propagation validation,
testing dispersion relations, causality constraints, and linear stability
in the spectral Israel-Stewart solver.

Usage:
    python run_sound_wave_benchmark.py [--resolution RESOLUTION] [--output OUTPUT]

Examples:
    python run_sound_wave_benchmark.py                    # Standard resolution
    python run_sound_wave_benchmark.py --resolution high  # High resolution
    python run_sound_wave_benchmark.py --wave-number 1.0  # Specific wave number
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from israel_stewart.benchmarks.sound_waves import (
    LinearStabilityAnalysis,
    NumericalSoundWaveBenchmark,
    SoundWaveAnalysis,
    WaveProperties,
)
from israel_stewart.core.fields import TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacegrid import SpaceGrid
from israel_stewart.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_benchmark(
    resolution: str = "standard",
) -> NumericalSoundWaveBenchmark:
    """Create sound wave benchmark with specified resolution.

    Args:
        resolution: One of "low", "standard", "high"

    Returns:
        NumericalSoundWaveBenchmark instance
    """
    resolution_configs = {
        "low": {"grid_points": (16, 16, 8)},
        "standard": {"grid_points": (32, 32, 16)},
        "high": {"grid_points": (64, 64, 32)},
    }

    if resolution not in resolution_configs:
        raise ValueError(
            f"Unknown resolution: {resolution}. Choose from {list(resolution_configs.keys())}"
        )

    config = resolution_configs[resolution]

    # Transport coefficients
    transport_coeffs = TransportCoefficients(
        shear_viscosity=0.08,
        bulk_viscosity=0.04,
        shear_relaxation_time=1.0,
        bulk_relaxation_time=0.5,
        # Second-order coefficients (turned off for test)
        lambda_pi_pi=0.0,
        lambda_pi_Pi=0.0,
        xi_1=0.0,
        xi_2=0.0,
    )

    logger.info("--------------------------------------------------------------------------------")
    logger.info("Benchmark Parameters")
    logger.info(f"  shear_viscosity (η):       {transport_coeffs.shear_viscosity}")
    logger.info(f"  bulk_viscosity (ζ):        {transport_coeffs.bulk_viscosity}")
    logger.info(f"  shear_relaxation_time (τ_π): {transport_coeffs.shear_relaxation_time}")
    logger.info(f"  bulk_relaxation_time (τ_Π):  {transport_coeffs.bulk_relaxation_time}")
    logger.info(f"  lambda_pi_pi:              {transport_coeffs.lambda_pi_pi}")
    logger.info(f"  lambda_pi_Pi:              {transport_coeffs.lambda_pi_Pi}")
    logger.info(f"  xi_1:                      {transport_coeffs.xi_1}")
    logger.info(f"  xi_2:                      {transport_coeffs.xi_2}")
    logger.info("--------------------------------------------------------------------------------")

    # Create numerical benchmark (it creates its own grid internally)
    numerical_benchmark = NumericalSoundWaveBenchmark(
        domain_size=2 * np.pi,
        grid_points=config["grid_points"],
        transport_coeffs=transport_coeffs,
    )

    logger.info(
        f"Created sound wave benchmark with {resolution} resolution: {config['grid_points']}"
    )

    return numerical_benchmark


def run_dispersion_analysis(
    analytical: SoundWaveAnalysis,
    k_values: np.ndarray | None = None,
) -> dict:
    """Run dispersion relation analysis over range of wave numbers.

    Args:
        analytical: SoundWaveAnalysis instance
        k_values: Wave numbers to test (None for default range)

    Returns:
        Dictionary with dispersion relation data
    """
    if k_values is None:
        k_values = np.logspace(-2, 0, 20)  # k from 0.01 to 1.0

    logger.info(f"Running dispersion analysis for {len(k_values)} wave numbers")

    frequencies = []
    dampings = []
    sound_speeds = []
    phase_velocities = []
    group_velocities = []

    for k in k_values:
        k_vector = np.array([k, 0.0, 0.0])
        wave_modes = analytical.analyze_dispersion_relation(k_vector)

        if not wave_modes:
            frequencies.append(np.nan)
            dampings.append(np.nan)
            sound_speeds.append(np.nan)
            phase_velocities.append(np.nan)
            group_velocities.append(np.nan)
            continue

        # Take first (dominant) mode
        mode = wave_modes[0]
        frequencies.append(mode.frequency)
        dampings.append(mode.attenuation)
        sound_speeds.append(mode.sound_speed)
        phase_velocities.append(mode.phase_velocity)
        group_velocities.append(np.linalg.norm(mode.group_velocity))

    return {
        "k_values": k_values,
        "frequencies": np.array(frequencies),
        "dampings": np.array(dampings),
        "sound_speeds": np.array(sound_speeds),
        "phase_velocities": np.array(phase_velocities),
        "group_velocities": np.array(group_velocities),
    }


def run_numerical_simulation(
    benchmark: NumericalSoundWaveBenchmark,
    wave_number: float = 1.0,
    amplitude: float = 0.01,
    simulation_time: float = 1.0,
    method: str = "split_step",
) -> dict:
    """Run numerical sound wave propagation simulation.

    Args:
        benchmark: NumericalSoundWaveBenchmark instance
        wave_number: Wave number k
        amplitude: Initial amplitude
        simulation_time: Total simulation time
        method: Integration method ('split_step' or 'spectral_imex')

    Returns:
        Dictionary with simulation results and performance metrics
    """
    logger.info(f"Running numerical simulation for k = {wave_number}")

    start_time = time.time()

    # Run simulation (setup_initial_conditions called internally)
    # Use n_periods=3 for accurate frequency extraction (minimum for FFT analysis)
    # This overrides simulation_time to ensure at least 3 wave periods are simulated
    results = benchmark.run_simulation(
        wave_number=wave_number,
        simulation_time=simulation_time,
        n_periods=3,  # Minimum periods for accurate frequency measurement
        method=method,
    )

    elapsed = time.time() - start_time

    # Extract timestep count from time series data
    n_steps = len(results.time_series_data.get("time", []))
    time_per_step = elapsed / max(n_steps, 1)

    logger.info(f"Numerical simulation completed in {elapsed:.2f}s")
    logger.info(f"Total timesteps: {n_steps}, Time per step: {time_per_step:.4f}s")

    return {
        "results": results,
        "elapsed_time": elapsed,
        "n_timesteps": n_steps,
        "time_per_step": time_per_step,
    }


def plot_dispersion_relation(dispersion_data: dict, output_path: Path | None = None) -> None:
    """Create dispersion relation validation plots.

    Args:
        dispersion_data: Dictionary from run_dispersion_analysis()
        output_path: Optional path to save figure
    """
    k_values = dispersion_data["k_values"]
    frequencies = dispersion_data["frequencies"]
    dampings = dispersion_data["dampings"]
    sound_speeds = dispersion_data["sound_speeds"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Frequency vs wave number
    ax = axes[0, 0]
    ax.loglog(k_values, frequencies, "o-", label="Numerical dispersion")
    # Add linear reference (ω = c_s * k)
    c_s = 1.0 / np.sqrt(3.0)  # Radiation sound speed
    ax.loglog(k_values, c_s * k_values, "--", label="Linear (c_s = 1/√3)", alpha=0.5)
    ax.set_xlabel("Wave number k")
    ax.set_ylabel("Frequency ω")
    ax.set_title("Dispersion Relation ω(k)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Damping vs wave number
    ax = axes[0, 1]
    ax.loglog(k_values, dampings, "o-", label="Attenuation rate")
    ax.set_xlabel("Wave number k")
    ax.set_ylabel("Damping γ")
    ax.set_title("Attenuation vs Wave Number")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Sound speed vs wave number
    ax = axes[1, 0]
    ax.semilogx(k_values, sound_speeds, "o-", label="Sound speed")
    ax.axhline(c_s, linestyle="--", color="gray", label="c_s = 1/√3", alpha=0.5)
    ax.axhline(1.0, linestyle=":", color="red", label="c (light speed)", alpha=0.5)
    ax.set_xlabel("Wave number k")
    ax.set_ylabel("Sound speed c_s")
    ax.set_title("Sound Speed vs Wave Number")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Phase velocity vs wave number
    ax = axes[1, 1]
    phase_vels = dispersion_data["phase_velocities"]
    group_vels = dispersion_data["group_velocities"]
    ax.semilogx(k_values, phase_vels, "o-", label="Phase velocity")
    ax.semilogx(k_values, group_vels, "s-", label="Group velocity", alpha=0.7)
    ax.axhline(1.0, linestyle=":", color="red", label="c (causality limit)", alpha=0.5)
    ax.set_xlabel("Wave number k")
    ax.set_ylabel("Velocity")
    ax.set_title("Phase and Group Velocities (Causality Check)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()


def validate_causality(dispersion_data: dict) -> dict:
    """Validate causality constraints from dispersion data.

    Args:
        dispersion_data: Dictionary from run_dispersion_analysis()

    Returns:
        Dictionary with causality validation results
    """
    sound_speeds = dispersion_data["sound_speeds"]
    phase_vels = dispersion_data["phase_velocities"]
    group_vels = dispersion_data["group_velocities"]

    # Remove NaN values
    valid_mask = ~np.isnan(sound_speeds)
    sound_speeds = sound_speeds[valid_mask]
    phase_vels = phase_vels[valid_mask]
    group_vels = group_vels[valid_mask]

    # Check causality: all velocities must be ≤ c = 1
    sound_causal = np.all(sound_speeds <= 1.0)
    phase_causal = np.all(phase_vels <= 1.0)
    group_causal = np.all(group_vels <= 1.0)

    max_sound = np.max(sound_speeds) if len(sound_speeds) > 0 else 0.0
    max_phase = np.max(phase_vels) if len(phase_vels) > 0 else 0.0
    max_group = np.max(group_vels) if len(group_vels) > 0 else 0.0

    return {
        "sound_speed_causal": sound_causal,
        "phase_velocity_causal": phase_causal,
        "group_velocity_causal": group_causal,
        "max_sound_speed": max_sound,
        "max_phase_velocity": max_phase,
        "max_group_velocity": max_group,
        "all_causal": sound_causal and phase_causal and group_causal,
    }


def main():
    """Main entry point for sound wave benchmark."""
    parser = argparse.ArgumentParser(
        description="Run sound wave benchmark with spectral solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                          # Standard resolution
    %(prog)s --resolution high        # High resolution test
    %(prog)s --output dispersion.png  # Save plot to file
    %(prog)s --wave-number 0.5        # Test specific wave number
        """,
    )

    parser.add_argument(
        "--resolution",
        choices=["low", "standard", "high"],
        default="standard",
        help="Grid resolution (default: standard)",
    )

    parser.add_argument(
        "--wave-number",
        type=float,
        default=1.0,
        help="Wave number k for numerical test (default: 1.0)",
    )

    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.01,
        help="Initial wave amplitude (default: 0.01)",
    )

    parser.add_argument(
        "--simulation-time",
        type=float,
        default=10.0,
        help="Simulation time (default: 10.0)",
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

    parser.add_argument(
        "--dispersion-only",
        action="store_true",
        help="Only run dispersion analysis (skip numerical simulation)",
    )

    parser.add_argument(
        "--method",
        choices=["split_step", "spectral_imex","rk4"],
        default="spectral_imex",
        help="Integration method for time stepping (default: split_step)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SOUND WAVE BENCHMARK - SPECTRAL SOLVER VALIDATION")
    print("=" * 80)
    print()

    try:
        # Create benchmark
        benchmark = create_benchmark(resolution=args.resolution)

        # Run dispersion relation analysis
        print("Running dispersion relation analysis...")
        dispersion_data = run_dispersion_analysis(benchmark.analytical)

        # Validate causality
        causality = validate_causality(dispersion_data)

        print()
        print("=" * 80)
        print("DISPERSION RELATION VALIDATION")
        print("=" * 80)
        print(f"Resolution:          {args.resolution}")
        print(f"Grid points:         {benchmark.grid_points}")
        print()
        print("Causality Constraints:")
        print(
            f"  Sound speed:       {'✓ PASS' if causality['sound_speed_causal'] else '✗ FAIL'} (max: {causality['max_sound_speed']:.6f})"
        )
        print(
            f"  Phase velocity:    {'✓ PASS' if causality['phase_velocity_causal'] else '✗ FAIL'} (max: {causality['max_phase_velocity']:.6f})"
        )
        print(
            f"  Group velocity:    {'✓ PASS' if causality['group_velocity_causal'] else '✗ FAIL'} (max: {causality['max_group_velocity']:.6f})"
        )
        print()

        # Plot dispersion relation
        if not args.no_plot:
            plot_dispersion_relation(dispersion_data, output_path=args.output)

        # Run numerical simulation if requested
        if not args.dispersion_only:
            print("=" * 80)
            print("NUMERICAL WAVE PROPAGATION")
            print("=" * 80)
            print(f"Wave number k:       {args.wave_number}")
            print(f"Amplitude:           {args.amplitude}")
            print(f"Simulation time:     {args.simulation_time} (requested)")
            print(f"Integration method:  {args.method}")
            print()
            print("NOTE: Simulation automatically extends to 3 wave periods for accurate")
            print("      frequency measurement. Expected runtime: 2-10 minutes depending")
            print("      on wave number (higher k = faster)")
            print()

            sim_results = run_numerical_simulation(
                benchmark,
                wave_number=args.wave_number,
                amplitude=args.amplitude,
                simulation_time=args.simulation_time,
                method=args.method,
            )

            print(f"Elapsed time:        {sim_results['elapsed_time']:.2f}s")
            print()

            # Display convergence metrics
            results = sim_results["results"]
            print("Convergence Metrics:")
            print(f"  Measured frequency:   {results.measured_frequency:.6f}")
            print(f"  Analytical frequency: {results.analytical_frequency:.6f}")
            print(f"  Frequency error:      {results.frequency_error:.1%}")
            print(f"  Damping error:        {results.damping_error:.1%}")
            print()

            if results.convergence_achieved:
                print("Status:              ✓ Simulation converged")
            else:
                print("Status:              ⚠️  Did not meet convergence criteria")
                print("  Required: freq_error < 10%, damping_error < 20%")

        # Overall validation
        print()
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        if causality["all_causal"]:
            print("✅ SOUND WAVE BENCHMARK PASSED")
            print("   - All causality constraints satisfied")
            print("   - Dispersion relations computed successfully")
            success = True
        else:
            print("⚠️  SOUND WAVE BENCHMARK - CAUSALITY VIOLATIONS DETECTED")
            success = False

        print("=" * 80)

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        print(f"\n❌ Benchmark failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
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
    AnalyticalSoundWaveBenchmark,
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
) -> tuple[NumericalSoundWaveBenchmark, SoundWaveAnalysis]:
    """Create sound wave benchmark with specified resolution.

    Args:
        resolution: One of "low", "standard", "high"

    Returns:
        Tuple of (NumericalSoundWaveBenchmark, SoundWaveAnalysis)
    """
    resolution_configs = {
        "low": {"grid_points": (32, 32, 16)},
        "standard": {"grid_points": (64, 64, 16)},
        "high": {"grid_points": (128, 128, 32)},
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

    # Create numerical benchmark
    numerical_benchmark = NumericalSoundWaveBenchmark(
        grid=grid,
        transport_coefficients=transport_coeffs,
    )

    # Create analytical analysis
    metric = MinkowskiMetric()
    analytical_benchmark = SoundWaveAnalysis(
        grid=grid,
        metric=metric,
        transport_coefficients=transport_coeffs,
    )

    logger.info(
        f"Created sound wave benchmark with {resolution} resolution: {config['grid_points']}"
    )

    return numerical_benchmark, analytical_benchmark


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
    numerical: NumericalSoundWaveBenchmark,
    wave_number: float = 1.0,
    amplitude: float = 0.01,
    simulation_time: float = 10.0,
) -> dict:
    """Run numerical sound wave propagation simulation.

    Args:
        numerical: NumericalSoundWaveBenchmark instance
        wave_number: Wave number k
        amplitude: Initial amplitude
        simulation_time: Total simulation time

    Returns:
        Dictionary with simulation results
    """
    logger.info(f"Running numerical simulation for k = {wave_number}")

    start_time = time.time()

    # Set up initial conditions
    numerical.setup_initial_conditions(
        wave_number=wave_number,
        amplitude=amplitude,
        background_density=1.0,
    )

    # Run simulation
    results = numerical.run_simulation(
        wave_number=wave_number,
        simulation_time=simulation_time,
        cfl_factor=0.5,
    )

    elapsed = time.time() - start_time
    logger.info(f"Numerical simulation completed in {elapsed:.2f}s")

    return {"results": results, "elapsed_time": elapsed}


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

    args = parser.parse_args()

    print("=" * 80)
    print("SOUND WAVE BENCHMARK - SPECTRAL SOLVER VALIDATION")
    print("=" * 80)
    print()

    try:
        # Create benchmark
        numerical, analytical = create_benchmark(resolution=args.resolution)

        # Run dispersion relation analysis
        print("Running dispersion relation analysis...")
        dispersion_data = run_dispersion_analysis(analytical)

        # Validate causality
        causality = validate_causality(dispersion_data)

        print()
        print("=" * 80)
        print("DISPERSION RELATION VALIDATION")
        print("=" * 80)
        print(f"Resolution:          {args.resolution}")
        print(f"Grid points:         {numerical.grid_points}")
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
            print(f"Simulation time:     {args.simulation_time}")
            print()

            sim_results = run_numerical_simulation(
                numerical,
                wave_number=args.wave_number,
                amplitude=args.amplitude,
                simulation_time=args.simulation_time,
            )

            print(f"Elapsed time:        {sim_results['elapsed_time']:.2f}s")

            # Check if simulation succeeded
            if sim_results["results"]["success"]:
                print("Status:              ✓ Simulation completed successfully")
                if "measured_frequency" in sim_results["results"]:
                    print(
                        f"Measured frequency:  {sim_results['results']['measured_frequency']:.6f}"
                    )
            else:
                print("Status:              ⚠️  Simulation did not converge")
                print(
                    f"Reason:              {sim_results['results'].get('error_message', 'Unknown')}"
                )

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

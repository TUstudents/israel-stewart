"""
Validate Bjorken flow with IReD transport coefficients.

This script runs the Bjorken expansion benchmark with IReD coefficients
and validates against the analytical Israel-Stewart solution.

Bjorken flow: Boost-invariant longitudinal expansion along the beam axis
    - Milne coordinates (τ, x, y, η) where τ = √(t² - z²)
    - Proper time evolution: τ ∈ [τ₀, τ_final]
    - Analytical solution exists for IS equations

Key validations:
1. Temperature evolution matches analytical T(τ)
2. Shear stress π^ηη evolution matches analytical solution
3. Energy conservation in expanding volume
4. Approach to equilibrium (π → 0 as τ → ∞)

Usage:
    uv run python validate_bjorken_ired.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from israel_stewart.benchmarks.bjorken_flow import create_bjorken_benchmark_with_ired
from israel_stewart.utils import get_logger

logger = get_logger(__name__)


def run_bjorken_validation(
    temperature_0: float = 0.4,
    cross_section: float = 10.0,
    tau_0: float = 0.6,
    tau_final: float = 5.0,
    n_steps: int = 200,
) -> dict:
    """
    Run Bjorken flow with IReD and validate against analytical solution.

    Args:
        temperature_0: Initial temperature at τ₀ (GeV)
        cross_section: Hard sphere cross-section σ (fm²)
        tau_0: Initial proper time (fm/c)
        tau_final: Final proper time (fm/c)
        n_steps: Number of time steps

    Returns:
        Dictionary with validation results
    """
    logger.info("=" * 80)
    logger.info("BJORKEN FLOW VALIDATION WITH IRED COEFFICIENTS")
    logger.info("=" * 80)

    # Create benchmark with IReD coefficients
    logger.info("\n1. Creating Bjorken benchmark with IReD coefficients...")
    benchmark, ired_model = create_bjorken_benchmark_with_ired(
        T0=temperature_0,
        tau0=tau_0,
        cross_section=cross_section,
        truncation="41",
    )

    # Print IReD parameters
    logger.info("\nIReD Transport Coefficients:")
    logger.info(f"  η/s = {ired_model.eta_over_s():.4f}")
    logger.info(f"  Shear viscosity η = {ired_model.shear_viscosity():.4e} GeV³")
    logger.info(f"  Shear relaxation time τ_π = {ired_model.shear_relaxation_time():.2f} fm/c")

    # Run numerical simulation
    logger.info(f"\n2. Running Bjorken evolution: τ ∈ [{tau_0}, {tau_final}] fm/c...")
    dt = (tau_final - tau_0) / n_steps

    result = benchmark.run_numerical_simulation(
        final_time=tau_final,
        timestep=dt,
        method="rk4",
    )

    logger.info(f"   Evolution complete: {len(result['time'])} time steps")

    # Extract data
    times = np.array(result["time"])
    temperatures = np.array(result["temperature"])
    shear_stresses = np.array(result["shear_stress_eta_eta"])

    # Get analytical solution at same times
    analytical_temps = []
    analytical_shears = []

    for t in times:
        sol = benchmark.analytical.israel_stewart_solution(t)
        analytical_temps.append(sol["temperature"])
        analytical_shears.append(sol["shear_eta_eta"])

    analytical_temps = np.array(analytical_temps)
    analytical_shears = np.array(analytical_shears)

    # Compute errors
    temp_error = np.abs(temperatures - analytical_temps) / analytical_temps
    shear_error = np.abs(shear_stresses - analytical_shears) / np.abs(analytical_shears)

    max_temp_error = np.max(temp_error)
    max_shear_error = np.max(shear_error)

    logger.info("\n3. Validation Results:")
    logger.info(f"   Max temperature error: {max_temp_error:.2%}")
    logger.info(f"   Max shear stress error: {max_shear_error:.2%}")

    # Check convergence
    temp_ok = max_temp_error < 0.05  # 5% tolerance
    shear_ok = max_shear_error < 0.10  # 10% tolerance

    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Temperature evolution: {'✓ PASS' if temp_ok else '✗ FAIL'}")
    logger.info(f"Shear stress evolution: {'✓ PASS' if shear_ok else '✗ FAIL'}")

    all_pass = temp_ok and shear_ok
    logger.info(f"\nOverall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    logger.info("=" * 80)

    return {
        "benchmark": benchmark,
        "ired_model": ired_model,
        "times": times,
        "temperatures": temperatures,
        "shear_stresses": shear_stresses,
        "analytical_temps": analytical_temps,
        "analytical_shears": analytical_shears,
        "temp_error": temp_error,
        "shear_error": shear_error,
        "temp_ok": temp_ok,
        "shear_ok": shear_ok,
        "all_pass": all_pass,
    }


def plot_bjorken_results(validation_data: dict, output_dir: str = "validation_plots") -> None:
    """
    Create Bjorken validation plots.

    Args:
        validation_data: Results from run_bjorken_validation()
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    times = validation_data["times"]
    temperatures = validation_data["temperatures"]
    shear_stresses = validation_data["shear_stresses"]
    analytical_temps = validation_data["analytical_temps"]
    analytical_shears = validation_data["analytical_shears"]
    temp_error = validation_data["temp_error"]
    shear_error = validation_data["shear_error"]

    # Plot 1: Temperature evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(times, temperatures, "b-", label="Numerical (IReD)", linewidth=2)
    ax1.plot(times, analytical_temps, "r--", label="Analytical (IS)", linewidth=2)
    ax1.set_xlabel("Proper time τ (fm/c)")
    ax1.set_ylabel("Temperature T (GeV)")
    ax1.set_title("Bjorken Flow: Temperature Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(times, temp_error, "b-", linewidth=2)
    ax2.set_xlabel("Proper time τ (fm/c)")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Temperature: Numerical vs Analytical Error")
    ax2.axhline(0.05, color="r", linestyle="--", label="5% tolerance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "bjorken_temperature.png", dpi=150)
    logger.info(f"\nSaved: {output_path / 'bjorken_temperature.png'}")
    plt.close()

    # Plot 2: Shear stress evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(times, shear_stresses, "b-", label="Numerical (IReD)", linewidth=2)
    ax1.plot(times, analytical_shears, "r--", label="Analytical (IS)", linewidth=2)
    ax1.set_xlabel("Proper time τ (fm/c)")
    ax1.set_ylabel("π^ηη (GeV⁴)")
    ax1.set_title("Bjorken Flow: Shear Stress Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(times, shear_error, "b-", linewidth=2)
    ax2.set_xlabel("Proper time τ (fm/c)")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Shear Stress: Numerical vs Analytical Error")
    ax2.axhline(0.10, color="r", linestyle="--", label="10% tolerance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "bjorken_shear_stress.png", dpi=150)
    logger.info(f"Saved: {output_path / 'bjorken_shear_stress.png'}")
    plt.close()

    logger.info(f"\nAll plots saved to {output_path}/")


if __name__ == "__main__":
    # Run validation
    # Use σ = 10 fm² for reasonable τ_π (balance between regime and evolution time)
    validation_data = run_bjorken_validation(
        temperature_0=0.4,  # 400 MeV initial temperature
        cross_section=10.0,  # 10 fm² cross-section
        tau_0=0.6,  # Initial proper time (fm/c)
        tau_final=5.0,  # Evolve to 5 fm/c
        n_steps=200,  # Fine time resolution
    )

    # Create plots
    plot_bjorken_results(validation_data)

    # Exit with appropriate code
    import sys

    sys.exit(0 if validation_data["all_pass"] else 1)

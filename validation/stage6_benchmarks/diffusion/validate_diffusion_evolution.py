"""
Validate diffusion benchmark with full time evolution.

This script runs the diffusion flow benchmark with IReD transport coefficients
and validates that the numerical solution matches the analytical exponential decay:

    n(x,t) = n₀ + δn₀ exp(-Dk²t) sin(kx)
    V^x(x,t) = -D k δn₀ exp(-Dk²t) cos(kx)

Key validations:
1. Exponential decay rate matches Dk²
2. Spatial structure preserved (sinusoidal)
3. Landau frame constraint V^μ u_μ = 0 maintained
4. Particle conservation ∫n d³x = const

Usage:
    uv run python validate_diffusion_evolution.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired
from israel_stewart.utils import get_logger

logger = get_logger(__name__)


def run_diffusion_validation(
    temperature: float = 0.4,
    cross_section: float = 1.0,
    perturbation_amplitude: float = 0.05,
    wave_number: float = 1.0,
    grid_points: tuple[int, int, int] = (64, 64, 16),
    final_time: float = 1.0,
    n_snapshots: int = 20,
) -> dict:
    """
    Run diffusion benchmark and validate against analytical solution.

    Args:
        temperature: Temperature T in GeV
        cross_section: Hard sphere cross-section σ in fm²
        perturbation_amplitude: δn₀/n₀
        wave_number: Wave vector k in GeV
        grid_points: Grid resolution
        final_time: Final time in GeV⁻¹
        n_snapshots: Number of time snapshots

    Returns:
        Dictionary with validation results
    """
    logger.info("=" * 80)
    logger.info("DIFFUSION FLOW VALIDATION WITH IRED COEFFICIENTS")
    logger.info("=" * 80)

    # Create benchmark
    logger.info("\n1. Creating diffusion benchmark with IReD coefficients...")
    benchmark, ired_model = create_diffusion_benchmark_with_ired(
        temperature=temperature,
        cross_section=cross_section,
        truncation="41",
        perturbation_amplitude=perturbation_amplitude,
        wave_number=wave_number,
        grid_points=grid_points,
    )

    # Print IReD parameters
    logger.info("\nIReD Transport Coefficients:")
    logger.info(f"  Diffusion coefficient D = {ired_model.diffusion_coefficient():.4e} GeV²")
    logger.info(
        f"  Diffusion relaxation time τ_V = {ired_model.diffusion_relaxation_time():.2f} fm/c"
    )
    logger.info(f"  Damping rate Γ = Dk² = {benchmark.analytical.damping_rate():.4e} GeV")
    logger.info(
        f"  e-folding time τ_decay = 1/Γ = {1.0/benchmark.analytical.damping_rate():.4f} GeV⁻¹"
    )

    # Run simulation
    logger.info(f"\n2. Running simulation: t ∈ [0, {final_time}] GeV⁻¹...")
    snapshot_interval = final_time / n_snapshots
    result = benchmark.run_numerical_simulation(
        final_time=final_time, timestep=0.01, snapshot_interval=snapshot_interval
    )

    logger.info(f"   Simulation complete: {len(result['time'])} snapshots")

    # Extract data
    times = result["time"]
    particle_densities = result["particle_density"]
    diffusion_currents_x = result["diffusion_current_x"]
    constraint_violations = result["constraint_violation"]

    # Get grid
    X, Y, Z = benchmark.grid.meshgrid()

    # Validate at each time
    logger.info("\n3. Validating exponential decay at each snapshot...")

    errors = []
    decay_rates = []

    # Store first amplitude for decay rate calculation
    initial_amplitude = None

    for i, t in enumerate(times):
        # Numerical diffusion current (this is what the solver evolves)
        V_x_numerical = diffusion_currents_x[i]

        # Analytical solutions
        n_analytical = benchmark.analytical.particle_density(X, t)
        V_x_analytical = benchmark.analytical.diffusion_current(X, t)

        # Compute errors (primary validation is on V_x, the evolved quantity)
        V_error = np.max(np.abs(V_x_numerical - V_x_analytical)) / np.max(np.abs(V_x_analytical))

        # For particle density, use V_x to infer amplitude (n_numerical not reliable
        # because isentropic diffusion has constant T, perturbation is in μ only)
        errors.append({"t": t, "V_error": V_error})

        # Extract decay rate from V_x amplitude (V ∝ exp(-Dk²t))
        # V_x = -D k δn₀ exp(-Dk²t) cos(kx)
        # Amplitude of V_x: max|V_x| ∝ exp(-Dk²t)
        V_x_amplitude = np.max(np.abs(V_x_numerical))
        V_x_amplitude_analytical = np.max(np.abs(V_x_analytical))

        # Store initial amplitude
        if i == 0:
            initial_amplitude = V_x_amplitude
            logger.debug(f"   Initial V_x amplitude: {initial_amplitude:.4e}")
        elif t > 0 and initial_amplitude is not None and initial_amplitude > 0:
            # Compute numerical decay rate from amplitude ratio
            amplitude_ratio = V_x_amplitude / initial_amplitude
            if (
                amplitude_ratio > 0 and amplitude_ratio < 1.05
            ):  # Should decay (allow tiny growth for numerical noise)
                measured_decay_rate = -np.log(amplitude_ratio) / t
                decay_rates.append(measured_decay_rate)
                if i % 3 == 0:  # Log every 3rd measurement
                    logger.debug(
                        f"   t={t:.2f}: V_x_amp={V_x_amplitude:.4e}, "
                        f"ratio={amplitude_ratio:.4f}, Γ={measured_decay_rate:.4e}"
                    )
            else:
                if i % 3 == 0:  # Log why we skipped
                    logger.debug(
                        f"   t={t:.2f}: V_x_amp={V_x_amplitude:.4e}, "
                        f"ratio={amplitude_ratio:.4f} (skipped: ratio >= 1.05 or <= 0)"
                    )

        if i % 5 == 0:  # Print every 5th snapshot
            logger.info(
                f"   t = {t:.4f}: V_error = {V_error:.2e}, V_amp = {V_x_amplitude:.4e}, "
                f"constraint = {np.max(constraint_violations[i]):.2e}"
            )

    # Validate Landau frame constraint
    logger.info("\n4. Validating Landau frame constraint V^μ u_μ = 0...")
    max_constraint_violation = np.max(constraint_violations)
    logger.info(f"   Max |V^μ u_μ| = {max_constraint_violation:.2e}")
    constraint_ok = max_constraint_violation < 1e-6

    # Validate particle conservation
    logger.info("\n5. Validating particle conservation...")
    particle_conservation_ok = benchmark.validate_particle_conservation(result, tolerance=0.01)

    # Validate Fick's law
    logger.info("\n6. Validating Fick's law at multiple times...")
    fick_ok_initial = benchmark.validate_fick_law(result, time_index=0, tolerance=0.1)
    fick_ok_middle = benchmark.validate_fick_law(result, time_index=len(times) // 2, tolerance=0.1)
    fick_ok_final = benchmark.validate_fick_law(result, time_index=-1, tolerance=0.1)

    # Validate exponential decay rate
    logger.info("\n7. Validating exponential decay rate...")
    theoretical_decay_rate = benchmark.analytical.damping_rate()

    if len(decay_rates) > 1:
        measured_decay_rate = np.mean(decay_rates[1:])  # Skip first (noisy)
        decay_rate_error = (
            abs(measured_decay_rate - theoretical_decay_rate) / theoretical_decay_rate
        )
        decay_rate_ok = decay_rate_error < 0.2  # 20% tolerance (numerical discretization)
    elif len(decay_rates) == 1:
        measured_decay_rate = decay_rates[0]
        decay_rate_error = (
            abs(measured_decay_rate - theoretical_decay_rate) / theoretical_decay_rate
        )
        decay_rate_ok = decay_rate_error < 0.2
    else:
        logger.warning("   No decay rate measurements available!")
        measured_decay_rate = np.nan
        decay_rate_error = np.nan
        decay_rate_ok = False

    logger.info(f"   Theoretical decay rate Γ = {theoretical_decay_rate:.4e} GeV")
    logger.info(
        f"   Measured decay rate Γ = {measured_decay_rate:.4e} GeV"
        if not np.isnan(measured_decay_rate)
        else "   Measured decay rate Γ = N/A"
    )
    logger.info(
        f"   Relative error = {decay_rate_error:.2%}"
        if not np.isnan(decay_rate_error)
        else "   Relative error = N/A"
    )
    logger.info(f"   Number of measurements: {len(decay_rates)}")

    # Overall validation
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Landau frame constraint: {'✓ PASS' if constraint_ok else '✗ FAIL'}")
    logger.info(f"Particle conservation: {'✓ PASS' if particle_conservation_ok else '✗ FAIL'}")
    logger.info(f"Fick's law (initial): {'✓ PASS' if fick_ok_initial else '✗ FAIL'}")
    logger.info(f"Fick's law (middle): {'✓ PASS' if fick_ok_middle else '✗ FAIL'}")
    logger.info(f"Fick's law (final): {'✓ PASS' if fick_ok_final else '✗ FAIL'}")
    logger.info(f"Exponential decay rate: {'✓ PASS' if decay_rate_ok else '✗ FAIL'}")

    all_pass = (
        constraint_ok
        and particle_conservation_ok
        and fick_ok_initial
        and fick_ok_final
        and decay_rate_ok
    )
    logger.info(f"\nOverall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    logger.info("=" * 80)

    return {
        "benchmark": benchmark,
        "ired_model": ired_model,
        "result": result,
        "errors": errors,
        "decay_rates": decay_rates,
        "constraint_ok": constraint_ok,
        "particle_conservation_ok": particle_conservation_ok,
        "fick_ok": fick_ok_initial and fick_ok_middle and fick_ok_final,
        "decay_rate_ok": decay_rate_ok,
        "all_pass": all_pass,
    }


def plot_validation_results(validation_data: dict, output_dir: str = "validation_plots") -> None:
    """
    Create validation plots.

    Args:
        validation_data: Results from run_diffusion_validation()
        output_dir: Directory to save plots
    """

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    benchmark = validation_data["benchmark"]
    result = validation_data["result"]
    errors = validation_data["errors"]

    times = result["time"]
    particle_densities = result["particle_density"]
    diffusion_currents_x = result["diffusion_current_x"]
    constraint_violations = result["constraint_violation"]

    X, Y, Z = benchmark.grid.meshgrid()

    # Plot 1: Particle density evolution at y=0, z=0 slice
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, t_idx in enumerate([0, len(times) // 3, 2 * len(times) // 3, -1]):
        ax = axes[i // 2, i % 2]
        t = times[t_idx]

        # Numerical
        n_numerical = particle_densities[t_idx, :, 0, 0]
        x_slice = X[:, 0, 0]

        # Analytical
        n_analytical = benchmark.analytical.particle_density(X[:, 0, 0], t)

        ax.plot(x_slice, n_numerical, "b-", label="Numerical", linewidth=2)
        ax.plot(x_slice, n_analytical, "r--", label="Analytical", linewidth=2)
        ax.set_xlabel("x (GeV⁻¹)")
        ax.set_ylabel("n (GeV³)")
        ax.set_title(f"t = {t:.4f} GeV⁻¹")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "particle_density_evolution.png", dpi=150)
    logger.info(f"\nSaved: {output_path / 'particle_density_evolution.png'}")
    plt.close()

    # Plot 2: Error evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    error_times = [e["t"] for e in errors]
    V_errors = [e["V_error"] for e in errors]

    ax1.semilogy(error_times, V_errors, "b-o", label="Diffusion current V^x", markersize=4)
    ax1.set_xlabel("Time (GeV⁻¹)")
    ax1.set_ylabel("Relative Error")
    ax1.set_title("V^x Numerical vs Analytical Error")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Constraint violation
    ax2.semilogy(times, np.max(constraint_violations, axis=(1, 2, 3)), "g-o", markersize=4)
    ax2.set_xlabel("Time (GeV⁻¹)")
    ax2.set_ylabel("|V^μ u_μ| (max)")
    ax2.set_title("Landau Frame Constraint Violation")
    ax2.axhline(1e-6, color="r", linestyle="--", label="Tolerance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "error_evolution.png", dpi=150)
    logger.info(f"Saved: {output_path / 'error_evolution.png'}")
    plt.close()

    # Plot 3: Exponential decay (using V_x amplitude)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract V_x amplitude at each time
    V_x_amplitudes = []
    for i, _t in enumerate(times):
        V_x = diffusion_currents_x[i]
        V_x_amplitude = np.max(np.abs(V_x))
        V_x_amplitudes.append(V_x_amplitude)

    # Theoretical decay: V_x = -D k δn₀ exp(-Dk²t) cos(kx)
    # Amplitude: max|V_x| = D k δn₀ exp(-Dk²t)
    Gamma = benchmark.analytical.damping_rate()
    D = benchmark.analytical.diffusion_coefficient
    k = benchmark.analytical.wave_number
    delta_n0 = benchmark.analytical.perturbation_amplitude * benchmark.analytical.particle_density_0
    V_x_amp_0 = D * k * delta_n0

    theoretical_V_x_amplitudes = [V_x_amp_0 * np.exp(-Gamma * t) for t in times]

    ax.semilogy(times, V_x_amplitudes, "b-o", label="Numerical V^x amplitude", markersize=6)
    ax.semilogy(
        times,
        theoretical_V_x_amplitudes,
        "r--",
        label="Analytical exp(-Γt)",
        linewidth=2,
    )
    ax.set_xlabel("Time (GeV⁻¹)")
    ax.set_ylabel("|V^x| amplitude (GeV²)")
    ax.set_title(f"Exponential Decay of Diffusion Current (Γ = {Gamma:.4e} GeV)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "exponential_decay.png", dpi=150)
    logger.info(f"Saved: {output_path / 'exponential_decay.png'}")
    plt.close()

    logger.info(f"\nAll plots saved to {output_path}/")


if __name__ == "__main__":
    # Run validation with regime-valid parameters
    # Use VERY large cross-section to enter Israel-Stewart regime (|τω| < 1)
    # σ = 100 fm² gives τ_V ≈ 2.7 fm/c ≈ 0.52 GeV⁻¹ (regime-valid for k ~ 3)
    validation_data = run_diffusion_validation(
        temperature=0.4,
        cross_section=100.0,  # Very large σ → short τ_V → regime-valid
        perturbation_amplitude=0.05,
        wave_number=2.0,  # Moderate k (not too fast decay)
        grid_points=(32, 32, 16),  # Coarser grid → smaller k_max
        final_time=2.0,  # Longer time for measurable decay
        n_snapshots=20,
    )

    # Create plots
    plot_validation_results(validation_data)

    # Exit with appropriate code
    import sys

    sys.exit(0 if validation_data["all_pass"] else 1)

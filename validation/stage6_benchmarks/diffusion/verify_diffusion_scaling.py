#!/usr/bin/env python3
"""Verify diffusion decay physics Γ = Dk² across parameter regimes.

This script tests diffusion at three regimes:
1. Fast regime (σ=10 fm²): D ≈ 0.016 GeV², Γ ≈ 4e-3, decay time ~ 250 GeV⁻¹
2. Medium regime (σ=100 fm²): D ≈ 0.0016 GeV², Γ ≈ 4e-4, decay time ~ 2500 GeV⁻¹
3. IReD regime (σ=1000 fm²): D ≈ 0.00016 GeV², Γ ≈ 4e-5, decay time ~ 25000 GeV⁻¹

For each regime, we:
- Initialize diffusion benchmark with appropriate σ
- Verify Fick's law at t=0: V^x = -D ∂_x(μ/T)
- Evolve for ~3 decay times (or max 100 GeV⁻¹)
- Extract decay rate from exponential fit
- Compare to analytical Γ = Dk²
"""

import sys

import numpy as np

from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired
from israel_stewart.equations.ired_simple import HardSphereIReD


def fit_exponential_decay(times, amplitudes):
    """Fit exponential decay A(t) = A₀ exp(-Γt) and return Γ.

    Uses log-linear regression: ln(A) = ln(A₀) - Γt
    """
    # Filter out any zero or negative amplitudes
    valid = amplitudes > 0
    if np.sum(valid) < 3:
        return np.nan

    t = times[valid]
    A = amplitudes[valid]

    # Log-linear fit: ln(A) = ln(A₀) - Γt
    log_A = np.log(A)
    coeffs = np.polyfit(t, log_A, 1)  # slope, intercept
    Gamma = -coeffs[0]  # Negative of slope

    return Gamma


def verify_ficks_law(fields, solver, D, k):
    """Verify Fick's law: V^x = -D ∂_x(μ/T) at t=0.

    Returns:
        correlation: Correlation coefficient between V^x and -D ∂_x(μ/T)
        relative_error: RMS relative error |V^x - (-D ∂_x(μ/T))| / |V^x|
    """
    # Get V^x
    V_x = fields.V_mu[..., 1]

    # Compute μ/T
    mu_over_T = fields.compute_chemical_potential_over_temperature()

    # Compute gradient ∂_x(μ/T)
    grad_mu_x, _, _ = solver.spectral.spatial_gradient(mu_over_T)

    # Fick's law: V^x = -D ∂_x(μ/T)
    V_x_expected = -D * grad_mu_x

    # Correlation
    corr = np.corrcoef(V_x.flat, V_x_expected.flat)[0, 1]

    # Relative RMS error
    error = np.sqrt(np.mean((V_x - V_x_expected) ** 2)) / np.sqrt(np.mean(V_x**2))

    return corr, error


def verify_regime(
    regime_name: str,
    cross_section: float,
    temperature: float = 0.4,
    wave_number: float = 0.5,
    perturbation_amplitude: float = 0.05,
    grid_points: tuple = (16, 16, 16),
    domain_size: float = 4 * np.pi,
    max_evolution_time: float = 100.0,
    n_steps: int = 50,
):
    """Verify diffusion physics for one parameter regime.

    Args:
        regime_name: Name of regime (e.g., "Fast", "Medium", "IReD")
        cross_section: Cross-section σ in fm²
        temperature: Temperature in GeV
        wave_number: Wave number k in GeV
        perturbation_amplitude: Dimensionless amplitude (0.05 = 5%)
        grid_points: Grid resolution
        domain_size: Spatial domain size
        max_evolution_time: Maximum evolution time in GeV⁻¹
        n_steps: Number of timesteps

    Returns:
        dict with results: D, Gamma_expected, Gamma_measured, error, etc.
    """
    print("\n" + "=" * 80)
    print(f"{regime_name.upper()} REGIME (σ = {cross_section:.0f} fm²)")
    print("=" * 80)

    # Create IReD model
    ired_model = HardSphereIReD(
        temperature=temperature, cross_section=cross_section, truncation="41"
    )

    # Get transport coefficients
    D = ired_model.diffusion_coefficient()
    tau_V = ired_model.diffusion_relaxation_time()
    Gamma_expected = D * wave_number**2

    print("\nTransport coefficients:")
    print(f"  D = {D:.6e} GeV²")
    print(f"  τ_V = {tau_V:.6e} GeV⁻¹")
    print(f"  k = {wave_number:.3f} GeV")
    print(f"  Γ_expected = Dk² = {Gamma_expected:.6e} GeV")
    print(f"  Decay time = 1/Γ = {1/Gamma_expected:.2e} GeV⁻¹")

    # Create benchmark
    benchmark, _ = create_diffusion_benchmark_with_ired(
        temperature=temperature,
        cross_section=cross_section,
        truncation="41",
        perturbation_amplitude=perturbation_amplitude,
        wave_number=wave_number,
        grid_points=grid_points,
        domain_size=domain_size,
    )

    fields = benchmark.fields
    solver = benchmark.solver

    # Verify Fick's law at t=0
    print("\nFick's law check at t=0:")
    corr, rel_error = verify_ficks_law(fields, solver, D, wave_number)
    print(f"  Correlation(V^x, -D∂_x(μ/T)): {corr:.6f}")
    print(f"  Relative RMS error: {rel_error:.2%}")

    if corr < 0.99:
        print(f"  ⚠ WARNING: Poor correlation ({corr:.4f} < 0.99)")
    if rel_error > 0.10:
        print(f"  ⚠ WARNING: Large error ({rel_error:.1%} > 10%)")

    # Check initial state
    print("\nInitial state:")
    print(f"  n: min={fields.n.min():.6e}, max={fields.n.max():.6e}, mean={fields.n.mean():.6e}")
    print(
        f"  V^x: min={fields.V_mu[..., 1].min():.6e}, max={fields.V_mu[..., 1].max():.6e}, mean={fields.V_mu[..., 1].mean():.6e}"
    )
    print(
        f"  T: min={fields.temperature.min():.4f}, max={fields.temperature.max():.4f}, mean={fields.temperature.mean():.4f}"
    )

    # Evolve and track amplitude
    times = []
    amplitudes = []

    def extract_amplitude(t, fields):
        V_x = fields.V_mu[..., 1]
        amplitude = np.sqrt(np.mean(V_x**2))
        times.append(t)
        amplitudes.append(amplitude)

    # Evolution time: 3 decay times or max_evolution_time
    t_final = min(3.0 / Gamma_expected, max_evolution_time)
    dt = t_final / n_steps

    print("\nTime evolution:")
    print(f"  t_final = {t_final:.2e} GeV⁻¹")
    print(f"  dt = {dt:.2e} GeV⁻¹")
    print(f"  n_steps = {n_steps}")
    print(
        f"  Expected decay: exp(-Γt) = exp(-{Gamma_expected * t_final:.4f}) = {np.exp(-Gamma_expected * t_final):.4f}"
    )

    benchmark.solver.evolve(t_final=t_final, dt=dt, method="rk4", callback=extract_amplitude)

    # Fit exponential decay
    times = np.array(times)
    amplitudes = np.array(amplitudes)

    print("\nAmplitude evolution:")
    print(f"  Initial: {amplitudes[0]:.6e}")
    print(f"  Final: {amplitudes[-1]:.6e}")
    print(f"  Ratio: {amplitudes[-1] / amplitudes[0]:.6f}")
    print(f"  Expected ratio: {np.exp(-Gamma_expected * t_final):.6f}")

    # Fit
    if len(amplitudes) > 5 and np.all(amplitudes > 0):
        Gamma_measured = fit_exponential_decay(times, amplitudes)
    else:
        Gamma_measured = np.nan
        print("  ⚠ WARNING: Insufficient data for fit")

    # Calculate error
    if not np.isnan(Gamma_measured):
        error = abs(Gamma_measured - Gamma_expected) / Gamma_expected
        print("\nDecay rate measurement:")
        print(f"  Γ_measured = {Gamma_measured:.6e} GeV")
        print(f"  Γ_expected = {Gamma_expected:.6e} GeV")
        print(f"  Relative error: {error:.1%}")

        if error < 0.10:
            print("  ✓ PASS: Error < 10%")
            success = True
        else:
            print("  ✗ FAIL: Error ≥ 10%")
            success = False

        if Gamma_measured < 0:
            print("  ⚠ WARNING: NEGATIVE DECAY RATE → EXPONENTIAL GROWTH!")
            success = False
    else:
        error = np.nan
        success = False

    # Check final state
    print("\nFinal state:")
    print(f"  n: min={fields.n.min():.6e}, max={fields.n.max():.6e}, mean={fields.n.mean():.6e}")
    print(
        f"  V^x: min={fields.V_mu[..., 1].min():.6e}, max={fields.V_mu[..., 1].max():.6e}, mean={fields.V_mu[..., 1].mean():.6e}"
    )

    return {
        "regime": regime_name,
        "cross_section": cross_section,
        "D": D,
        "tau_V": tau_V,
        "Gamma_expected": Gamma_expected,
        "Gamma_measured": Gamma_measured,
        "error": error,
        "success": success,
        "ficks_law_corr": corr,
        "ficks_law_error": rel_error,
        "t_final": t_final,
        "amplitude_ratio": amplitudes[-1] / amplitudes[0] if len(amplitudes) > 0 else np.nan,
    }


def main():
    """Run verification across all three regimes."""
    print("\n" + "=" * 80)
    print("DIFFUSION SCALING VERIFICATION: Γ = Dk²")
    print("=" * 80)
    print("\nThis script tests diffusion physics at three parameter regimes:")
    print("  1. Fast (σ=10 fm²): D ~ 0.016 GeV², Γ ~ 4e-3, observable in ~250 GeV⁻¹")
    print("  2. Medium (σ=100 fm²): D ~ 0.0016 GeV², Γ ~ 4e-4, observable in ~2500 GeV⁻¹")
    print("  3. IReD (σ=1000 fm²): D ~ 0.00016 GeV², Γ ~ 4e-5, needs ~25000 GeV⁻¹")

    # Test parameters
    temperature = 0.4  # GeV
    wave_number = 0.5  # GeV
    perturbation_amplitude = 0.05  # 5%
    grid_points = (16, 16, 16)  # Small grid for speed
    domain_size = 4 * np.pi  # Large enough for k=0.5

    results = []

    # Regime 1: Fast (100× faster than IReD)
    result = verify_regime(
        regime_name="Fast",
        cross_section=10.0,  # σ=10 fm² → D ≈ 0.016 GeV²
        temperature=temperature,
        wave_number=wave_number,
        perturbation_amplitude=perturbation_amplitude,
        grid_points=grid_points,
        domain_size=domain_size,
        max_evolution_time=10.0,  # Should see decay in ~3 GeV⁻¹
        n_steps=50,
    )
    results.append(result)

    # Regime 2: Medium (10× faster than IReD)
    result = verify_regime(
        regime_name="Medium",
        cross_section=100.0,  # σ=100 fm² → D ≈ 0.0016 GeV²
        temperature=temperature,
        wave_number=wave_number,
        perturbation_amplitude=perturbation_amplitude,
        grid_points=grid_points,
        domain_size=domain_size,
        max_evolution_time=100.0,  # Should see decay in ~30 GeV⁻¹
        n_steps=50,
    )
    results.append(result)

    # Regime 3: IReD (actual regime)
    result = verify_regime(
        regime_name="IReD",
        cross_section=1000.0,  # σ=1000 fm² → D ≈ 0.00016 GeV²
        temperature=temperature,
        wave_number=wave_number,
        perturbation_amplitude=perturbation_amplitude,
        grid_points=grid_points,
        domain_size=domain_size,
        max_evolution_time=100.0,  # Limited time
        n_steps=50,
    )
    results.append(result)

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"\n{'Regime':<10} {'σ (fm²)':<12} {'D (GeV²)':<12} {'Γ_exp':<12} {'Γ_meas':<12} {'Error':<12} {'Status':<10}"
    )
    print("-" * 90)

    for r in results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(
            f"{r['regime']:<10} {r['cross_section']:<12.1f} {r['D']:<12.6e} "
            f"{r['Gamma_expected']:<12.6e} {r['Gamma_measured']:<12.6e} "
            f"{r['error']:<12.1%} {status:<10}"
        )

    # Check scaling law
    print("\n" + "=" * 80)
    print("SCALING LAW CHECK: Γ should scale linearly with D")
    print("=" * 80)

    if all(not np.isnan(r["Gamma_measured"]) for r in results):
        # Check if Γ_measured / Γ_expected is consistent
        ratios = [r["Gamma_measured"] / r["Gamma_expected"] for r in results]
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        print("\nΓ_measured / Γ_expected:")
        for r, ratio in zip(results, ratios):
            print(f"  {r['regime']}: {ratio:.6f}")
        print(f"\nMean: {mean_ratio:.6f}")
        print(f"Std: {std_ratio:.6f}")

        if std_ratio < 0.1 * mean_ratio:
            print(f"✓ Scaling consistent across regimes (std/mean = {std_ratio/mean_ratio:.1%})")
        else:
            print(f"⚠ Scaling inconsistent (std/mean = {std_ratio/mean_ratio:.1%})")

        # Check if mean ratio is close to 1
        if abs(mean_ratio - 1.0) < 0.1:
            print(f"✓ Measured Γ matches expected (mean ratio = {mean_ratio:.3f})")
        else:
            print(f"⚠ Systematic bias in Γ (mean ratio = {mean_ratio:.3f})")
    else:
        print("⚠ Some regimes failed to produce valid measurements")

    # Exit code
    n_pass = sum(r["success"] for r in results)
    n_total = len(results)

    print(f"\n{'=' * 80}")
    print(f"OVERALL: {n_pass}/{n_total} regimes passed")
    print(f"{'=' * 80}\n")

    if n_pass == n_total:
        print("✓ All regimes validate Γ = Dk² scaling law")
        sys.exit(0)
    else:
        print("✗ Some regimes failed validation")
        sys.exit(1)


if __name__ == "__main__":
    main()

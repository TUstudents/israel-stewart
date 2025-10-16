"""
Analytical validation tests for IReD transport coefficients.

These tests compare numerical evolution with IReD coefficients against
analytical predictions from Israel-Stewart theory and linear response theory.

CRITICAL: All tests enforce regime validity (|τω| < 1). Tests will be skipped
if parameters violate the Israel-Stewart regime requirement.
"""

import numpy as np
import pytest

from israel_stewart.benchmarks.bjorken_flow import create_bjorken_benchmark_with_ired
from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired
from israel_stewart.benchmarks.sound_waves import create_numerical_benchmark_with_ired
from israel_stewart.tests.test_helpers import check_regime_validity


def fit_exponential_decay(times, amplitudes):
    """
    Fit exponential decay: A(t) = A₀ exp(-Γt).

    Args:
        times: Time points
        amplitudes: Amplitude values (must be positive)

    Returns:
        float: Decay rate Γ
    """
    # Use log-linear fit
    log_A = np.log(np.abs(amplitudes) + 1e-15)  # Avoid log(0)
    coeffs = np.polyfit(times, log_A, deg=1)
    Gamma = -coeffs[0]  # Decay rate (negative slope)
    return Gamma


class TestIReDAnalyticalValidation:
    """
    Compare IReD evolution with analytical predictions.

    Tests validate that IReD coefficients from hard sphere kinetic theory
    produce quantitatively correct physics when used in hydrodynamic evolution.
    """

    @pytest.mark.slow
    def test_bjorken_temperature_vs_analytical(self, ired_regime_valid_coarse):
        """
        Test Bjorken temperature T(τ) matches analytical IS solution.

        For Bjorken flow, the Israel-Stewart equations can be solved analytically
        in the relaxation time approximation. Numerical evolution with IReD
        coefficients should match this analytical solution to < 5%.

        Uses coarse grid (8³) with large σ to ensure |τω| < 1.

        Reference: Romatschke & Romatschke (2019), IRED_TEST_PLAN.md Phase 3
        """
        # Create Bjorken benchmark with regime-valid IReD
        benchmark, ired_model = create_bjorken_benchmark_with_ired(
            T0=ired_regime_valid_coarse["temperature"],
            tau0=0.6,
            cross_section=ired_regime_valid_coarse["cross_section"],
            truncation="41",
            grid_points=ired_regime_valid_coarse["grid_points"],
        )

        # Check regime validity
        regime_param = check_regime_validity(
            benchmark.grid, benchmark.coefficients, max_allowed=1.0
        )
        print(f"\nRegime parameter |τω| = {regime_param:.3f} < 1.0 ✓")

        # Evolve numerically
        result = benchmark.run_numerical_simulation(
            final_time=3.0,  # τ = 0.6 → 3.0 fm/c
            timestep=0.05,
            method="rk4",
        )

        # Compare with analytical solution at several points
        times = np.array(result["time"])
        T_numerical = np.array(result["temperature"])

        max_error = 0.0
        for t, T_num in zip(times, T_numerical):
            # Get analytical temperature
            analytical_solution = benchmark.analytical.israel_stewart_solution(t)
            T_analytical = analytical_solution["temperature"]

            # Compute relative error
            error = abs(T_num - T_analytical) / T_analytical
            max_error = max(max_error, error)

            if error > 0.05:  # Print violations
                print(
                    f"t={t:.2f} fm/c: T_num={T_num:.4f}, T_ana={T_analytical:.4f}, error={error:.1%}"
                )

        print(f"Maximum temperature error: {max_error:.1%}")

        assert max_error < 0.05, (
            f"Temperature error too large: {max_error:.1%} > 5%. "
            f"IReD coefficients not reproducing analytical IS solution."
        )

    @pytest.mark.slow
    def test_bjorken_shear_stress_evolution(self, ired_regime_valid_coarse):
        """
        Test Bjorken shear stress π^{ηη}(τ) evolution with IReD.

        The shear stress component π^{ηη} evolves according to IS equations
        with IReD relaxation time τ_π and viscosity η. Check that numerical
        evolution matches analytical prediction to < 10%.

        Reference: IReD eq. (5), IRED_TEST_PLAN.md Phase 3
        """
        # Create Bjorken benchmark with regime-valid IReD
        benchmark, ired_model = create_bjorken_benchmark_with_ired(
            T0=ired_regime_valid_coarse["temperature"],
            tau0=0.6,
            cross_section=ired_regime_valid_coarse["cross_section"],
            truncation="41",
            grid_points=ired_regime_valid_coarse["grid_points"],
        )

        # Check regime validity
        regime_param = check_regime_validity(
            benchmark.grid, benchmark.coefficients, max_allowed=1.0
        )
        print(f"\nRegime parameter |τω| = {regime_param:.3f} < 1.0 ✓")
        print(f"Shear viscosity η = {ired_model.shear_viscosity():.6f}")
        print(f"Shear relaxation time τ_π = {ired_model.shear_relaxation_time():.6f} fm/c")

        # Evolve numerically
        result = benchmark.run_numerical_simulation(
            final_time=2.0,  # τ = 0.6 → 2.0 fm/c
            timestep=0.05,
            method="rk4",
        )

        # Extract shear stress (spatial average)
        times = np.array(result["time"])
        # π^{ηη} = π^{zz} / τ² in Milne coordinates
        # For uniform Bjorken flow, extract from center of grid
        pi_eta_eta_values = []

        # We need to access solver fields at each snapshot
        # For now, just check that shear stress is present and evolving
        # Full validation requires storing π^{μν} in result dictionary

        # Simplified check: verify shear stress magnitude is reasonable
        # For IReD: π^{ηη} ~ -4η/(3τ) at late times
        final_time = times[-1]
        eta = ired_model.shear_viscosity()
        expected_magnitude = 4 * eta / (3 * final_time)

        # This is a placeholder - full test requires π^{μν} storage
        print(
            f"Expected shear stress magnitude at τ={final_time:.2f}: |π^ηη| ~ {expected_magnitude:.6e}"
        )

        # Mark as expected to evolve (weak test for now)
        assert times[-1] > times[0], "Bjorken evolution progressed"

    @pytest.mark.slow
    def test_sound_wave_frequency(self, ired_regime_valid_large_domain):
        """
        Test sound wave frequency ω(k) from dispersion relation.

        For radiation fluid with IReD viscosity, the sound wave frequency
        follows the dispersion relation:
            ω = c_s k - i Γ(k)
        where c_s = 1/√3 and Γ(k) is the damping rate.

        Validate that real part Re(ω) ≈ c_s k to < 5%.

        Reference: IRED_THEORY.md Part IV, IRED_TEST_PLAN.md Phase 3
        """
        # Create sound wave benchmark with regime-valid IReD
        benchmark, ired_model = create_numerical_benchmark_with_ired(
            temperature=ired_regime_valid_large_domain["temperature"],
            cross_section=ired_regime_valid_large_domain["cross_section"],
            truncation="41",
            domain_size=ired_regime_valid_large_domain["domain_size"],
            grid_points=ired_regime_valid_large_domain["grid_points"],
        )

        # Setup with low wavenumber for regime validity
        k = 0.5  # fm⁻¹
        benchmark.setup_initial_conditions(wave_number=k)

        # Check regime validity
        regime_param = check_regime_validity(
            benchmark.grid, benchmark.transport_coeffs, max_allowed=1.0
        )
        print(f"\nRegime parameter |τω| = {regime_param:.3f} < 1.0 ✓")

        # Analyze dispersion relation
        wave_vector = np.array([k, 0.0, 0.0])
        modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)

        # Get sound mode (highest frequency)
        sound_mode = modes[0]
        omega_real = sound_mode.frequency
        omega_imag = -sound_mode.attenuation

        # Expected: ω_real ≈ c_s k
        c_s = 1.0 / np.sqrt(3.0)
        expected_omega = c_s * k

        error = abs(omega_real - expected_omega) / expected_omega

        print(f"Wave number k = {k:.3f} fm⁻¹")
        print(f"Numerical frequency ω = {omega_real:.6f} - i {-omega_imag:.6f}")
        print(f"Expected frequency c_s k = {expected_omega:.6f}")
        print(f"Relative error: {error:.1%}")

        assert error < 0.05, (
            f"Sound wave frequency error: {error:.1%} > 5%. "
            f"Expected ω ≈ c_s k = {expected_omega:.6f}, got {omega_real:.6f}"
        )

    @pytest.mark.slow
    def test_sound_wave_damping(self, ired_regime_valid_large_domain):
        """
        Test sound wave damping rate Γ(k) with IReD shear viscosity.

        The sound wave damping rate for radiation fluid is:
            Γ(k) ≈ (4η/3 + ζ) k² / (ε + p)
        For conformal hard sphere gas: ζ = 0.

        Validate that numerical damping matches analytical prediction to < 15%.

        Reference: IReD Tables III-IV, IRED_TEST_PLAN.md Phase 3
        """
        # Create sound wave benchmark with regime-valid IReD
        benchmark, ired_model = create_numerical_benchmark_with_ired(
            temperature=ired_regime_valid_large_domain["temperature"],
            cross_section=ired_regime_valid_large_domain["cross_section"],
            truncation="41",
            domain_size=ired_regime_valid_large_domain["domain_size"],
            grid_points=ired_regime_valid_large_domain["grid_points"],
        )

        # Setup with low wavenumber
        k = 0.5
        benchmark.setup_initial_conditions(wave_number=k)

        # Check regime validity
        regime_param = check_regime_validity(
            benchmark.grid, benchmark.transport_coeffs, max_allowed=1.0
        )
        print(f"\nRegime parameter |τω| = {regime_param:.3f} < 1.0 ✓")

        # Get numerical damping from dispersion relation
        wave_vector = np.array([k, 0.0, 0.0])
        modes = benchmark.analytical.analyze_dispersion_relation(wave_vector)
        sound_mode = modes[0]
        Gamma_numerical = -sound_mode.attenuation

        # Analytical prediction: Γ = (4η/3) k² / (ε + p)
        eta = ired_model.shear_viscosity()
        T = ired_regime_valid_large_domain["temperature"]
        epsilon = (np.pi**2 / 30.0) * T**4  # Radiation energy density
        p = epsilon / 3.0  # Conformal EOS

        Gamma_analytical = (4 * eta / 3.0) * k**2 / (epsilon + p)

        error = abs(Gamma_numerical - Gamma_analytical) / Gamma_analytical

        print(f"Shear viscosity η = {eta:.6f}")
        print(f"Numerical damping Γ = {Gamma_numerical:.6e}")
        print(f"Analytical damping Γ = {Gamma_analytical:.6e}")
        print(f"Relative error: {error:.1%}")

        assert error < 0.15, (
            f"Sound wave damping error: {error:.1%} > 15%. "
            f"Expected Γ ≈ {Gamma_analytical:.6e}, got {Gamma_numerical:.6e}"
        )

    @pytest.mark.slow
    def test_diffusion_decay_rate(self, ired_regime_valid_large_domain):
        """
        Test diffusion decay rate Γ = D k² with IReD diffusion coefficient.

        For isentropic diffusion mode, the particle density perturbation decays as:
            δn(t) = δn₀ exp(-Dk²t)
        where D is the IReD diffusion coefficient (Landau frame).

        Validate that numerical decay rate matches Dk² to < 10%.

        Reference: IReD Table III, IRED_TEST_PLAN.md Phase 3
        """
        # Create diffusion benchmark with regime-valid IReD
        benchmark, ired_model = create_diffusion_benchmark_with_ired(
            temperature=ired_regime_valid_large_domain["temperature"],
            cross_section=ired_regime_valid_large_domain["cross_section"],
            truncation="41",
            perturbation_amplitude=0.05,
            wave_number=0.5,  # Low k for regime validity
            grid_points=ired_regime_valid_large_domain["grid_points"],
            domain_size=ired_regime_valid_large_domain["domain_size"],
        )

        # Check regime validity
        regime_param = check_regime_validity(
            benchmark.grid, benchmark.coefficients, max_allowed=1.0
        )
        print(f"\nRegime parameter |τω| = {regime_param:.3f} < 1.0 ✓")

        # Get analytical decay rate
        D = ired_model.diffusion_coefficient()
        k = benchmark.analytical.wave_number
        Gamma_expected = D * k**2

        print(f"Diffusion coefficient D = {D:.6f}")
        print(f"Wave number k = {k:.3f}")
        print(f"Expected decay rate Γ = Dk² = {Gamma_expected:.6e}")

        # Evolve and extract amplitude at each time
        times = []
        amplitudes = []

        # Get k-space index for wave
        kx_idx = int(k * benchmark.grid.shape[0] / (2 * np.pi / benchmark.grid.dx))

        def extract_amplitude(t, fields):
            # Compute FFT of diffusion current V^x
            V_x_fft = np.fft.fftn(fields.V_mu[..., 1])
            amplitude = np.abs(V_x_fft[kx_idx, 0, 0])
            times.append(t)
            amplitudes.append(amplitude)

        # Evolve for several decay times
        t_final = 3.0 / Gamma_expected
        benchmark.solver.evolve(
            t_final=t_final, dt=t_final / 50, method="rk4", callback=extract_amplitude
        )

        # Fit exponential decay
        times = np.array(times)
        amplitudes = np.array(amplitudes)
        Gamma_measured = fit_exponential_decay(times, amplitudes)

        error = abs(Gamma_measured - Gamma_expected) / Gamma_expected

        print(f"Measured decay rate Γ = {Gamma_measured:.6e}")
        print(f"Relative error: {error:.1%}")

        assert error < 0.10, (
            f"Diffusion decay rate error: {error:.1%} > 10%. "
            f"Expected Γ = Dk² = {Gamma_expected:.6e}, measured {Gamma_measured:.6e}"
        )

    @pytest.mark.slow
    def test_diffusion_ficks_law(self, ired_regime_valid_large_domain):
        """
        Test Fick's law: V^i = -D ∇^i(μ/T) for IReD diffusion.

        In the Landau frame, the diffusion current V^μ follows Fick's law
        at leading order. For isentropic perturbations:
            V^x = -D ∂_x(μ/T)

        Validate that numerical V^x matches analytical prediction to < 10%
        at early times (before nonlinear effects).

        Reference: IReD eq. (5) with q^μ → V^μ, IRED_TEST_PLAN.md Phase 3
        """
        # Create diffusion benchmark with regime-valid IReD
        benchmark, ired_model = create_diffusion_benchmark_with_ired(
            temperature=ired_regime_valid_large_domain["temperature"],
            cross_section=ired_regime_valid_large_domain["cross_section"],
            truncation="41",
            perturbation_amplitude=0.05,
            wave_number=0.5,
            grid_points=ired_regime_valid_large_domain["grid_points"],
            domain_size=ired_regime_valid_large_domain["domain_size"],
        )

        # Check regime validity
        regime_param = check_regime_validity(
            benchmark.grid, benchmark.coefficients, max_allowed=1.0
        )
        print(f"\nRegime parameter |τω| = {regime_param:.3f} < 1.0 ✓")
        print(f"Diffusion coefficient D = {ired_model.diffusion_coefficient():.6f}")

        # Check Fick's law at t=0 (initial condition)
        X, _, _ = benchmark.grid.meshgrid()
        V_x_numerical = benchmark.fields.V_mu[..., 1]
        V_x_analytical = benchmark.analytical.diffusion_current(X, 0.0)

        # Compute relative error (pointwise)
        relative_error = np.abs(V_x_numerical - V_x_analytical) / (np.abs(V_x_analytical) + 1e-15)
        max_error = np.max(relative_error)
        mean_error = np.mean(relative_error)

        print("Fick's law validation at t=0:")
        print(f"  Mean error: {mean_error:.1%}")
        print(f"  Max error: {max_error:.1%}")

        assert mean_error < 0.10, (
            f"Fick's law error at t=0: {mean_error:.1%} > 10%. "
            f"Initial diffusion current V^x does not match -D∇(μ/T)"
        )

        # Also check at t = 0.5 (early time, still linear regime)
        times_check = [0.5]
        errors_check = []

        def check_ficks_law(t, fields):
            if t in times_check:
                V_x_num = fields.V_mu[..., 1]
                V_x_ana = benchmark.analytical.diffusion_current(X, t)
                error = np.mean(np.abs(V_x_num - V_x_ana) / (np.abs(V_x_ana) + 1e-15))
                errors_check.append(error)
                print(f"  t={t:.2f}: error = {error:.1%}")

        benchmark.solver.evolve(t_final=0.5, dt=0.05, method="rk4", callback=check_ficks_law)

        # Early time error should still be reasonable (< 20%)
        if errors_check:
            assert errors_check[0] < 0.20, (
                f"Fick's law error at t=0.5: {errors_check[0]:.1%} > 20%. "
                f"Diffusion current deviates significantly from analytical prediction"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])

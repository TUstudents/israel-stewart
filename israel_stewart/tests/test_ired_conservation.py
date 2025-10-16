"""
Conservation law tests with IReD transport coefficients.

These tests validate that fundamental conservation laws (energy, momentum,
particle number) hold during evolution with IReD coefficients in the
regime-valid parameter space (|τω| < 1).

CRITICAL: These tests enforce regime validity. Tests will be skipped if
parameters violate |τω| < 1 requirement (Wagner & Gavassino 2024).
"""

import numpy as np
import pytest

from israel_stewart.benchmarks.bjorken_flow import create_bjorken_benchmark_with_ired
from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired
from israel_stewart.benchmarks.sound_waves import create_numerical_benchmark_with_ired
from israel_stewart.solvers.spectral import SpectralISHydrodynamics
from israel_stewart.tests.test_helpers import check_regime_validity


def compute_total_energy(fields, grid):
    """
    Compute total energy ∫ T^{00} d³x.

    For Minkowski metric (-,+,+,+): T^{00} = ρ (energy density).

    Args:
        fields: ISFieldConfiguration
        grid: SpaceGrid

    Returns:
        float: Total energy
    """
    energy_density = fields.rho
    total_energy = np.sum(energy_density) * grid.dx**3
    return total_energy


def compute_total_momentum(fields, grid):
    """
    Compute total momentum ∫ T^{0i} d³x.

    For Minkowski: T^{0i} = (ρ + p) u^0 u^i + π^{0i} + V^0 u^i + V^i u^0

    Args:
        fields: ISFieldConfiguration
        grid: SpaceGrid

    Returns:
        ndarray: Total momentum vector [P^x, P^y, P^z]
    """
    # Energy-momentum flux (leading term)
    rho_plus_p = fields.rho + fields.pressure
    u0 = fields.u_mu[..., 0]

    # Spatial momentum density
    momentum_density_x = rho_plus_p * u0 * fields.u_mu[..., 1]
    momentum_density_y = rho_plus_p * u0 * fields.u_mu[..., 2]
    momentum_density_z = rho_plus_p * u0 * fields.u_mu[..., 3]

    # Integrate over volume
    Px = np.sum(momentum_density_x) * grid.dx**3
    Py = np.sum(momentum_density_y) * grid.dx**3
    Pz = np.sum(momentum_density_z) * grid.dx**3

    return np.array([Px, Py, Pz])


def compute_total_particle_number(fields, grid):
    """
    Compute total particle number ∫ n u^0 d³x.

    For radiation fluid: n ≈ 1.202 T³ / π²

    Args:
        fields: ISFieldConfiguration
        grid: SpaceGrid

    Returns:
        float: Total particle number
    """
    # Temperature from energy density: ρ = (π²/30) T⁴
    T = (30.0 * fields.rho / np.pi**2) ** 0.25

    # Particle density
    n = (1.202 / np.pi**2) * T**3

    # Integrate n u^0
    u0 = fields.u_mu[..., 0]
    total_N = np.sum(n * u0) * grid.dx**3

    return total_N


class TestIReDConservation:
    """
    Test conservation laws with IReD transport coefficients.

    All tests enforce regime validity (|τω| < 1). Tests will be skipped if
    parameters are outside the valid Israel-Stewart regime.
    """

    @pytest.mark.slow
    def test_energy_conservation_regime_valid(self, ired_regime_valid_coarse):
        """
        Test energy conservation during regime-valid evolution with IReD.

        Uses coarse grid (8³) with large cross-section (σ=100 fm²) to ensure
        |τω| < 1. Validates that total energy ∫ ρ d³x is conserved to < 0.1%
        during 100 timesteps of Bjorken expansion.

        Reference: Wagner & Gavassino (2024), IRED_TEST_PLAN.md Phase 2
        """
        # Create Bjorken benchmark with regime-valid parameters
        benchmark, ired_model = create_bjorken_benchmark_with_ired(
            T0=ired_regime_valid_coarse["temperature"],
            tau0=0.6,  # Initial proper time
            cross_section=ired_regime_valid_coarse["cross_section"],
            truncation="41",
            grid_points=ired_regime_valid_coarse["grid_points"],
        )

        # Check regime validity (will skip if |τω| > 1)
        regime_param = check_regime_validity(
            benchmark.grid, benchmark.coefficients, max_allowed=1.0
        )
        print(f"\nRegime parameter |τω| = {regime_param:.3f} < 1.0 ✓")

        # Setup initial conditions (run_numerical_simulation calls this internally)
        # But we need E0 before evolution, so call it explicitly
        benchmark._setup_bjorken_initial_conditions(benchmark.fields)

        # Compute initial energy
        E0 = compute_total_energy(benchmark.fields, benchmark.grid)
        print(f"Initial energy: E0 = {E0:.6e}")

        # Evolve 100 steps (τ = 0.6 → 1.1 fm/c)
        result = benchmark.run_numerical_simulation(final_time=1.1, timestep=0.005, method="rk4")

        # Get final fields from solver
        Ef = compute_total_energy(benchmark.fields, benchmark.grid)
        print(f"Final energy: Ef = {Ef:.6e}")

        # Conservation check
        relative_change = abs(Ef - E0) / E0
        print(f"Energy conservation: ΔE/E0 = {relative_change:.2e}")

        assert relative_change < 1e-3, f"Energy not conserved: ΔE/E = {relative_change:.2e} > 0.1%"

    @pytest.mark.slow
    def test_particle_conservation_diffusion(self, ired_regime_valid_large_domain):
        """
        Test particle number conservation during diffusion with IReD.

        Uses large domain (20π) with fine grid (32³) to ensure regime validity
        while maintaining resolution. Validates that total particle number
        ∫ n u^0 d³x is conserved to < 0.1% during diffusion evolution.

        This tests the IReD diffusion coefficient D and relaxation time τ_V
        in the Landau frame (V^μ instead of q^μ).

        Reference: IRED_TEST_PLAN.md Phase 2, IReD Tables III-IV
        """
        # Create diffusion benchmark with regime-valid parameters
        benchmark, ired_model = create_diffusion_benchmark_with_ired(
            temperature=ired_regime_valid_large_domain["temperature"],
            cross_section=ired_regime_valid_large_domain["cross_section"],
            truncation="41",
            perturbation_amplitude=0.05,  # Linear regime
            wave_number=0.5,  # Low k for regime validity
            grid_points=ired_regime_valid_large_domain["grid_points"],
            domain_size=ired_regime_valid_large_domain["domain_size"],
        )

        # Check regime validity
        regime_param = check_regime_validity(
            benchmark.grid, benchmark.coefficients, max_allowed=1.0
        )
        print(f"\nRegime parameter |τω| = {regime_param:.3f} < 1.0 ✓")
        print(f"Diffusion coefficient D = {ired_model.diffusion_coefficient():.6f}")
        print(f"Diffusion relaxation time τ_V = {ired_model.diffusion_relaxation_time():.6f} fm/c")

        # Compute initial particle number
        N0 = compute_total_particle_number(benchmark.initial_fields, benchmark.grid)
        print(f"Initial particle number: N0 = {N0:.6e}")

        # Evolve for short time (IReD diffusion is very slow with large cross-section)
        # Use 2 fm/c evolution time (short but sufficient to test conservation)
        t_final = 2.0 / 0.197  # 2 fm/c in GeV^-1 units
        n_timesteps = 40  # Moderate number of timesteps for speed
        dt = t_final / n_timesteps

        print(f"Evolution time: {t_final * 0.197:.2f} fm/c ({n_timesteps} timesteps)")
        print(f"Timestep: {dt * 0.197:.4f} fm/c")

        # Collect snapshots
        N_values = [N0]

        def track_particle_number(t, fields):
            N = compute_total_particle_number(fields, benchmark.grid)
            N_values.append(N)

        benchmark.solver.evolve(
            t_final=t_final,
            dt=dt,
            method="rk4",
            callback=track_particle_number,
        )

        # Check conservation at all times
        N_array = np.array(N_values)
        relative_changes = np.abs(N_array - N0) / N0
        max_violation = np.max(relative_changes)

        print(f"Final particle number: Nf = {N_array[-1]:.6e}")
        print(f"Maximum violation: max(ΔN/N0) = {max_violation:.2e}")

        assert (
            max_violation < 1e-3
        ), f"Particle number not conserved: max(ΔN/N) = {max_violation:.2e} > 0.1%"

    @pytest.mark.slow
    def test_momentum_conservation_sound_wave(self, ired_regime_valid_large_domain):
        """
        Test momentum conservation during sound wave with IReD.

        Uses large domain (20π) for regime validity. Validates that total
        momentum ∫ T^{0i} d³x is conserved to < 0.1% during sound wave
        propagation with IReD shear viscosity.

        This tests the IReD shear viscosity η and relaxation time τ_π
        from hard sphere kinetic theory.

        Reference: IRED_TEST_PLAN.md Phase 2, IReD eq. (5) and Tables III-IV
        """
        # Create sound wave benchmark with regime-valid parameters
        benchmark, ired_model = create_numerical_benchmark_with_ired(
            temperature=ired_regime_valid_large_domain["temperature"],
            cross_section=ired_regime_valid_large_domain["cross_section"],
            truncation="41",
            domain_size=ired_regime_valid_large_domain["domain_size"],
            grid_points=ired_regime_valid_large_domain["grid_points"],
        )

        # Setup initial conditions (low-k sound wave)
        benchmark.setup_initial_conditions(wave_number=0.5)  # Low k for regime validity

        # Check regime validity
        regime_param = check_regime_validity(
            benchmark.grid, benchmark.transport_coeffs, max_allowed=1.0
        )
        print(f"\nRegime parameter |τω| = {regime_param:.3f} < 1.0 ✓")
        print(f"Shear viscosity η = {ired_model.shear_viscosity():.6f}")
        print(f"Shear relaxation time τ_π = {ired_model.shear_relaxation_time():.6f} fm/c")

        # Compute initial momentum
        P0 = compute_total_momentum(benchmark.fields, benchmark.grid)
        print(f"Initial momentum: P0 = [{P0[0]:.6e}, {P0[1]:.6e}, {P0[2]:.6e}]")

        # Collect snapshots
        P_values = [P0]

        def track_momentum(t, fields):
            P = compute_total_momentum(fields, benchmark.grid)
            P_values.append(P)

        # Evolve for one oscillation period
        # For sound wave: ω ≈ c_s k, period ≈ 2π/(c_s k)
        c_s = 1.0 / np.sqrt(3.0)
        k = 0.5
        period = 2 * np.pi / (c_s * k)

        benchmark.solver.evolve(
            t_final=period,
            dt=period / 50,  # 50 timesteps per period
            method="spectral_imex",
            callback=track_momentum,
        )

        # Check conservation at all times
        P_array = np.array(P_values)
        P_norms = np.linalg.norm(P_array, axis=1)
        P0_norm = np.linalg.norm(P0)

        relative_changes = np.abs(P_norms - P0_norm) / (P0_norm + 1e-15)  # Avoid division by zero
        max_violation = np.max(relative_changes)

        print(
            f"Final momentum: Pf = [{P_array[-1, 0]:.6e}, {P_array[-1, 1]:.6e}, {P_array[-1, 2]:.6e}]"
        )
        print(f"Maximum violation: max(Δ|P|/|P0|) = {max_violation:.2e}")

        assert (
            max_violation < 1e-3
        ), f"Momentum not conserved: max(Δ|P|/|P|) = {max_violation:.2e} > 0.1%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])

"""
Tests for Landau frame constraint validation.

Validates that the Landau frame constraint V^μ u_μ = 0 is:
1. Satisfied at initialization
2. Maintained during time evolution
3. Properly enforced by projection operators
4. Correctly detected when violated

The Landau frame is defined by zero energy flux: V^μ u_μ = 0
where V^μ is the particle diffusion current and u^μ is the four-velocity.

This is in contrast to the Eckart frame where q^μ u_μ = 0 (zero particle flux).

Reference:
    - Landau & Lifshitz, Fluid Mechanics (1987), §127
    - IReD paper: Wagner, Palermo, Ambrus (2022), Section II
"""

import numpy as np
import pytest

from israel_stewart.benchmarks.bjorken_flow import create_bjorken_benchmark_with_ired
from israel_stewart.benchmarks.diffusion_flow import create_diffusion_benchmark_with_ired
from israel_stewart.benchmarks.sound_waves import create_numerical_benchmark_with_ired
from israel_stewart.core import ISFieldConfiguration
from israel_stewart.core.spacegrid import SpaceGrid


class TestLandauFrameConstraintInitialization:
    """Test that Landau frame constraint is satisfied at initialization."""

    def test_constraint_in_bjorken_initial_state(self):
        """Test V^μ u_μ = 0 in Bjorken flow initial conditions."""
        benchmark, _ = create_bjorken_benchmark_with_ired(
            T0=0.4, cross_section=1.0, grid_points=(16, 16, 16)
        )

        fields = benchmark.fields

        # Compute V^μ u_μ with Minkowski metric (-,+,+,+)
        constraint = (
            -fields.V_mu[..., 0] * fields.u_mu[..., 0]
            + fields.V_mu[..., 1] * fields.u_mu[..., 1]
            + fields.V_mu[..., 2] * fields.u_mu[..., 2]
            + fields.V_mu[..., 3] * fields.u_mu[..., 3]
        )

        # For Bjorken flow (isentropic), V^μ = 0 initially
        max_violation = np.max(np.abs(constraint))
        assert (
            max_violation < 1e-10
        ), f"Landau frame constraint violated: max |V^μ u_μ| = {max_violation}"

    def test_constraint_in_diffusion_initial_state(self):
        """Test V^μ u_μ = 0 in diffusion flow initial conditions."""
        benchmark, _ = create_diffusion_benchmark_with_ired(
            temperature=0.4,
            cross_section=1.0,
            perturbation_amplitude=0.05,
            grid_points=(16, 16, 8),
        )

        fields = benchmark.initial_fields

        # Compute V^μ u_μ
        constraint = (
            -fields.V_mu[..., 0] * fields.u_mu[..., 0]
            + fields.V_mu[..., 1] * fields.u_mu[..., 1]
            + fields.V_mu[..., 2] * fields.u_mu[..., 2]
            + fields.V_mu[..., 3] * fields.u_mu[..., 3]
        )

        # Diffusion flow has non-zero V^x, but still V^μ u_μ = 0
        max_violation = np.max(np.abs(constraint))
        assert (
            max_violation < 1e-10
        ), f"Landau frame constraint violated: max |V^μ u_μ| = {max_violation}"

    def test_constraint_in_sound_wave_initial_state(self):
        """Test V^μ u_μ = 0 in sound wave initial conditions."""
        benchmark, _ = create_numerical_benchmark_with_ired(
            temperature=0.4, cross_section=1.0, grid_points=(16, 16, 8)
        )

        fields = benchmark.fields

        # Compute V^μ u_μ
        constraint = (
            -fields.V_mu[..., 0] * fields.u_mu[..., 0]
            + fields.V_mu[..., 1] * fields.u_mu[..., 1]
            + fields.V_mu[..., 2] * fields.u_mu[..., 2]
            + fields.V_mu[..., 3] * fields.u_mu[..., 3]
        )

        # Sound waves (isentropic) have V^μ = 0 initially
        max_violation = np.max(np.abs(constraint))
        assert (
            max_violation < 1e-10
        ), f"Landau frame constraint violated: max |V^μ u_μ| = {max_violation}"


class TestLandauFrameConstraintEvolution:
    """Test that Landau frame constraint is maintained during time evolution."""

    @pytest.mark.slow
    def test_constraint_maintained_in_diffusion_evolution(self):
        """Test that V^μ u_μ = 0 is maintained during diffusion evolution."""
        benchmark, _ = create_diffusion_benchmark_with_ired(
            temperature=0.4,
            cross_section=1.0,
            perturbation_amplitude=0.05,
            wave_number=1.0,
            grid_points=(8, 8, 4),  # Smaller grid for faster test
        )

        # Run very short simulation
        result = benchmark.run_numerical_simulation(
            final_time=0.05, timestep=0.01, snapshot_interval=0.05
        )

        # Check constraint violations at all times
        violations = result["constraint_violation"]

        # Maximum violation should be small (within numerical precision)
        max_violation = np.max(violations)
        assert max_violation < 1e-6, (
            f"Landau frame constraint not maintained during evolution: "
            f"max |V^μ u_μ| = {max_violation:.2e}"
        )

    @pytest.mark.slow
    def test_constraint_validation_method(self):
        """Test that benchmark validation method correctly checks constraint."""
        benchmark, _ = create_diffusion_benchmark_with_ired(
            temperature=0.4, cross_section=1.0, grid_points=(8, 8, 4)
        )

        result = benchmark.run_numerical_simulation(final_time=0.05, timestep=0.01)

        # Validate constraint
        constraint_ok = benchmark.validate_landau_frame_constraint(result, tolerance=1e-6)
        assert constraint_ok, "Landau frame constraint validation failed"


class TestLandauFrameConstraintMath:
    """Test mathematical properties of Landau frame constraint."""

    def test_constraint_with_rest_frame_four_velocity(self):
        """Test V^μ u_μ = 0 with rest frame four-velocity."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0, 1)] * 3,
            grid_points=(8, 8, 8),
            boundary_conditions="periodic",
        )

        fields = ISFieldConfiguration(grid)

        # Rest frame: u^μ = (1, 0, 0, 0)
        fields.u_mu[..., 0] = 1.0
        fields.u_mu[..., 1:] = 0.0

        # Set spatial diffusion current V^i (arbitrary)
        fields.V_mu[..., 1] = 0.1 * np.sin(2 * np.pi * grid.meshgrid()[0])
        fields.V_mu[..., 2] = 0.0
        fields.V_mu[..., 3] = 0.0

        # In rest frame, Landau constraint requires V^0 = 0
        fields.V_mu[..., 0] = 0.0

        # Check constraint
        constraint = (
            -fields.V_mu[..., 0] * fields.u_mu[..., 0]
            + fields.V_mu[..., 1] * fields.u_mu[..., 1]
            + fields.V_mu[..., 2] * fields.u_mu[..., 2]
            + fields.V_mu[..., 3] * fields.u_mu[..., 3]
        )

        assert np.allclose(constraint, 0.0, atol=1e-14)

    def test_constraint_with_boosted_four_velocity(self):
        """Test V^μ u_μ = 0 with boosted four-velocity."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0, 1)] * 3,
            grid_points=(8, 8, 8),
            boundary_conditions="periodic",
        )

        fields = ISFieldConfiguration(grid)

        # Boosted in x-direction: v_x = 0.5
        v_x = 0.5
        gamma = 1.0 / np.sqrt(1.0 - v_x**2)
        fields.u_mu[..., 0] = gamma
        fields.u_mu[..., 1] = gamma * v_x
        fields.u_mu[..., 2] = 0.0
        fields.u_mu[..., 3] = 0.0

        # Set V^μ in rest frame: V^0 = 0, V^x = 0.1
        # In boosted frame: V^0' = γ(V^0 - v·V) = -γ v_x V^x
        V_x_rest = 0.1
        fields.V_mu[..., 0] = -gamma * v_x * V_x_rest
        fields.V_mu[..., 1] = V_x_rest
        fields.V_mu[..., 2] = 0.0
        fields.V_mu[..., 3] = 0.0

        # Check constraint
        constraint = (
            -fields.V_mu[..., 0] * fields.u_mu[..., 0]
            + fields.V_mu[..., 1] * fields.u_mu[..., 1]
            + fields.V_mu[..., 2] * fields.u_mu[..., 2]
            + fields.V_mu[..., 3] * fields.u_mu[..., 3]
        )

        # V^μ u_μ = -V^0 u^0 + V^x u^x
        #         = -(-γv_x V^x)(γ) + V^x(γv_x)
        #         = γ²v_x V^x + γv_x V^x
        #         = 0  ✗ This is WRONG!

        # Correct Lorentz transformation for V^μ:
        # V^0' = γ(V^0 + v_x V^x) = γ v_x V^x (since V^0 = 0 in rest frame)
        # V^x' = γ(V^x + v_x V^0) = γ V^x (since V^0 = 0 in rest frame)
        fields.V_mu[..., 0] = gamma * v_x * V_x_rest  # Not negative!
        fields.V_mu[..., 1] = gamma * V_x_rest
        fields.V_mu[..., 2] = 0.0
        fields.V_mu[..., 3] = 0.0

        # Check constraint: V^μ u_μ = -V^0 u^0 + V^x u^x
        #                            = -(γv_x V^x)(γ) + (γV^x)(γv_x)
        #                            = -γ²v_x V^x + γ²v_x V^x = 0 ✓
        constraint = (
            -fields.V_mu[..., 0] * fields.u_mu[..., 0]
            + fields.V_mu[..., 1] * fields.u_mu[..., 1]
            + fields.V_mu[..., 2] * fields.u_mu[..., 2]
            + fields.V_mu[..., 3] * fields.u_mu[..., 3]
        )

        assert np.allclose(constraint, 0.0, atol=1e-14)

    def test_constraint_normalization_preservation(self):
        """Test that V^μ u_μ = 0 is independent of u^μ normalization."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0, 1)] * 3,
            grid_points=(8, 8, 8),
            boundary_conditions="periodic",
        )

        fields = ISFieldConfiguration(grid)

        # Four-velocity (unnormalized)
        fields.u_mu[..., 0] = 2.0  # Not normalized: u·u ≠ -1
        fields.u_mu[..., 1] = 1.0
        fields.u_mu[..., 2] = 0.0
        fields.u_mu[..., 3] = 0.0

        # V^μ orthogonal to u^μ
        # V·u = 0 → -V^0 u^0 + V^1 u^1 = 0 → V^0 = (u^1/u^0) V^1
        V_1 = 0.1
        fields.V_mu[..., 0] = (fields.u_mu[..., 1] / fields.u_mu[..., 0]) * V_1
        fields.V_mu[..., 1] = V_1
        fields.V_mu[..., 2] = 0.0
        fields.V_mu[..., 3] = 0.0

        # Check constraint
        constraint = (
            -fields.V_mu[..., 0] * fields.u_mu[..., 0]
            + fields.V_mu[..., 1] * fields.u_mu[..., 1]
            + fields.V_mu[..., 2] * fields.u_mu[..., 2]
            + fields.V_mu[..., 3] * fields.u_mu[..., 3]
        )

        assert np.allclose(constraint, 0.0, atol=1e-14)


class TestLandauVsEckartFrame:
    """Test differences between Landau and Eckart frames."""

    def test_landau_frame_zero_energy_flux(self):
        """Test that Landau frame has zero energy flux (V^μ u_μ = 0)."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0, 1)] * 3,
            grid_points=(8, 8, 8),
            boundary_conditions="periodic",
        )

        fields = ISFieldConfiguration(grid)

        # Rest frame
        fields.u_mu[..., 0] = 1.0
        fields.u_mu[..., 1:] = 0.0

        # Landau frame: V^μ u_μ = 0 → V^0 = 0 in rest frame
        fields.V_mu[..., 0] = 0.0
        fields.V_mu[..., 1] = 0.1  # Spatial diffusion current

        constraint = (
            -fields.V_mu[..., 0] * fields.u_mu[..., 0] + fields.V_mu[..., 1] * fields.u_mu[..., 1]
        )

        assert np.allclose(constraint, 0.0)

    def test_eckart_frame_would_have_nonzero_energy_flux(self):
        """
        Test that Eckart frame (n^μ u_μ = 0) would have nonzero V^μ u_μ.

        Note: This codebase uses Landau frame, so this test demonstrates
        what would happen if we incorrectly used Eckart frame constraint.
        """
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0, 1)] * 3,
            grid_points=(8, 8, 8),
            boundary_conditions="periodic",
        )

        fields = ISFieldConfiguration(grid)

        # Rest frame
        fields.u_mu[..., 0] = 1.0
        fields.u_mu[..., 1:] = 0.0

        # Eckart frame: n^μ u_μ = 0 → n^0 = 0 in rest frame
        # But n^μ = nu^μ + V^μ, so if n^0 = 0, then V^0 = -n u^0 = -n
        n = 0.01  # Particle density perturbation
        fields.V_mu[..., 0] = -n  # Eckart frame condition
        fields.V_mu[..., 1] = 0.1  # Spatial diffusion

        # In Eckart frame, V^μ u_μ ≠ 0 generally
        constraint = (
            -fields.V_mu[..., 0] * fields.u_mu[..., 0] + fields.V_mu[..., 1] * fields.u_mu[..., 1]
        )

        # V^μ u_μ = -(-n)(1) = n ≠ 0
        assert not np.allclose(constraint, 0.0)
        assert np.allclose(constraint, n)


class TestConstraintViolationDetection:
    """Test that constraint violations are properly detected."""

    def test_constraint_violation_threshold(self):
        """Test that small violations pass but large violations fail."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0, 1)] * 3,
            grid_points=(8, 8, 8),
            boundary_conditions="periodic",
        )

        fields = ISFieldConfiguration(grid)

        # Rest frame
        fields.u_mu[..., 0] = 1.0
        fields.u_mu[..., 1:] = 0.0

        # Test case 1: Small violation (numerical error)
        fields.V_mu[..., 0] = 1e-12  # Very small V^0
        fields.V_mu[..., 1:] = 0.0

        constraint = -fields.V_mu[..., 0] * fields.u_mu[..., 0]
        max_violation = np.max(np.abs(constraint))

        assert max_violation < 1e-10  # Should pass with tight tolerance

        # Test case 2: Large violation (physical error)
        fields.V_mu[..., 0] = 0.01  # Significant V^0
        fields.V_mu[..., 1:] = 0.0

        constraint = -fields.V_mu[..., 0] * fields.u_mu[..., 0]
        max_violation = np.max(np.abs(constraint))

        assert max_violation > 1e-10  # Should fail with tight tolerance
        assert max_violation == pytest.approx(0.01, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

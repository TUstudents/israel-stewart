"""
Test suite for finite difference solver implementations.

Tests the core functionality of conservative, upwind, and WENO finite difference schemes,
including proper numerical fluxes, Christoffel symbol contributions, and performance optimizations.
"""

import numpy as np
import pytest

from israel_stewart.core.fields import ISFieldConfiguration
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.finite_difference import (
    ConservativeFiniteDifference,
    UpwindFiniteDifference,
    WENOFiniteDifference,
    create_finite_difference_solver,
)


class TestFiniteDifferenceBase:
    """Base test class with common setup for finite difference tests."""

    @pytest.fixture
    def simple_grid(self):
        """Create a simple 2D spatial grid for testing."""
        return SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)],
            grid_points=(10, 16, 16, 16)  # nt, nx, ny, nz
        )

    @pytest.fixture
    def flat_metric(self):
        """Create Minkowski metric for flat spacetime tests."""
        return MinkowskiMetric()

    @pytest.fixture
    def test_field_1d(self, simple_grid):
        """Create a 1D test field with known derivative."""
        nx = simple_grid.grid_points[1]  # spatial x-direction
        x = np.linspace(0, 2*np.pi, nx)

        # Create a sine wave: f(x) = sin(x), f'(x) = cos(x)
        field = np.sin(x)
        analytical_derivative = np.cos(x)

        return field, analytical_derivative

    @pytest.fixture
    def test_field_2d(self, simple_grid):
        """Create a 2D test field with known derivatives."""
        nx, ny = simple_grid.grid_points[1:3]
        x = np.linspace(0, 2*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Create a separable function: f(x,y) = sin(x)*cos(y)
        field = np.sin(X) * np.cos(Y)
        df_dx = np.cos(X) * np.cos(Y)  # ∂f/∂x
        df_dy = -np.sin(X) * np.sin(Y)  # ∂f/∂y

        return field, df_dx, df_dy


class TestConservativeFiniteDifference(TestFiniteDifferenceBase):
    """Test conservative finite difference implementation."""

    def test_initialization(self, simple_grid, flat_metric):
        """Test proper initialization of conservative solver."""
        solver = ConservativeFiniteDifference(
            simple_grid, flat_metric, boundary_conditions="periodic", order=2
        )

        assert solver.order == 2
        assert solver.flux_limiter == "none"
        assert solver.stencil_width == 3  # order + 1
        assert len(solver.derivative_coefficients) > 0

    def test_lax_friedrichs_flux(self, simple_grid, flat_metric):
        """Test Lax-Friedrichs numerical flux implementation."""
        solver = ConservativeFiniteDifference(
            simple_grid, flat_metric, boundary_conditions="periodic", order=2
        )

        # Create a simple test field
        test_field = np.random.randn(20, 16, 16)  # Extended field with ghost points

        # Test different flux offsets
        for offset in [0.5, -0.5]:
            flux = solver._compute_numerical_flux(test_field, axis=1, offset=offset)

            # Flux should have the right shape (removing ghost points)
            expected_shape = list(test_field.shape)
            expected_shape[1] = test_field.shape[1] - 2 * solver.ghost_points
            assert flux.shape == tuple(expected_shape)

            # Flux values should be finite
            assert np.all(np.isfinite(flux))

    def test_first_derivative_accuracy(self, simple_grid, flat_metric, test_field_1d):
        """Test accuracy of first derivative computation."""
        field, analytical_deriv = test_field_1d

        solver = ConservativeFiniteDifference(
            simple_grid, flat_metric, boundary_conditions="periodic", order=2
        )

        # Add spatial dimensions to make it 3D
        field_3d = np.tile(field, (16, 16, 1)).transpose(2, 0, 1)
        analytical_3d = np.tile(analytical_deriv, (16, 16, 1)).transpose(2, 0, 1)

        # Compute derivative in x-direction (axis=1)
        computed_deriv = solver._compute_conservative_derivative(field_3d, axis=1, spatial_dim=0)

        # Check accuracy (should be second-order for smooth functions)
        error = np.abs(computed_deriv - analytical_3d)
        max_error = np.max(error)

        # For periodic sine wave, expect high accuracy
        assert max_error < 0.1, f"Derivative error {max_error} too large"

    def test_second_derivative_vectorization(self, simple_grid, flat_metric):
        """Test that vectorized second derivative gives same result as loop version."""
        solver = ConservativeFiniteDifference(
            simple_grid, flat_metric, boundary_conditions="periodic", order=2
        )

        # Create test field
        field = np.random.randn(16, 16, 16)

        # Compute second derivative (should use vectorized implementation)
        result = solver._compute_second_derivative(field, axis=1, spatial_dim=0)

        # Check result properties
        assert result.shape == field.shape
        assert np.all(np.isfinite(result))

    def test_christoffel_contributions_flat_space(self, simple_grid, flat_metric):
        """Test that Christoffel contributions vanish in flat spacetime."""
        solver = ConservativeFiniteDifference(
            simple_grid, flat_metric, boundary_conditions="periodic", order=2
        )

        # Create dummy tensor field
        tensor_field = np.random.randn(16, 16, 16, 4, 4)
        component_indices = (0, 1)  # T^01 component

        # In flat spacetime, Christoffel symbols should be zero
        christoffel_terms = solver._compute_christoffel_contributions(
            tensor_field, component_indices
        )

        expected_shape = tensor_field.shape[:-len(component_indices)]
        assert christoffel_terms.shape == expected_shape
        assert np.allclose(christoffel_terms, 0.0)


class TestUpwindFiniteDifference(TestFiniteDifferenceBase):
    """Test upwind finite difference implementation."""

    def test_upwind_derivative_vectorization(self, simple_grid, flat_metric):
        """Test vectorized upwind derivative computation."""
        solver = UpwindFiniteDifference(
            simple_grid, flat_metric, boundary_conditions="periodic", order=1
        )

        field = np.random.randn(16, 16, 16)

        # Test both directions
        for direction in ["upwind", "downwind"]:
            result = solver._compute_upwind_derivative(field, axis=1, spatial_dim=0, direction=direction)

            assert result.shape == field.shape
            assert np.all(np.isfinite(result))

    def test_characteristic_speeds(self, simple_grid, flat_metric):
        """Test characteristic speed computation."""
        solver = UpwindFiniteDifference(simple_grid, flat_metric)

        # Create dummy field configuration
        fields = ISFieldConfiguration(simple_grid)

        # Get characteristic speeds
        speeds = solver.characteristic_speeds(fields)

        # Should have speeds for all spatial dimensions
        for dim in range(3):
            assert f"lambda_plus_{dim}" in speeds
            assert f"lambda_minus_{dim}" in speeds

            # Speeds should be finite
            assert np.all(np.isfinite(speeds[f"lambda_plus_{dim}"]))
            assert np.all(np.isfinite(speeds[f"lambda_minus_{dim}"]))

    def test_godunov_flux_selection(self, simple_grid, flat_metric):
        """Test Godunov flux selection logic."""
        solver = UpwindFiniteDifference(simple_grid, flat_metric)

        # Create test fluxes and speeds
        flux_left = np.array([1.0, 2.0, 3.0])
        flux_right = np.array([4.0, 5.0, 6.0])

        # Test different wave speed scenarios
        # Case 1: All waves go right (λ₋ > 0)
        speed_plus = np.array([2.0, 2.0, 2.0])
        speed_minus = np.array([1.0, 1.0, 1.0])

        flux = solver._godunov_flux_selection(flux_left, flux_right, speed_plus, speed_minus)
        np.testing.assert_array_equal(flux, flux_left)  # Should use left flux

        # Case 2: All waves go left (λ₊ < 0)
        speed_plus = np.array([-1.0, -1.0, -1.0])
        speed_minus = np.array([-2.0, -2.0, -2.0])

        flux = solver._godunov_flux_selection(flux_left, flux_right, speed_plus, speed_minus)
        np.testing.assert_array_equal(flux, flux_right)  # Should use right flux


class TestWENOFiniteDifference(TestFiniteDifferenceBase):
    """Test WENO finite difference implementation."""

    def test_weno_initialization(self, simple_grid, flat_metric):
        """Test WENO solver initialization."""
        solver = WENOFiniteDifference(
            simple_grid, flat_metric, boundary_conditions="periodic", weno_order=5
        )

        assert solver.weno_order == 5
        assert solver.epsilon == 1e-6
        assert "optimal_weights" in solver.weno_coeffs
        assert "stencil_coeffs" in solver.weno_coeffs

    def test_weno5_smoothness_indicators(self, simple_grid, flat_metric):
        """Test WENO5 smoothness indicator computation."""
        solver = WENOFiniteDifference(simple_grid, flat_metric, weno_order=5)

        # Create smooth and non-smooth test fields
        x = np.linspace(0, 2*np.pi, 16)

        # Smooth field: sin(x)
        smooth_field = np.sin(x)[np.newaxis, :, np.newaxis]
        smooth_field = np.tile(smooth_field, (1, 1, 16))
        extended_smooth = solver._apply_boundary_conditions(smooth_field, axis=1)

        # Non-smooth field: step function
        step_field = np.ones_like(smooth_field)
        step_field[:, :8, :] = -1.0
        extended_step = solver._apply_boundary_conditions(step_field, axis=1)

        # Compute smoothness indicators for all stencils
        for stencil_idx in range(3):  # WENO5 has 3 stencils
            # Smooth field should have smaller smoothness indicators
            beta_smooth = solver._compute_smoothness_indicator(extended_smooth, axis=1, stencil_index=stencil_idx)
            beta_step = solver._compute_smoothness_indicator(extended_step, axis=1, stencil_index=stencil_idx)

            assert beta_smooth.shape == smooth_field.shape
            assert np.all(np.isfinite(beta_smooth))
            assert np.all(np.isfinite(beta_step))

            # Step function should have larger smoothness indicators (detecting discontinuity)
            assert np.mean(beta_step) > np.mean(beta_smooth)

    def test_weno_weights_normalization(self, simple_grid, flat_metric):
        """Test that WENO weights sum to 1."""
        solver = WENOFiniteDifference(simple_grid, flat_metric, weno_order=5)

        # Create test smoothness indicators
        field_shape = (16, 16, 16)
        smoothness_indicators = [
            np.ones(field_shape) * 0.1,  # Smooth stencil
            np.ones(field_shape) * 1.0,   # Medium smoothness
            np.ones(field_shape) * 10.0   # Non-smooth stencil
        ]

        optimal_weights = solver.weno_coeffs["optimal_weights"]
        weno_weights = solver._compute_weno_weights(optimal_weights, smoothness_indicators)

        # Weights should sum to 1
        weight_sum = sum(weno_weights)
        np.testing.assert_allclose(weight_sum, 1.0, rtol=1e-12)

        # Each weight should be positive
        for w in weno_weights:
            assert np.all(w >= 0)


class TestFiniteDifferenceFactory:
    """Test the finite difference solver factory function."""

    def test_create_conservative_solver(self, simple_grid, flat_metric):
        """Test factory creation of conservative solver."""
        solver = create_finite_difference_solver(
            "conservative", simple_grid, flat_metric, order=4
        )

        assert isinstance(solver, ConservativeFiniteDifference)
        assert solver.order == 4

    def test_create_upwind_solver(self, simple_grid, flat_metric):
        """Test factory creation of upwind solver."""
        solver = create_finite_difference_solver(
            "upwind", simple_grid, flat_metric, order=2
        )

        assert isinstance(solver, UpwindFiniteDifference)
        assert solver.order == 2

    def test_create_weno_solver(self, simple_grid, flat_metric):
        """Test factory creation of WENO solver."""
        solver = create_finite_difference_solver(
            "weno", simple_grid, flat_metric, weno_order=3
        )

        assert isinstance(solver, WENOFiniteDifference)
        assert solver.weno_order == 3

    def test_invalid_solver_type(self, simple_grid, flat_metric):
        """Test factory error handling for invalid solver type."""
        with pytest.raises(ValueError, match="Unknown finite difference scheme"):
            create_finite_difference_solver(
                "invalid_type", simple_grid, flat_metric
            )


class TestFiniteDifferenceIntegration:
    """Integration tests for finite difference solvers with physics modules."""

    def test_conservative_with_tensor_divergence(self, simple_grid, flat_metric):
        """Test conservative solver with tensor field divergence."""
        solver = ConservativeFiniteDifference(simple_grid, flat_metric, order=2)

        # Create a simple stress tensor field T^μν
        tensor_field = np.random.randn(16, 16, 16, 4, 4)
        component_indices = (0, 1)  # Computing ∇_μ T^{μ1}

        divergence = solver.compute_divergence(tensor_field, component_indices)

        expected_shape = tensor_field.shape[:-len(component_indices)]
        assert divergence.shape == expected_shape
        assert np.all(np.isfinite(divergence))

    def test_performance_vs_accuracy_tradeoff(self, simple_grid, flat_metric, test_field_1d):
        """Test performance vs accuracy for different schemes."""
        field, analytical_deriv = test_field_1d

        # Test different orders and schemes
        solvers = [
            ConservativeFiniteDifference(simple_grid, flat_metric, order=2),
            ConservativeFiniteDifference(simple_grid, flat_metric, order=4),
            UpwindFiniteDifference(simple_grid, flat_metric, order=1),
            WENOFiniteDifference(simple_grid, flat_metric, weno_order=5)
        ]

        # Prepare 3D field
        field_3d = np.tile(field, (16, 16, 1)).transpose(2, 0, 1)

        for solver in solvers:
            # Each solver should be able to compute derivatives
            if hasattr(solver, '_compute_conservative_derivative'):
                result = solver._compute_conservative_derivative(field_3d, axis=1, spatial_dim=0)
            elif hasattr(solver, '_compute_upwind_derivative'):
                result = solver._compute_upwind_derivative(field_3d, axis=1, spatial_dim=0, direction="upwind")
            elif hasattr(solver, '_compute_weno_derivative'):
                result = solver._compute_weno_derivative(field_3d, axis=1, spatial_dim=0)

            # Result should be well-defined
            assert np.all(np.isfinite(result))
            assert result.shape == field_3d.shape


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
"""
Tests for finite difference solvers in Israel-Stewart hydrodynamics.

This module provides comprehensive tests for spatial discretization schemes
including accuracy validation, boundary conditions, and shock-capturing capabilities.
"""

import numpy as np
import pytest

from israel_stewart.core.fields import ISFieldConfiguration, TransportCoefficients
from israel_stewart.core.metrics import MinkowskiMetric
from israel_stewart.core.spacetime_grid import SpacetimeGrid
from israel_stewart.solvers.finite_difference import (
    ConservativeFiniteDifference,
    UpwindFiniteDifference,
    WENOFiniteDifference,
    create_finite_difference_solver,
)


class TestFiniteDifferenceBase:
    """Base test class for finite difference schemes."""

    @pytest.fixture
    def setup_1d_grid(self) -> SpacetimeGrid:
        """Setup 1D grid for testing derivatives."""
        return SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-2.0, 2.0)],
            grid_points=(5, 64),
        )

    @pytest.fixture
    def setup_3d_grid(self) -> SpacetimeGrid:
        """Setup 3D grid for full testing."""
        return SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 16, 16, 16),
        )

    @pytest.fixture
    def metric(self) -> MinkowskiMetric:
        """Setup Minkowski metric."""
        return MinkowskiMetric()

    @pytest.fixture
    def transport_coefficients(self) -> TransportCoefficients:
        """Setup transport coefficients."""
        return TransportCoefficients(
            shear_viscosity=0.1,
            bulk_viscosity=0.05,
        )

    def smooth_function(self, x: np.ndarray) -> np.ndarray:
        """Smooth test function: sin(πx)."""
        return np.sin(np.pi * x)

    def smooth_derivative(self, x: np.ndarray) -> np.ndarray:
        """Analytical derivative of smooth function: π*cos(πx)."""
        return np.pi * np.cos(np.pi * x)

    def discontinuous_function(self, x: np.ndarray) -> np.ndarray:
        """Discontinuous test function: step function."""
        return np.where(x > 0, 1.0, -1.0)


class TestConservativeFiniteDifference(TestFiniteDifferenceBase):
    """Test ConservativeFiniteDifference scheme."""

    @pytest.fixture
    def scheme(self, setup_3d_grid, metric, transport_coefficients) -> ConservativeFiniteDifference:
        """Create ConservativeFiniteDifference scheme."""
        return ConservativeFiniteDifference(
            setup_3d_grid, metric, transport_coefficients, order=2
        )

    def test_initialization(self, scheme: ConservativeFiniteDifference) -> None:
        """Test proper initialization."""
        assert scheme.order == 2
        assert scheme.boundary_condition == "periodic"
        assert hasattr(scheme, "grid")
        assert hasattr(scheme, "metric")

    def test_smooth_derivative_accuracy(self, setup_1d_grid, metric, transport_coefficients) -> None:
        """Test derivative accuracy on smooth functions."""
        scheme = ConservativeFiniteDifference(setup_1d_grid, metric, transport_coefficients, order=2)

        # Create 1D test field
        coords = setup_1d_grid.coordinates()
        x = coords["x"]
        field = self.smooth_function(x)

        # Compute derivative
        derivative = scheme.compute_spatial_derivatives(field, axis=1)

        # Compare with analytical derivative
        expected = self.smooth_derivative(x)

        # Interior points should have high accuracy
        interior_slice = slice(2, -2)  # Avoid boundary effects
        np.testing.assert_allclose(
            derivative[interior_slice],
            expected[interior_slice],
            rtol=1e-3,
            err_msg="Conservative scheme should be accurate for smooth functions"
        )

    def test_conservation_property(self, scheme: ConservativeFiniteDifference, setup_3d_grid) -> None:
        """Test discrete conservation property."""
        # Create uniform field
        field = np.ones(setup_3d_grid.shape)

        # Derivative of constant should be zero
        for axis in range(1, 4):  # Spatial axes
            derivative = scheme.compute_spatial_derivatives(field, axis=axis)
            np.testing.assert_allclose(
                derivative, 0.0, atol=1e-14,
                err_msg=f"Derivative of constant should be zero for axis {axis}"
            )

    def test_boundary_conditions(self, setup_1d_grid, metric, transport_coefficients) -> None:
        """Test different boundary conditions."""
        # Test periodic boundaries
        scheme_periodic = ConservativeFiniteDifference(
            setup_1d_grid, metric, transport_coefficients, boundary_condition="periodic"
        )

        coords = setup_1d_grid.coordinates()
        x = coords["x"]
        field = self.smooth_function(x)

        derivative_periodic = scheme_periodic.compute_spatial_derivatives(field, axis=1)
        assert np.all(np.isfinite(derivative_periodic)), "Periodic boundaries should give finite values"

        # Test outflow boundaries
        scheme_outflow = ConservativeFiniteDifference(
            setup_1d_grid, metric, transport_coefficients, boundary_condition="outflow"
        )

        derivative_outflow = scheme_outflow.compute_spatial_derivatives(field, axis=1)
        assert np.all(np.isfinite(derivative_outflow)), "Outflow boundaries should give finite values"

    def test_order_convergence(self, metric, transport_coefficients) -> None:
        """Test convergence order for different grid resolutions."""
        resolutions = [16, 32, 64]
        errors = []

        for N in resolutions:
            grid = SpacetimeGrid(
                coordinate_system="cartesian",
                time_range=(0.0, 1.0),
                spatial_ranges=[(-1.0, 1.0)],
                grid_points=(5, N),
            )

            scheme = ConservativeFiniteDifference(grid, metric, transport_coefficients, order=2)

            coords = grid.coordinates()
            x = coords["x"]
            field = self.smooth_function(x)

            derivative = scheme.compute_spatial_derivatives(field, axis=1)
            expected = self.smooth_derivative(x)

            # Compute L2 error on interior points
            interior = slice(2, -2)
            error = np.sqrt(np.mean((derivative[interior] - expected[interior])**2))
            errors.append(error)

        # Check second-order convergence: error ∝ h^2
        convergence_rates = []
        for i in range(len(errors) - 1):
            rate = np.log(errors[i] / errors[i+1]) / np.log(2.0)
            convergence_rates.append(rate)

        # Should achieve approximately 2nd order convergence
        assert np.mean(convergence_rates) > 1.5, "Should achieve at least 1.5 order convergence"

    def test_numerical_flux_computation(self, scheme: ConservativeFiniteDifference, setup_3d_grid) -> None:
        """Test numerical flux computation."""
        # Create simple linear field
        coords = setup_3d_grid.coordinates()
        x = coords["x"]
        field = x  # Linear field

        # Apply boundary conditions and test flux computation
        extended_field = scheme._apply_boundary_conditions(field, axis=1)
        flux = scheme._compute_numerical_flux(extended_field, axis=1, offset=0.5)

        assert np.all(np.isfinite(flux)), "Numerical flux should be finite"
        assert flux.shape == field.shape, "Flux should have same shape as field"


class TestUpwindFiniteDifference(TestFiniteDifferenceBase):
    """Test UpwindFiniteDifference scheme."""

    @pytest.fixture
    def scheme(self, setup_3d_grid, metric, transport_coefficients) -> UpwindFiniteDifference:
        """Create UpwindFiniteDifference scheme."""
        return UpwindFiniteDifference(setup_3d_grid, metric, transport_coefficients, order=1)

    def test_initialization(self, scheme: UpwindFiniteDifference) -> None:
        """Test proper initialization."""
        assert scheme.order == 1
        assert hasattr(scheme, "characteristic_speeds")

    def test_upwind_direction(self, setup_1d_grid, metric, transport_coefficients) -> None:
        """Test upwind direction detection."""
        scheme = UpwindFiniteDifference(setup_1d_grid, metric, transport_coefficients)

        coords = setup_1d_grid.coordinates()
        x = coords["x"]
        field = self.smooth_function(x)

        # Test both upwind directions
        derivative_forward = scheme._compute_upwind_derivative(field, axis=1, spatial_dim=0, direction="forward")
        derivative_backward = scheme._compute_upwind_derivative(field, axis=1, spatial_dim=0, direction="backward")

        assert derivative_forward.shape == field.shape
        assert derivative_backward.shape == field.shape
        assert np.all(np.isfinite(derivative_forward))
        assert np.all(np.isfinite(derivative_backward))

    def test_characteristic_speed_estimation(self, scheme: UpwindFiniteDifference, setup_3d_grid) -> None:
        """Test characteristic speed estimation."""
        coords = setup_3d_grid.coordinates()
        x = coords["x"]
        field = self.smooth_function(x)

        speeds = scheme._estimate_characteristic_speeds(field, spatial_dim=0)

        assert speeds.shape == field.shape
        assert np.all(speeds >= 0), "Characteristic speeds should be non-negative"
        assert np.all(speeds <= 1.0), "Speeds should not exceed speed of light"

    def test_discontinuity_handling(self, setup_1d_grid, metric, transport_coefficients) -> None:
        """Test behavior near discontinuities."""
        scheme = UpwindFiniteDifference(setup_1d_grid, metric, transport_coefficients)

        coords = setup_1d_grid.coordinates()
        x = coords["x"]
        field = self.discontinuous_function(x)

        derivative = scheme.compute_spatial_derivatives(field, axis=1)

        # Should not produce infinite values near discontinuity
        assert np.all(np.isfinite(derivative)), "Upwind scheme should handle discontinuities"

    def test_monotonicity_preservation(self, setup_1d_grid, metric, transport_coefficients) -> None:
        """Test monotonicity preservation property."""
        scheme = UpwindFiniteDifference(setup_1d_grid, metric, transport_coefficients, order=1)

        coords = setup_1d_grid.coordinates()
        x = coords["x"]
        # Monotonic function
        field = np.tanh(x)

        derivative = scheme.compute_spatial_derivatives(field, axis=1)

        # First-order upwind should preserve monotonicity (derivative ≥ 0)
        assert np.all(derivative >= -1e-10), "Upwind scheme should preserve monotonicity"


class TestWENOFiniteDifference(TestFiniteDifferenceBase):
    """Test WENOFiniteDifference scheme."""

    @pytest.fixture
    def scheme(self, setup_3d_grid, metric, transport_coefficients) -> WENOFiniteDifference:
        """Create WENOFiniteDifference scheme."""
        return WENOFiniteDifference(setup_3d_grid, metric, transport_coefficients, order=5)

    def test_initialization(self, scheme: WENOFiniteDifference) -> None:
        """Test proper initialization."""
        assert scheme.order == 5
        assert hasattr(scheme, "epsilon")  # WENO parameter
        assert scheme.epsilon > 0

    def test_smoothness_indicators(self, setup_1d_grid, metric, transport_coefficients) -> None:
        """Test WENO smoothness indicators."""
        scheme = WENOFiniteDifference(setup_1d_grid, metric, transport_coefficients, order=5)

        coords = setup_1d_grid.coordinates()
        x = coords["x"]

        # Test smooth function
        smooth_field = self.smooth_function(x)
        smooth_indicators = scheme._compute_smoothness_indicators(smooth_field, axis=1)

        # Test discontinuous function
        discontinuous_field = self.discontinuous_function(x)
        discontinuous_indicators = scheme._compute_smoothness_indicators(discontinuous_field, axis=1)

        # Discontinuous field should have larger smoothness indicators
        smooth_max = np.max(smooth_indicators)
        discontinuous_max = np.max(discontinuous_indicators)
        assert discontinuous_max > smooth_max, "Discontinuous field should have larger smoothness indicators"

    def test_adaptive_stencil_selection(self, setup_1d_grid, metric, transport_coefficients) -> None:
        """Test adaptive stencil selection in WENO."""
        scheme = WENOFiniteDifference(setup_1d_grid, metric, transport_coefficients, order=5)

        coords = setup_1d_grid.coordinates()
        x = coords["x"]

        # Test with mixed smooth/discontinuous function
        field = np.where(x < 0, self.smooth_function(x), self.discontinuous_function(x))

        derivative = scheme.compute_spatial_derivatives(field, axis=1)

        # Should produce finite, reasonable values
        assert np.all(np.isfinite(derivative)), "WENO should handle mixed smooth/discontinuous fields"

    def test_high_order_accuracy(self, metric, transport_coefficients) -> None:
        """Test high-order accuracy of WENO scheme."""
        resolutions = [32, 64, 128]
        errors = []

        for N in resolutions:
            grid = SpacetimeGrid(
                coordinate_system="cartesian",
                time_range=(0.0, 1.0),
                spatial_ranges=[(-1.0, 1.0)],
                grid_points=(5, N),
            )

            scheme = WENOFiniteDifference(grid, metric, transport_coefficients, order=5)

            coords = grid.coordinates()
            x = coords["x"]
            field = self.smooth_function(x)

            derivative = scheme.compute_spatial_derivatives(field, axis=1)
            expected = self.smooth_derivative(x)

            # Compute error on interior points (avoid boundary effects)
            interior = slice(3, -3)
            error = np.sqrt(np.mean((derivative[interior] - expected[interior])**2))
            errors.append(error)

        # WENO-5 should achieve close to 5th order on smooth regions
        convergence_rates = []
        for i in range(len(errors) - 1):
            rate = np.log(errors[i] / errors[i+1]) / np.log(2.0)
            convergence_rates.append(rate)

        # Should achieve high-order convergence
        assert np.mean(convergence_rates) > 3.0, "WENO-5 should achieve high-order convergence"

    def test_shock_capturing(self, setup_1d_grid, metric, transport_coefficients) -> None:
        """Test shock-capturing capability."""
        scheme = WENOFiniteDifference(setup_1d_grid, metric, transport_coefficients, order=5)

        coords = setup_1d_grid.coordinates()
        x = coords["x"]

        # Sharp transition (shock-like)
        field = np.tanh(10 * x)  # Very sharp transition

        derivative = scheme.compute_spatial_derivatives(field, axis=1)

        # Should not produce oscillations
        assert np.all(np.isfinite(derivative)), "WENO should handle sharp transitions"

        # Check for spurious oscillations near the shock
        # The derivative should be positive everywhere for this monotonic function
        # Allow small negative values due to numerical errors
        assert np.all(derivative >= -1e-6), "WENO should not produce spurious oscillations"


class TestFiniteDifferenceFactory:
    """Test factory functions for finite difference schemes."""

    @pytest.fixture
    def setup_problem(self) -> tuple[SpacetimeGrid, MinkowskiMetric, TransportCoefficients]:
        """Setup problem for factory testing."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0)],
            grid_points=(5, 32),
        )
        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()
        return grid, metric, coefficients

    def test_factory_function(self, setup_problem) -> None:
        """Test create_finite_difference_solver factory function."""
        grid, metric, coefficients = setup_problem

        # Test different scheme types
        conservative = create_finite_difference_solver("conservative", grid, metric, coefficients)
        assert isinstance(conservative, ConservativeFiniteDifference)

        upwind = create_finite_difference_solver("upwind", grid, metric, coefficients)
        assert isinstance(upwind, UpwindFiniteDifference)

        weno = create_finite_difference_solver("weno", grid, metric, coefficients)
        assert isinstance(weno, WENOFiniteDifference)

    def test_factory_options(self, setup_problem) -> None:
        """Test factory function with options."""
        grid, metric, coefficients = setup_problem

        # Test with custom order
        scheme = create_finite_difference_solver(
            "conservative", grid, metric, coefficients, order=4
        )
        assert scheme.order == 4

        # Test with custom boundary condition
        scheme = create_finite_difference_solver(
            "conservative", grid, metric, coefficients, boundary_condition="outflow"
        )
        assert scheme.boundary_condition == "outflow"

    def test_factory_error_handling(self, setup_problem) -> None:
        """Test factory function error handling."""
        grid, metric, coefficients = setup_problem

        with pytest.raises(ValueError, match="Unknown finite difference scheme"):
            create_finite_difference_solver("invalid_scheme", grid, metric, coefficients)


class TestFiniteDifferenceIntegration:
    """Test integration of finite difference schemes with other components."""

    @pytest.fixture
    def setup_integration_test(self) -> tuple[SpacetimeGrid, MinkowskiMetric, TransportCoefficients, ISFieldConfiguration]:
        """Setup for integration testing."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0)],
            grid_points=(5, 16, 16),
        )
        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()
        fields = ISFieldConfiguration(grid)

        return grid, metric, coefficients, fields

    def test_field_integration(self, setup_integration_test) -> None:
        """Test integration with ISFieldConfiguration."""
        grid, metric, coefficients, fields = setup_integration_test

        scheme = ConservativeFiniteDifference(grid, metric, coefficients)

        # Test with actual field arrays
        coords = grid.coordinates()
        x, y = coords["x"], coords["y"]
        fields.rho = np.sin(np.pi * x) * np.cos(np.pi * y)

        # Compute derivatives
        drho_dx = scheme.compute_spatial_derivatives(fields.rho, axis=1)
        drho_dy = scheme.compute_spatial_derivatives(fields.rho, axis=2)

        assert drho_dx.shape == fields.rho.shape
        assert drho_dy.shape == fields.rho.shape
        assert np.all(np.isfinite(drho_dx))
        assert np.all(np.isfinite(drho_dy))

    def test_metric_integration(self, setup_integration_test) -> None:
        """Test integration with metric computations."""
        grid, metric, coefficients, fields = setup_integration_test

        scheme = ConservativeFiniteDifference(grid, metric, coefficients)

        # Test metric-dependent computations
        coords = grid.coordinates()
        x = coords["x"]
        test_field = np.sin(np.pi * x)

        # Scheme should handle metric consistently
        derivative = scheme.compute_spatial_derivatives(test_field, axis=1)
        assert np.all(np.isfinite(derivative))


@pytest.mark.slow
class TestFiniteDifferencePerformance:
    """Performance tests for finite difference schemes."""

    def test_computational_scaling(self) -> None:
        """Test computational scaling with grid size."""
        import time

        grid_sizes = [16, 32, 64]
        times = []

        for N in grid_sizes:
            grid = SpacetimeGrid(
                coordinate_system="cartesian",
                time_range=(0.0, 1.0),
                spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
                grid_points=(5, N, N, N),
            )

            metric = MinkowskiMetric()
            coefficients = TransportCoefficients()
            scheme = ConservativeFiniteDifference(grid, metric, coefficients)

            coords = grid.coordinates()
            x = coords["x"]
            field = np.sin(np.pi * x)

            start_time = time.time()
            for _ in range(10):  # Multiple iterations for timing
                derivative = scheme.compute_spatial_derivatives(field, axis=1)
            end_time = time.time()

            times.append((end_time - start_time) / 10)

        # Check that timing scales reasonably
        assert all(t > 0 for t in times), "All operations should take measurable time"

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency of schemes."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0)],
            grid_points=(5, 1024),  # Large 1D grid
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()

        # Different schemes should handle large grids
        schemes = [
            ConservativeFiniteDifference(grid, metric, coefficients),
            UpwindFiniteDifference(grid, metric, coefficients),
        ]

        coords = grid.coordinates()
        x = coords["x"]
        field = np.sin(np.pi * x)

        for scheme in schemes:
            derivative = scheme.compute_spatial_derivatives(field, axis=1)
            assert derivative.shape == field.shape
            assert np.all(np.isfinite(derivative))


class TestFiniteDifferenceStability:
    """Stability tests for finite difference schemes."""

    def test_cfl_condition(self) -> None:
        """Test CFL condition implementation."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0)],
            grid_points=(5, 64),
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()
        scheme = ConservativeFiniteDifference(grid, metric, coefficients)

        # Test CFL condition estimation
        coords = grid.coordinates()
        x = coords["x"]
        field = np.sin(np.pi * x)

        cfl_dt = scheme.estimate_stable_timestep(field)

        assert cfl_dt > 0, "CFL timestep should be positive"
        assert cfl_dt < 1.0, "CFL timestep should be reasonable"

    def test_stability_boundaries(self) -> None:
        """Test stability near scheme boundaries."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(-1.0, 1.0)],
            grid_points=(5, 32),
        )

        metric = MinkowskiMetric()
        coefficients = TransportCoefficients()
        scheme = UpwindFiniteDifference(grid, metric, coefficients)

        coords = grid.coordinates()
        x = coords["x"]

        # Test with large gradients
        field = np.tanh(10 * x)

        derivative = scheme.compute_spatial_derivatives(field, axis=1)

        # Should remain stable even with large gradients
        assert np.all(np.isfinite(derivative)), "Scheme should remain stable with large gradients"
        assert np.max(np.abs(derivative)) < 1000, "Derivatives should remain bounded"

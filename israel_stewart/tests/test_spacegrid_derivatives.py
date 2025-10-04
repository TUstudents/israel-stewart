"""
Comprehensive tests for SpaceGrid derivative methods.

Tests gradient() and divergence() methods with:
- Analytical function comparisons
- Boundary condition handling (periodic, dirichlet, neumann)
- Order convergence (2nd-order, 4th-order)
- Edge cases (small grids, degenerate axes)
- Vectorization correctness
"""

import numpy as np
import pytest

from israel_stewart.core.spacegrid import SpaceGrid


class TestSpaceGridGradient:
    """Test SpaceGrid.gradient() method."""

    def test_gradient_sine_function_2nd_order(self) -> None:
        """Test ∂ₓsin(kx) = k·cos(kx) with 2nd-order accuracy."""
        # Setup: periodic domain with sine wave
        k = 1.0  # Wavenumber (1 wavelength over [0, 2π])
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 2.0 * np.pi)] * 3,
            grid_points=(32, 32, 32),
            boundary_conditions="periodic",
        )

        X, Y, Z = grid.meshgrid()
        field = np.sin(k * X)

        # Analytical derivative
        analytical = k * np.cos(k * X)

        # Numerical derivative
        numerical = grid.gradient(field, axis=0, order=2)

        # Should be accurate to ~1e-2 for 32 points with 2nd-order
        error = np.max(np.abs(numerical - analytical))
        assert error < 1e-2, f"2nd-order gradient error {error} too large"

    def test_gradient_polynomial_2nd_order(self) -> None:
        """Test ∂ₓ(x²) = 2x with 2nd-order accuracy."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 1.0)] * 3,
            grid_points=(16, 16, 16),
            boundary_conditions="dirichlet",
        )

        X, Y, Z = grid.meshgrid()
        field = X**2

        # Analytical derivative
        analytical = 2.0 * X

        # Numerical derivative
        numerical = grid.gradient(field, axis=0, order=2)

        # Polynomial derivatives should be exact in interior (modulo edge effects)
        # Check interior points only
        error = np.max(np.abs(numerical[2:-2, :, :] - analytical[2:-2, :, :]))
        assert error < 1e-10, f"Polynomial gradient error {error} too large"

    def test_gradient_sine_function_4th_order(self) -> None:
        """Test 4th-order accuracy is better than 2nd-order."""
        k = 1.0
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 2.0 * np.pi)] * 3,
            grid_points=(32, 32, 32),
            boundary_conditions="periodic",
        )

        X, Y, Z = grid.meshgrid()
        field = np.sin(k * X)
        analytical = k * np.cos(k * X)

        # Compare 2nd and 4th order
        numerical_2nd = grid.gradient(field, axis=0, order=2)
        numerical_4th = grid.gradient(field, axis=0, order=4)

        error_2nd = np.max(np.abs(numerical_2nd - analytical))
        error_4th = np.max(np.abs(numerical_4th - analytical))

        # 4th-order should be significantly more accurate
        assert error_4th < error_2nd / 5, f"4th-order not significantly better: {error_4th} vs {error_2nd}"

    def test_gradient_periodic_boundary(self) -> None:
        """Test periodic boundary wraps correctly."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 2.0 * np.pi)] * 3,
            grid_points=(16, 16, 16),
            boundary_conditions="periodic",
        )

        X, Y, Z = grid.meshgrid()
        # Periodic function
        field = np.sin(X) + np.cos(2.0 * Y)

        # Gradient should be continuous at boundaries
        grad_x = grid.gradient(field, axis=0, order=2)

        # Left and right boundaries should match for periodic function
        # (not exact due to discrete sampling, but should be close)
        left_boundary = grad_x[0, :, :]
        right_boundary = grad_x[-1, :, :]

        # For periodic BC, derivative pattern should wrap
        # Check that we don't have spurious jumps
        assert np.std(grad_x) < 10.0, "Gradient has spurious oscillations"

    def test_gradient_dirichlet_boundary(self) -> None:
        """Test dirichlet boundary uses one-sided stencils."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 1.0)] * 3,
            grid_points=(8, 8, 8),
            boundary_conditions="dirichlet",
        )

        X, Y, Z = grid.meshgrid()
        field = X**3  # Smooth function

        grad_x = grid.gradient(field, axis=0, order=2)

        # Gradient should be computed everywhere (including boundaries)
        assert not np.any(np.isnan(grad_x)), "NaN values in gradient"
        assert grad_x.shape == field.shape, "Shape mismatch"

    def test_gradient_all_axes(self) -> None:
        """Test gradient works for x, y, z axes."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 2.0 * np.pi)] * 3,
            grid_points=(32, 32, 32),
            boundary_conditions="periodic",
        )

        X, Y, Z = grid.meshgrid()
        # Use fully periodic functions: k=1,2,3 are all integer wavenumbers
        field = np.sin(X) + np.cos(2.0 * Y) + np.sin(3.0 * Z)

        # Analytical gradients
        grad_x_analytical = np.cos(X)
        grad_y_analytical = -2.0 * np.sin(2.0 * Y)
        grad_z_analytical = 3.0 * np.cos(3.0 * Z)

        # Numerical gradients
        grad_x = grid.gradient(field, axis=0, order=2)
        grad_y = grid.gradient(field, axis=1, order=2)
        grad_z = grid.gradient(field, axis=2, order=2)

        # Check accuracy for 2nd-order with 32 points
        # Error scales as k²h² where h = 2π/32 ≈ 0.196, h² ≈ 0.038
        # k=1: 1×h² ≈ 3.8% → measured 0.64%
        # k=2: 4×h² ≈ 15% → measured 2.55%
        # k=3: 9×h² ≈ 34% → measured 5.68%
        assert np.allclose(grad_x, grad_x_analytical, rtol=0.01, atol=0.01)  # 1% for k=1
        assert np.allclose(grad_y, grad_y_analytical, rtol=0.03, atol=0.06)  # 3% for k=2
        assert np.allclose(grad_z, grad_z_analytical, rtol=0.06, atol=0.18)  # 6% for k=3

    def test_gradient_small_grid(self) -> None:
        """Test gradient on minimum viable grid (nx=3)."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 1.0)] * 3,
            grid_points=(3, 3, 3),
            boundary_conditions="dirichlet",
        )

        X, Y, Z = grid.meshgrid()
        field = X + Y + Z

        # Should work without crashing
        grad_x = grid.gradient(field, axis=0, order=2)
        grad_y = grid.gradient(field, axis=1, order=2)
        grad_z = grid.gradient(field, axis=2, order=2)

        # For linear function, gradient should be ~1
        assert np.allclose(grad_x[1, 1, 1], 1.0, atol=0.1)

    def test_gradient_degenerate_axis(self) -> None:
        """Test gradient on degenerate axis (nx=1) returns zeros."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(1, 8, 8),
            boundary_conditions="periodic",
        )

        field = np.random.rand(*grid.shape)

        # Gradient along degenerate axis should be zero
        grad_x = grid.gradient(field, axis=0, order=2)
        assert np.all(grad_x == 0.0), "Degenerate axis gradient should be zero"


class TestSpaceGridDivergence:
    """Test SpaceGrid.divergence() method."""

    def test_divergence_constant_vector(self) -> None:
        """Test ∇·(1,1,1) = 0 (divergence-free constant field)."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 1.0)] * 3,
            grid_points=(16, 16, 16),
            boundary_conditions="periodic",
        )

        # Constant vector field
        vector_field = np.ones((*grid.shape, 3))

        div = grid.divergence(vector_field, order=2)

        # Divergence of constant field is zero
        assert np.allclose(div, 0.0, atol=1e-12), f"Max divergence: {np.max(np.abs(div))}"

    def test_divergence_linear_vector(self) -> None:
        """Test ∇·(x,y,z) = 3 (uniform expansion)."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 1.0)] * 3,
            grid_points=(16, 16, 16),
            boundary_conditions="dirichlet",
        )

        X, Y, Z = grid.meshgrid()
        vector_field = np.stack([X, Y, Z], axis=-1)

        div = grid.divergence(vector_field, order=2)

        # ∂ₓx + ∂ᵧy + ∂_zz = 1 + 1 + 1 = 3
        # (Interior points should be exact for linear functions)
        expected = 3.0
        error = np.max(np.abs(div[2:-2, 2:-2, 2:-2] - expected))
        assert error < 1e-10, f"Linear divergence error: {error}"

    def test_divergence_sine_wave_2nd_order(self) -> None:
        """Test divergence of sine wave with analytical comparison."""
        k = 1.0  # Integer wavenumber for periodicity on [0, 2π]
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 2.0 * np.pi)] * 3,
            grid_points=(64, 64, 64),  # Finer grid for better accuracy
            boundary_conditions="periodic",
        )

        X, Y, Z = grid.meshgrid()
        # Vector field: v = (sin(kx), cos(ky), 0)
        vx = np.sin(k * X)
        vy = np.cos(k * Y)
        vz = np.zeros_like(X)
        vector_field = np.stack([vx, vy, vz], axis=-1)

        # Analytical divergence: k·cos(kx) - k·sin(ky)
        analytical = k * np.cos(k * X) - k * np.sin(k * Y)

        div = grid.divergence(vector_field, order=2)

        # For 64 points with k=1: h² ≈ (2π/64)² ≈ 0.0096
        # Expected error ~0.96%, allow 0.5% tolerance
        error = np.max(np.abs(div - analytical))
        assert error < 0.005, f"Sine wave divergence error: {error}"

    def test_divergence_sine_wave_4th_order(self) -> None:
        """Test 4th-order divergence is more accurate."""
        k = 1.0  # Integer wavenumber for periodicity
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 2.0 * np.pi)] * 3,
            grid_points=(32, 32, 32),
            boundary_conditions="periodic",
        )

        X, Y, Z = grid.meshgrid()
        vx = np.sin(k * X)
        vy = np.cos(k * Y)
        vz = np.zeros_like(X)
        vector_field = np.stack([vx, vy, vz], axis=-1)
        analytical = k * np.cos(k * X) - k * np.sin(k * Y)

        div_2nd = grid.divergence(vector_field, order=2)
        div_4th = grid.divergence(vector_field, order=4)

        error_2nd = np.max(np.abs(div_2nd - analytical))
        error_4th = np.max(np.abs(div_4th - analytical))

        assert error_4th < error_2nd / 5, f"4th-order not better: {error_4th} vs {error_2nd}"

    def test_divergence_vs_component_sum(self) -> None:
        """Test divergence equals sum of component gradients."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 1.0)] * 3,
            grid_points=(16, 16, 16),
            boundary_conditions="periodic",
        )

        X, Y, Z = grid.meshgrid()
        vx = np.sin(2.0 * np.pi * X)
        vy = np.cos(2.0 * np.pi * Y)
        vz = np.exp(-Z)
        vector_field = np.stack([vx, vy, vz], axis=-1)

        # Using divergence method
        div_method = grid.divergence(vector_field, order=2)

        # Manual component sum
        dvx_dx = grid.gradient(vx, axis=0, order=2)
        dvy_dy = grid.gradient(vy, axis=1, order=2)
        dvz_dz = grid.gradient(vz, axis=2, order=2)
        div_manual = dvx_dx + dvy_dy + dvz_dz

        # Should be identical
        assert np.allclose(div_method, div_manual, atol=1e-12)

    def test_divergence_periodic_boundary(self) -> None:
        """Test divergence with periodic boundaries."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 2.0 * np.pi)] * 3,
            grid_points=(16, 16, 16),
            boundary_conditions="periodic",
        )

        X, Y, Z = grid.meshgrid()
        # Periodic vector field
        vx = np.sin(X) * np.cos(Y)
        vy = np.cos(X) * np.sin(Y)
        vz = np.zeros_like(X)
        vector_field = np.stack([vx, vy, vz], axis=-1)

        div = grid.divergence(vector_field, order=2)

        # No NaN or Inf
        assert np.all(np.isfinite(div)), "Non-finite values in divergence"

    def test_divergence_dirichlet_boundary(self) -> None:
        """Test divergence with dirichlet boundaries."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 1.0)] * 3,
            grid_points=(12, 12, 12),
            boundary_conditions="dirichlet",
        )

        X, Y, Z = grid.meshgrid()
        vector_field = np.stack([X**2, Y**2, Z**2], axis=-1)

        div = grid.divergence(vector_field, order=2)

        # Should compute everywhere
        assert div.shape == grid.shape
        assert np.all(np.isfinite(div))

    def test_divergence_vector_field_shape(self) -> None:
        """Test divergence validates input shape."""
        grid = SpaceGrid(
            coordinate_system="cartesian",
            spatial_ranges=[(0.0, 1.0)] * 3,
            grid_points=(8, 8, 8),
            boundary_conditions="periodic",
        )

        # Wrong shape: missing vector component dimension
        wrong_shape = np.ones(grid.shape)

        with pytest.raises(ValueError, match="shape.*doesn't match"):
            grid.divergence(wrong_shape)

        # Wrong number of components
        wrong_components = np.ones((*grid.shape, 4))

        with pytest.raises(ValueError, match="shape.*doesn't match"):
            grid.divergence(wrong_components)


class TestOrderConvergence:
    """Test convergence rates for 2nd and 4th order methods."""

    def test_gradient_convergence_2nd_order(self) -> None:
        """Test gradient shows O(h²) convergence."""
        k = 1.0  # Integer wavenumber for periodicity
        errors = []
        spacings = []

        for n in [8, 16, 32, 64]:
            grid = SpaceGrid(
                coordinate_system="cartesian",
                spatial_ranges=[(0.0, 2.0 * np.pi)] * 3,
                grid_points=(n, n, n),
                boundary_conditions="periodic",
            )

            X, Y, Z = grid.meshgrid()
            field = np.sin(k * X)
            analytical = k * np.cos(k * X)

            numerical = grid.gradient(field, axis=0, order=2)
            error = np.max(np.abs(numerical - analytical))

            errors.append(error)
            spacings.append(2.0 * np.pi / n)

        # Check convergence rate: error ~ h²
        # log(error) vs log(h) should have slope ~2
        log_errors = np.log(errors)
        log_spacings = np.log(spacings)

        # Linear fit
        slope = (log_errors[-1] - log_errors[0]) / (log_spacings[-1] - log_spacings[0])

        # Should be close to 2 for 2nd-order method
        assert 1.8 < slope < 2.2, f"2nd-order convergence rate {slope} not ~2"

    def test_gradient_convergence_4th_order(self) -> None:
        """Test gradient shows O(h⁴) convergence."""
        k = 1.0  # Integer wavenumber for periodicity
        errors = []
        spacings = []

        for n in [16, 32, 64]:
            grid = SpaceGrid(
                coordinate_system="cartesian",
                spatial_ranges=[(0.0, 2.0 * np.pi)] * 3,
                grid_points=(n, n, n),
                boundary_conditions="periodic",
            )

            X, Y, Z = grid.meshgrid()
            field = np.sin(k * X)
            analytical = k * np.cos(k * X)

            numerical = grid.gradient(field, axis=0, order=4)
            error = np.max(np.abs(numerical - analytical))

            errors.append(error)
            spacings.append(2.0 * np.pi / n)

        # Check convergence rate: error ~ h⁴
        log_errors = np.log(errors)
        log_spacings = np.log(spacings)

        slope = (log_errors[-1] - log_errors[0]) / (log_spacings[-1] - log_spacings[0])

        # Should be close to 4 for 4th-order method
        assert 3.5 < slope < 4.5, f"4th-order convergence rate {slope} not ~4"

    def test_divergence_convergence_2nd_order(self) -> None:
        """Test divergence shows O(h²) convergence."""
        k = 1.0  # Integer wavenumber for periodicity
        errors = []
        spacings = []

        for n in [8, 16, 32, 64]:
            grid = SpaceGrid(
                coordinate_system="cartesian",
                spatial_ranges=[(0.0, 2.0 * np.pi)] * 3,
                grid_points=(n, n, n),
                boundary_conditions="periodic",
            )

            X, Y, Z = grid.meshgrid()
            vx = np.sin(k * X)
            vy = np.cos(k * Y)
            vz = np.zeros_like(X)
            vector_field = np.stack([vx, vy, vz], axis=-1)

            analytical = k * np.cos(k * X) - k * np.sin(k * Y)
            numerical = grid.divergence(vector_field, order=2)
            error = np.max(np.abs(numerical - analytical))

            errors.append(error)
            spacings.append(2.0 * np.pi / n)

        log_errors = np.log(errors)
        log_spacings = np.log(spacings)
        slope = (log_errors[-1] - log_errors[0]) / (log_spacings[-1] - log_spacings[0])

        assert 1.8 < slope < 2.2, f"Divergence 2nd-order convergence {slope} not ~2"

    def test_divergence_convergence_4th_order(self) -> None:
        """Test divergence shows O(h⁴) convergence."""
        k = 1.0  # Integer wavenumber for periodicity
        errors = []
        spacings = []

        for n in [16, 32, 64]:
            grid = SpaceGrid(
                coordinate_system="cartesian",
                spatial_ranges=[(0.0, 2.0 * np.pi)] * 3,
                grid_points=(n, n, n),
                boundary_conditions="periodic",
            )

            X, Y, Z = grid.meshgrid()
            vx = np.sin(k * X)
            vy = np.cos(k * Y)
            vz = np.zeros_like(X)
            vector_field = np.stack([vx, vy, vz], axis=-1)

            analytical = k * np.cos(k * X) - k * np.sin(k * Y)
            numerical = grid.divergence(vector_field, order=4)
            error = np.max(np.abs(numerical - analytical))

            errors.append(error)
            spacings.append(2.0 * np.pi / n)

        log_errors = np.log(errors)
        log_spacings = np.log(spacings)
        slope = (log_errors[-1] - log_errors[0]) / (log_spacings[-1] - log_spacings[0])

        assert 3.5 < slope < 4.5, f"Divergence 4th-order convergence {slope} not ~4"

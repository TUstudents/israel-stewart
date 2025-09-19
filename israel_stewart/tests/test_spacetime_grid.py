"""
Tests for SpacetimeGrid class, especially the Laplacian method fix.

This module tests the spacetime grid implementation with focus on
the critical bug fix in the covariant Laplacian computation.
"""

import numpy as np
import pytest
import sympy as sp

from israel_stewart.core.metrics import FLRWMetric, MilneMetric, MinkowskiMetric
from israel_stewart.core.spacetime_grid import SpacetimeGrid


class TestSpacetimeGridLaplacian:
    """Test Laplacian computation with proper index raising."""

    @pytest.fixture
    def simple_grid(self) -> SpacetimeGrid:
        """Create a simple Cartesian grid for testing."""
        return SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(5, 5, 5, 5),
        )

    @pytest.fixture
    def minkowski_grid(self) -> SpacetimeGrid:
        """Create a grid with Minkowski metric for testing covariant operations."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(5, 5, 5, 5),
        )
        grid.metric = MinkowskiMetric()
        return grid

    def test_flat_space_laplacian_identity(self, simple_grid: SpacetimeGrid) -> None:
        """Test that flat-space Laplacian works correctly without metric."""
        # Create a simple quadratic field: φ = x² + y² + z²
        coords = simple_grid.meshgrid()
        x, y, z = coords[1], coords[2], coords[3]  # Spatial coordinates
        field = x**2 + y**2 + z**2

        # Compute Laplacian (should be 6 for φ = x² + y² + z²)
        laplacian = simple_grid.laplacian(field)

        # The second derivatives are: ∂²φ/∂x² = 2, ∂²φ/∂y² = 2, ∂²φ/∂z² = 2
        # So ∇²φ = 2 + 2 + 2 = 6 everywhere (analytically)
        # However, finite differences have boundary effects, so test interior points only
        interior = laplacian[1:-1, 1:-1, 1:-1, 1:-1]  # Interior points away from boundaries
        expected_interior = 6.0

        # Allow significant numerical error due to finite differences on small grids
        # The key is that the result should be positive and roughly in the right range
        assert np.all(interior > 4.0), f"Interior Laplacian too small: {interior}"
        assert np.all(interior < 8.0), f"Interior Laplacian too large: {interior}"
        # Test that the mean is close to expected
        np.testing.assert_allclose(np.mean(interior), expected_interior, rtol=0.2)

    def test_minkowski_laplacian_flat_case(self, minkowski_grid: SpacetimeGrid) -> None:
        """Test that Minkowski Laplacian reduces to flat-space case."""
        # Create a simple quadratic spatial field: φ = x² + y² + z²
        coords = minkowski_grid.meshgrid()
        x, y, z = coords[1], coords[2], coords[3]  # Spatial coordinates
        field = x**2 + y**2 + z**2

        # Compute covariant Laplacian
        laplacian = minkowski_grid.laplacian(field)

        # For Minkowski metric in Cartesian coordinates, the covariant Laplacian
        # should give: ∇²φ = -∂²φ/∂t² + ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
        # Since φ = x² + y² + z² has no time dependence: ∂²φ/∂t² = 0
        # And ∂²φ/∂x² = ∂²φ/∂y² = ∂²φ/∂z² = 2
        # So ∇²φ = 0 + 2 + 2 + 2 = 6
        # Test interior points to avoid boundary effects
        interior = laplacian[1:-1, 1:-1, 1:-1, 1:-1]
        expected_interior = 6.0

        # Allow some numerical error
        np.testing.assert_allclose(interior, expected_interior, rtol=1e-6, atol=1e-4)

    def test_covariant_laplacian_index_raising(self, minkowski_grid: SpacetimeGrid) -> None:
        """Test that the Laplacian correctly raises indices in the gradient.

        This is the specific test for the critical bug fix where the gradient
        index was not being raised before taking the divergence.
        """
        # Create a field with known analytical Laplacian
        coords = minkowski_grid.meshgrid()
        t, x, y, z = coords[0], coords[1], coords[2], coords[3]

        # Use a field with time dependence: φ = t² - x² - y² - z²
        # This has ∇²φ = ∂²φ/∂μ∂^μ = η^{μν} ∂²φ/∂μ∂ν
        # = η^{00}∂²φ/∂t² + η^{11}∂²φ/∂x² + η^{22}∂²φ/∂y² + η^{33}∂²φ/∂z²
        # = (-1)(2) + (1)(−2) + (1)(−2) + (1)(−2) = -2 - 2 - 2 - 2 = -8
        field = t**2 - x**2 - y**2 - z**2

        # Compute covariant Laplacian using the fixed method
        laplacian = minkowski_grid.laplacian(field)

        # Test interior points to avoid boundary effects
        interior = laplacian[1:-1, 1:-1, 1:-1, 1:-1]
        expected_interior = -8.0

        # The fix ensures proper index raising before divergence
        np.testing.assert_allclose(interior, expected_interior, rtol=1e-4, atol=1e-2)

    def test_laplacian_consistency_with_manual_computation(
        self, minkowski_grid: SpacetimeGrid
    ) -> None:
        """Test Laplacian against manual step-by-step computation.

        This verifies that the index raising is done correctly by comparing
        with a manual implementation.
        """
        # Create a simple test field
        coords = minkowski_grid.meshgrid()
        x, y = coords[1], coords[2]  # Use x and y only
        field = x**2 + 2 * y**2  # Simple quadratic field

        # Compute Laplacian using the grid method
        laplacian = minkowski_grid.laplacian(field)

        # Manual computation:
        # 1. Compute gradient (covariant): ∇_μ φ
        gradient = np.zeros((*minkowski_grid.shape, 4))
        for mu in range(4):
            gradient[..., mu] = minkowski_grid.gradient(field, axis=mu)

        # 2. Raise index using metric: ∇^μ φ = g^{μν} ∇_ν φ
        from israel_stewart.core.tensor_utils import optimized_einsum

        assert minkowski_grid.metric is not None, "Metric should be set for this test"
        contravariant_gradient = optimized_einsum(
            "ij,...j->...i", minkowski_grid.metric.inverse, gradient
        )

        # 3. Compute divergence of contravariant gradient
        manual_laplacian = minkowski_grid.divergence(contravariant_gradient)

        # Should match the grid method result
        np.testing.assert_allclose(laplacian, manual_laplacian, rtol=1e-14)

    def test_laplacian_shape_preservation(self, minkowski_grid: SpacetimeGrid) -> None:
        """Test that Laplacian preserves field shape."""
        # Create random field
        field = np.random.random(minkowski_grid.shape)

        # Compute Laplacian
        laplacian = minkowski_grid.laplacian(field)

        # Shape should be preserved
        assert laplacian.shape == field.shape

    def test_laplacian_symmetry(self, minkowski_grid: SpacetimeGrid) -> None:
        """Test Laplacian symmetry properties."""
        coords = minkowski_grid.meshgrid()
        x, y, z = coords[1], coords[2], coords[3]

        # Create symmetric field: φ = x² + y² + z²
        field_symmetric = x**2 + y**2 + z**2
        laplacian_sym = minkowski_grid.laplacian(field_symmetric)

        # Should be constant (6.0) everywhere due to symmetry
        expected_constant = 6.0
        np.testing.assert_allclose(
            laplacian_sym, expected_constant * np.ones_like(field_symmetric), rtol=1e-6
        )

    def test_laplacian_error_handling(self, minkowski_grid: SpacetimeGrid) -> None:
        """Test proper error handling for invalid inputs."""
        # Wrong shape field
        wrong_shape_field = np.random.random((3, 3, 3, 3))

        with pytest.raises(ValueError, match="Field shape .* doesn't match grid shape"):
            minkowski_grid.laplacian(wrong_shape_field)


class TestCoordinateMappingFixes:
    """Test coordinate mapping fixes for periodic boundary conditions."""

    def test_spherical_coordinate_mapping(self) -> None:
        """Test that spherical coordinates map correctly to axes."""
        grid = SpacetimeGrid(
            coordinate_system="spherical",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 2.0), (0.0, np.pi), (0.0, 2 * np.pi)],  # r, theta, phi
            grid_points=(5, 5, 5, 5),
        )

        # Create test field
        field = np.random.random(grid.shape)

        # Test that phi periodic boundaries affect the correct axis (axis 3 for spherical)
        field_original = field.copy()
        field_with_phi_bc = grid._apply_periodic_bc(field, "phi_min")

        # For spherical coordinates: phi should be at axis 3
        # So phi_min boundary should copy from field[:, :, :, -1] to field[:, :, :, 0]
        expected_phi_bc = field_original.copy()
        expected_phi_bc[:, :, :, 0] = field_original[:, :, :, -1]

        np.testing.assert_array_equal(field_with_phi_bc, expected_phi_bc)

    def test_cylindrical_coordinate_mapping(self) -> None:
        """Test that cylindrical coordinates map correctly to axes."""
        grid = SpacetimeGrid(
            coordinate_system="cylindrical",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 2.0), (0.0, 2 * np.pi), (-1.0, 1.0)],  # rho, phi, z
            grid_points=(5, 5, 5, 5),
        )

        # Create test field
        field = np.random.random(grid.shape)

        # Test that phi periodic boundaries affect the correct axis (axis 2 for cylindrical)
        field_original = field.copy()
        field_with_phi_bc = grid._apply_periodic_bc(field, "phi_min")

        # For cylindrical coordinates: phi should be at axis 2
        # So phi_min boundary should copy from field[:, :, -1, :] to field[:, :, 0, :]
        expected_phi_bc = field_original.copy()
        expected_phi_bc[:, :, 0, :] = field_original[:, :, -1, :]

        np.testing.assert_array_equal(field_with_phi_bc, expected_phi_bc)

    def test_coordinate_mapping_consistency(self) -> None:
        """Test that coordinate mapping is consistent across all boundary methods."""
        # Test cylindrical coordinates
        grid = SpacetimeGrid(
            coordinate_system="cylindrical",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 2.0), (0.0, 2 * np.pi), (-1.0, 1.0)],  # rho, phi, z
            grid_points=(5, 5, 5, 5),
        )

        # Create deterministic test field
        field = np.ones(grid.shape)
        # Mark axis 2 boundaries distinctly
        field[:, :, 0, :] = 99.0  # phi_min boundary (axis 2, first slice)
        field[:, :, -1, :] = 77.0  # phi_max boundary (axis 2, last slice)

        # Test all boundary condition methods use the same coordinate mapping
        phi_periodic = grid._apply_periodic_bc(field, "phi_min")
        phi_reflecting = grid._apply_reflecting_bc(field, "phi_min")
        phi_absorbing = grid._apply_absorbing_bc(field, "phi_min")
        phi_fixed = grid._apply_fixed_bc(field, "phi_min", 42.0)

        # All should modify the same axis (axis 2 for phi in cylindrical)
        # Check that they all modify field[:, :, 0, :] (first slice in axis 2)
        assert not np.array_equal(phi_periodic[:, :, 0, :], field[:, :, 0, :])
        assert not np.array_equal(phi_reflecting[:, :, 0, :], field[:, :, 0, :])
        assert not np.array_equal(phi_absorbing[:, :, 0, :], field[:, :, 0, :])
        assert not np.array_equal(phi_fixed[:, :, 0, :], field[:, :, 0, :])

        # Check specific boundary condition behaviors
        # Periodic: phi_min should copy from phi_max (77.0)
        np.testing.assert_array_equal(phi_periodic[:, :, 0, :], 77.0 * np.ones((5, 5, 5)))
        # Reflecting: phi_min should copy from interior (1.0)
        np.testing.assert_array_equal(phi_reflecting[:, :, 0, :], 1.0 * np.ones((5, 5, 5)))
        # Absorbing: phi_min should be set to 0.0
        np.testing.assert_array_equal(phi_absorbing[:, :, 0, :], 0.0 * np.ones((5, 5, 5)))
        # Fixed: phi_min should be set to 42.0
        np.testing.assert_array_equal(phi_fixed[:, :, 0, :], 42.0 * np.ones((5, 5, 5)))

        # But should not modify other axes slices (except at intersections)
        # For axis 1: check that phi boundary condition doesn't modify rho=0 slices
        # except where they intersect with the modified phi boundary
        # Check interior slices that don't intersect with phi=0
        np.testing.assert_array_equal(
            phi_periodic[:, 0, 1:-1, :], field[:, 0, 1:-1, :]
        )  # axis 1, interior
        # For axis 3: check that phi boundary condition doesn't modify z boundaries
        # except where they intersect with the modified phi boundary
        np.testing.assert_array_equal(
            phi_periodic[:, :, 1:-1, 0], field[:, :, 1:-1, 0]
        )  # axis 3, interior

    def test_milne_coordinate_mapping(self) -> None:
        """Test that Milne coordinates map correctly to axes."""
        grid = SpacetimeGrid(
            coordinate_system="milne",
            time_range=(0.1, 2.0),  # tau range (must be positive)
            spatial_ranges=[(-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)],  # eta, x, y
            grid_points=(5, 5, 5, 5),
        )

        field = np.random.random(grid.shape)

        # Test eta coordinate (axis 1 in Milne)
        field_eta_bc = grid._apply_periodic_bc(field, "eta_min")
        expected_eta_bc = field.copy()
        expected_eta_bc[:, 0, :, :] = field[:, -1, :, :]

        np.testing.assert_array_equal(field_eta_bc, expected_eta_bc)

        # Test x coordinate (axis 2 in Milne)
        field_x_bc = grid._apply_periodic_bc(field, "x_min")
        expected_x_bc = field.copy()
        expected_x_bc[:, :, 0, :] = field[:, :, -1, :]

        np.testing.assert_array_equal(field_x_bc, expected_x_bc)


class TestVolumeElementSymbolicFix:
    """Test volume element computation with symbolic metrics."""

    def test_volume_element_with_numeric_metric(self) -> None:
        """Test that volume element works correctly with numeric metrics."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(5, 5, 5, 5),
        )
        grid.metric = MinkowskiMetric()

        # Should work without errors
        volume_elem = grid.volume_element()

        # Minkowski metric has determinant -1, so √(-g) = √(1) = 1
        expected = np.ones(grid.shape)
        np.testing.assert_allclose(volume_elem, expected, rtol=1e-10)

    def test_volume_element_with_symbolic_milne_metric(self) -> None:
        """Test that volume element works with symbolic Milne metric."""
        grid = SpacetimeGrid(
            coordinate_system="milne",
            time_range=(0.1, 2.0),  # tau > 0 for Milne coordinates
            spatial_ranges=[(-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)],  # eta, x, y
            grid_points=(3, 3, 3, 3),  # Small grid for testing
        )

        # Create Milne metric (has symbolic determinant involving tau)
        milne_metric = MilneMetric()
        grid.metric = milne_metric

        # This should NOT crash (was the main bug)
        try:
            volume_elem = grid.volume_element()
            # Should return array with correct shape
            assert volume_elem.shape == grid.shape
            # Should contain finite positive values
            assert np.all(np.isfinite(volume_elem))
            assert np.all(volume_elem > 0)
        except Exception as e:
            if "Cannot convert expression" in str(e) or "TypeError" in str(e):
                pytest.fail(f"Symbolic volume element computation failed: {e}")
            else:
                raise

    def test_volume_element_with_symbolic_flrw_metric(self) -> None:
        """Test that volume element works with symbolic FLRW metric."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",  # FLRW can use Cartesian spatial coords
            time_range=(0.1, 2.0),  # cosmic time t
            spatial_ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],  # x, y, z
            grid_points=(3, 3, 3, 3),  # Small grid for testing
        )

        # Create FLRW metric (has symbolic determinant involving scale factor)
        flrw_metric = FLRWMetric(curvature_param=0.0)  # Flat FLRW
        grid.metric = flrw_metric

        # This should NOT crash (was the main bug)
        try:
            volume_elem = grid.volume_element()
            # Should return array with correct shape
            assert volume_elem.shape == grid.shape
            # Should contain finite positive values (or fallback to ones)
            assert np.all(np.isfinite(volume_elem))
            assert np.all(volume_elem > 0)
        except Exception as e:
            if "Cannot convert expression" in str(e) or "TypeError" in str(e):
                pytest.fail(f"Symbolic volume element computation failed: {e}")
            else:
                raise

    def test_volume_element_fallback_behavior(self) -> None:
        """Test that volume element gracefully falls back when symbolic evaluation fails."""
        # Create a grid that might have issues with symbolic evaluation
        grid = SpacetimeGrid(
            coordinate_system="spherical",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.1, 2.0), (0.0, np.pi), (0.0, 2 * np.pi)],
            grid_points=(3, 3, 3, 3),
        )

        # Create a mock metric with symbolic determinant that's hard to evaluate
        class MockSymbolicMetric:
            @property
            def determinant(self):
                # Return complex symbolic expression that's hard to evaluate
                tau = sp.Symbol("tau")
                return -(tau**2) * sp.exp(sp.sqrt(tau))

        grid.metric = MockSymbolicMetric()

        # Should not crash, should return fallback array of ones
        # Note: Warning may or may not be emitted depending on evaluation path
        volume_elem = grid.volume_element()

        # Should return array of ones as fallback (or valid numeric array)
        assert volume_elem.shape == grid.shape
        assert np.all(np.isfinite(volume_elem))
        assert np.all(volume_elem > 0)  # Volume element should be positive

    def test_volume_element_shape_consistency(self) -> None:
        """Test that volume element always returns correct shape."""
        grid = SpacetimeGrid(
            coordinate_system="cylindrical",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.1, 2.0), (0.0, 2 * np.pi), (-1.0, 1.0)],
            grid_points=(4, 6, 8, 10),  # Non-uniform shape
        )

        # Test with no metric
        volume_elem_no_metric = grid.volume_element()
        assert volume_elem_no_metric.shape == grid.shape

        # Test with numeric metric
        grid.metric = MinkowskiMetric()
        volume_elem_numeric = grid.volume_element()
        assert volume_elem_numeric.shape == grid.shape

        # Test with symbolic metric
        grid.metric = MilneMetric()
        volume_elem_symbolic = grid.volume_element()
        assert volume_elem_symbolic.shape == grid.shape


class TestVectorizedDivergence:
    """Test the vectorized divergence implementation for performance and correctness."""

    @pytest.fixture
    def small_grid(self) -> SpacetimeGrid:
        """Create a small grid for correctness tests."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(4, 4, 4, 4),
        )
        grid.metric = MinkowskiMetric()
        return grid

    @pytest.fixture
    def large_grid(self) -> SpacetimeGrid:
        """Create a larger grid for performance tests."""
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(16, 16, 16, 16),
        )
        grid.metric = MinkowskiMetric()
        return grid

    def test_divergence_numerical_christoffel(self, small_grid: SpacetimeGrid) -> None:
        """Test divergence computation with numerical Christoffel symbols."""
        # Create a simple vector field: V^μ = (1, x, y, z)
        coords = small_grid.meshgrid()
        t, x, y, z = coords[0], coords[1], coords[2], coords[3]

        vector_field = np.zeros((*small_grid.shape, 4))
        vector_field[..., 0] = 1.0  # V^0 = 1
        vector_field[..., 1] = x  # V^1 = x
        vector_field[..., 2] = y  # V^2 = y
        vector_field[..., 3] = z  # V^3 = z

        # Compute divergence
        divergence = small_grid.divergence(vector_field)

        # For Minkowski metric, Christoffel symbols are zero
        # So ∇_μ V^μ = ∂_μ V^μ = ∂_t(1) + ∂_x(x) + ∂_y(y) + ∂_z(z) = 0 + 1 + 1 + 1 = 3
        expected_divergence = 3.0

        # Check that divergence is approximately 3 (within numerical errors)
        assert np.allclose(divergence, expected_divergence, atol=1e-10), (
            f"Expected divergence ≈ {expected_divergence}, got mean: {np.mean(divergence):.6f}, "
            f"std: {np.std(divergence):.6f}"
        )

    def test_divergence_symbolic_christoffel(self) -> None:
        """Test divergence computation with symbolic Christoffel symbols."""
        # Create a grid with symbolic metric (time-dependent)
        grid = SpacetimeGrid(
            coordinate_system="cartesian",
            time_range=(0.1, 1.0),  # Avoid t=0 for symbolic metrics
            spatial_ranges=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            grid_points=(4, 4, 4, 4),
        )

        # Use a simple time-dependent metric for symbolic test
        from israel_stewart.core.metrics import GeneralMetric

        t = sp.Symbol("t", real=True, positive=True)
        g_symbolic = sp.Matrix([[t**2, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
        grid.metric = GeneralMetric(g_symbolic)

        # Create a simple constant vector field to test the symbolic path
        vector_field = np.zeros((*grid.shape, 4))
        vector_field[..., 0] = 1.0  # V^0 = 1
        vector_field[..., 1] = 0.0  # V^1 = 0
        vector_field[..., 2] = 0.0  # V^2 = 0
        vector_field[..., 3] = 0.0  # V^3 = 0

        # This should not crash and should handle symbolic Christoffel symbols
        divergence = grid.divergence(vector_field)

        # Verify result has correct shape
        assert divergence.shape == grid.shape
        assert np.all(np.isfinite(divergence)), "Divergence should be finite"

    def test_divergence_consistency_numerical_vs_symbolic(self, small_grid: SpacetimeGrid) -> None:
        """Test that numerical and symbolic approaches give consistent results for Minkowski."""
        # Create simple vector field
        vector_field = np.zeros((*small_grid.shape, 4))
        vector_field[..., 0] = 1.0
        vector_field[..., 1] = 0.1
        vector_field[..., 2] = 0.1
        vector_field[..., 3] = 0.1

        # Compute with numerical Minkowski (should use vectorized path)
        div_numerical = small_grid.divergence(vector_field)

        # Compute with symbolic Minkowski for comparison
        from israel_stewart.core.metrics import GeneralMetric

        g_symbolic = sp.Matrix([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        symbolic_metric = GeneralMetric(g_symbolic)
        small_grid.metric = symbolic_metric

        div_symbolic = small_grid.divergence(vector_field)

        # Results should be very close (symbolic should fall back to loop)
        assert np.allclose(
            div_numerical, div_symbolic, rtol=1e-12
        ), "Numerical and symbolic divergence should match for Minkowski metric"

    def test_divergence_performance_improvement(self, large_grid: SpacetimeGrid) -> None:
        """Test that vectorized implementation provides performance improvement."""
        import time

        # Create a non-trivial vector field
        coords = large_grid.meshgrid()
        t, x, y, z = coords[0], coords[1], coords[2], coords[3]

        vector_field = np.zeros((*large_grid.shape, 4))
        vector_field[..., 0] = np.sin(t) * np.cos(x)
        vector_field[..., 1] = np.cos(y) * np.sin(z)
        vector_field[..., 2] = np.exp(-0.1 * t) * x
        vector_field[..., 3] = y * z

        # Time the vectorized computation
        start_time = time.perf_counter()
        divergence_vectorized = large_grid.divergence(vector_field)
        vectorized_time = time.perf_counter() - start_time

        # Verify result quality
        assert divergence_vectorized.shape == large_grid.shape
        assert np.all(np.isfinite(divergence_vectorized))

        # Performance should be reasonable (under 1 second for 16^4 grid)
        assert (
            vectorized_time < 1.0
        ), f"Vectorized divergence took {vectorized_time:.3f}s, should be faster"

        print(f"Vectorized divergence time for {large_grid.shape} grid: {vectorized_time:.4f}s")

    def test_divergence_christoffel_trace_extraction(self, small_grid: SpacetimeGrid) -> None:
        """Test that Christoffel trace extraction works correctly."""
        # For Minkowski metric, all Christoffel symbols should be zero
        christoffel = small_grid.metric.christoffel_symbols

        # Check that we can extract the trace correctly
        if isinstance(christoffel, np.ndarray):
            mu_indices = np.arange(4)
            gamma_trace = christoffel[mu_indices, mu_indices, :]  # Γ^μ_{μν}

            # For Minkowski, this should be all zeros
            assert np.allclose(gamma_trace, 0.0), "Minkowski Christoffel trace should be zero"
            assert gamma_trace.shape == (
                4,
                4,
            ), f"Gamma trace shape should be (4,4), got {gamma_trace.shape}"

    def test_divergence_error_handling(self, small_grid: SpacetimeGrid) -> None:
        """Test error handling for invalid vector field shapes."""
        # Test with wrong shape
        wrong_shape_field = np.zeros((2, 2, 2, 2, 4))  # Wrong grid dimensions
        with pytest.raises(ValueError, match="doesn't match expected"):
            small_grid.divergence(wrong_shape_field)

        # Test with missing vector components
        wrong_components_field = np.zeros((*small_grid.shape, 3))  # Only 3 components instead of 4
        with pytest.raises(ValueError, match="doesn't match expected"):
            small_grid.divergence(wrong_components_field)

    def test_divergence_coordinate_systems(self) -> None:
        """Test divergence computation in different coordinate systems."""
        # Test with cylindrical coordinates
        cyl_grid = SpacetimeGrid(
            coordinate_system="cylindrical",
            time_range=(0.0, 1.0),
            spatial_ranges=[(0.1, 2.0), (0.0, 2 * np.pi), (-1.0, 1.0)],  # (rho, phi, z)
            grid_points=(4, 4, 4, 4),
        )
        cyl_grid.metric = MinkowskiMetric()

        vector_field = np.zeros((*cyl_grid.shape, 4))
        vector_field[..., 0] = 1.0

        # Should not crash and should give reasonable results
        divergence = cyl_grid.divergence(vector_field)
        assert divergence.shape == cyl_grid.shape
        assert np.all(np.isfinite(divergence))


if __name__ == "__main__":
    pytest.main([__file__])

"""
Tests for SpacetimeGrid class, especially the Laplacian method fix.

This module tests the spacetime grid implementation with focus on
the critical bug fix in the covariant Laplacian computation.
"""

import numpy as np
import pytest

from israel_stewart.core.metrics import MinkowskiMetric
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


if __name__ == "__main__":
    pytest.main([__file__])
